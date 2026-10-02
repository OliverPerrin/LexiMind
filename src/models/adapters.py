"""Opt-in native LoRA and same-base effective-delta merging.

All specialists must clone one complete, identically laid-out initialization,
including private heads. Tensor hashes do not establish label semantics, data
admission, or a training budget. Those remain external study requirements.
No checkpoint/model is loaded, trained, or promoted by this library.

LoRA: https://arxiv.org/abs/2106.09685
Task arithmetic: https://arxiv.org/abs/2212.04089v3
TIES: https://arxiv.org/abs/2306.01708v2, Algorithm 1.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadAttention
from .heads import ClassificationHead, TokenClassificationHead


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1 or type(self.seed) is not int:
            raise ValueError("LoRA requires a positive integer rank and integer seed")
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if (
            isinstance(self.dropout, bool)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("LoRA dropout must be finite and in [0, 1)")


class LoRALinear(nn.Linear):
    """Retain native weight/bias names; add only low-rank factors and dropout."""

    def __init__(self, base: nn.Linear, config: LoRAConfig, *, seed: int):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(config.rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(initial.to(device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, config.rank)))
        self.lora_dropout = nn.Dropout(config.dropout)
        self.train(base.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.lora_dropout(inputs), self.lora_A), self.lora_B)
        return F.linear(inputs, self.weight, self.bias) + update * (
            self.config.alpha / self.config.rank
        )

    def effective_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * (self.config.alpha / self.config.rank)


@dataclass(frozen=True)
class AdapterBinding:
    config: LoRAConfig
    shared_projections: tuple[str, ...]
    private_projections: tuple[str, ...]
    private_heads: tuple[str, ...]
    private_parameters: tuple[str, ...]
    base_sha256: str
    architecture_sha256: str
    frozen_sha256: str


@dataclass(frozen=True)
class EffectiveDelta:
    task_id: str
    binding: AdapterBinding
    shared: dict[str, torch.Tensor]
    private_deltas: dict[str, torch.Tensor]
    private_state: dict[str, torch.Tensor]

    def metadata(self) -> dict:
        """JSON-safe binding and payload digests; tensors stay in a state dict."""
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "binding": asdict(self.binding),
            **{
                name: {
                    key: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _tensor_hash(value),
                    }
                    for key, value in getattr(self, name).items()
                }
                for name in ("shared", "private_deltas", "private_state")
            },
        }


@dataclass(frozen=True)
class MergedDelta:
    shared: dict[str, torch.Tensor]
    specialists: tuple[EffectiveDelta, ...]
    method: str
    weights: tuple[float, ...]
    density: float
    scale: float


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    if value.layout != torch.strided or value.device.type == "meta" or value.is_quantized:
        raise ValueError("Adapter evidence requires materialized, dense, unquantized state")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Adapter state and deltas must be finite")
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _base_state(model: nn.Module) -> dict[str, torch.Tensor]:
    factors = {
        f"{name}.{factor}"
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for factor in ("lora_A", "lora_B")
    }
    return {
        name: value
        for name, value in model.state_dict(keep_vars=True).items()
        if name not in factors
    }


def _fingerprints(model: nn.Module, private_parameters: Sequence[str] = ()) -> tuple[str, str, str]:
    state = _base_state(model)
    adapted = {name for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    modules = []
    for name, module in model.named_modules(remove_duplicate=False):
        if any(name.startswith(parent + ".") for parent in adapted):
            continue
        kind = nn.Linear if isinstance(module, LoRALinear) else type(module)
        attributes = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_")
            and key != "training"
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        modules.append(
            (name, f"{kind.__module__}.{kind.__qualname__}", module.extra_repr(), attributes)
        )
    first_names: dict[int, str] = {}
    layout = {
        name: (list(value.shape), str(value.dtype), first_names.setdefault(id(value), name))
        for name, value in state.items()
    }
    architecture = _digest({"modules": modules, "layout": layout})
    hashes = {name: _tensor_hash(value) for name, value in state.items()}
    full = _digest({"architecture": architecture, "tensors": hashes})
    frozen = _digest(
        {name: value for name, value in hashes.items() if name not in private_parameters}
    )
    return full, architecture, frozen


def _reject_mutable_aliases(model: nn.Module, mutable: Sequence[str]) -> None:
    aliases: dict[tuple, list[str]] = {}
    for name, value in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        if value.layout != torch.strided or value.device.type == "meta" or value.is_quantized:
            raise ValueError("Adapters require dense unquantized native parameters")
        key = (value.device, value.untyped_storage().data_ptr())
        if value.numel():
            aliases.setdefault(key, []).append(name)
    if any(len(names) > 1 and set(names).intersection(mutable) for names in aliases.values()):
        raise ValueError(
            "Adapted projections/private heads must not alias other parameters or buffers"
        )


def _projection(model: nn.Module, path: str, scope: str, *, adapted: bool = False) -> nn.Linear:
    parent_name, _, leaf = path.rpartition(".")
    if not path.startswith(scope + ".") or leaf not in {"W_Q", "W_K", "W_V", "W_O"}:
        raise ValueError(f"Explicit {scope} attention projection paths are required")
    parent = model.get_submodule(parent_name)
    projection = model.get_submodule(path)
    expected = LoRALinear if adapted else nn.Linear
    if not isinstance(parent, MultiHeadAttention) or type(projection) is not expected:
        raise ValueError(
            "Only native dense attention Linear projections are supported; quantized/subclass targets are rejected"
        )
    if parent.use_lora:
        raise ValueError("Legacy attention LoRA cannot be combined with projection adapters")
    if not projection.weight.is_floating_point() or projection.weight.is_quantized:
        raise ValueError("LoRA targets must be unquantized floating weights")
    return projection


def _private_parameters(model: nn.Module, heads: Sequence[str]) -> tuple[str, ...]:
    parameters = []
    for name in heads:
        head = model.get_submodule(name)
        if name.split(".")[0] in {"encoder", "decoder"} or type(head) not in {
            ClassificationHead,
            TokenClassificationHead,
        }:
            raise ValueError("Private heads must be explicitly named native classifier heads")
        names = [
            f"{name}.{child + '.' if child else ''}{parameter}"
            for child, module in head.named_modules()
            if type(module) is nn.Linear
            for parameter, _ in module.named_parameters(recurse=False)
        ]
        if not names:
            raise ValueError("A private head needs native Linear classifier/pooler parameters")
        parameters.extend(names)
    return tuple(parameters)


def attach_lora(
    model: nn.Module,
    *,
    shared_projections: Sequence[str],
    config: LoRAConfig,
    private_projections: Sequence[str] = (),
    private_heads: Sequence[str] = (),
) -> AdapterBinding:
    """Attach before optimizer creation; all checks/allocation precede mutation.

    Shared scope is encoder attention; decoder attention and selected native
    classifier/pooler Linear parameters are private. Everything else is frozen.
    """
    shared, private, heads = (
        tuple(shared_projections),
        tuple(private_projections),
        tuple(private_heads),
    )
    paths = shared + private + heads
    if not shared or len(set(paths)) != len(paths) or any(not name for name in paths):
        raise ValueError("Provide unique nonempty allowlists and at least one shared projection")
    if any(second.startswith(first + ".") for first in paths for second in paths):
        raise ValueError("Parent/child adapter allowlists must not overlap")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Attach LoRA only once to a pristine base")
    projections = {
        name: _projection(model, name, scope)
        for scope, names in (("encoder", shared), ("decoder", private))
        for name in names
    }
    private_parameters = _private_parameters(model, heads)
    mutable = tuple(f"{name}.weight" for name in projections) + tuple(private_parameters)
    _reject_mutable_aliases(model, mutable)
    full, architecture, frozen = _fingerprints(model, private_parameters)
    replacements = {
        name: LoRALinear(module, config, seed=config.seed + index)
        for index, (name, module) in enumerate(projections.items())
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, replacement in replacements.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
    for name in private_parameters:
        model.get_parameter(name).requires_grad_(True)
    binding = AdapterBinding(
        config, shared, private, heads, tuple(private_parameters), full, architecture, frozen
    )
    # A private head can legitimately change during adaptation, so the later
    # frozen-state fingerprint cannot recover its initialization. Keep the
    # original attachment receipt outside state_dict; restoring adapter weights
    # requires recreating attachment on the same full base first.
    model._leximind_adapter_binding = binding
    return binding


@torch.no_grad()
def extract_effective_delta(
    model: nn.Module, binding: AdapterBinding, *, task_id: str
) -> EffectiveDelta:
    """Extract alpha/r * B@A, checking frozen base evidence; never merge factors."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("A nonempty specialist task_id is required")
    if (
        not isinstance(binding, AdapterBinding)
        or getattr(model, "_leximind_adapter_binding", None) != binding
    ):
        raise ValueError("Extraction requires the exact binding recorded at adapter attachment")
    _, architecture, frozen = _fingerprints(model, binding.private_parameters)
    if architecture != binding.architecture_sha256 or frozen != binding.frozen_sha256:
        raise ValueError("Specialist architecture or frozen base changed after adapter attachment")
    outputs = []
    for scope, paths in (
        ("encoder", binding.shared_projections),
        ("decoder", binding.private_projections),
    ):
        values = {}
        for name in paths:
            module = _projection(model, name, scope, adapted=True)
            if module.config != binding.config or module.lora_dropout.p != binding.config.dropout:
                raise ValueError("Adapter configuration changed after attachment")
            if (
                module.lora_A.shape != (binding.config.rank, module.in_features)
                or module.lora_B.shape != (module.out_features, binding.config.rank)
                or module.lora_A.dtype != module.weight.dtype
                or module.lora_B.dtype != module.weight.dtype
            ):
                raise ValueError(
                    "LoRA factor shapes/dtypes differ from the bound native projection"
                )
            value = module.effective_delta().detach().cpu().clone()
            _tensor_hash(value)
            values[name + ".weight"] = value
        outputs.append(values)
    state = {
        name: model.get_parameter(name).detach().cpu().clone()
        for name in binding.private_parameters
    }
    _reject_mutable_aliases(model, tuple(outputs[0]) + tuple(outputs[1]) + tuple(state))
    return EffectiveDelta(task_id, binding, outputs[0], outputs[1], state)


def _compatible(specialists: Sequence[EffectiveDelta]) -> tuple[EffectiveDelta, ...]:
    values = tuple(specialists)
    if not values or len({value.task_id for value in values}) != len(values):
        raise ValueError("Merging requires uniquely named specialist deltas")
    first = values[0]
    for value in values:
        if (
            value.binding.base_sha256,
            value.binding.architecture_sha256,
            value.binding.config,
            value.binding.shared_projections,
            tuple(value.shared),
        ) != (
            first.binding.base_sha256,
            first.binding.architecture_sha256,
            first.binding.config,
            first.binding.shared_projections,
            tuple(first.shared),
        ):
            raise ValueError(
                "Specialists must share the exact full base, layout, LoRA config, and shared allowlist"
            )
        if set(value.shared) != {name + ".weight" for name in value.binding.shared_projections}:
            raise ValueError("Shared effective deltas must match the bound projection allowlist")
        for name, tensor in value.shared.items():
            _tensor_hash(tensor)
            if tensor.ndim != 2 or not tensor.is_floating_point():
                raise ValueError("Effective weight deltas must be floating matrices")
            if tensor.shape != first.shared[name].shape or tensor.dtype != first.shared[name].dtype:
                raise ValueError("Shared delta shapes and dtypes must match")
        if set(value.private_deltas) != {
            name + ".weight" for name in value.binding.private_projections
        } or set(value.private_state) != set(value.binding.private_parameters):
            raise ValueError("Private payload differs from its bound allowlists")
        for tensor in (*value.private_deltas.values(), *value.private_state.values()):
            _tensor_hash(tensor)
    return values


@torch.no_grad()
def merge_deltas(
    specialists: Sequence[EffectiveDelta],
    *,
    method: str,
    weights: Sequence[float] | None = None,
    density: float = 1.0,
    scale: float = 1.0,
) -> MergedDelta:
    """Merge shared effective matrices; private components remain task-indexed.

    TIES trims globally within each specialist's shared vector, retaining
    ceil(density*N) entries. Equal magnitudes prefer earlier allowlist/flat
    indices. Exact sign-vote ties produce zero; aligned means exclude zeros.
    These discrete tie choices are explicit. TIES requires unweighted votes.
    """
    values = _compatible(specialists)
    if method not in {"task_arithmetic", "ties"}:
        raise ValueError("Unknown merge method")
    if not math.isfinite(scale) or not math.isfinite(density) or not 0 < density <= 1:
        raise ValueError("Merge scale must be finite and density must be in (0, 1]")
    coefficients = tuple(weights) if weights is not None else (1.0,) * len(values)
    if len(coefficients) != len(values) or any(
        not math.isfinite(weight) for weight in coefficients
    ):
        raise ValueError("Provide one finite task-arithmetic coefficient per specialist")
    if method == "ties" and weights is not None:
        raise ValueError("TIES uses unweighted sign votes; choose only density and scale")
    if method == "task_arithmetic" and density != 1:
        raise ValueError("Task arithmetic does not trim deltas; density must be one")
    names = tuple(values[0].shared)
    vectors = torch.stack(
        [
            torch.cat([value.shared[name].detach().cpu().reshape(-1) for name in names])
            for value in values
        ]
    )
    # Preserve fp64, otherwise merge in fp32 and cast back to the base dtype.
    working = vectors if vectors.dtype == torch.float64 else vectors.float()
    if method == "task_arithmetic":
        merged = (working * working.new_tensor(coefficients)[:, None]).sum(dim=0) * scale
    else:
        keep = math.ceil(density * working.shape[1])
        indices = working.abs().argsort(dim=1, descending=True, stable=True)[:, :keep]
        trimmed = torch.zeros_like(working).scatter_(1, indices, working.gather(1, indices))
        elected = trimmed.sum(dim=0).sign()
        aligned = (trimmed.sign() == elected) & (trimmed != 0)
        merged = (trimmed * aligned).sum(dim=0) / aligned.sum(dim=0).clamp(min=1) * scale
    merged = merged.to(vectors.dtype)
    _tensor_hash(merged)
    shared, offset = {}, 0
    for name in names:
        template = values[0].shared[name]
        shared[name] = (
            merged[offset : offset + template.numel()]
            .reshape(template.shape)
            .to(template.dtype)
            .clone()
        )
        offset += template.numel()
    return MergedDelta(shared, values, method, coefficients, density, scale)


@torch.no_grad()
def materialize_merge(
    base: nn.Module, merged: MergedDelta, *, private_tasks: Sequence[str] = ()
) -> nn.Module:
    """Return an independent dense copy with legacy keys; never mutate the base.

    Caller selects which specialist's private components to retain. Overlapping
    private destinations are rejected, not averaged or silently overwritten.
    """
    specialists = _compatible(merged.specialists)
    if any(isinstance(module, LoRALinear) for module in base.modules()):
        raise ValueError("Materialization requires the pristine unadapted full base")
    full, architecture, _ = _fingerprints(base)
    if (
        full != specialists[0].binding.base_sha256
        or architecture != specialists[0].binding.architecture_sha256
    ):
        raise ValueError("Materialization base does not match the bound full initialization")
    for name in specialists[0].binding.shared_projections:
        _projection(base, name, "encoder")
    if len(set(private_tasks)) != len(private_tasks):
        raise ValueError("Select each private specialist only once")
    by_task = {value.task_id: value for value in specialists}
    updates = dict(merged.shared)
    if set(updates) != set(specialists[0].shared):
        raise ValueError("Merged shared keys differ from the bound allowlist")
    absolute: set[str] = set()
    for task in private_tasks:
        if task not in by_task:
            raise ValueError("Unknown private specialist")
        value = by_task[task]
        for name in value.binding.private_projections:
            _projection(base, name, "decoder")
        if (
            _private_parameters(base, value.binding.private_heads)
            != value.binding.private_parameters
        ):
            raise ValueError(
                "Private parameter inventory differs from the native classifier allowlist"
            )
        if set(value.private_deltas) != {
            name + ".weight" for name in value.binding.private_projections
        } or set(value.private_state) != set(value.binding.private_parameters):
            raise ValueError("Private payload differs from its bound allowlists")
        private = value.private_deltas | value.private_state
        if set(updates).intersection(private):
            raise ValueError("Selected private specialist components overlap")
        updates.update(private)
        absolute.update(value.private_state)
    _reject_mutable_aliases(base, tuple(updates))
    ready = {}
    for name, value in updates.items():
        target = base.get_parameter(name)
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError("Materialized delta/state shapes and dtypes must match the base")
        _tensor_hash(value)
        prepared = value.to(target.device) if name in absolute else target + value.to(target.device)
        _tensor_hash(prepared)
        ready[name] = prepared
    result = deepcopy(base)
    for name, value in ready.items():
        result.get_parameter(name).copy_(value)
    return result
