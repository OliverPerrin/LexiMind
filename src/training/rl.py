"""Auditable policy objectives over caller-scored response token log probabilities.

This module collects no rollouts and admits no reward sources. Masks describe
response tokens only: include the first sampled EOS, exclude prompt and padding.
The rollout/scoring adapter must establish those token boundaries and verify the
evidence identified by RewardProvenance; hashes alone do not establish truth.

References: Dr.GRPO https://arxiv.org/abs/2503.20783v2;
DAPO https://arxiv.org/abs/2503.14476v2;
GSPO https://arxiv.org/abs/2507.18071v2;
BPO https://arxiv.org/abs/2609.15987v1;
RLP https://arxiv.org/abs/2510.01265v2;
sampled reference penalty https://arxiv.org/abs/2402.03300v3;
offline DPO https://arxiv.org/abs/2305.18290v3.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RewardProvenance:
    verifier_id: str
    verifier_revision: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.verifier_id,
                self.verifier_revision,
            )
        ):
            raise ValueError("Reward provenance requires a verifier ID and revision")
        if not isinstance(self.evidence_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.evidence_sha256
        ):
            raise ValueError("Reward evidence_sha256 must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class RolloutContract:
    behavior_policy_revision: str
    sampling_temperature: float
    reward_provenance: tuple[RewardProvenance, ...]
    sampling_top_p: float = 1.0
    sampling_top_k: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.behavior_policy_revision, str)
            or not self.behavior_policy_revision.strip()
        ):
            raise ValueError("An explicit behavior policy revision is required")
        _positive(self.sampling_temperature, "sampling_temperature")
        _positive(self.sampling_top_p, "sampling_top_p")
        if (
            self.sampling_top_p != 1.0
            or type(self.sampling_top_k) is not int
            or self.sampling_top_k != 0
        ):
            raise ValueError(
                "Only full-support temperature sampling is supported (top_p=1, top_k=0)"
            )
        if not isinstance(self.reward_provenance, tuple):
            raise ValueError("reward_provenance must be an immutable tuple")


@dataclass(frozen=True)
class GroupRelativeConfig:
    max_response_tokens: int
    method: str = "dr_grpo"
    clip_low: float | None = None
    clip_high: float | None = None
    kl_coefficient: float = 0.0
    bpo_epsilon: float = 0.1
    bpo_weight_cap: float = 3.0

    def __post_init__(self) -> None:
        defaults = {"dr_grpo": (0.2, 0.2), "dapo": (0.2, 0.28), "gspo": (3e-4, 4e-4)}
        if self.method not in {*defaults, "bpo_experimental", "rlp"}:
            raise ValueError("Unknown group-relative method")
        if type(self.max_response_tokens) is not int or self.max_response_tokens < 1:
            raise ValueError("max_response_tokens must be a positive integer")
        if self.method in {"bpo_experimental", "rlp"} and (
            self.clip_low is None or self.clip_high is None
        ):
            raise ValueError(
                "BPO/RLP requires explicit clipping bounds; the paper does not specify defaults"
            )
        for index, name in enumerate(("clip_low", "clip_high")):
            if getattr(self, name) is None:
                object.__setattr__(self, name, defaults[self.method][index])
        assert self.clip_low is not None and self.clip_high is not None
        _positive(self.clip_low, "clip_low", allow_zero=True)
        _positive(self.clip_high, "clip_high", allow_zero=True)
        if self.clip_low >= 1:
            raise ValueError("clip_low must be in [0, 1)")
        _positive(self.kl_coefficient, "kl_coefficient", allow_zero=True)
        if self.method in {"bpo_experimental", "rlp"} and self.kl_coefficient:
            raise ValueError("BPO/RLP follows the paper's no-KL objective")
        _positive(self.bpo_epsilon, "bpo_epsilon")
        _positive(self.bpo_weight_cap, "bpo_weight_cap")


@dataclass(frozen=True)
class PolicyObjective:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    sampled_reference_penalty: torch.Tensor
    advantages: torch.Tensor
    has_signal: bool


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")


def _finite(values: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")


def _provenance(values: Sequence[RewardProvenance], size: int) -> None:
    if len(values) != size or any(not isinstance(value, RewardProvenance) for value in values):
        raise ValueError(
            "Each response or preference pair requires verifier/evidence provenance metadata"
        )


def _response_log_probs(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if (
        values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not values.is_floating_point()
    ):
        raise ValueError(f"{name} must be a nonempty floating B x T tensor")
    if mask.dtype != torch.bool or mask.shape != values.shape or mask.device != values.device:
        raise ValueError(
            "response_mask must be bool, match B x T, and share the log-probability device"
        )
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("Response masks must be contiguous prefixes with right padding")
    observed = values[mask]
    _finite(observed, name)
    if bool((observed > 0).any()):
        raise ValueError(f"{name} must contain normalized log probabilities, not positive logits")
    # At least fp32 arithmetic; retain fp64 for numerical/gradient checks.
    values = values.to(torch.float32) if values.dtype in {torch.float16, torch.bfloat16} else values
    return torch.where(mask, values, 0)


def _advantages(rewards: torch.Tensor, group_ids: torch.Tensor, method: str) -> torch.Tensor:
    if (
        group_ids.dtype not in {torch.int32, torch.int64}
        or group_ids.shape != rewards.shape
        or group_ids.device != rewards.device
    ):
        raise ValueError("group_ids must be integer B-vectors on the rewards device")
    if bool((group_ids < 0).any()):
        raise ValueError("group_ids must be nonnegative")
    _, inverse, counts = torch.unique(group_ids, return_inverse=True, return_counts=True)
    if bool((counts < 2).any()) or not bool((counts == counts[0]).all()):
        raise ValueError("Prompt groups must have equal sizes of at least two responses")
    # Subtract a group-local anchor before summing. Without this, rounding a
    # constant reward's mean can manufacture a nonzero standardized advantage.
    anchors = rewards.new_full((len(counts),), float("inf")).scatter_reduce_(
        0, inverse, rewards, reduce="amin"
    )
    shifted = rewards - anchors[inverse]
    totals = rewards.new_zeros(len(counts)).scatter_add_(0, inverse, shifted)
    centered = shifted - (totals / counts)[inverse]
    if method not in {"dr_grpo", "rlp"}:
        # Sample standard deviation (correction=1), explicit for reproducibility.
        # Scale before squaring to avoid under/overflow for nonflat rewards.
        scale = rewards.new_zeros(len(counts)).scatter_reduce_(
            0, inverse, centered.abs(), reduce="amax"
        )[inverse]
        centered = centered / torch.where(scale > 0, scale, 1)
        squares = rewards.new_zeros(len(counts)).scatter_add_(0, inverse, centered.square())
        std = (squares / (counts - 1)).sqrt()[inverse]
        _finite(std, "Group standard deviations")
        centered = centered / torch.where(std > 0, std, 1)
    elif method == "rlp":
        centered = centered * (len(rewards) / (len(rewards) - len(counts)))
    _finite(centered, "Group advantages")
    return centered


def group_relative_loss(
    policy_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    config: GroupRelativeConfig,
    rollout: RolloutContract,
    expected_behavior_revision: str,
    policy_temperature: float,
    reference_log_probs: torch.Tensor | None = None,
    reference_temperature: float | None = None,
) -> PolicyObjective:
    """Clipped token/sequence objectives with explicit collection contracts.

    Dr.GRPO uses centered rewards and B*fixed-cap normalization. DAPO-style
    uses sample-standardized rewards and a token mean. GSPO uses the geometric
    mean likelihood ratio and equal response weights. Experimental BPO uses
    detached complement-probability weights, not the PPO likelihood-ratio loss.
    RLP uses G/(G-1) centered rewards and equal-response token means; its mask
    covers thought tokens only, never the observed target used for the reward.

    Optional k3 = exp(logref-logp) - (logref-logp) - 1 is a sampled reference
    penalty, not an unbiased full KL estimate on stored behavior trajectories.
    It uses the method's token reduction (per-response token means for GSPO).
    KL is an explicit extension for DAPO/GSPO and is disabled by default.
    """
    if (
        not isinstance(expected_behavior_revision, str)
        or expected_behavior_revision != rollout.behavior_policy_revision
    ):
        raise ValueError("Behavior policy revision does not match the expected rollout snapshot")
    _positive(policy_temperature, "policy_temperature")
    if policy_temperature != rollout.sampling_temperature:
        raise ValueError("Policy and behavior log probabilities must use the sampling temperature")
    current = _response_log_probs(policy_log_probs, response_mask, "policy_log_probs")
    old = _response_log_probs(behavior_log_probs.detach(), response_mask, "behavior_log_probs")
    size = current.shape[0]
    if (
        rewards.shape != (size,)
        or not rewards.is_floating_point()
        or rewards.device != current.device
    ):
        raise ValueError("rewards must be a floating B-vector on the policy device")
    rewards = rewards.detach().to(current.dtype)
    _finite(rewards, "rewards")
    _provenance(rollout.reward_provenance, size)
    advantages = _advantages(rewards, group_ids, config.method)
    lengths = response_mask.sum(dim=1)
    if bool((lengths > config.max_response_tokens).any()):
        raise ValueError("Response exceeds the configured fixed response cap")
    observed = bool(response_mask.any())
    if observed and bool((lengths == 0).any()):
        raise ValueError("Each sampled response must include at least one token, including its EOS")
    if config.kl_coefficient and reference_log_probs is None:
        raise ValueError("A nonzero KL coefficient requires explicit reference log probabilities")
    reference = None
    if reference_log_probs is not None:
        if reference_temperature != policy_temperature:
            raise ValueError(
                "Reference scoring temperature must be explicit and match policy scoring"
            )
        reference = _response_log_probs(
            reference_log_probs.detach(), response_mask, "reference_log_probs"
        )
    if not observed:
        zero = current.sum()
        return PolicyObjective(
            zero, zero.detach(), zero.detach(), torch.zeros_like(advantages), False
        )

    assert config.clip_low is not None and config.clip_high is not None
    low, high = 1 - config.clip_low, 1 + config.clip_high
    advantage = advantages[:, None]

    def reduce_tokens(values: torch.Tensor) -> torch.Tensor:
        if config.method in {"gspo", "rlp", "bpo_experimental"}:
            return ((values * response_mask).sum(dim=1) / lengths).mean()
        denominator = (
            size * config.max_response_tokens if config.method == "dr_grpo" else lengths.sum()
        )
        return (values * response_mask).sum() / denominator

    if config.method == "bpo_experimental":
        # epsilon + (1-p), evaluated without cancellation as p approaches one.
        weight = (config.bpo_epsilon - old.expm1()) / (config.bpo_epsilon - current.expm1())
        _finite(weight, "BPO complement-probability weights")
        active = ~(((advantage > 0) & (weight > high)) | ((advantage < 0) & (weight < low)))
        token_loss = (
            -advantage * active * weight.detach().clamp(max=config.bpo_weight_cap) * current
        )
        policy_loss = reduce_tokens(token_loss)
    else:
        log_ratio = current - old
        if config.method == "gspo":
            log_ratio = log_ratio.sum(dim=1, keepdim=True) / lengths[:, None]
        ratio = log_ratio.exp()
        _finite(ratio, "Importance ratios (overflow; no silent log-ratio clamping is applied)")
        if bool((ratio == 0).any()):
            raise ValueError(
                "Importance ratios underflowed to zero; wider scoring precision is required"
            )
        active = ~(((advantage > 0) & (ratio > high)) | ((advantage < 0) & (ratio < low)))
        token_loss = -torch.minimum(ratio * advantage, ratio.clamp(low, high) * advantage)
        policy_loss = token_loss.mean() if config.method == "gspo" else reduce_tokens(token_loss)
    signal = bool((active & (advantage != 0) & response_mask).any())
    penalty = current.new_zeros(())
    if config.kl_coefficient:
        assert reference is not None
        difference = reference - current
        token_penalty = difference.expm1() - difference
        _finite(token_penalty, "Sampled reference penalty")
        penalty = reduce_tokens(token_penalty)
        signal |= bool((difference[response_mask] != 0).any())
    loss = policy_loss + config.kl_coefficient * penalty
    _finite(loss, "Policy objective")
    return PolicyObjective(loss, policy_loss.detach(), penalty.detach(), advantages, signal)


def offline_preference_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    *,
    beta: float,
    provenance: Sequence[RewardProvenance],
) -> torch.Tensor:
    """Original offline DPO, using summed response log probabilities (not RL).

    Chosen/rejected pairs and reference scores must belong to the same prompt
    and scoring/tokenization contract; the adapter validates that association.
    """
    _positive(beta, "beta")
    chosen = _response_log_probs(chosen_log_probs, chosen_mask, "chosen_log_probs")
    rejected = _response_log_probs(rejected_log_probs, rejected_mask, "rejected_log_probs")
    ref_chosen = _response_log_probs(
        reference_chosen_log_probs.detach(), chosen_mask, "reference_chosen_log_probs"
    )
    ref_rejected = _response_log_probs(
        reference_rejected_log_probs.detach(), rejected_mask, "reference_rejected_log_probs"
    )
    if chosen.shape[0] != rejected.shape[0] or chosen.device != rejected.device:
        raise ValueError("Chosen and rejected responses must share batch size and device")
    if not bool(chosen_mask.any(dim=1).all() & rejected_mask.any(dim=1).all()):
        raise ValueError("Every preference response must contain observed tokens including EOS")
    _provenance(provenance, chosen.shape[0])
    margin = beta * ((chosen - ref_chosen).sum(dim=1) - (rejected - ref_rejected).sum(dim=1))
    _finite(margin, "Preference log-odds margin")
    return -F.logsigmoid(margin).mean()


def information_gain_reward(
    reasoned_target_log_probs: torch.Tensor, ema_target_log_probs: torch.Tensor
) -> torch.Tensor:
    """RLP Eq5: detached next-token log-evidence difference per sampled position.

    Each element scores the SAME observed target token with/without a thought;
    the adapter must verify that pairing and corpus provenance. No span sum,
    normalization, clipping, scorer execution, or gradient through scorers occurs.
    """
    if reasoned_target_log_probs.ndim != 1 or ema_target_log_probs.ndim != 1:
        raise ValueError("Information gain requires one observed-target log probability per row")
    mask = torch.ones_like(reasoned_target_log_probs[:, None], dtype=torch.bool)
    reasoned = _response_log_probs(
        reasoned_target_log_probs.detach()[:, None], mask, "reasoned_target_log_probs"
    )
    baseline = _response_log_probs(
        ema_target_log_probs.detach()[:, None], mask, "ema_target_log_probs"
    )
    reward = (reasoned - baseline).squeeze(1)
    _finite(reward, "Information-gain rewards")
    return reward


@torch.no_grad()
def update_ema_parameters(
    teacher: torch.nn.Module, policy: torch.nn.Module, *, decay: float
) -> None:
    """Update a separate matching teacher after a policy optimizer step.

    The decay is always caller-supplied; decay=0 initializes an exact copy.
    Parameters are averaged once (including tied weights). Buffers are copied,
    a local implementation choice not specified by the RLP paper. Training/eval
    mode and requires_grad flags are preserved; no model or teacher is executed.
    All structural/numerical checks precede mutation so invalid inputs fail closed.
    """
    _positive(decay, "decay", allow_zero=True)
    if decay >= 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if teacher is policy:
        raise ValueError("EMA teacher and policy must be separate nonaliasing modules")

    def architecture(model):
        return {
            name: (
                type(module),
                module.extra_repr(),
                {
                    key: value
                    for key, value in vars(module).items()
                    if not key.startswith("_")
                    and key != "training"
                    and isinstance(value, (str, int, float, bool, type(None)))
                },
            )
            for name, module in model.named_modules(remove_duplicate=False)
        }

    def state(model):
        return {
            ("parameter", name): value
            for name, value in model.named_parameters(remove_duplicate=False)
        } | {("buffer", name): value for name, value in model.named_buffers(remove_duplicate=False)}

    targets, sources = state(teacher), state(policy)
    if architecture(teacher) != architecture(policy) or targets.keys() != sources.keys():
        raise ValueError("EMA requires matching module architecture and named state")
    aliases, storages = [], []
    finite: dict[torch.device, list[torch.Tensor]] = {}
    for values in (targets, sources):
        seen: dict[int, tuple[str, str]] = {}
        storage_ids: dict[tuple[torch.device, int], int] = {}
        names = {}
        for name, value in values.items():
            if value.layout != torch.strided or value.device.type == "meta":
                raise ValueError("EMA requires materialized dense state")
            if name[0] == "parameter" and not value.is_floating_point():
                raise ValueError("EMA parameters must be real floating point")
            names[name] = seen.setdefault(id(value), name)
            if value.numel():
                storage = (value.device, value.untyped_storage().data_ptr())
                if storage in storage_ids and storage_ids[storage] != id(value):
                    raise ValueError("EMA does not support distinct tensors sharing storage views")
                storage_ids[storage] = id(value)
            if (value.is_floating_point() or value.is_complex()) and names[name] == name:
                finite.setdefault(value.device, []).append(torch.isfinite(value).all())
        aliases.append(names)
        storages.append(set(storage_ids))
    if aliases[0] != aliases[1] or storages[0] & storages[1]:
        raise ValueError("EMA requires matching tied weights and disjoint policy/teacher storage")
    for name, target in targets.items():
        source = sources[name]
        if (
            target.shape != source.shape
            or target.dtype != source.dtype
            or target.device != source.device
        ):
            raise ValueError("EMA state shapes, dtypes, and devices must match")
    if any(not bool(torch.stack(flags).all()) for flags in finite.values()):
        raise ValueError("EMA state must be finite before updating")
    for name, target in targets.items():
        if aliases[0][name] != name:
            continue
        if name[0] == "parameter" and decay:
            target.mul_(decay).add_(sources[name], alpha=1 - decay)
        else:
            target.copy_(sources[name])
