"""
Training utilities for LexiMind.

Provides reproducibility helpers including seed management for stdlib, PyTorch,
and NumPy random number generators with thread-safe spawning support.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch._functorch.config

_seed_sequence: Optional[np.random.SeedSequence] = None
_seed_lock = threading.Lock()
_spawn_counter = 0
_thread_local = threading.local()


def set_seed(seed: int) -> np.random.Generator:
    """Seed stdlib/Torch RNGs and initialise this thread's NumPy generator."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base_seq = np.random.SeedSequence(seed)
    child = base_seq.spawn(1)[0]
    rng = np.random.default_rng(child)

    global _seed_sequence, _spawn_counter
    with _seed_lock:
        _seed_sequence = base_seq
        _spawn_counter = 1
    _thread_local.rng = rng
    return rng


def freeze_encoder_layers(encoder: torch.nn.Module, count: int) -> int:
    """Freeze the native embedding and an explicitly bounded layer prefix."""
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= len(encoder.layers)
    ):
        raise ValueError("freeze_encoder_layers must be an integer within the encoder layer count")
    if count == 0:
        return 0
    parameters = {
        p for module in [encoder.embedding, *encoder.layers[:count]] for p in module.parameters()
    }
    for parameter in parameters:
        parameter.requires_grad_(False)
    return sum(p.numel() for p in parameters)


def prepare_training_runtime(cfg):
    """Resolve the opt-in offline book recipe before allocating model tensors."""
    device = torch.device(cfg.device)
    recipe = cfg.training.get("book_lora")
    if not recipe or not recipe.get("enabled", False):
        return device, None
    if cfg.get("resume_from") and not str(cfg.resume_from).endswith(".adapter.pt"):
        raise ValueError(
            "book_lora resume requires its base-bound .adapter.pt artifact; merged weights are for ordinary inference"
        )
    if (
        device.type not in {"mps", "cuda"}
        or (device.type == "mps" and not torch.backends.mps.is_available())
        or (device.type == "cuda" and not torch.cuda.is_available())
    ):
        raise ValueError(
            "book_lora requires an available MPS or CUDA device; no automatic fallback"
        )
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("book_lora requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    if recipe.get("dtype") != "float32":
        raise ValueError("book_lora requires float32")
    if (
        list(cfg.training.trainer.tasks) != ["topic"]
        or cfg.data.get("topic_problem_type") != "multi_label"
    ):
        raise ValueError("book_lora requires only the multi_label topic task")
    expected = dict(
        d_model=768,
        vocab_size=32128,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_attention_heads=12,
        ffn_dim=2048,
        activation="gated-gelu-tanh",
        use_pretrained=True,
    )
    if any(cfg.model.get(key) != value for key, value in expected.items()):
        raise ValueError("book_lora requires the faithful native FLAN-T5-base architecture")
    if not cfg.training.get(
        "use_relative_position_bias", cfg.model.get("use_relative_position_bias")
    ):
        raise ValueError("book_lora requires relative position bias")
    if (
        cfg.training.get("compile_encoder")
        or cfg.training.get("compile_decoder")
        or cfg.training.get("freeze_encoder_layers", 0)
    ):
        raise ValueError("book_lora uses eager adapters, without a separate prefix freeze")
    if cfg.training.trainer.get("use_pcgrad", False):
        raise ValueError("book_lora is a single-task recipe without PCGrad")
    if recipe.get("base_repo") != "google/flan-t5-base":
        raise ValueError("book_lora requires the pinned FLAN-T5-base repository")
    revision, digest = recipe.get("base_revision", ""), recipe.get("base_weight_sha256", "")
    if len(revision) != 40 or len(digest) != 64:
        raise ValueError("book_lora requires an immutable base revision and weight SHA-256")
    threads, fraction = recipe.get("threads"), recipe.get("mps_memory_fraction")
    if (
        type(threads) is not int
        or not 1 <= threads <= 8
        or type(fraction) not in (float, int)
        or not 0 < fraction <= 0.4
    ):
        raise ValueError("book_lora requires bounded threads and MPS memory fraction <= 0.4")
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(recipe.base_repo, revision=revision, local_files_only=True))
    hasher = hashlib.sha256()
    with (snapshot / "model.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError("Cached base weights differ from the pinned checkpoint")
    torch.set_num_threads(threads)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(fraction)
    return device, snapshot


def build_training_model(
    cfg,
    tokenizer,
    *,
    num_emotions,
    num_topics,
    topic_problem_type="single_label",
    snapshot=None,
    compile_modules=True,
):
    """One model, adapter, freeze and compilation recipe for train and profile."""
    from src.models.adapters import LoRAConfig, attach_lora
    from src.models.factory import ModelConfig, build_multitask_model

    training, settings = cfg.training, cfg.model
    checkpointing = training.get(
        "gradient_checkpointing", settings.get("gradient_checkpointing", False)
    )
    relative = training.get(
        "use_relative_position_bias", settings.get("use_relative_position_bias", False)
    )
    model_config = ModelConfig(
        d_model=settings.d_model,
        vocab_size=settings.get("vocab_size"),
        num_encoder_layers=settings.num_encoder_layers,
        num_decoder_layers=settings.num_decoder_layers,
        num_attention_heads=settings.num_attention_heads,
        ffn_dim=settings.ffn_dim,
        dropout=settings.dropout,
        use_pretrained=settings.use_pretrained,
        pretrained_model_name=str(snapshot) if snapshot else settings.pretrained_model_name,
        activation=settings.get("activation", "gelu"),
        use_relative_position_bias=relative,
        gradient_checkpointing=checkpointing,
        use_learned_pos_enc=False if snapshot else settings.get("use_learned_pos_enc", True),
    )
    model = build_multitask_model(
        tokenizer,
        num_emotions=num_emotions,
        num_topics=num_topics,
        config=model_config,
        topic_problem_type=topic_problem_type,
    )
    model._leximind_training_model_config = asdict(model_config)
    binding = None
    if snapshot is not None:
        recipe = training.book_lora
        tokenization = asdict(tokenizer.config)
        tokenization.update(max_length=min(256, tokenizer.config.max_length), padding="longest")
        # Encoding mutates only these runtime controls in the fast tokenizer.
        # Hash the remaining backend, including normalization and token pieces.
        backend = json.loads(tokenizer.tokenizer.backend_tokenizer.to_str())
        backend.pop("padding", None)
        backend.pop("truncation", None)
        model._leximind_training_tokenizer_contract = {
            "schema_version": 1,
            "tokenizer_config": tokenization,
            "pad_to_multiple_of": 8,
            "padding_side": tokenizer.tokenizer.padding_side,
            "truncation_side": tokenizer.tokenizer.truncation_side,
            "input_format": "book_title_description_v1",
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "base_repo": recipe.base_repo,
            "base_revision": recipe.base_revision,
            "vocab_sha256": hashlib.sha256(
                json.dumps(
                    tokenizer.tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "backend_sha256": hashlib.sha256(
                json.dumps(backend, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        paths = [
            f"encoder.layers.{i}.self_attn.W_{projection}"
            for i in range(settings.num_encoder_layers)
            for projection in ("Q", "V")
        ]
        binding = attach_lora(
            model,
            shared_projections=paths,
            private_heads=["head_topic"],
            config=LoRAConfig(
                rank=recipe.rank, alpha=recipe.alpha, dropout=recipe.dropout, seed=cfg.seed
            ),
        )
        print(f"  Encoder Q/V LoRA rank={recipe.rank}; topic head trainable; decoder frozen")
    else:
        freeze_encoder_layers(model.encoder, training.get("freeze_encoder_layers", 0))
    model.to(torch.device(cfg.device), dtype=torch.float32)
    gc.collect()
    if compile_modules:
        compile_training_model(model, cfg)
    print(
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}; trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model, binding


def compile_training_model(model, cfg):
    """Compile after any weights-only restore; preserve legacy wrapper ordering."""
    training = cfg.training
    if training.trainer.get("use_pcgrad", False):
        torch._functorch.config.donated_buffer = False
    checkpointing = training.get(
        "gradient_checkpointing", cfg.model.get("gradient_checkpointing", False)
    )
    mode = "default" if checkpointing else "reduce-overhead"
    for name in ("encoder", "decoder"):
        if training.get(f"compile_{name}", True):
            setattr(
                model,
                name,
                torch.compile(getattr(model, name), mode=mode, dynamic=mode == "default"),
            )


def build_training_optimizer(model, cfg, device):
    """AdamW state is allocated only for parameters that can receive updates."""
    settings = cfg.training.get("optimizer", {})
    fused = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(settings.get("lr", 3e-5)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
        eps=float(settings.get("eps", 1e-8)),
        betas=tuple(float(v) for v in settings.get("betas", (0.9, 0.999))),
        fused=fused,
        foreach=False if device.type == "mps" else None,
    )


def save_training_checkpoint(model, path, binding=None, label_metadata=None):
    """Save native merged weights plus an explicitly separate adapter artifact."""
    from src.models.adapters import extract_effective_delta
    from src.utils.atomic import atomic_write
    from src.utils.io import normalize_state_dict, save_state
    from src.utils.labels import load_label_metadata, save_label_metadata

    if binding is None:
        save_state(model, path)
        return
    if label_metadata is None:
        raise ValueError("LoRA checkpoints require their exact ordered label metadata")
    paired_labels = Path(path).parent / "labels.json"
    if paired_labels.exists() and load_label_metadata(paired_labels) != label_metadata:
        raise ValueError("Checkpoint directory contains a different ordered label contract")
    if not paired_labels.exists():
        save_label_metadata(label_metadata, paired_labels)
    tokenizer_contract = model._leximind_training_tokenizer_contract
    tokenizer_path = Path(path).parent / "tokenizer_config.json"
    tokenizer_content = (json.dumps(tokenizer_contract, indent=2) + "\n").encode()
    if tokenizer_path.exists() and tokenizer_path.read_bytes() != tokenizer_content:
        raise ValueError(
            "Different tokenizer contract already exists; use a fresh checkpoint directory"
        )
    atomic_write(tokenizer_path, lambda stream: stream.write(tokenizer_content))
    configuration = dict(model._leximind_training_model_config)
    configuration["use_pretrained"] = False
    configuration_path = Path(path).parent / "model_config.yaml"
    content = (json.dumps(configuration, indent=2) + "\n").encode()
    if configuration_path.exists() and configuration_path.read_bytes() != content:
        raise ValueError(
            "Different inference model config already exists; use a fresh checkpoint directory"
        )
    atomic_write(configuration_path, lambda stream: stream.write(content))
    delta = extract_effective_delta(model, binding, task_id="topic")
    state = normalize_state_dict(model.state_dict())
    for name in list(state):
        if name.endswith((".lora_A", ".lora_B")):
            del state[name]
    for name, update in delta.shared.items():
        state[name] = state[name].detach().cpu() + update
    # Move each ordinary tensor separately: no duplicate full-model clone on MPS.
    for name in state:
        state[name] = state[name].detach().cpu()
    atomic_write(path, lambda stream: torch.save(state, stream))
    artifact = {
        "schema_version": 1,
        "kind": "book_lora_adapter_artifact",
        "resume_supported": "weights_only",
        "binding": asdict(binding),
        "merged_checkpoint": Path(path).name,
        "model_config": configuration,
        "label_metadata": asdict(label_metadata),
        "tokenizer_contract": tokenizer_contract,
        "trainable_state": {
            name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad
        },
    }
    atomic_write(Path(path).with_suffix(".adapter.pt"), lambda stream: torch.save(artifact, stream))


def load_training_adapter(model, path, binding, label_metadata=None):
    """Restore factors/head only after recreating the identical pinned base."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    from src.utils.labels import load_label_metadata

    paired_labels = Path(path).parent / "labels.json"
    if (
        label_metadata is None
        or not paired_labels.exists()
        or load_label_metadata(paired_labels) != label_metadata
        or not isinstance(state, dict)
        or state.get("label_metadata") != asdict(label_metadata)
    ):
        raise ValueError(
            "Adapter artifact, current dataset and paired checkpoint label contracts must match"
        )
    expected_tokens = dict(model._leximind_training_tokenizer_contract)
    expected_tokens["tokenizer_config"] = dict(expected_tokens["tokenizer_config"])
    expected_tokens["tokenizer_config"].pop("pretrained_model_name", None)
    recorded_tokens = state.get("tokenizer_contract")
    if isinstance(recorded_tokens, dict) and isinstance(
        recorded_tokens.get("tokenizer_config"), dict
    ):
        recorded_tokens = dict(recorded_tokens)
        recorded_tokens["tokenizer_config"] = dict(recorded_tokens["tokenizer_config"])
        recorded_tokens["tokenizer_config"].pop("pretrained_model_name", None)
    if recorded_tokens != expected_tokens:
        raise ValueError(
            "Adapter artifact differs from the current tokenizer vocabulary or encoding contract"
        )
    expected = dict(model._leximind_training_model_config)
    expected["use_pretrained"] = False
    # Snapshot paths differ between the Mac and an RTX host. Actual base bytes
    # and architecture are bound by the adapter fingerprints, not this path.
    expected.pop("pretrained_model_name", None)
    recorded = state.get("model_config") if isinstance(state, dict) else None
    if isinstance(recorded, dict):
        recorded = dict(recorded)
        recorded.pop("pretrained_model_name", None)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or state.get("kind") != "book_lora_adapter_artifact"
        or state.get("binding") != asdict(binding)
        or recorded != expected
    ):
        raise ValueError(
            "Adapter artifact differs from the pinned base, architecture, seed or LoRA binding"
        )
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    values = state.get("trainable_state")
    if not isinstance(values, dict) or values.keys() != parameters.keys():
        raise ValueError("Adapter artifact has different trainable parameter names")
    for name, value in values.items():
        parameter = parameters[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].copy_(value)
