"""
Inference pipeline factory for LexiMind.

Assembles a complete inference pipeline from saved checkpoints, tokenizer
artifacts, and label metadata. Handles model loading and configuration.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Tuple

import torch

from ..data.tokenization import Tokenizer, TokenizerConfig
from ..models.factory import build_multitask_model, load_model_config
from ..utils.io import load_state
from ..utils.labels import LabelMetadata, load_label_metadata
from .pipeline import InferenceConfig, InferencePipeline


def create_inference_pipeline(
    checkpoint_path: str | Path,
    labels_path: str | Path,
    *,
    tokenizer_config: TokenizerConfig | None = None,
    tokenizer_dir: str | Path | None = None,
    model_config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    summary_max_length: int | None = None,
) -> Tuple[InferencePipeline, LabelMetadata]:
    """Build an :class:`InferencePipeline` from saved model and label metadata."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    labels = load_label_metadata(labels_path)
    paired_labels = checkpoint.parent / "labels.json"
    if paired_labels.exists() and load_label_metadata(paired_labels) != labels:
        raise ValueError("Supplied labels differ from the checkpoint directory's label contract")

    paired_config = checkpoint.parent / "model_config.yaml"
    if model_config_path is None:
        model_config_path = (
            paired_config
            if paired_config.exists()
            else Path(__file__).resolve().parent.parent.parent / "configs" / "model" / "base.yaml"
        )
    model_config = load_model_config(model_config_path)
    if paired_config.exists():
        checkpoint_config = load_model_config(paired_config)
        # These fields only control training or initial weight loading; inference
        # rebuilds the saved weights without fetching a pretrained model.
        ignored = {"use_pretrained", "pretrained_model_name", "gradient_checkpointing"}
        supplied = asdict(model_config)
        paired = asdict(checkpoint_config)
        conflicts = [key for key in paired if key not in ignored and supplied[key] != paired[key]]
        if conflicts:
            raise ValueError(
                "Supplied model config conflicts with checkpoint model_config.yaml: "
                + ", ".join(conflicts)
            )

    resolved_tokenizer_config: TokenizerConfig | None
    tokenizer_contract = None
    paired_tokenizer = checkpoint.parent / "tokenizer_config.json"
    if paired_tokenizer.exists():
        tokenizer_contract = json.loads(paired_tokenizer.read_text(encoding="utf-8"))
        if (
            tokenizer_contract.get("schema_version") != 1
            or tokenizer_contract.get("input_format") != labels.topic_input_format
        ):
            raise ValueError("Checkpoint tokenizer contract does not match label input format")
        bound_config = TokenizerConfig(**tokenizer_contract["tokenizer_config"])
        if tokenizer_config is not None:
            supplied = asdict(tokenizer_config)
            bound = asdict(bound_config)
            conflicts = [
                key
                for key in bound
                if key != "pretrained_model_name" and supplied[key] != bound[key]
            ]
            if conflicts:
                raise ValueError(
                    "Supplied tokenizer config conflicts with checkpoint: " + ", ".join(conflicts)
                )
        chosen_source = (
            str(tokenizer_dir)
            if tokenizer_dir is not None
            else tokenizer_config.pretrained_model_name
            if tokenizer_config is not None
            else bound_config.pretrained_model_name
        )
        resolved_tokenizer_config = replace(bound_config, pretrained_model_name=chosen_source)
    else:
        resolved_tokenizer_config = tokenizer_config
        if resolved_tokenizer_config is None:
            default_dir = (
                Path(__file__).resolve().parent.parent.parent / "artifacts" / "hf_tokenizer"
            )
            local_tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else default_dir
            if not local_tokenizer_dir.exists():
                raise ValueError(
                    "No tokenizer configuration provided and default tokenizer directory "
                    f"'{local_tokenizer_dir}' not found. Please provide tokenizer_config parameter or set tokenizer_dir."
                )
            resolved_tokenizer_config = TokenizerConfig(
                pretrained_model_name=str(local_tokenizer_dir)
            )

    tokenizer = Tokenizer(resolved_tokenizer_config)
    if tokenizer_contract is not None:
        for key in ("truncation_side", "padding_side"):
            if getattr(tokenizer.tokenizer, key) != tokenizer_contract[key]:
                raise ValueError("Supplied tokenizer differs from checkpoint: " + key)
        for key in ("vocab_size", "pad_token_id", "eos_token_id", "bos_token_id"):
            if getattr(tokenizer, key) != tokenizer_contract[key]:
                raise ValueError("Supplied tokenizer differs from checkpoint: " + key)
        vocabulary = json.dumps(
            tokenizer.tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        )
        if hashlib.sha256(vocabulary.encode()).hexdigest() != tokenizer_contract["vocab_sha256"]:
            raise ValueError("Supplied tokenizer vocabulary differs from checkpoint")
        backend = json.loads(tokenizer.tokenizer.backend_tokenizer.to_str())
        for mutable_setting in ("padding", "truncation"):
            backend.pop(mutable_setting, None)
        serialized_backend = json.dumps(backend, sort_keys=True, separators=(",", ":"))
        if (
            hashlib.sha256(serialized_backend.encode()).hexdigest()
            != tokenizer_contract["backend_sha256"]
        ):
            raise ValueError("Supplied tokenizer processing differs from checkpoint")

    model = build_multitask_model(
        tokenizer,
        num_emotions=labels.emotion_size,
        num_topics=labels.topic_size,
        config=model_config,
        load_pretrained=False,
        topic_problem_type=labels.topic_problem_type,
    )

    # Load checkpoint - weights will load separately since factory doesn't tie them
    load_state(model, str(checkpoint))

    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    if summary_max_length is not None:
        pipeline_config = InferenceConfig(summary_max_length=summary_max_length, device=device_str)
    else:
        pipeline_config = InferenceConfig(device=device_str)

    pipeline = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        config=pipeline_config,
        emotion_labels=labels.emotion,
        topic_labels=labels.topic,
        topic_problem_type=labels.topic_problem_type,
        topic_input_format=labels.topic_input_format,
        device=device,
        book_pad_to_multiple_of=(
            tokenizer_contract["pad_to_multiple_of"] if tokenizer_contract else None
        ),
    )
    return pipeline, labels
