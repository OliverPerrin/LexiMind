"""Helpers to assemble an inference pipeline from saved artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from ..data.preprocessing import TextPreprocessor
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

    resolved_tokenizer_config = tokenizer_config
    if resolved_tokenizer_config is None:
        default_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "hf_tokenizer"
        chosen_dir = Path(tokenizer_dir) if tokenizer_dir is not None else default_dir
        local_tokenizer_dir = chosen_dir
        if local_tokenizer_dir.exists():
            resolved_tokenizer_config = TokenizerConfig(
                pretrained_model_name=str(local_tokenizer_dir)
            )
        else:
            raise ValueError(
                "No tokenizer configuration provided and default tokenizer directory "
                f"'{local_tokenizer_dir}' not found. Please provide tokenizer_config parameter or set tokenizer_dir."
            )

    tokenizer = Tokenizer(resolved_tokenizer_config)

    # Default to base config if not specified (checkpoint was trained with base config)
    if model_config_path is None:
        model_config_path = (
            Path(__file__).resolve().parent.parent.parent / "configs" / "model" / "base.yaml"
        )

    model_config = load_model_config(model_config_path)
    model = build_multitask_model(
        tokenizer,
        num_emotions=labels.emotion_size,
        num_topics=labels.topic_size,
        config=model_config,
        load_pretrained=False,
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
        device=device,
        preprocessor=TextPreprocessor(tokenizer=tokenizer, lowercase=tokenizer.config.lower),
    )
    return pipeline, labels
