"""Rebuild and export the trained multitask model for downstream use."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import build_multitask_model, load_model_config
from src.utils.config import load_yaml
from src.utils.labels import load_label_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LexiMind model weights")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to the trained checkpoint.")
    parser.add_argument("--output", default="outputs/model.pt", help="Output path for the exported state dict.")
    parser.add_argument("--labels", default="artifacts/labels.json", help="Label metadata JSON produced after training.")
    parser.add_argument("--model-config", default="configs/model/base.yaml", help="Model architecture configuration.")
    parser.add_argument("--data-config", default="configs/data/datasets.yaml", help="Data configuration (for tokenizer settings).")
    return parser.parse_args()


def main() -> None:
    """Export multitask model weights from a training checkpoint to a standalone state dict."""
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    labels = load_label_metadata(args.labels)
    data_cfg = load_yaml(args.data_config).data
    tokenizer_section = data_cfg.get("tokenizer", {})
    tokenizer_config = TokenizerConfig(
        pretrained_model_name=tokenizer_section.get("pretrained_model_name", "facebook/bart-base"),
        max_length=int(tokenizer_section.get("max_length", 512)),
        lower=bool(tokenizer_section.get("lower", False)),
    )
    tokenizer = Tokenizer(tokenizer_config)

    model = build_multitask_model(
        tokenizer,
        num_emotions=labels.emotion_size,
        num_topics=labels.topic_size,
        config=load_model_config(args.model_config),
    )

    raw_state = torch.load(checkpoint, map_location="cpu")
    if isinstance(raw_state, dict):
        if "model_state_dict" in raw_state and isinstance(raw_state["model_state_dict"], dict):
            state_dict = raw_state["model_state_dict"]
        elif "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
    else:
        raise TypeError(f"Unsupported checkpoint format: expected dict, got {type(raw_state)!r}")
    model.load_state_dict(state_dict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    main()
