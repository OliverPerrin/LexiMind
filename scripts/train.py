"""End-to-end training entrypoint for the LexiMind multitask model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    build_emotion_dataloader,
    build_summarization_dataloader,
    build_topic_dataloader,
)
from src.data.dataset import (
    EmotionDataset,
    SummarizationDataset,
    TopicDataset,
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import build_multitask_model, load_model_config
from src.training.trainer import Trainer, TrainerConfig
from src.training.utils import set_seed
from src.utils.config import load_yaml
from src.utils.io import save_state
from src.utils.labels import LabelMetadata, save_label_metadata


SplitExamples = Dict[str, list]


SPLIT_ALIASES: Dict[str, Sequence[str]] = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}


def _read_examples(data_dir: Path, loader) -> SplitExamples:
    splits: SplitExamples = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        found = False
        for alias in aliases:
            for extension in ("jsonl", "json"):
                candidate = data_dir / f"{alias}.{extension}"
                if candidate.exists():
                    splits[canonical] = loader(str(candidate))
                    found = True
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"Missing {canonical} split under {data_dir}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LexiMind multitask transformer")
    parser.add_argument("--data-config", default="configs/data/datasets.yaml", help="Path to data configuration YAML.")
    parser.add_argument("--training-config", default="configs/training/default.yaml", help="Path to training hyperparameter YAML.")
    parser.add_argument("--model-config", default="configs/model/base.yaml", help="Path to model architecture YAML.")
    parser.add_argument("--checkpoint-out", default="checkpoints/best.pt", help="Where to store the trained checkpoint.")
    parser.add_argument("--labels-out", default="artifacts/labels.json", help="Where to persist label vocabularies.")
    parser.add_argument("--history-out", default="outputs/training_history.json", help="Where to write training history.")
    parser.add_argument("--device", default="cpu", help="Training device identifier (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_cfg = load_yaml(args.data_config).data
    training_cfg = load_yaml(args.training_config).data
    model_cfg = load_model_config(args.model_config)

    summarization_dir = Path(data_cfg["processed"]["summarization"])
    emotion_dir = Path(data_cfg["processed"]["emotion"])
    topic_dir = Path(data_cfg["processed"]["topic"])

    summarization_splits = _read_examples(summarization_dir, load_summarization_jsonl)
    emotion_splits = _read_examples(emotion_dir, load_emotion_jsonl)
    topic_splits = _read_examples(topic_dir, load_topic_jsonl)

    tokenizer_section = data_cfg.get("tokenizer", {})
    tokenizer_config = TokenizerConfig(
        pretrained_model_name=tokenizer_section.get("pretrained_model_name", "facebook/bart-base"),
        max_length=int(tokenizer_section.get("max_length", 512)),
        lower=bool(tokenizer_section.get("lower", False)),
    )
    tokenizer = Tokenizer(tokenizer_config)

    summarization_train = SummarizationDataset(summarization_splits["train"])
    summarization_val = SummarizationDataset(summarization_splits["val"])

    emotion_train = EmotionDataset(emotion_splits["train"])
    emotion_val = EmotionDataset(emotion_splits["val"], binarizer=emotion_train.binarizer)

    topic_train = TopicDataset(topic_splits["train"])
    topic_val = TopicDataset(topic_splits["val"], encoder=topic_train.encoder)

    dataloader_args = training_cfg.get("dataloader", {})
    batch_size = int(dataloader_args.get("batch_size", 8))
    shuffle = bool(dataloader_args.get("shuffle", True))
    max_length = tokenizer.config.max_length

    train_loaders = {
        "summarization": build_summarization_dataloader(
            summarization_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_source_length=max_length,
            max_target_length=max_length,
        ),
        "emotion": build_emotion_dataloader(
            emotion_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
        ),
        "topic": build_topic_dataloader(
            topic_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
        ),
    }

    val_loaders = {
        "summarization": build_summarization_dataloader(
            summarization_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_source_length=max_length,
            max_target_length=max_length,
        ),
        "emotion": build_emotion_dataloader(
            emotion_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        ),
        "topic": build_topic_dataloader(
            topic_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        ),
    }

    device = torch.device(args.device)
    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emotion_train.emotion_classes),
        num_topics=len(topic_train.topic_classes),
        config=model_cfg,
    ).to(device)

    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 3.0e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer_cfg = training_cfg.get("trainer", {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=TrainerConfig(
            max_epochs=int(trainer_cfg.get("max_epochs", 1)),
            gradient_clip_norm=float(trainer_cfg.get("gradient_clip_norm", 1.0)),
            logging_interval=int(trainer_cfg.get("logging_interval", 50)),
            task_weights=trainer_cfg.get("task_weights"),
        ),
        device=device,
        tokenizer=tokenizer,
    )

    history = trainer.fit(train_loaders, val_loaders)

    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_state(model, str(checkpoint_path))

    labels_path = Path(args.labels_out)
    save_label_metadata(
        LabelMetadata(
            emotion=emotion_train.emotion_classes,
            topic=topic_train.topic_classes,
        ),
        labels_path,
    )

    history_path = Path(args.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training complete. Checkpoint saved to {checkpoint_path}")
    print(f"Label metadata saved to {labels_path}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
