"""Evaluate the multitask model on processed validation/test splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.inference.factory import create_inference_pipeline
from src.training.metrics import accuracy, multilabel_f1, rouge_like
from src.utils.config import load_yaml


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}


def _read_split(root: Path, split: str, loader) -> list:
    aliases = SPLIT_ALIASES.get(split, (split,))
    for alias in aliases:
        for ext in ("jsonl", "json"):
            candidate = root / f"{alias}.{ext}"
            if candidate.exists():
                return loader(str(candidate))
    raise FileNotFoundError(f"Missing {split} split under {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the LexiMind multitask model")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to the trained checkpoint.")
    parser.add_argument("--labels", default="artifacts/labels.json", help="Label metadata JSON.")
    parser.add_argument("--data-config", default="configs/data/datasets.yaml", help="Data configuration YAML.")
    parser.add_argument("--model-config", default="configs/model/base.yaml", help="Model architecture YAML.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation.")    
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation/classification during evaluation.")
    return parser.parse_args()


def chunks(items: List, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config).data

    pipeline, metadata = create_inference_pipeline(
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        tokenizer_config=None,
        model_config_path=args.model_config,
        device=args.device,
    )

    summarization_dir = Path(data_cfg["processed"]["summarization"])
    emotion_dir = Path(data_cfg["processed"]["emotion"])
    topic_dir = Path(data_cfg["processed"]["topic"])

    summary_examples = _read_split(summarization_dir, args.split, load_summarization_jsonl)
    emotion_examples = _read_split(emotion_dir, args.split, load_emotion_jsonl)
    topic_examples = _read_split(topic_dir, args.split, load_topic_jsonl)

    emotion_binarizer = MultiLabelBinarizer(classes=metadata.emotion)
    # Ensure scikit-learn initializes the attributes using metadata ordering.
    emotion_binarizer.fit([[label] for label in metadata.emotion])

    # Summarization
    summaries_pred = []
    summaries_ref = []
    for batch in chunks(summary_examples, args.batch_size):
        inputs = [example.source for example in batch]
        summaries_pred.extend(pipeline.summarize(inputs))
        summaries_ref.extend([example.summary for example in batch])
    rouge_score = rouge_like(summaries_pred, summaries_ref)

    # Emotion
    emotion_preds_tensor = []
    emotion_target_tensor = []
    label_to_index = {label: idx for idx, label in enumerate(metadata.emotion)}
    for batch in chunks(emotion_examples, args.batch_size):
        inputs = [example.text for example in batch]
        predictions = pipeline.predict_emotions(inputs)
        target_matrix = emotion_binarizer.transform([list(example.emotions) for example in batch])
        for pred, target_row in zip(predictions, target_matrix):
            vector = torch.zeros(len(metadata.emotion), dtype=torch.float32)
            for label in pred.labels:
                idx = label_to_index.get(label)
                if idx is not None:
                    vector[idx] = 1.0
            emotion_preds_tensor.append(vector)
            emotion_target_tensor.append(torch.tensor(target_row, dtype=torch.float32))
    emotion_f1 = multilabel_f1(torch.stack(emotion_preds_tensor), torch.stack(emotion_target_tensor))

    # Topic
    topic_preds = []
    topic_targets = []
    for batch in chunks(topic_examples, args.batch_size):
        inputs = [example.text for example in batch]
        predictions = pipeline.predict_topics(inputs)
        topic_preds.extend([pred.label for pred in predictions])
        topic_targets.extend([example.topic for example in batch])
    topic_accuracy = accuracy(topic_preds, topic_targets)

    print(json.dumps(
        {
            "split": args.split,
            "rouge_like": rouge_score,
            "emotion_f1": emotion_f1,
            "topic_accuracy": topic_accuracy,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
