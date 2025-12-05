"""
Evaluation script for LexiMind.

Computes ROUGE/BLEU for summarization, multi-label F1 for emotion,
and accuracy with confusion matrix for topic classification.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_emotion_jsonl, load_summarization_jsonl, load_topic_jsonl
from src.inference.factory import create_inference_pipeline
from src.training.metrics import (
    accuracy,
    calculate_bleu,
    classification_report_dict,
    get_confusion_matrix,
    multilabel_f1,
    rouge_like,
)
from src.utils.config import load_yaml

# --------------- Data Loading ---------------

SPLIT_ALIASES = {"train": ("train",), "val": ("val", "validation"), "test": ("test",)}


def load_split(root: Path, split: str, loader: Callable[[str], List[Any]]) -> List[Any]:
    """Load a dataset split, checking aliases."""
    for alias in SPLIT_ALIASES.get(split, (split,)):
        for ext in ("jsonl", "json"):
            path = root / f"{alias}.{ext}"
            if path.exists():
                return list(loader(str(path)))
    raise FileNotFoundError(f"Missing {split} split in {root}")


def chunks(items: List, size: int):
    """Yield batches of items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------- Visualization ---------------


def plot_confusion_matrix(cm, labels, path: Path) -> None:
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Topic Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------- Main ---------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LexiMind")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--labels", default="artifacts/labels.json")
    p.add_argument("--data-config", default="configs/data/datasets.yaml")
    p.add_argument("--model-config", default="configs/model/base.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=148)  # Larger batch for inference (no grads)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading model...")
    pipeline, metadata = create_inference_pipeline(
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        tokenizer_config=None,
        model_config_path=args.model_config,
        device=args.device,
    )

    # Load data
    data_cfg = load_yaml(args.data_config).data
    summ_data = load_split(
        Path(data_cfg["processed"]["summarization"]), args.split, load_summarization_jsonl
    )
    emot_data = load_split(Path(data_cfg["processed"]["emotion"]), args.split, load_emotion_jsonl)
    topic_data = load_split(Path(data_cfg["processed"]["topic"]), args.split, load_topic_jsonl)

    print(f"\nEvaluating on {args.split} split:")
    print(f"  Summarization: {len(summ_data)} samples")
    print(f"  Emotion: {len(emot_data)} samples")
    print(f"  Topic: {len(topic_data)} samples")

    # --------------- Summarization ---------------

    print("\nSummarization...")
    preds, refs = [], []
    for batch in tqdm(list(chunks(summ_data, args.batch_size)), desc="Summarization", unit="batch"):
        preds.extend(pipeline.summarize([ex.source for ex in batch]))
        refs.extend([ex.summary for ex in batch])

    rouge = rouge_like(preds, refs)
    bleu = calculate_bleu(preds, refs)
    print(f"  ROUGE-like: {rouge:.4f}, BLEU: {bleu:.4f}")

    # --------------- Emotion ---------------

    print("\nEmotion Classification...")
    binarizer = MultiLabelBinarizer(classes=metadata.emotion)
    binarizer.fit([[label] for label in metadata.emotion])
    label_idx = {label: i for i, label in enumerate(metadata.emotion)}

    pred_vecs, target_vecs = [], []
    for batch in tqdm(list(chunks(emot_data, args.batch_size)), desc="Emotion", unit="batch"):
        emotion_results = pipeline.predict_emotions([ex.text for ex in batch], threshold=0.3)
        targets = binarizer.transform([list(ex.emotions) for ex in batch])

        for pred, target in zip(emotion_results, targets, strict=False):
            vec = torch.zeros(len(metadata.emotion))
            for lbl in pred.labels:
                if lbl in label_idx:
                    vec[label_idx[lbl]] = 1.0
            pred_vecs.append(vec)
            target_vecs.append(torch.tensor(target, dtype=torch.float32))

    emotion_f1 = multilabel_f1(torch.stack(pred_vecs), torch.stack(target_vecs))
    print(f"  F1 (macro): {emotion_f1:.4f}")

    # --------------- Topic ---------------

    print("\nTopic Classification...")
    topic_pred_labels: List[str] = []
    topic_true_labels: List[str] = []
    for batch in tqdm(list(chunks(topic_data, args.batch_size)), desc="Topic", unit="batch"):
        topic_results = pipeline.predict_topics([ex.text for ex in batch])
        topic_pred_labels.extend([r.label for r in topic_results])
        topic_true_labels.extend([ex.topic for ex in batch])

    topic_acc = accuracy(topic_pred_labels, topic_true_labels)
    topic_report = classification_report_dict(
        topic_pred_labels, topic_true_labels, labels=metadata.topic
    )
    topic_cm = get_confusion_matrix(topic_pred_labels, topic_true_labels, labels=metadata.topic)
    print(f"  Accuracy: {topic_acc:.4f}")

    # Save confusion matrix
    cm_path = output_dir / "topic_confusion_matrix.png"
    plot_confusion_matrix(topic_cm, metadata.topic, cm_path)
    print(f"  Confusion matrix saved: {cm_path}")

    # --------------- Save Results ---------------

    results = {
        "split": args.split,
        "summarization": {"rouge_like": rouge, "bleu": bleu},
        "emotion": {"f1_macro": emotion_f1},
        "topic": {"accuracy": topic_acc, "classification_report": topic_report},
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.perf_counter() - start_time
    print(f"\n{'=' * 50}")
    print(f"Evaluation complete in {total_time:.1f}s")
    print(f"Report saved: {report_path}")
    print(f"{'=' * 50}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
