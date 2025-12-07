"""
Dataset download script for LexiMind.

Downloads training datasets from HuggingFace Hub and Project Gutenberg:
- GoEmotions: 28 emotion labels (43K samples)
- Yahoo Answers: 10 topic labels (1.4M samples, subset to 200K)
- CNN/DailyMail + BookSum: Summarization (100K + 9.6K samples)
- Gutenberg: Classic books for inference demos

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from datasets import ClassLabel, DatasetDict, load_dataset
from datasets import Sequence as DatasetSequence
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml

DOWNLOAD_TIMEOUT = 60

# --------------- Label Definitions ---------------

EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

TOPIC_LABELS = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]


# --------------- Utility Functions ---------------


def _normalize_label(label: object, label_names: list[str]) -> str:
    """Convert a label index or raw value into a string name.

    - Valid integer indices are mapped to label_names.
    - Everything else is stringified for robustness.
    """

    if isinstance(label, int) and 0 <= label < len(label_names):
        return label_names[label]
    return str(label)


def _emotion_records(dataset_split: Any, label_names: list[str]) -> list[dict[str, object]]:
    """Yield emotion records with resilient label handling."""

    records: list[dict[str, object]] = []
    for row in dataset_split:
        text = str(getattr(row, "text", None) or row.get("text", ""))
        raw_labels = getattr(row, "label", None) or row.get("label") or row.get("labels", [])

        # Normalize to list
        if isinstance(raw_labels, list):
            label_values = raw_labels
        elif raw_labels is None:
            label_values = []
        else:
            label_values = [raw_labels]

        emotions = [_normalize_label(lbl, label_names) for lbl in label_values]
        if text:
            records.append({"text": text, "emotions": emotions})
    return records


def _topic_records(dataset_split: Any, label_names: list[str]) -> list[dict[str, object]]:
    """Yield topic records with resilient label handling."""

    records: list[dict[str, object]] = []
    for row in dataset_split:
        text = str(getattr(row, "text", None) or row.get("text", ""))
        raw_label = getattr(row, "label", None) or row.get("label") or row.get("topic")

        if isinstance(raw_label, list):
            label_value = raw_label[0] if raw_label else ""
        else:
            label_value = raw_label

        topic = _normalize_label(label_value, label_names) if label_value is not None else ""
        if text:
            records.append({"text": text, "topic": topic})
    return records


def _write_jsonl(records: list[dict], destination: Path, desc: str = "Writing") -> None:
    """Write records to JSONL file with progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=desc, leave=False):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def gutenberg_download(url: str, output_path: str) -> None:
    """Download a text file from Project Gutenberg."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            content = response.read()
            target.write_bytes(content)
    except (URLError, socket.timeout, OSError) as e:
        raise RuntimeError(f"Failed to download '{url}': {e}") from e


# --------------- Emotion Dataset (GoEmotions) ---------------


def download_emotion_dataset(output_dir: Path, config: dict) -> None:
    """Download GoEmotions dataset with 28 emotion labels."""
    print("\n�� Downloading GoEmotions (28 emotions)...")

    dataset_name = config.get("dataset", "google-research-datasets/go_emotions")
    dataset_config = config.get("config", "simplified")

    ds = cast(DatasetDict, load_dataset(dataset_name, dataset_config))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get label names from dataset
    label_feature = ds["train"].features.get("labels")
    inner_feature = getattr(label_feature, "feature", None)
    if isinstance(label_feature, DatasetSequence) and isinstance(inner_feature, ClassLabel):
        label_names = cast(list[str], inner_feature.names)
    else:
        label_names = EMOTION_LABELS

    for split_name, split in ds.items():
        records = []
        for item in tqdm(split, desc=f"Processing {split_name}", leave=False):
            row = cast(dict[str, Any], item)
            text = row.get("text", "")
            label_indices = row.get("labels", [])
            # Convert indices to label names
            emotions = [label_names[i] for i in label_indices if 0 <= i < len(label_names)]
            if text and emotions:
                records.append({"text": text, "emotions": emotions})

        output_path = output_dir / f"{split_name}.jsonl"
        _write_jsonl(records, output_path, f"Writing {split_name}")
        print(f"   ✓ {split_name}: {len(records):,} samples -> {output_path}")

    # Save label names
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(label_names, indent=2))
    print(f"   ✓ Labels ({len(label_names)}): {labels_path}")


# --------------- Topic Dataset (Yahoo Answers) ---------------


def download_topic_dataset(output_dir: Path, config: dict) -> None:
    """Download Yahoo Answers dataset with 10 topic labels."""
    print("\n📥 Downloading Yahoo Answers (10 topics)...")

    dataset_name = config.get("dataset", "yahoo_answers_topics")
    max_samples = config.get("max_samples", 200000)

    ds = cast(DatasetDict, load_dataset(dataset_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get label names
    label_feature = ds["train"].features.get("topic")
    if isinstance(label_feature, ClassLabel):
        label_names = label_feature.names
    else:
        label_names = TOPIC_LABELS

    for split_name, split in ds.items():
        # Determine sample limit for this split
        if split_name == "train":
            limit = max_samples
        else:
            limit = min(len(split), max_samples // 10)

        # Random sample if needed
        indices = list(range(len(split)))
        if len(indices) > limit:
            random.seed(42)
            indices = random.sample(indices, limit)

        records = []
        for idx in tqdm(indices, desc=f"Processing {split_name}", leave=False):
            item = cast(dict[str, Any], split[idx])
            # Combine question and best answer for richer text
            question = item.get("question_title", "") + " " + item.get("question_content", "")
            answer = item.get("best_answer", "")
            text = (question + " " + answer).strip()

            topic_idx = item.get("topic", 0)
            topic = label_names[topic_idx] if 0 <= topic_idx < len(label_names) else str(topic_idx)

            if text and len(text) > 50:  # Filter very short texts
                records.append({"text": text, "topic": topic})

        output_path = output_dir / f"{split_name}.jsonl"
        _write_jsonl(records, output_path, f"Writing {split_name}")
        print(f"   ✓ {split_name}: {len(records):,} samples -> {output_path}")

    # Save label names
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(label_names, indent=2))
    print(f"   ✓ Labels ({len(label_names)}): {labels_path}")


# --------------- Summarization Dataset (CNN/DailyMail + BookSum) ---------------


def download_summarization_datasets(output_dir: Path, config: list[dict]) -> None:
    """Download summarization datasets (CNN/DailyMail and BookSum)."""
    print("\n📥 Downloading Summarization datasets...")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_train, all_val, all_test = [], [], []

    for ds_config in config:
        name = ds_config.get("name", "unknown")
        dataset_name = ds_config.get("dataset")
        dataset_config = ds_config.get("config")
        source_field = ds_config.get("source_field", "article")
        target_field = ds_config.get("target_field", "highlights")
        max_samples = ds_config.get("max_samples")

        print(f"\n   Loading {name}...")

        if not dataset_name:
            print(f"      ✗ Skipping {name}: no dataset specified")
            continue

        if dataset_config:
            ds = cast(DatasetDict, load_dataset(str(dataset_name), str(dataset_config)))
        else:
            ds = cast(DatasetDict, load_dataset(str(dataset_name)))

        for split_name, split in ds.items():
            split_str = str(split_name)
            # Determine limit
            limit = max_samples if max_samples else len(split)
            if split_str != "train":
                limit = min(len(split), limit // 10)

            indices = list(range(min(len(split), limit)))

            records = []
            for idx in tqdm(indices, desc=f"{name}/{split_str}", leave=False):
                item = cast(dict[str, Any], split[idx])
                source = item.get(source_field, "")
                target = item.get(target_field, "")

                if source and target and len(str(source)) > 100:
                    records.append({"source": source, "summary": target})

            # Route to appropriate split
            if "train" in split_str:
                all_train.extend(records)
            elif "val" in split_str or "validation" in split_str:
                all_val.extend(records)
            else:
                all_test.extend(records)

            print(f"      ✓ {split_name}: {len(records):,} samples")

    # Write combined files
    if all_train:
        _write_jsonl(all_train, output_dir / "train.jsonl", "Writing train")
        print(f"   ✓ Combined train: {len(all_train):,} samples")
    if all_val:
        _write_jsonl(all_val, output_dir / "validation.jsonl", "Writing validation")
        print(f"   ✓ Combined validation: {len(all_val):,} samples")
    if all_test:
        _write_jsonl(all_test, output_dir / "test.jsonl", "Writing test")
        print(f"   ✓ Combined test: {len(all_test):,} samples")


# --------------- Book Downloads (Gutenberg) ---------------


def download_books(books_dir: Path, config: list[dict]) -> None:
    """Download classic books from Project Gutenberg."""
    print("\n📥 Downloading Gutenberg books...")

    books_dir.mkdir(parents=True, exist_ok=True)

    for book in config:
        name = book.get("name", "unknown")
        url = book.get("url")
        output = book.get("output", str(books_dir / f"{name}.txt"))

        if not url:
            continue

        output_path = Path(output)
        if output_path.exists():
            print(f"   ✓ {name}: already exists")
            continue

        try:
            print(f"   ⏳ {name}: downloading...")
            gutenberg_download(url, str(output_path))
            print(f"   ✓ {name}: {output_path}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")


# --------------- Main Entry Point ---------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LexiMind training datasets")
    parser.add_argument(
        "--config", default="configs/data/datasets.yaml", help="Dataset config path"
    )
    parser.add_argument(
        "--skip-summarization", action="store_true", help="Skip summarization datasets"
    )
    parser.add_argument("--skip-emotion", action="store_true", help="Skip emotion dataset")
    parser.add_argument("--skip-topic", action="store_true", help="Skip topic dataset")
    parser.add_argument("--skip-books", action="store_true", help="Skip Gutenberg books")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_yaml(str(config_path)).data
    raw_paths = config.get("raw", {})
    downloads = config.get("downloads", {})

    print("=" * 60)
    print("LexiMind Dataset Download")
    print("=" * 60)

    # Download emotion dataset
    if not args.skip_emotion:
        emotion_config = downloads.get("emotion", {})
        emotion_dir = Path(raw_paths.get("emotion", "data/raw/emotion"))
        download_emotion_dataset(emotion_dir, emotion_config)

    # Download topic dataset
    if not args.skip_topic:
        topic_config = downloads.get("topic", {})
        topic_dir = Path(raw_paths.get("topic", "data/raw/topic"))
        download_topic_dataset(topic_dir, topic_config)

    # Download summarization datasets
    if not args.skip_summarization:
        summ_config = downloads.get("summarization", [])
        if isinstance(summ_config, list):
            summ_dir = Path(raw_paths.get("summarization", "data/raw/summarization"))
            download_summarization_datasets(summ_dir, summ_config)

    # Download books
    if not args.skip_books:
        books_config = downloads.get("books", [])
        if isinstance(books_config, list):
            books_dir = Path(raw_paths.get("books", "data/raw/books"))
            download_books(books_dir, books_config)

    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
