"""Download datasets used by LexiMind."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, cast

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import gutenberg_download, kaggle_download
from src.utils.config import load_yaml


DEFAULT_SUMMARIZATION_DATASET = "gowrishankarp/newspaper-text-summarization-cnn-dailymail"
DEFAULT_EMOTION_DATASET = "dair-ai/emotion"
DEFAULT_TOPIC_DATASET = "ag_news"
DEFAULT_BOOK_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
DEFAULT_BOOK_OUTPUT = "data/raw/books/pride_and_prejudice.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets required for LexiMind training")
    parser.add_argument(
        "--config",
        default="configs/data/datasets.yaml",
        help="Path to the dataset configuration YAML.",
    )
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip downloading the Kaggle summarization dataset.")
    parser.add_argument("--skip-book", action="store_true", help="Skip downloading Gutenberg book texts.")
    return parser.parse_args()


def _safe_load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml(str(config_path)).data


def _write_jsonl(records: Iterable[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emotion_records(dataset_split: Dataset, label_names: list[str] | None) -> Iterator[dict[str, object]]:
    for item in dataset_split:
        data = dict(item)
        text = data.get("text", "")
        label_value = data.get("label")
        def resolve_label(index: object) -> str:
            if isinstance(index, int) and label_names and 0 <= index < len(label_names):
                return label_names[index]
            return str(index)

        if isinstance(label_value, list):
            labels = [resolve_label(idx) for idx in label_value]
        else:
            labels = [resolve_label(label_value)]
        yield {"text": text, "emotions": labels}


def _topic_records(dataset_split: Dataset, label_names: list[str] | None) -> Iterator[dict[str, object]]:
    for item in dataset_split:
        data = dict(item)
        text = data.get("text") or data.get("content") or ""
        label_value = data.get("label")
        def resolve_topic(raw: object) -> str:
            if label_names:
                idx: int | None = None
                if isinstance(raw, int):
                    idx = raw
                elif isinstance(raw, str):
                    try:
                        idx = int(raw)
                    except ValueError:
                        idx = None
                if idx is not None and 0 <= idx < len(label_names):
                    return label_names[idx]
            return str(raw) if raw is not None else ""

        if isinstance(label_value, list):
            topic = resolve_topic(label_value[0]) if label_value else ""
        else:
            topic = resolve_topic(label_value)
        yield {"text": text, "topic": topic}


def main() -> None:
    args = parse_args()
    config = _safe_load_config(args.config)

    raw_paths = config.get("raw", {}) if isinstance(config, dict) else {}
    downloads_cfg = config.get("downloads", {}) if isinstance(config, dict) else {}

    summarization_cfg = downloads_cfg.get("summarization", {}) if isinstance(downloads_cfg, dict) else {}
    summarization_dataset = summarization_cfg.get("dataset", DEFAULT_SUMMARIZATION_DATASET)
    summarization_output = summarization_cfg.get("output", raw_paths.get("summarization", "data/raw/summarization"))

    if not args.skip_kaggle and summarization_dataset:
        print(f"Downloading summarization dataset '{summarization_dataset}' -> {summarization_output}")
        kaggle_download(summarization_dataset, summarization_output)
    else:
        print("Skipping Kaggle summarization download.")

    books_root = Path(raw_paths.get("books", "data/raw/books"))
    books_root.mkdir(parents=True, exist_ok=True)

    books_entries: list[dict[str, object]] = []
    if isinstance(downloads_cfg, dict):
        raw_entries = downloads_cfg.get("books")
        if isinstance(raw_entries, list):
            books_entries = [entry for entry in raw_entries if isinstance(entry, dict)]

    if not args.skip_book:
        if not books_entries:
            books_entries = [
                {
                    "name": "pride_and_prejudice",
                    "url": DEFAULT_BOOK_URL,
                    "output": DEFAULT_BOOK_OUTPUT,
                }
            ]
        for entry in books_entries:
            name = str(entry.get("name") or "gutenberg_text")
            url = str(entry.get("url") or DEFAULT_BOOK_URL)
            output_value = entry.get("output")
            destination = Path(output_value) if isinstance(output_value, str) and output_value else books_root / f"{name}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Gutenberg text '{name}' from {url} -> {destination}")
            gutenberg_download(url, str(destination))
    else:
        print("Skipping Gutenberg downloads.")
    emotion_cfg = downloads_cfg.get("emotion", {}) if isinstance(downloads_cfg, dict) else {}
    emotion_name = emotion_cfg.get("dataset", DEFAULT_EMOTION_DATASET)
    emotion_dir = Path(raw_paths.get("emotion", "data/raw/emotion"))
    emotion_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading emotion dataset '{emotion_name}' -> {emotion_dir}")
    emotion_dataset = cast(DatasetDict, load_dataset(emotion_name))
    first_emotion_key = next(iter(emotion_dataset.keys()), None) if emotion_dataset else None
    emotion_label_feature = (
        emotion_dataset[first_emotion_key].features.get("label")
        if first_emotion_key is not None
        else None
    )
    emotion_label_names = emotion_label_feature.names if isinstance(emotion_label_feature, ClassLabel) else None
    for split_name, split in emotion_dataset.items():
        output_path = emotion_dir / f"{str(split_name)}.jsonl"
        _write_jsonl(_emotion_records(split, emotion_label_names), output_path)

    topic_cfg = downloads_cfg.get("topic", {}) if isinstance(downloads_cfg, dict) else {}
    topic_name = topic_cfg.get("dataset", DEFAULT_TOPIC_DATASET)
    topic_dir = Path(raw_paths.get("topic", "data/raw/topic"))
    topic_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading topic dataset '{topic_name}' -> {topic_dir}")
    topic_dataset = cast(DatasetDict, load_dataset(topic_name))
    first_topic_key = next(iter(topic_dataset.keys()), None) if topic_dataset else None
    topic_label_feature = (
        topic_dataset[first_topic_key].features.get("label")
        if first_topic_key is not None
        else None
    )
    topic_label_names = topic_label_feature.names if isinstance(topic_label_feature, ClassLabel) else None
    for split_name, split in topic_dataset.items():
        output_path = topic_dir / f"{str(split_name)}.jsonl"
        _write_jsonl(_topic_records(split, topic_label_names), output_path)

    print("Download routine finished.")


if __name__ == "__main__":
    main()
