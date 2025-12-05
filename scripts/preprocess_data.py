"""
Data preprocessing script for LexiMind.

Transforms raw datasets into standardized JSONL splits for training. Handles
summarization, emotion classification, topic classification, and book paragraph
extraction with text cleaning.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import BasicTextCleaner
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess datasets configured for LexiMind")
    parser.add_argument(
        "--config",
        default="configs/data/datasets.yaml",
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split size for topic dataset when no validation split is present.",
    )
    parser.add_argument(
        "--seed", type=int, default=17, help="Random seed for deterministic splitting."
    )
    return parser.parse_args()


def _resolve_csv(base: Path, filename: str) -> Path | None:
    primary = base / filename
    if primary.exists():
        return primary
    nested = base / "cnn_dailymail" / filename
    if nested.exists():
        return nested
    return None


def _write_jsonl(records: Iterable[Dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            yield json.loads(row)


def preprocess_books(
    raw_dir: Path,
    processed_dir: Path,
    cleaner: BasicTextCleaner,
    *,
    min_tokens: int = 30,
) -> None:
    if not raw_dir.exists():
        print(f"Skipping book preprocessing (missing directory: {raw_dir})")
        return

    processed_dir.mkdir(parents=True, exist_ok=True)
    index: list[Dict[str, object]] = []

    for book_path in sorted(raw_dir.glob("*.txt")):
        text = book_path.read_text(encoding="utf-8").lstrip("\ufeff")
        normalized = text.replace("\r\n", "\n")
        paragraphs = [
            paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()
        ]

        records: list[Dict[str, object]] = []
        for paragraph_id, paragraph in enumerate(paragraphs):
            cleaned = cleaner.transform([paragraph])[0]
            tokens = cleaned.split()
            if len(tokens) < min_tokens:
                continue
            record = {
                "book": book_path.stem,
                "title": book_path.stem.replace("_", " ").title(),
                "paragraph_id": paragraph_id,
                "text": paragraph,
                "clean_text": cleaned,
                "token_count": len(tokens),
                "char_count": len(paragraph),
            }
            records.append(record)

        if not records:
            print(f"No suitably sized paragraphs found in {book_path}; skipping.")
            continue

        output_path = processed_dir / f"{book_path.stem}.jsonl"
        print(f"Writing book segments for '{book_path.stem}' to {output_path}")
        _write_jsonl(records, output_path)
        index.append(
            {
                "book": book_path.stem,
                "title": records[0]["title"],
                "paragraphs": len(records),
                "source": str(book_path),
                "output": str(output_path),
            }
        )

    if index:
        index_path = processed_dir / "index.json"
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
        print(f"Book index written to {index_path}")


def preprocess_summarization(raw_dir: Path, processed_dir: Path) -> None:
    if not raw_dir.exists():
        print(f"Skipping summarization preprocessing (missing directory: {raw_dir})")
        return

    for split in ("train", "validation", "test"):
        source_path = _resolve_csv(raw_dir, f"{split}.csv")
        if source_path is None:
            print(f"Skipping summarization split '{split}' (file not found)")
            continue

        output_path = processed_dir / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing summarization split '{split}' to {output_path}")
        with (
            source_path.open("r", encoding="utf-8", newline="") as source_handle,
            output_path.open("w", encoding="utf-8") as sink,
        ):
            reader = csv.DictReader(source_handle)
            for row in reader:
                article = row.get("article") or row.get("Article") or ""
                highlights = row.get("highlights") or row.get("summary") or ""
                payload = {"source": article.strip(), "summary": highlights.strip()}
                sink.write(json.dumps(payload, ensure_ascii=False) + "\n")


def preprocess_emotion(raw_dir: Path, processed_dir: Path, cleaner: BasicTextCleaner) -> None:
    if not raw_dir.exists():
        print(f"Skipping emotion preprocessing (missing directory: {raw_dir})")
        return

    split_aliases: Dict[str, Sequence[str]] = {
        "train": ("train",),
        "val": ("val", "validation"),
        "test": ("test",),
    }

    for split, aliases in split_aliases.items():
        source_path: Path | None = None
        for alias in aliases:
            for extension in ("jsonl", "txt", "csv"):
                candidate = raw_dir / f"{alias}.{extension}"
                if candidate.exists():
                    source_path = candidate
                    break
            if source_path is not None:
                break
        if source_path is None:
            print(f"Skipping emotion split '{split}' (file not found)")
            continue

        assert source_path is not None
        path = source_path

        def iter_records(path: Path = path) -> Iterator[Dict[str, object]]:
            if path.suffix == ".jsonl":
                for row in _read_jsonl(path):
                    raw_text = str(row.get("text", ""))
                    text = cleaner.transform([raw_text])[0]
                    labels = row.get("emotions") or row.get("labels") or []
                    if isinstance(labels, str):
                        labels = [label.strip() for label in labels.split(",") if label.strip()]
                    elif isinstance(labels, Sequence):
                        labels = [str(label) for label in labels]
                    else:
                        labels = [str(labels)] if labels else []
                    if not labels:
                        labels = ["neutral"]
                    yield {"text": text, "emotions": labels}
            else:
                delimiter = ";" if path.suffix == ".txt" else ","
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle, delimiter=delimiter)
                    for csv_row in reader:
                        if not csv_row:
                            continue
                        raw_text = str(csv_row[0])
                        text = cleaner.transform([raw_text])[0]
                        raw_labels = csv_row[1] if len(csv_row) > 1 else ""
                        labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
                        if not labels:
                            labels = ["neutral"]
                        yield {"text": text, "emotions": labels}

        output_path = processed_dir / f"{split}.jsonl"
        print(f"Writing emotion split '{split}' to {output_path}")
        _write_jsonl(iter_records(), output_path)


def preprocess_topic(
    raw_dir: Path,
    processed_dir: Path,
    cleaner: BasicTextCleaner,
    val_ratio: float,
    seed: int,
) -> None:
    if not raw_dir.exists():
        print(f"Skipping topic preprocessing (missing directory: {raw_dir})")
        return

    def locate(*names: str) -> Path | None:
        for name in names:
            candidate = raw_dir / name
            if candidate.exists():
                return candidate
        return None

    train_path = locate("train.jsonl", "train.csv")
    if train_path is None:
        print(f"Skipping topic preprocessing (missing train split in {raw_dir})")
        return

    assert train_path is not None

    def load_topic_rows(path: Path) -> list[Tuple[str, str]]:
        rows: list[Tuple[str, str]] = []
        if path.suffix == ".jsonl":
            for record in _read_jsonl(path):
                text = str(record.get("text") or record.get("content") or "")
                topic = record.get("topic") or record.get("label")
                cleaned_text = cleaner.transform([text])[0]
                rows.append((cleaned_text, str(topic).strip()))
        else:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    topic = row.get("Class Index") or row.get("topic") or row.get("label")
                    title = str(row.get("Title") or "")
                    description = str(row.get("Description") or row.get("text") or "")
                    text = " ".join(filter(None, (title, description)))
                    cleaned_text = cleaner.transform([text])[0]
                    rows.append((cleaned_text, str(topic).strip()))
        return rows

    train_rows = load_topic_rows(train_path)
    if not train_rows:
        print("No topic training rows found; skipping topic preprocessing.")
        return

    texts = [row[0] for row in train_rows]
    topics = [row[1] for row in train_rows]

    validation_path = locate("val.jsonl", "validation.jsonl", "val.csv", "validation.csv")
    has_validation = validation_path is not None

    if has_validation and validation_path:
        val_rows = load_topic_rows(validation_path)
        train_records = train_rows
    else:
        train_texts, val_texts, train_topics, val_topics = train_test_split(
            texts,
            topics,
            test_size=val_ratio,
            random_state=seed,
            stratify=topics,
        )
        train_records = list(zip(train_texts, train_topics, strict=False))
        val_rows = list(zip(val_texts, val_topics, strict=False))

    def to_records(pairs: Sequence[Tuple[str, str]]) -> Iterator[Dict[str, object]]:
        for text, topic in pairs:
            yield {"text": text, "topic": topic}

    print(f"Writing topic train split to {processed_dir / 'train.jsonl'}")
    _write_jsonl(to_records(train_records), processed_dir / "train.jsonl")
    print(f"Writing topic val split to {processed_dir / 'val.jsonl'}")
    _write_jsonl(to_records(val_rows), processed_dir / "val.jsonl")

    test_path = locate("test.jsonl", "test.csv")
    if test_path is not None:
        test_rows = load_topic_rows(test_path)
        print(f"Writing topic test split to {processed_dir / 'test.jsonl'}")
        _write_jsonl(to_records(test_rows), processed_dir / "test.jsonl")
    else:
        print(f"Skipping topic test split (missing test split in {raw_dir})")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config).data

    raw_cfg = config.get("raw", {})
    processed_cfg = config.get("processed", {})

    books_raw = Path(raw_cfg.get("books", "data/raw/books"))
    summarization_raw = Path(raw_cfg.get("summarization", "data/raw/summarization"))
    emotion_raw = Path(raw_cfg.get("emotion", "data/raw/emotion"))
    topic_raw = Path(raw_cfg.get("topic", "data/raw/topic"))

    books_processed = Path(processed_cfg.get("books", "data/processed/books"))
    summarization_processed = Path(
        processed_cfg.get("summarization", "data/processed/summarization")
    )
    emotion_processed = Path(processed_cfg.get("emotion", "data/processed/emotion"))
    topic_processed = Path(processed_cfg.get("topic", "data/processed/topic"))

    cleaner = BasicTextCleaner()

    preprocess_books(books_raw, books_processed, cleaner)
    preprocess_summarization(summarization_raw, summarization_processed)
    preprocess_emotion(emotion_raw, emotion_processed, cleaner)
    preprocess_topic(topic_raw, topic_processed, cleaner, val_ratio=args.val_ratio, seed=args.seed)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
