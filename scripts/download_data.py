#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""
Dataset download script for LexiMind.

Downloads and prepares training datasets:
- CNN/DailyMail + BookSum for summarization (news + literary)
- Project Gutenberg books for additional literary training
- GoEmotions for emotion classification (28 labels)
- AG News for topic classification (4 labels: World, Sports, Business, Sci/Tech)

Usage:
    python scripts/download_data.py              # Download all
    python scripts/download_data.py --task topic # Download specific task
    python scripts/download_data.py --max-books 30000 --max-gutenberg 20000

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# Label definitions
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

TOPIC_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def write_jsonl(records: list[dict[str, Any]], path: Path, desc: str = "Writing") -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=desc, leave=False):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):,} samples → {path}")


def download_summarization(max_news: int = 80000, max_books: int = 30000) -> None:
    """Download CNN/DailyMail + BookSum for summarization."""
    print("\n📰 Downloading Summarization...")
    out_dir = OUTPUT_DIR / "summarization"
    
    all_train: list[dict[str, Any]] = []
    all_val: list[dict[str, Any]] = []
    all_test: list[dict[str, Any]] = []
    
    # CNN/DailyMail - great for news summarization
    print("  Loading CNN/DailyMail...")
    cnn = load_dataset("cnn_dailymail", "3.0.0")
    
    for split_name in cnn.keys():
        split = str(split_name)
        data = cnn[split_name]
        limit = max_news if "train" in split else max_news // 10
        indices = random.sample(range(len(data)), min(len(data), limit))
        
        records: list[dict[str, Any]] = []
        for i in indices:
            item = data[i]
            article = item["article"]
            highlights = item["highlights"]
            if article and highlights:
                records.append({"source": article, "summary": highlights})
        
        if "train" in split:
            all_train.extend(records)
        elif "val" in split:
            all_val.extend(records)
        else:
            all_test.extend(records)
        print(f"    {split}: {len(records):,}")
    
    # BookSum - literary text summarization (chapters → summaries)
    print("  Loading BookSum...")
    booksum = load_dataset("kmfoda/booksum")
    
    for split_name in booksum.keys():
        split = str(split_name)
        data = booksum[split_name]
        limit = max_books if "train" in split else max_books // 10
        indices = random.sample(range(len(data)), min(len(data), limit))
        
        records = []
        for i in indices:
            item = data[i]
            chapter = item.get("chapter", "")
            summary = item.get("summary_text") or item.get("summary", "")
            if chapter and summary and len(chapter) > 300:
                # Truncate very long chapters to fit model context
                records.append({"source": chapter[:4000], "summary": summary})
        
        if "train" in split:
            all_train.extend(records)
        elif "val" in split:
            all_val.extend(records)
        else:
            all_test.extend(records)
        print(f"    {split}: {len(records):,}")
    
    random.shuffle(all_train)
    write_jsonl(all_train, out_dir / "train.jsonl", "train")
    write_jsonl(all_val, out_dir / "validation.jsonl", "validation")
    write_jsonl(all_test, out_dir / "test.jsonl", "test")


# Patterns to filter out Gutenberg boilerplate
GUTENBERG_JUNK_PATTERNS = [
    r"Project Gutenberg",
    r"www\.gutenberg\.org",
    r"This ebook is for the use of",
    r"You may copy it, give it away",
    r"Gutenberg License",
    r"^\*\*\* START OF",
    r"^\*\*\* END OF",
    r"Produced by",
    r"Transcriber's Note",
    r"Editor's Note",
    r"TABLE OF CONTENTS",
    r"CONTENTS\s*$",
    r"^\s*CHAPTER\s+[IVXLC\d]+",
    r"^\s*Chapter\s+[IVXLC\d]+",
    r"^\s*BOOK\s+[IVXLC\d]+",
    r"^\s*PART\s+[IVXLC\d]+",
    r"^\s*PREFACE\s*$",
    r"^\s*INTRODUCTION\s*$",
    r"^\s*EPILOGUE\s*$",
    r"^\s*PROLOGUE\s*$",
    r"^\s*APPENDIX",
    r"^\s*INDEX\s*$",
    r"^\s*FOOTNOTES?\s*$",
    r"^\s*\[Illustration",
    r"^\s*\[Transcriber",
    r"E-text prepared by",
    r"Internet Archive",
    r"This file was produced",
    r"Distributed Proofreaders",
    r"^\s*_+\s*$",  # Lines of underscores
    r"^\s*\*+\s*$",  # Lines of asterisks
]
GUTENBERG_JUNK_REGEX = re.compile("|".join(GUTENBERG_JUNK_PATTERNS), re.IGNORECASE)


def is_clean_prose(text: str) -> bool:
    """Check if text is clean literary prose (not boilerplate/metadata)."""
    # Must be substantial
    if len(text) < 300 or len(text) > 3000:
        return False
    
    # Skip if contains Gutenberg boilerplate
    if GUTENBERG_JUNK_REGEX.search(text):
        return False
    
    # Must have actual sentences (prose check)
    # Good prose has periods, commas, and lowercase letters
    if text.count('.') < 2:
        return False
    
    # Skip if mostly uppercase (headers, titles)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if uppercase_ratio > 0.3:
        return False
    
    # Skip if too many numbers (tables, dates, page numbers)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.1:
        return False
    
    return True


def download_gutenberg(max_samples: int = 20000) -> None:
    """
    Download Project Gutenberg books for literary language modeling.
    
    Uses the standardized_gutenberg dataset which has clean, parsed books.
    Creates paragraph-level chunks for training diversity.
    Filters out boilerplate (headers, licenses, TOC, etc).
    """
    print("\n📚 Downloading Gutenberg Books...")
    out_dir = OUTPUT_DIR / "books"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Gutenberg dataset - has ~60K books
    print("  Loading standardized_gutenberg dataset...")
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        # Fallback to alternative dataset
        print("  Trying alternative: pg19...")
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    books_processed = 0
    chunks_filtered = 0
    
    # Sample books randomly
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    print("  Processing books into clean prose chunks...")
    for i in tqdm(indices, desc="Books", leave=False):
        if len(records) >= max_samples:
            break
            
        item = gutenberg[i]
        # Handle both uppercase (sedthh/gutenberg_english) and lowercase (pg19) keys
        text = item.get("TEXT", "") or item.get("text", "") or item.get("content", "")
        metadata = item.get("METADATA", {}) or {}
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        if not title:
            title = item.get("title", f"Book_{i}")
        
        if not text or len(text) < 1000:
            continue
        
        # Split into paragraphs for diverse training samples
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            
            # Use strict filtering for clean prose only
            if is_clean_prose(para):
                records.append({
                    "text": para,
                    "title": title,
                    "type": "gutenberg"
                })
                if len(records) >= max_samples:
                    break
            else:
                chunks_filtered += 1
        
        books_processed += 1
    
    # Split into train/val/test (90/5/5)
    random.shuffle(records)
    n = len(records)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)
    
    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]
    
    write_jsonl(train_records, out_dir / "train.jsonl", "train")
    write_jsonl(val_records, out_dir / "validation.jsonl", "validation")
    write_jsonl(test_records, out_dir / "test.jsonl", "test")
    
    print(f"  ✓ {books_processed:,} books → {len(records):,} clean prose chunks")
    print(f"  ✓ Filtered out {chunks_filtered:,} boilerplate/metadata chunks")


def download_emotions() -> None:
    """Download GoEmotions for emotion classification."""
    print("\n😊 Downloading Emotions...")
    out_dir = OUTPUT_DIR / "emotion"
    
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    for split_name in ds.keys():
        split = str(split_name)
        data = ds[split_name]
        
        records: list[dict[str, Any]] = []
        for item in tqdm(data, desc=split, leave=False):
            text = item.get("text", "")
            label_ids = item.get("labels", [])
            if text and label_ids:
                emotions = [EMOTION_LABELS[i] for i in label_ids if 0 <= i < len(EMOTION_LABELS)]
                if emotions:
                    records.append({"text": text, "emotions": emotions})
        write_jsonl(records, out_dir / f"{split}.jsonl", split)
    
    (out_dir / "labels.json").write_text(json.dumps(EMOTION_LABELS, indent=2))
    print(f"  ✓ {len(EMOTION_LABELS)} emotion labels saved")


def download_topics(max_samples: int = 100000) -> None:
    """Download AG News for topic classification (4 clean categories)."""
    print("\n📂 Downloading Topics...")
    out_dir = OUTPUT_DIR / "topic"
    
    ds = load_dataset("fancyzhx/ag_news")
    train_data = ds["train"]
    test_data = ds["test"]
    
    # Split train into train/val
    all_idx = list(range(len(train_data)))
    random.shuffle(all_idx)
    train_idx = all_idx[:max_samples]
    val_idx = all_idx[max_samples:max_samples + max_samples // 10]
    
    splits_config = [
        ("train", train_idx, train_data),
        ("validation", val_idx, train_data),
        ("test", list(range(len(test_data))), test_data),
    ]
    
    for split_name, indices, data in splits_config:
        records: list[dict[str, Any]] = []
        for i in tqdm(indices, desc=split_name, leave=False):
            item = data[i]
            text = item.get("text", "")
            label = item.get("label", 0)
            if text and len(text) > 50:
                records.append({"text": text, "topic": TOPIC_LABELS[label]})
        write_jsonl(records, out_dir / f"{split_name}.jsonl", split_name)
    
    (out_dir / "labels.json").write_text(json.dumps(TOPIC_LABELS, indent=2))
    print(f"  ✓ {len(TOPIC_LABELS)} topic labels saved")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LexiMind datasets")
    parser.add_argument(
        "--task", 
        choices=["all", "summarization", "emotion", "topic", "gutenberg"],
        default="all", 
        help="Dataset to download"
    )
    parser.add_argument("--max-news", type=int, default=80000, help="Max news articles")
    parser.add_argument("--max-books", type=int, default=30000, help="Max BookSum chapters")
    parser.add_argument("--max-gutenberg", type=int, default=20000, help="Max Gutenberg chunks")
    parser.add_argument("--max-topics", type=int, default=100000, help="Max topic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("LexiMind Dataset Download")
    print("=" * 60)
    
    if args.task in ["all", "summarization"]:
        download_summarization(args.max_news, args.max_books)
    if args.task in ["all", "gutenberg"]:
        download_gutenberg(args.max_gutenberg)
    if args.task in ["all", "emotion"]:
        download_emotions()
    if args.task in ["all", "topic"]:
        download_topics(args.max_topics)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
