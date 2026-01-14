#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""
Dataset download script for LexiMind.

Focus: Books, Academic Papers, Technical Writing
- NO news articles (overdone, dated)
- YES literary text, research, technical writing

Datasets:
- BookSum for literary summarization
- arXiv for academic paper summarization  
- Project Gutenberg for literary language
- GoEmotions for emotion classification (28 labels)
- Custom topic classification: Fiction, Science, Technology, etc.

Usage:
    python scripts/download_data.py              # Download all
    python scripts/download_data.py --task arxiv # Download specific task
    python scripts/download_data.py --max-arxiv 50000

Author: Oliver Perrin
Date: January 2026
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

# ============== LABEL DEFINITIONS ==============

# 28 emotions from GoEmotions - works for all text types
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# New topic labels for books + papers + blogs
TOPIC_LABELS = [
    "Fiction",           # Novels, short stories, literary fiction
    "Science",           # Physics, chemistry, biology, nature
    "Technology",        # CS, engineering, programming, AI/ML
    "Philosophy",        # Ethics, logic, metaphysics, epistemology
    "History",           # Historical texts, biographies, memoirs
    "Psychology",        # Mind, behavior, self-help, mental health
    "Business",          # Economics, finance, entrepreneurship
    "Arts",              # Music, visual arts, film, architecture
]

# arXiv category → our topic mapping
ARXIV_CATEGORY_MAP = {
    # Computer Science
    "cs.AI": "Technology", "cs.CL": "Technology", "cs.CV": "Technology",
    "cs.LG": "Technology", "cs.NE": "Technology", "cs.RO": "Technology",
    "cs.SE": "Technology", "cs.PL": "Technology", "cs.DB": "Technology",
    "cs.DS": "Technology", "cs.CR": "Technology", "cs.DC": "Technology",
    "cs.HC": "Technology", "cs.IR": "Technology", "cs.IT": "Technology",
    "cs.MA": "Technology", "cs.MM": "Technology", "cs.NI": "Technology",
    "cs.OS": "Technology", "cs.PF": "Technology", "cs.SY": "Technology",
    # Physics
    "physics": "Science", "astro-ph": "Science", "cond-mat": "Science",
    "gr-qc": "Science", "hep-ex": "Science", "hep-lat": "Science",
    "hep-ph": "Science", "hep-th": "Science", "math-ph": "Science",
    "nlin": "Science", "nucl-ex": "Science", "nucl-th": "Science",
    "quant-ph": "Science",
    # Math
    "math": "Science",
    # Biology/Medicine
    "q-bio": "Science", "stat": "Science",
    # Economics/Finance
    "econ": "Business", "q-fin": "Business",
    # Electrical Engineering
    "eess": "Technology",
}

# Gutenberg subject → our topic mapping
GUTENBERG_SUBJECT_MAP = {
    "fiction": "Fiction", "novel": "Fiction", "stories": "Fiction",
    "poetry": "Arts", "drama": "Arts", "plays": "Arts",
    "science": "Science", "physics": "Science", "chemistry": "Science",
    "biology": "Science", "nature": "Science", "astronomy": "Science",
    "philosophy": "Philosophy", "ethics": "Philosophy", "logic": "Philosophy",
    "history": "History", "biography": "History", "memoir": "History",
    "psychology": "Psychology", "mind": "Psychology",
    "economics": "Business", "business": "Business", "finance": "Business",
    "art": "Arts", "music": "Arts", "architecture": "Arts",
    "technology": "Technology", "engineering": "Technology",
}


def write_jsonl(records: list[dict[str, Any]], path: Path, desc: str = "Writing") -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=desc, leave=False):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):,} samples → {path}")


# ============== SUMMARIZATION: BOOKS + ARXIV ==============

def download_booksum(max_samples: int = 40000) -> list[dict[str, Any]]:
    """Download BookSum - literary chapter summarization."""
    print("\n📖 Loading BookSum (literary summarization)...")
    
    all_records: list[dict[str, Any]] = []
    booksum = load_dataset("kmfoda/booksum")
    
    for split_name in booksum.keys():
        split = str(split_name)
        data = booksum[split_name]
        limit = max_samples if "train" in split else max_samples // 10
        indices = random.sample(range(len(data)), min(len(data), limit))
        
        records = []
        for i in tqdm(indices, desc=f"BookSum {split}", leave=False):
            item = data[i]
            chapter = item.get("chapter", "")
            summary = item.get("summary_text") or item.get("summary", "")
            if chapter and summary and len(chapter) > 300:
                records.append({
                    "source": chapter[:4000],
                    "summary": summary,
                    "type": "literary",
                    "split": split,
                })
        all_records.extend(records)
        print(f"    {split}: {len(records):,}")
    
    return all_records


def clean_arxiv_text(text: str) -> str:
    """Clean arXiv LaTeX-style text to make it more readable."""
    import re
    # Remove LaTeX math placeholders
    text = re.sub(r'@xmath\d+', '', text)
    text = re.sub(r'@xcite', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()


def download_arxiv_summarization(max_samples: int = 50000) -> list[dict[str, Any]]:
    """
    Download arXiv papers for academic summarization only.
    Note: This dataset doesn't have categories, so can't be used for topic classification.
    
    Returns: summarization_records
    """
    print("\n🎓 Loading arXiv (academic papers for summarization)...")
    
    print("  Loading dataset (this may take a minute)...")
    arxiv = load_dataset("ccdv/arxiv-summarization", split="train")
    
    summ_records: list[dict[str, Any]] = []
    
    indices = list(range(len(arxiv)))
    random.shuffle(indices)
    
    print("  Processing papers...")
    for i in tqdm(indices[:max_samples * 2], desc="arXiv", leave=False):
        if len(summ_records) >= max_samples:
            break
            
        item = arxiv[i]
        
        # Get abstract and article
        abstract = item.get("abstract", "")
        article = item.get("article", "")
        
        if not abstract or len(abstract) < 100:
            continue
        
        # Clean LaTeX artifacts
        abstract = clean_arxiv_text(abstract)
        article = clean_arxiv_text(article)
        
        # Skip if still has too many weird characters after cleaning
        if '@' in abstract or '@' in article[:500]:
            continue
        
        # Summarization: article → abstract
        if article and len(article) > 500:
            summ_records.append({
                "source": article[:4000],
                "summary": abstract,
                "type": "academic",
            })
    
    print(f"    Summarization: {len(summ_records):,}")
    
    return summ_records


def download_topics_from_datasets(max_samples: int = 50000) -> list[dict[str, Any]]:
    """
    Download topic classification data from multiple sources with real categories.
    
    Sources:
    - 20 Newsgroups (classic topic classification)
    - Wikipedia (article categories)
    """
    print("\n📂 Loading topic classification datasets...")
    
    records: list[dict[str, Any]] = []
    
    # 20 Newsgroups - classic topic dataset
    print("  Loading 20 Newsgroups...")
    try:
        newsgroups = load_dataset("SetFit/20_newsgroups", split="train")
        
        # Map 20 newsgroups categories to our 8 topics
        newsgroup_map = {
            # Science
            "sci.crypt": "Science", "sci.electronics": "Science",
            "sci.med": "Science", "sci.space": "Science",
            # Technology  
            "comp.graphics": "Technology", "comp.os.ms-windows.misc": "Technology",
            "comp.sys.ibm.pc.hardware": "Technology", "comp.sys.mac.hardware": "Technology",
            "comp.windows.x": "Technology",
            # Philosophy/Religion
            "alt.atheism": "Philosophy", "soc.religion.christian": "Philosophy",
            "talk.religion.misc": "Philosophy",
            # History/Politics
            "talk.politics.guns": "History", "talk.politics.mideast": "History",
            "talk.politics.misc": "History",
            # Business
            "misc.forsale": "Business",
            # Sports/Recreation
            "rec.autos": "Arts", "rec.motorcycles": "Arts",
            "rec.sport.baseball": "Arts", "rec.sport.hockey": "Arts",
        }
        
        for item in tqdm(newsgroups, desc="20 Newsgroups", leave=False):
            if len(records) >= max_samples:
                break
            label_name = item.get("label_text", "")
            text = item.get("text", "")
            
            if label_name in newsgroup_map and text and len(text) > 100:
                records.append({
                    "text": text[:1500],
                    "topic": newsgroup_map[label_name],
                    "source": "newsgroups",
                })
        
        print(f"    20 Newsgroups: {len(records):,}")
    except Exception as e:
        print(f"    20 Newsgroups failed: {e}")
    
    # Add from Gutenberg for Fiction
    gutenberg_topics = download_gutenberg_topics(max_samples // 4)
    records.extend(gutenberg_topics)
    
    # Add from scientific papers abstract dataset for more Science/Tech
    print("  Loading scientific papers...")
    try:
        sci_papers = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
        sci_count = 0
        for item in tqdm(sci_papers, desc="Scientific papers", leave=False, total=max_samples//4):
            if sci_count >= max_samples // 4:
                break
            abstract = item.get("abstract", "")
            if abstract and len(abstract) > 100:
                # Alternate between Science and Technology
                topic = "Science" if sci_count % 2 == 0 else "Technology"
                records.append({
                    "text": abstract[:1500],
                    "topic": topic,
                    "source": "scientific_papers",
                })
                sci_count += 1
        print(f"    Scientific papers: {sci_count:,}")
    except Exception as e:
        print(f"    Scientific papers failed: {e}")
    
    return records


def download_summarization(max_books: int = 40000, max_arxiv: int = 50000) -> None:
    """Download all summarization data (books + arxiv, NO news)."""
    print("\n📝 Downloading Summarization Data...")
    out_dir = OUTPUT_DIR / "summarization"
    
    all_records: list[dict[str, Any]] = []
    
    # BookSum - literary
    book_records = download_booksum(max_books)
    all_records.extend(book_records)
    
    # arXiv - academic (summarization only, no categories in this dataset)
    arxiv_summ = download_arxiv_summarization(max_arxiv)
    all_records.extend(arxiv_summ)
    
    # Shuffle and split
    random.shuffle(all_records)
    
    # Split by original split if available, else 90/5/5
    train_records = [r for r in all_records if r.get("split", "train") == "train" or "split" not in r]
    val_records = [r for r in all_records if r.get("split") == "validation"]
    test_records = [r for r in all_records if r.get("split") == "test"]
    
    # If no split info, do 90/5/5
    if len(val_records) < 100:
        n = len(train_records)
        random.shuffle(train_records)
        val_records = train_records[int(n*0.9):int(n*0.95)]
        test_records = train_records[int(n*0.95):]
        train_records = train_records[:int(n*0.9)]
    
    # Remove split key before saving
    for r in train_records + val_records + test_records:
        r.pop("split", None)
    
    write_jsonl(train_records, out_dir / "train.jsonl", "train")
    write_jsonl(val_records, out_dir / "validation.jsonl", "val")
    write_jsonl(test_records, out_dir / "test.jsonl", "test")
    
    print(f"\n  ✓ Total summarization: {len(train_records) + len(val_records) + len(test_records):,}")


# ============== TOPIC CLASSIFICATION ==============

def download_topics(max_samples: int = 50000) -> None:
    """
    Download topic classification data from multiple sources.
    
    Sources:
    - 20 Newsgroups (classic topic dataset)
    - Gutenberg books (Fiction)
    - Scientific papers (Science, Technology)
    """
    print("\n📂 Downloading Topic Classification...")
    out_dir = OUTPUT_DIR / "topic"
    
    # Get topic records from various sources
    all_records = download_topics_from_datasets(max_samples)
    
    # Balance topics
    topic_counts: dict[str, list] = {t: [] for t in TOPIC_LABELS}
    for r in all_records:
        topic = r.get("topic")
        if topic in topic_counts:
            topic_counts[topic].append(r)
    
    # Print distribution before balancing
    print("\n  Topic distribution (before balancing):")
    for topic, records in topic_counts.items():
        print(f"    {topic}: {len(records):,}")
    
    # Balance to min count (with some tolerance) - only from topics that have data
    counts_with_data = [len(v) for v in topic_counts.values() if v]
    if not counts_with_data:
        print("  ⚠️ No topic data found!")
        return
    
    min_count = min(counts_with_data)
    target_count = min(min_count, max_samples // len(TOPIC_LABELS))
    
    balanced: list[dict[str, Any]] = []
    for topic, records in topic_counts.items():
        if records:
            random.shuffle(records)
            balanced.extend(records[:target_count])
    
    random.shuffle(balanced)
    
    # Split 90/5/5
    n = len(balanced)
    train_records = balanced[:int(n*0.9)]
    val_records = balanced[int(n*0.9):int(n*0.95)]
    test_records = balanced[int(n*0.95):]
    
    write_jsonl(train_records, out_dir / "train.jsonl", "train")
    write_jsonl(val_records, out_dir / "validation.jsonl", "val")
    write_jsonl(test_records, out_dir / "test.jsonl", "test")
    
    # Save labels - only labels that have data
    used_labels = [t for t in TOPIC_LABELS if topic_counts.get(t)]
    (out_dir / "labels.json").write_text(json.dumps(used_labels, indent=2))
    print(f"\n  ✓ {len(used_labels)} topic labels with data: {used_labels}")


def download_gutenberg_topics(max_samples: int = 30000) -> list[dict[str, Any]]:
    """Extract topic-labeled samples from Gutenberg books."""
    print("\n📚 Loading Gutenberg for topic classification...")
    
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        print("  Trying pg19...")
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    for i in tqdm(indices, desc="Gutenberg topics", leave=False):
        if len(records) >= max_samples:
            break
        
        item = gutenberg[i]
        text = item.get("TEXT", "") or item.get("text", "")
        metadata = item.get("METADATA", {}) or {}
        
        if not text or len(text) < 1000:
            continue
        
        # Try to determine topic from metadata
        subjects = ""
        if isinstance(metadata, dict):
            subjects = str(metadata.get("subjects", "")).lower()
            subjects += " " + str(metadata.get("subject", "")).lower()
            subjects += " " + str(metadata.get("category", "")).lower()
        
        topic = None
        for keyword, mapped_topic in GUTENBERG_SUBJECT_MAP.items():
            if keyword in subjects:
                topic = mapped_topic
                break
        
        # Default fiction for novels without clear subject
        if not topic and ("novel" in subjects or not subjects.strip()):
            topic = "Fiction"
        
        if topic:
            # Get a clean paragraph as sample
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs[5:]:  # Skip front matter
                para = para.strip()
                if 200 < len(para) < 1500 and para.count('.') >= 2:
                    records.append({
                        "text": para,
                        "topic": topic,
                        "source": "gutenberg",
                    })
                    break
    
    print(f"    Gutenberg topics: {len(records):,}")
    return records


# ============== EMOTIONS (unchanged) ==============

def download_emotions() -> None:
    """Download GoEmotions for emotion classification."""
    print("\n😊 Downloading Emotions (GoEmotions)...")
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


# ============== GUTENBERG BOOKS (for language modeling) ==============

GUTENBERG_JUNK_PATTERNS = [
    r"Project Gutenberg", r"www\.gutenberg\.org", r"This ebook is for",
    r"Gutenberg License", r"^\*\*\* START OF", r"^\*\*\* END OF",
    r"Produced by", r"Transcriber's Note", r"TABLE OF CONTENTS",
    r"^\s*CHAPTER\s+[IVXLC\d]+", r"^\s*Chapter\s+[IVXLC\d]+",
    r"^\s*BOOK\s+[IVXLC\d]+", r"^\s*PREFACE\s*$", r"^\s*INTRODUCTION\s*$",
    r"E-text prepared by", r"Internet Archive", r"Distributed Proofreaders",
]
GUTENBERG_JUNK_REGEX = re.compile("|".join(GUTENBERG_JUNK_PATTERNS), re.IGNORECASE)


def is_clean_prose(text: str) -> bool:
    """Check if text is clean literary prose."""
    if len(text) < 300 or len(text) > 3000:
        return False
    if GUTENBERG_JUNK_REGEX.search(text):
        return False
    if text.count('.') < 2:
        return False
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if uppercase_ratio > 0.3:
        return False
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.1:
        return False
    return True


def download_gutenberg(max_samples: int = 30000) -> None:
    """Download Gutenberg books for language modeling."""
    print("\n📚 Downloading Gutenberg Books...")
    out_dir = OUTPUT_DIR / "books"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    for i in tqdm(indices, desc="Books", leave=False):
        if len(records) >= max_samples:
            break
        
        item = gutenberg[i]
        text = item.get("TEXT", "") or item.get("text", "")
        metadata = item.get("METADATA", {}) or {}
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        if not title:
            title = item.get("title", f"Book_{i}")
        
        if not text or len(text) < 1000:
            continue
        
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para = para.strip()
            if is_clean_prose(para):
                records.append({"text": para, "title": title, "type": "gutenberg"})
                if len(records) >= max_samples:
                    break
    
    random.shuffle(records)
    n = len(records)
    write_jsonl(records[:int(n*0.9)], out_dir / "train.jsonl", "train")
    write_jsonl(records[int(n*0.9):int(n*0.95)], out_dir / "validation.jsonl", "val")
    write_jsonl(records[int(n*0.95):], out_dir / "test.jsonl", "test")


# ============== MAIN ==============

def main() -> None:
    parser = argparse.ArgumentParser(description="Download LexiMind datasets")
    parser.add_argument(
        "--task",
        choices=["all", "summarization", "emotion", "topic", "gutenberg"],
        default="all",
        help="Dataset to download"
    )
    parser.add_argument("--max-books", type=int, default=40000, help="Max BookSum samples")
    parser.add_argument("--max-arxiv", type=int, default=50000, help="Max arXiv samples")
    parser.add_argument("--max-gutenberg", type=int, default=30000, help="Max Gutenberg chunks")
    parser.add_argument("--max-topics", type=int, default=50000, help="Max topic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("LexiMind Dataset Download")
    print("Books + Academic Papers + Topic Classification")
    print("=" * 60)
    
    if args.task in ["all", "summarization"]:
        download_summarization(args.max_books, args.max_arxiv)
    if args.task in ["all", "emotion"]:
        download_emotions()
    if args.task in ["all", "topic"]:
        download_topics(args.max_topics)
    if args.task in ["all", "gutenberg"]:
        download_gutenberg(args.max_gutenberg)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
