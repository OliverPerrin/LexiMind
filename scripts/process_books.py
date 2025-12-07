"""
Process book collection with LexiMind model.

Analyzes each book to generate:
- Overall topic classification
- Dominant emotions
- Concise summary

Results are saved to data/processed/books/library.json for future use.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.factory import create_inference_pipeline
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# --------------- Configuration ---------------

BOOKS_DIR = PROJECT_ROOT / "data" / "raw" / "books"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "books" / "library.json"

# Chunk books into manageable sections for analysis
MAX_CHUNK_LENGTH = 1000  # characters per chunk
MAX_CHUNKS = 5  # analyze first N chunks to get representative sample


# --------------- Book Processing ---------------


def clean_text(text: str) -> str:
    """Clean and normalize book text."""
    # Remove Project Gutenberg headers/footers (common patterns)
    lines = text.split("\n")
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if "START OF" in line.upper() and "PROJECT GUTENBERG" in line.upper():
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if "END OF" in lines[i].upper() and "PROJECT GUTENBERG" in lines[i].upper():
            end_idx = i
            break

    text = "\n".join(lines[start_idx:end_idx])

    # Basic cleanup
    text = text.strip()
    text = " ".join(text.split())  # normalize whitespace

    return text


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_LENGTH) -> list[str]:
    """Split text into chunks for analysis."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_book(book_path: Path, pipeline) -> dict:
    """Analyze a single book and return metadata."""
    logger.info(f"Processing {book_path.name}...")

    # Read and clean
    try:
        text = book_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error(f"Failed to read {book_path.name}: {exc}")
        return {}

    text = clean_text(text)

    if not text or len(text) < 100:
        logger.warning(f"Skipping {book_path.name} - insufficient content")
        return {}

    # Chunk and sample
    chunks = chunk_text(text)
    sample_chunks = chunks[: min(MAX_CHUNKS, len(chunks))]

    logger.info(f"  Analyzing {len(sample_chunks)} chunks (of {len(chunks)} total)...")

    # Run inference on chunks
    try:
        topics = pipeline.predict_topics(sample_chunks)
        emotions = pipeline.predict_emotions(sample_chunks, threshold=0.3)
        summaries = pipeline.summarize(sample_chunks, max_length=64)

        # Aggregate results
        # Topic: most common prediction
        topic_counts: dict[str, int] = {}
        for t in topics:
            topic_counts[t.label] = topic_counts.get(t.label, 0) + 1
        dominant_topic = max(topic_counts.items(), key=lambda x: x[1])[0]

        # Emotion: aggregate top emotions
        all_emotions: dict[str, list[float]] = {}
        for emotion in emotions:
            for label, score in zip(emotion.labels, emotion.scores, strict=False):
                if label not in all_emotions:
                    all_emotions[label] = []
                all_emotions[label].append(score)

        # Average scores and take top 3
        emotion_scores = {
            label: sum(scores) / len(scores) for label, scores in all_emotions.items()
        }
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        # Summary: combine first few chunk summaries
        combined_summary = " ".join(summaries[:3])

        result: dict[str, object] = {
            "title": book_path.stem.replace("_", " ").title(),
            "filename": book_path.name,
            "topic": dominant_topic,
            "emotions": [{"label": label, "score": float(score)} for label, score in top_emotions],
            "summary": combined_summary,
            "word_count": len(text.split()),
            "chunks_analyzed": len(sample_chunks),
        }

        logger.info(
            f"  ✓ {result['title']}: {result['topic']} | "
            f"{', '.join(str(e['label']) for e in result['emotions'][:2] if isinstance(e, dict))}"  # type: ignore[index]
        )

        return result

    except Exception as exc:
        logger.error(f"Analysis failed for {book_path.name}: {exc}", exc_info=True)
        return {}


# --------------- Main ---------------


def main():
    """Process all books and save library."""
    logger.info("Loading inference pipeline...")

    pipeline, label_metadata = create_inference_pipeline(
        tokenizer_dir="artifacts/hf_tokenizer/",
        checkpoint_path="checkpoints/best.pt",
        labels_path="artifacts/labels.json",
    )

    logger.info("Finding books...")
    book_files = sorted(BOOKS_DIR.glob("*.txt"))

    if not book_files:
        logger.error(f"No books found in {BOOKS_DIR}")
        return

    logger.info(f"Found {len(book_files)} books")

    # Process each book
    library = []
    for book_path in book_files:
        result = process_book(book_path, pipeline)
        if result:
            library.append(result)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "books": library,
                "metadata": {
                    "total_books": len(library),
                    "chunk_size": MAX_CHUNK_LENGTH,
                    "chunks_per_book": MAX_CHUNKS,
                },
            },
            f,
            indent=2,
        )

    logger.info(f"\n✓ Library saved to {OUTPUT_PATH}")
    logger.info(f"  Processed {len(library)} books")

    # Print summary
    print("\n" + "=" * 60)
    print("BOOK LIBRARY SUMMARY")
    print("=" * 60)

    for book in library:
        print(f"\n📚 {book['title']}")
        print(f"   Topic: {book['topic']}")
        emotions_str = ", ".join(f"{e['label']} ({e['score']:.0%})" for e in book["emotions"])
        print(f"   Emotions: {emotions_str}")
        print(f"   Summary: {book['summary'][:100]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
