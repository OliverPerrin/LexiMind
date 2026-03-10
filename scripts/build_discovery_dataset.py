"""Build a discovery dataset for the HuggingFace Space demo.

This script samples from the already-filtered training data (processed by
download_data.py), runs model inference to generate summaries/topics/emotions,
and uploads the result to HuggingFace Datasets.

Data sources (only domains the model was trained on):
  - ArXiv academic papers (summarization training data)
  - Project Gutenberg / Goodreads literary works (summarization training data)

The training data has already been filtered by download_data.py for:
  - English content only
  - Quality text (no metadata, errata, technical manuals)
  - No Shakespeare/plays (excluded titles)
  - Proper book descriptions (from Goodreads, not plot summaries)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.inference.factory import create_inference_pipeline  # noqa: E402

# --------------- Data Loading ---------------


def load_academic_papers(data_dir: Path, max_samples: int = 500) -> list[dict[str, Any]]:
    """Load academic paper samples from the summarization training data."""
    results: list[dict[str, Any]] = []

    for split in ["train", "test"]:
        summ_file = data_dir / "summarization" / f"{split}.jsonl"
        if not summ_file.exists():
            print(f"  Warning: {summ_file} not found")
            continue

        with open(summ_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") != "academic":
                    continue
                text = item.get("source", "")
                if len(text) < 500:
                    continue
                results.append(
                    {
                        "text": text[:2000],
                        "title": item.get("title", "Research Paper")[:150],
                        "reference_summary": item.get("summary", "")[:500],
                    }
                )

    random.shuffle(results)
    results = results[:max_samples]

    samples = []
    for i, item in enumerate(results):
        samples.append(
            {
                "id": f"paper_{i}",
                "title": item["title"],
                "text": item["text"],
                "source_type": "academic",
                "dataset": "arxiv",
                "reference_summary": item["reference_summary"],
            }
        )

    print(f"  Loaded {len(samples)} academic papers")
    return samples


def load_literary(data_dir: Path, max_samples: int = 500) -> list[dict[str, Any]]:
    """Load literary samples (Project Gutenberg / Goodreads) from training data."""
    literary: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for split in ["train", "test"]:
        summ_file = data_dir / "summarization" / f"{split}.jsonl"
        if not summ_file.exists():
            print(f"  Warning: {summ_file} not found")
            continue

        with open(summ_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") != "literary":
                    continue
                title = item.get("title", "")
                if not title or title in seen_titles:
                    continue
                text = item.get("source", "")
                summary = item.get("summary", "")
                if len(text) < 300 or len(summary) < 50:
                    continue
                seen_titles.add(title)
                literary.append(
                    {
                        "text": text[:2000],
                        "title": title,
                        "reference_summary": summary[:600],
                    }
                )

    random.shuffle(literary)
    literary = literary[:max_samples]

    samples = []
    for i, item in enumerate(literary):
        samples.append(
            {
                "id": f"literary_{i}",
                "title": item["title"],
                "text": item["text"],
                "source_type": "literary",
                "dataset": "gutenberg",
                "reference_summary": item["reference_summary"],
            }
        )

    print(f"  Loaded {len(samples)} literary works (unique titles)")
    return samples


# --------------- Inference ---------------


def run_inference(pipeline: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run model inference on all samples to get summaries, topics, and emotions."""
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running inference"):
        text = sample["text"]

        # Get model predictions
        summaries = pipeline.summarize([text])
        topics = pipeline.predict_topics([text])
        emotions = pipeline.predict_emotions([text])

        summary = summaries[0] if summaries else ""
        topic = topics[0] if topics else None
        emotion = emotions[0] if emotions else None

        # Primary emotion (highest confidence)
        primary_emotion = "neutral"
        emotion_confidence = 0.0
        if emotion and emotion.labels:
            primary_emotion = emotion.labels[0]
            emotion_confidence = emotion.scores[0]

        result = {
            "id": sample["id"],
            "title": sample["title"],
            "text": text,
            "source_type": sample["source_type"],
            "dataset": sample["dataset"],
            "topic": topic.label if topic else "Unknown",
            "topic_confidence": topic.confidence if topic else 0.0,
            "emotion": primary_emotion,
            "emotion_confidence": emotion_confidence,
            "generated_summary": summary,
            "reference_summary": sample.get("reference_summary", ""),
        }
        results.append(result)

    # Print distribution stats
    topic_dist: dict[str, int] = defaultdict(int)
    emotion_dist: dict[str, int] = defaultdict(int)
    for r in results:
        topic_dist[r["topic"]] += 1
        emotion_dist[r["emotion"]] += 1

    print(f"\nTopic distribution: {dict(topic_dist)}")
    print(f"Emotion distribution: {dict(emotion_dist)}")

    return results


# --------------- Main ---------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build discovery dataset for the HuggingFace Space demo"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--num-papers", type=int, default=500, help="Academic papers to sample")
    parser.add_argument("--num-literary", type=int, default=500, help="Literary works to sample")
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()

    random.seed(42)

    # ── Load data ──
    print("Loading data samples...")
    print("  Sources: ArXiv papers, Gutenberg/Goodreads books")
    print("  (No news articles or social posts — model is trained on papers & books)\n")

    papers = load_academic_papers(args.data_dir, args.num_papers)
    literary = load_literary(args.data_dir, args.num_literary)

    all_samples = papers + literary
    random.shuffle(all_samples)

    print(f"\nTotal samples: {len(all_samples)} ({len(papers)} papers, {len(literary)} literary)")

    if not all_samples:
        print("ERROR: No samples loaded! Check if data/processed exists and has data.")
        print("Run: python scripts/download_data.py --task summarization")
        return

    # ── Run model inference ──
    print(f"\nLoading model from {args.checkpoint}...")
    labels_path = Path("artifacts/labels.json")
    pipeline, _labels = create_inference_pipeline(
        args.checkpoint, labels_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Running inference on all samples...")
    results = run_inference(pipeline, all_samples)

    # ── Save locally ──
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in results:
            # Remove internal fields
            item.pop("_ground_truth_emotion", None)
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} items")

    # ── Push to Hub ──
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        # Re-read to ensure clean (no internal fields)
        clean: list[dict[str, Any]] = []
        with open(args.output) as f:
            for line in f:
                clean.append(json.loads(line))
        dataset = Dataset.from_list(clean)
        dataset.push_to_hub(
            args.hub_repo,
            private=False,
            commit_message=(f"Rebuild discovery dataset: {len(clean)} items (papers, books)"),
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")

    print("\nDone!")


if __name__ == "__main__":
    main()
