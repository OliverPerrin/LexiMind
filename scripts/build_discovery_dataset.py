#!/usr/bin/env python3
"""Build a discovery dataset for the HuggingFace Space demo.

This script samples from the already-filtered training data (processed by
download_data.py), runs inference to generate descriptions/topics/emotions,
and uploads the result to HuggingFace Datasets.

The training data has already been filtered for:
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

import torch
from datasets import Dataset
from tqdm import tqdm

from src.inference.factory import create_inference_pipeline


# --------------- Data Loading ---------------

def load_academic_papers(data_dir: Path, max_samples: int = 300) -> list[dict]:
    """Load academic paper samples from the training data."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    if not summ_file.exists():
        print(f"  Warning: {summ_file} not found")
        return []
    
    academic = []
    with open(summ_file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") != "academic":
                continue
            
            text = item.get("source", "")
            if len(text) < 500:
                continue
            
            # Use title from data
            title = item.get("title", "Research Paper")
            
            academic.append({
                "text": text[:2000],
                "title": title,
                "reference_summary": item.get("summary", "")[:500]
            })
    
    random.seed(42)
    samples = random.sample(academic, min(max_samples, len(academic)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"paper_{i}",
            "title": item["title"],
            "text": item["text"],
            "source_type": "academic",
            "dataset": "arxiv",
            "reference_summary": item["reference_summary"]
        })
    
    print(f"  Loaded {len(results)} academic papers")
    return results


def load_literary(data_dir: Path, max_samples: int = 300) -> list[dict]:
    """Load literary samples from the training data.
    
    Training data now contains Goodreads descriptions (back-cover style)
    instead of plot summaries.
    """
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    if not summ_file.exists():
        print(f"  Warning: {summ_file} not found")
        return []
    
    literary = []
    seen_titles = set()
    
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
            
            if len(text) < 500 or len(summary) < 50:
                continue
            
            seen_titles.add(title)
            literary.append({
                "text": text[:2000],
                "title": title,
                "reference_summary": summary[:600]
            })
    
    random.seed(42)
    samples = random.sample(literary, min(max_samples, len(literary)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"literary_{i}",
            "title": item["title"],
            "text": item["text"],
            "source_type": "literary",
            "dataset": "goodreads",
            "reference_summary": item["reference_summary"],
        })
    
    print(f"  Loaded {len(results)} literary works (unique titles)")
    return results


# --------------- Inference ---------------

def run_inference(pipeline: Any, samples: list[dict]) -> list[dict]:
    """Run model inference on all samples."""
    results = []
    
    for sample in tqdm(samples, desc="Running inference"):
        text = sample["text"]
        
        # Get model predictions using correct pipeline methods
        summaries = pipeline.summarize([text])
        topics = pipeline.predict_topics([text])
        emotions = pipeline.predict_emotions([text])
        
        # Extract first result from each list
        summary = summaries[0] if summaries else ""
        topic = topics[0] if topics else None
        emotion = emotions[0] if emotions else None
        
        # Get primary emotion (highest confidence if any detected)
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
    topic_dist = defaultdict(int)
    emotion_dist = defaultdict(int)
    for r in results:
        topic_dist[r["topic"]] += 1
        emotion_dist[r["emotion"]] += 1
    
    print(f"\nTopic distribution: {dict(topic_dist)}")
    print(f"Emotion distribution: {dict(emotion_dist)}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build discovery dataset for HuggingFace Space")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--num-papers", type=int, default=300, help="Number of academic papers")
    parser.add_argument("--num-literary", type=int, default=300, help="Number of literary works")
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()
    
    print("Loading data samples from training data...")
    print("(Data has already been filtered by download_data.py)")
    
    # Load samples from training data
    papers = load_academic_papers(args.data_dir, args.num_papers)
    literary = load_literary(args.data_dir, args.num_literary)
    
    all_samples = papers + literary
    print(f"\nTotal samples: {len(all_samples)} ({len(papers)} papers, {len(literary)} literary)")
    
    if not all_samples:
        print("ERROR: No samples loaded! Check if data/processed exists and has data.")
        print("Run: python scripts/download_data.py --task summarization")
        return
    
    # Load model and run inference
    print(f"\nLoading model from {args.checkpoint}...")
    labels_path = Path("artifacts/labels.json")
    pipeline, labels = create_inference_pipeline(
        args.checkpoint,
        labels_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Running inference on all samples...")
    results = run_inference(pipeline, all_samples)
    
    # Save locally
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    # Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        dataset = Dataset.from_list(results)
        dataset.push_to_hub(
            args.hub_repo,
            private=False,
            commit_message="Rebuild with Goodreads descriptions (back-cover style)"
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
