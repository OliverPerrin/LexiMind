#!/usr/bin/env python3
"""Build a discovery dataset for the HuggingFace Space demo.

This script extracts a diverse sample of books and academic papers from the
training data, runs inference to generate summaries/topics/emotions, and
uploads the result to HuggingFace Datasets.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.factory import create_inference_pipeline


def load_books(data_dir: Path, max_per_book: int = 1) -> list[dict]:
    """Load book excerpts, taking max_per_book excerpts per unique book."""
    books_file = data_dir / "books" / "train.jsonl"
    
    # Group by title
    by_title = defaultdict(list)
    with open(books_file) as f:
        for line in f:
            item = json.loads(line)
            by_title[item["title"]].append(item)
    
    # Sample from each book
    samples = []
    for title, excerpts in by_title.items():
        # Pick the longest excerpts (more interesting content)
        excerpts.sort(key=lambda x: len(x["text"]), reverse=True)
        for excerpt in excerpts[:max_per_book]:
            samples.append({
                "id": f"book_{title}",
                "title": title.replace("Book_", "Gutenberg #"),
                "text": excerpt["text"][:2000],  # Truncate for reasonable size
                "source_type": "literary",
                "dataset": "gutenberg"
            })
    
    return samples


def load_academic_papers(data_dir: Path, max_samples: int = 100) -> list[dict]:
    """Load academic paper samples from summarization data."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    academic = []
    with open(summ_file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") == "academic":
                academic.append(item)
    
    # Sample randomly
    random.seed(42)
    samples = random.sample(academic, min(max_samples, len(academic)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"paper_{i}",
            "title": f"Academic Paper #{i+1}",
            "text": item["source"][:2000],  # Truncate
            "source_type": "academic",
            "dataset": "arxiv",
            "reference_summary": item.get("summary", "")[:500]
        })
    
    return results


def load_literary(data_dir: Path, max_samples: int = 50) -> list[dict]:
    """Load literary samples from summarization data (BookSum)."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    literary = []
    with open(summ_file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") == "literary":
                literary.append(item)
    
    # Sample randomly
    random.seed(42)
    samples = random.sample(literary, min(max_samples, len(literary)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"literary_{i}",
            "title": f"Literary Excerpt #{i+1}",
            "text": item["source"][:2000],
            "source_type": "literary", 
            "dataset": "booksum",
            "reference_summary": item.get("summary", "")[:500]
        })
    
    return results


def run_inference(pipeline, labels, samples: list[dict]) -> list[dict]:
    """Run the model to generate summaries, topics, and emotions for each sample."""
    results = []
    
    for sample in tqdm(samples, desc="Running inference"):
        text = sample["text"]
        
        # Generate summary
        summaries = pipeline.summarize([text], max_length=150)
        summary = summaries[0] if summaries else ""
        
        # Get topic
        topic_results = pipeline.predict_topics([text])
        if topic_results:
            topic = topic_results[0].label
            topic_conf = topic_results[0].confidence
        else:
            topic, topic_conf = "unknown", 0.0
        
        # Get emotion (use lower threshold to catch more emotions)
        emotion_results = pipeline.predict_emotions([text], threshold=0.3)
        if emotion_results and emotion_results[0].labels:
            # Take top emotion
            top_idx = emotion_results[0].scores.index(max(emotion_results[0].scores))
            emotion = emotion_results[0].labels[top_idx]
            emotion_conf = emotion_results[0].scores[top_idx]
        else:
            emotion, emotion_conf = "neutral", 0.5
        
        sample_with_predictions = {
            **sample,
            "generated_summary": summary,
            "topic": topic,
            "topic_confidence": round(topic_conf, 3),
            "emotion": emotion,
            "emotion_confidence": round(emotion_conf, 3)
        }
        results.append(sample_with_predictions)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--num-books", type=int, default=100, help="Number of books to include")
    parser.add_argument("--num-papers", type=int, default=80, help="Number of academic papers")
    parser.add_argument("--num-literary", type=int, default=20, help="Number of BookSum excerpts")
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()
    
    print("Loading data samples...")
    
    # Load samples from each source
    books = load_books(args.data_dir, max_per_book=1)
    random.seed(42)
    books = random.sample(books, min(args.num_books, len(books)))
    print(f"  Books: {len(books)}")
    
    papers = load_academic_papers(args.data_dir, args.num_papers)
    print(f"  Academic papers: {len(papers)}")
    
    literary = load_literary(args.data_dir, args.num_literary)
    print(f"  Literary (BookSum): {len(literary)}")
    
    all_samples = books + papers + literary
    print(f"  Total: {len(all_samples)}")
    
    # Load model and run inference
    print(f"\nLoading model from {args.checkpoint}...")
    labels_path = Path("artifacts/labels.json")
    pipeline, labels = create_inference_pipeline(
        args.checkpoint,
        labels_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Running inference on all samples...")
    results = run_inference(pipeline, labels, all_samples)
    
    # Save locally
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    # Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        
        # Convert to HF Dataset
        dataset = Dataset.from_list(results)
        
        # Push
        dataset.push_to_hub(
            args.hub_repo,
            private=False,
            commit_message="Add discovery dataset with summaries, topics, and emotions"
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")
    
    print("\nDone!")
    print(f"Sample topics: {set(r['topic'] for r in results[:20])}")
    print(f"Sample emotions: {set(r['emotion'] for r in results[:20])}")


if __name__ == "__main__":
    main()
