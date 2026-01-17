#!/usr/bin/env python3
"""
Comprehensive evaluation script for LexiMind.

Evaluates all three tasks with full metrics:
- Summarization: ROUGE-1/2/L, BLEU-4, BERTScore
- Emotion: Multi-label F1, Precision, Recall
- Topic: Accuracy, Macro F1, Per-class metrics

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --skip-bertscore  # Faster, skip BERTScore

Author: Oliver Perrin
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from src.data.dataloader import (
    build_emotion_dataloader,
    build_summarization_dataloader,
    build_topic_dataloader,
)
from src.data.dataset import (
    EmotionDataset,
    SummarizationDataset,
    TopicDataset,
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.inference.factory import create_inference_pipeline
from src.training.metrics import (
    calculate_all_summarization_metrics,
    calculate_bertscore,
    calculate_bleu,
    calculate_rouge,
    multilabel_f1,
)


def evaluate_summarization(
    pipeline,
    data_path: Path,
    max_samples: int | None = None,
    include_bertscore: bool = True,
    batch_size: int = 8,
) -> dict:
    """Evaluate summarization with comprehensive metrics."""
    print("\n" + "=" * 60)
    print("SUMMARIZATION EVALUATION")
    print("=" * 60)
    
    # Load data (returns SummarizationExample dataclass objects)
    data = load_summarization_jsonl(str(data_path))
    if max_samples:
        data = data[:max_samples]
    print(f"Evaluating on {len(data)} samples...")
    
    # Generate summaries
    predictions = []
    references = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Generating summaries"):
        batch = data[i:i + batch_size]
        sources = [ex.source for ex in batch]
        refs = [ex.summary for ex in batch]
        
        preds = pipeline.summarize(sources)
        predictions.extend(preds)
        references.extend(refs)
    
    # Calculate metrics
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge(predictions, references)
    
    print("Calculating BLEU score...")
    bleu = calculate_bleu(predictions, references)
    
    metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu4": bleu,
        "num_samples": len(predictions),
    }
    
    if include_bertscore:
        print("Calculating BERTScore (this may take a few minutes)...")
        bert_scores = calculate_bertscore(predictions, references)
        metrics["bertscore_precision"] = bert_scores["precision"]
        metrics["bertscore_recall"] = bert_scores["recall"]
        metrics["bertscore_f1"] = bert_scores["f1"]
    
    # Print results
    print("\n" + "-" * 40)
    print("SUMMARIZATION RESULTS:")
    print("-" * 40)
    print(f"  ROUGE-1:     {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:     {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:     {metrics['rougeL']:.4f}")
    print(f"  BLEU-4:      {metrics['bleu4']:.4f}")
    if include_bertscore:
        print(f"  BERTScore P: {metrics['bertscore_precision']:.4f}")
        print(f"  BERTScore R: {metrics['bertscore_recall']:.4f}")
        print(f"  BERTScore F: {metrics['bertscore_f1']:.4f}")
    
    # Show examples
    print("\n" + "-" * 40)
    print("SAMPLE OUTPUTS:")
    print("-" * 40)
    for i in range(min(3, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"  Source:    {data[i].source[:100]}...")
        print(f"  Generated: {predictions[i][:150]}...")
        print(f"  Reference: {references[i][:150]}...")
    
    return metrics


def evaluate_emotion(
    pipeline,
    data_path: Path,
    max_samples: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Evaluate emotion detection with multi-label metrics."""
    print("\n" + "=" * 60)
    print("EMOTION DETECTION EVALUATION")
    print("=" * 60)
    
    # Load data (returns EmotionExample dataclass objects)
    data = load_emotion_jsonl(str(data_path))
    if max_samples:
        data = data[:max_samples]
    print(f"Evaluating on {len(data)} samples...")
    
    # Get predictions
    all_preds = []
    all_refs = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Predicting emotions"):
        batch = data[i:i + batch_size]
        texts = [ex.text for ex in batch]
        refs = [set(ex.emotions) for ex in batch]
        
        preds = pipeline.predict_emotions(texts)
        pred_sets = [set(p.labels) for p in preds]
        
        all_preds.extend(pred_sets)
        all_refs.extend(refs)
    
    # Calculate metrics
    # Convert to binary arrays for sklearn
    all_emotions = sorted(pipeline.emotion_labels)
    
    def to_binary(emotion_sets, labels):
        return [[1 if e in es else 0 for e in labels] for es in emotion_sets]
    
    pred_binary = torch.tensor(to_binary(all_preds, all_emotions))
    ref_binary = torch.tensor(to_binary(all_refs, all_emotions))
    
    # Multi-label F1
    f1 = multilabel_f1(pred_binary, ref_binary)
    
    # Per-sample metrics
    sample_f1s = []
    for pred, ref in zip(all_preds, all_refs):
        if len(pred) == 0 and len(ref) == 0:
            sample_f1s.append(1.0)
        elif len(pred) == 0 or len(ref) == 0:
            sample_f1s.append(0.0)
        else:
            intersection = len(pred & ref)
            precision = intersection / len(pred) if pred else 0
            recall = intersection / len(ref) if ref else 0
            if precision + recall > 0:
                sample_f1s.append(2 * precision * recall / (precision + recall))
            else:
                sample_f1s.append(0.0)
    
    avg_f1 = sum(sample_f1s) / len(sample_f1s)
    
    metrics = {
        "multilabel_f1": f1,
        "sample_avg_f1": avg_f1,
        "num_samples": len(all_preds),
        "num_classes": len(all_emotions),
    }
    
    # Print results
    print("\n" + "-" * 40)
    print("EMOTION DETECTION RESULTS:")
    print("-" * 40)
    print(f"  Multi-label F1:  {metrics['multilabel_f1']:.4f}")
    print(f"  Sample Avg F1:   {metrics['sample_avg_f1']:.4f}")
    print(f"  Num Classes:     {metrics['num_classes']}")
    
    return metrics


def evaluate_topic(
    pipeline,
    data_path: Path,
    max_samples: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Evaluate topic classification."""
    print("\n" + "=" * 60)
    print("TOPIC CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    # Load data (returns TopicExample dataclass objects)
    data = load_topic_jsonl(str(data_path))
    if max_samples:
        data = data[:max_samples]
    print(f"Evaluating on {len(data)} samples...")
    
    # Get predictions
    all_preds = []
    all_refs = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Predicting topics"):
        batch = data[i:i + batch_size]
        texts = [ex.text for ex in batch]
        refs = [ex.topic for ex in batch]
        
        preds = pipeline.predict_topics(texts)
        pred_labels = [p.label for p in preds]
        
        all_preds.extend(pred_labels)
        all_refs.extend(refs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_refs, all_preds)
    macro_f1 = f1_score(all_refs, all_preds, average="macro", zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": len(all_preds),
    }
    
    # Print results
    print("\n" + "-" * 40)
    print("TOPIC CLASSIFICATION RESULTS:")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    
    # Classification report
    print("\n" + "-" * 40)
    print("PER-CLASS METRICS:")
    print("-" * 40)
    print(classification_report(all_refs, all_preds, zero_division=0))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LexiMind model")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/labels.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_report.json"))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore (faster)")
    parser.add_argument("--summarization-only", action="store_true")
    parser.add_argument("--emotion-only", action="store_true")
    parser.add_argument("--topic-only", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LexiMind Evaluation")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, labels = create_inference_pipeline(
        args.checkpoint,
        args.labels,
        device=device,
    )
    print(f"  Device: {device}")
    print(f"  Topics: {labels.topic}")
    print(f"  Emotions: {len(labels.emotion)} classes")
    
    results = {}
    
    # Determine which tasks to evaluate
    eval_all = not (args.summarization_only or args.emotion_only or args.topic_only)
    
    # Evaluate summarization
    if eval_all or args.summarization_only:
        val_path = args.data_dir / "summarization" / "validation.jsonl"
        if not val_path.exists():
            val_path = args.data_dir / "summarization" / "val.jsonl"
        if val_path.exists():
            results["summarization"] = evaluate_summarization(
                pipeline, val_path,
                max_samples=args.max_samples,
                include_bertscore=not args.skip_bertscore,
            )
        else:
            print(f"Warning: summarization validation data not found, skipping")
    
    # Evaluate emotion
    if eval_all or args.emotion_only:
        val_path = args.data_dir / "emotion" / "validation.jsonl"
        if not val_path.exists():
            val_path = args.data_dir / "emotion" / "val.jsonl"
        if val_path.exists():
            results["emotion"] = evaluate_emotion(
                pipeline, val_path,
                max_samples=args.max_samples,
            )
        else:
            print(f"Warning: emotion validation data not found, skipping")
    
    # Evaluate topic
    if eval_all or args.topic_only:
        val_path = args.data_dir / "topic" / "validation.jsonl"
        if not val_path.exists():
            val_path = args.data_dir / "topic" / "val.jsonl"
        if val_path.exists():
            results["topic"] = evaluate_topic(
                pipeline, val_path,
                max_samples=args.max_samples,
            )
        else:
            print(f"Warning: topic validation data not found, skipping")
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {args.output}")
    
    # Final summary
    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Time: {elapsed/60:.1f} minutes")
    
    if "summarization" in results:
        s = results["summarization"]
        print(f"\n  Summarization:")
        print(f"    ROUGE-1: {s['rouge1']:.4f}")
        print(f"    ROUGE-L: {s['rougeL']:.4f}")
        if "bertscore_f1" in s:
            print(f"    BERTScore F1: {s['bertscore_f1']:.4f}")
    
    if "emotion" in results:
        print(f"\n  Emotion:")
        print(f"    Multi-label F1: {results['emotion']['multilabel_f1']:.4f}")
    
    if "topic" in results:
        print(f"\n  Topic:")
        print(f"    Accuracy: {results['topic']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
