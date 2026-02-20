#!/usr/bin/env python3
"""
Comprehensive evaluation script for LexiMind.

Evaluates all three tasks with full metrics:
- Summarization: ROUGE-1/2/L, BLEU-4, BERTScore, per-domain breakdown
- Emotion: Sample-avg F1, Macro F1, Micro F1, per-class metrics, threshold tuning
- Topic: Accuracy, Macro F1, Per-class metrics, bootstrap confidence intervals

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --skip-bertscore  # Faster, skip BERTScore
    python scripts/evaluate.py --tune-thresholds  # Tune per-class emotion thresholds
    python scripts/evaluate.py --bootstrap        # Compute confidence intervals

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

from src.data.dataset import (
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.inference.factory import create_inference_pipeline
from src.training.metrics import (
    bootstrap_confidence_interval,
    calculate_bertscore,
    calculate_bleu,
    calculate_rouge,
    multilabel_f1,
    multilabel_macro_f1,
    multilabel_micro_f1,
    multilabel_per_class_metrics,
    tune_per_class_thresholds,
)


def evaluate_summarization(
    pipeline,
    data_path: Path,
    max_samples: int | None = None,
    include_bertscore: bool = True,
    batch_size: int = 8,
    compute_bootstrap: bool = False,
) -> dict:
    """Evaluate summarization with comprehensive metrics and per-domain breakdown."""
    print("\n" + "=" * 60)
    print("SUMMARIZATION EVALUATION")
    print("=" * 60)
    
    # Load data - try to get domain info from the raw JSONL
    raw_data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    data = load_summarization_jsonl(str(data_path))
    if max_samples:
        data = data[:max_samples]
        raw_data = raw_data[:max_samples]
    print(f"Evaluating on {len(data)} samples...")
    
    # Generate summaries
    predictions = []
    references = []
    domains = []  # Track domain for per-domain breakdown
    
    for i in tqdm(range(0, len(data), batch_size), desc="Generating summaries"):
        batch = data[i:i + batch_size]
        sources = [ex.source for ex in batch]
        refs = [ex.summary for ex in batch]
        
        preds = pipeline.summarize(sources)
        predictions.extend(preds)
        references.extend(refs)
        
        # Track domain if available
        for j in range(len(batch)):
            idx = i + j
            if idx < len(raw_data):
                domain = raw_data[idx].get("type", raw_data[idx].get("domain", "unknown"))
                domains.append(domain)
            else:
                domains.append("unknown")
    
    # Calculate overall metrics
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge(predictions, references)
    
    print("Calculating BLEU score...")
    bleu = calculate_bleu(predictions, references)
    
    metrics: dict = {
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
    
    # Per-domain breakdown
    unique_domains = sorted(set(domains))
    if len(unique_domains) > 1:
        print("\nComputing per-domain breakdown...")
        domain_metrics = {}
        for domain in unique_domains:
            if domain == "unknown":
                continue
            d_preds = [p for p, d in zip(predictions, domains, strict=True) if d == domain]
            d_refs = [r for r, d in zip(references, domains, strict=True) if d == domain]
            if not d_preds:
                continue
            d_rouge = calculate_rouge(d_preds, d_refs)
            d_bleu = calculate_bleu(d_preds, d_refs)
            dm: dict = {
                "num_samples": len(d_preds),
                "rouge1": d_rouge["rouge1"],
                "rouge2": d_rouge["rouge2"],
                "rougeL": d_rouge["rougeL"],
                "bleu4": d_bleu,
            }
            if include_bertscore:
                d_bert = calculate_bertscore(d_preds, d_refs)
                dm["bertscore_f1"] = d_bert["f1"]
            domain_metrics[domain] = dm
        metrics["per_domain"] = domain_metrics
    
    # Bootstrap confidence intervals
    if compute_bootstrap:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            per_sample_r1 = []
            per_sample_rL = []
            for pred, ref in zip(predictions, references, strict=True):
                scores = scorer.score(ref, pred)
                per_sample_r1.append(scores['rouge1'].fmeasure)
                per_sample_rL.append(scores['rougeL'].fmeasure)
            r1_mean, r1_lo, r1_hi = bootstrap_confidence_interval(per_sample_r1)
            rL_mean, rL_lo, rL_hi = bootstrap_confidence_interval(per_sample_rL)
            metrics["rouge1_ci"] = {"mean": r1_mean, "lower": r1_lo, "upper": r1_hi}
            metrics["rougeL_ci"] = {"mean": rL_mean, "lower": rL_lo, "upper": rL_hi}
        except ImportError:
            pass
    
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
    
    if "per_domain" in metrics:
        print("\n  Per-Domain Breakdown:")
        for domain, dm in metrics["per_domain"].items():
            bs_str = f", BS-F1={dm['bertscore_f1']:.4f}" if "bertscore_f1" in dm else ""
            print(f"    {domain} (n={dm['num_samples']}): R1={dm['rouge1']:.4f}, RL={dm['rougeL']:.4f}, B4={dm['bleu4']:.4f}{bs_str}")
    
    if "rouge1_ci" in metrics:
        ci = metrics["rouge1_ci"]
        print(f"\n  ROUGE-1 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
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
    tune_thresholds: bool = False,
    compute_bootstrap: bool = False,
) -> dict:
    """Evaluate emotion detection with comprehensive multi-label metrics.
    
    Reports sample-averaged F1, macro F1, micro F1, and per-class breakdown.
    Optionally tunes per-class thresholds on the evaluation set.
    """
    print("\n" + "=" * 60)
    print("EMOTION DETECTION EVALUATION")
    print("=" * 60)
    
    # Load data (returns EmotionExample dataclass objects)
    data = load_emotion_jsonl(str(data_path))
    if max_samples:
        data = data[:max_samples]
    print(f"Evaluating on {len(data)} samples...")
    
    # Get predictions - collect raw logits for threshold tuning
    all_preds = []
    all_refs = []
    all_logits_list = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Predicting emotions"):
        batch = data[i:i + batch_size]
        texts = [ex.text for ex in batch]
        refs = [set(ex.emotions) for ex in batch]
        
        preds = pipeline.predict_emotions(texts)
        pred_sets = [set(p.labels) for p in preds]
        
        all_preds.extend(pred_sets)
        all_refs.extend(refs)
        
        # Also get raw logits for threshold tuning
        if tune_thresholds:
            encoded = pipeline.tokenizer.batch_encode(texts)
            input_ids = encoded["input_ids"].to(pipeline.device)
            attention_mask = encoded["attention_mask"].to(pipeline.device)
            with torch.inference_mode():
                logits = pipeline.model.forward("emotion", {"input_ids": input_ids, "attention_mask": attention_mask})
                all_logits_list.append(logits.cpu())
    
    # Calculate metrics
    all_emotions = sorted(pipeline.emotion_labels)
    
    def to_binary(emotion_sets, labels):
        return [[1 if e in es else 0 for e in labels] for es in emotion_sets]
    
    pred_binary = torch.tensor(to_binary(all_preds, all_emotions))
    ref_binary = torch.tensor(to_binary(all_refs, all_emotions))
    
    # Core metrics: sample-avg F1, macro F1, micro F1
    sample_f1 = multilabel_f1(pred_binary, ref_binary)
    macro_f1 = multilabel_macro_f1(pred_binary, ref_binary)
    micro_f1 = multilabel_micro_f1(pred_binary, ref_binary)
    
    # Per-class metrics
    per_class = multilabel_per_class_metrics(pred_binary, ref_binary, class_names=all_emotions)
    
    metrics: dict = {
        "sample_avg_f1": sample_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "num_samples": len(all_preds),
        "num_classes": len(all_emotions),
        "per_class": per_class,
    }
    
    # Per-class threshold tuning
    if tune_thresholds and all_logits_list:
        print("\nTuning per-class thresholds...")
        all_logits = torch.cat(all_logits_list, dim=0)
        best_thresholds, tuned_macro_f1 = tune_per_class_thresholds(all_logits, ref_binary)
        metrics["tuned_thresholds"] = {
            name: thresh for name, thresh in zip(all_emotions, best_thresholds, strict=True)
        }
        metrics["tuned_macro_f1"] = tuned_macro_f1
        
        # Also compute tuned sample-avg F1
        probs = torch.sigmoid(all_logits)
        tuned_preds = torch.zeros_like(probs)
        for c, t in enumerate(best_thresholds):
            tuned_preds[:, c] = (probs[:, c] >= t).float()
        metrics["tuned_sample_avg_f1"] = multilabel_f1(tuned_preds, ref_binary)
        metrics["tuned_micro_f1"] = multilabel_micro_f1(tuned_preds, ref_binary)
    
    # Bootstrap confidence intervals
    if compute_bootstrap:
        # Compute per-sample F1 for bootstrapping
        per_sample_f1s = []
        for pred, ref in zip(all_preds, all_refs, strict=True):
            if len(pred) == 0 and len(ref) == 0:
                per_sample_f1s.append(1.0)
            elif len(pred) == 0 or len(ref) == 0:
                per_sample_f1s.append(0.0)
            else:
                intersection = len(pred & ref)
                p = intersection / len(pred) if pred else 0
                r = intersection / len(ref) if ref else 0
                per_sample_f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        mean, lo, hi = bootstrap_confidence_interval(per_sample_f1s)
        metrics["sample_f1_ci"] = {"mean": mean, "lower": lo, "upper": hi}
    
    # Print results
    print("\n" + "-" * 40)
    print("EMOTION DETECTION RESULTS:")
    print("-" * 40)
    print(f"  Sample-avg F1: {metrics['sample_avg_f1']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:      {metrics['micro_f1']:.4f}")
    print(f"  Num Classes:   {metrics['num_classes']}")
    
    if "tuned_macro_f1" in metrics:
        print("\n  After per-class threshold tuning:")
        print(f"    Tuned Macro F1:      {metrics['tuned_macro_f1']:.4f}")
        print(f"    Tuned Sample-avg F1: {metrics['tuned_sample_avg_f1']:.4f}")
        print(f"    Tuned Micro F1:      {metrics['tuned_micro_f1']:.4f}")
    
    if "sample_f1_ci" in metrics:
        ci = metrics["sample_f1_ci"]
        print(f"\n  Sample F1 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
    # Print top-10 per-class performance
    print("\n  Per-class F1 (top 10 by support):")
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["support"], reverse=True)
    for name, m in sorted_classes[:10]:
        print(f"    {name:20s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    
    return metrics


def evaluate_topic(
    pipeline,
    data_path: Path,
    max_samples: int | None = None,
    batch_size: int = 32,
    compute_bootstrap: bool = False,
) -> dict:
    """Evaluate topic classification with per-class metrics and optional bootstrap CI."""
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
    
    metrics: dict = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": len(all_preds),
    }
    
    # Bootstrap confidence intervals for accuracy
    if compute_bootstrap:
        per_sample_correct = [1.0 if p == r else 0.0 for p, r in zip(all_preds, all_refs, strict=True)]
        mean, lo, hi = bootstrap_confidence_interval(per_sample_correct)
        metrics["accuracy_ci"] = {"mean": mean, "lower": lo, "upper": hi}
    
    # Print results
    print("\n" + "-" * 40)
    print("TOPIC CLASSIFICATION RESULTS:")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    
    if "accuracy_ci" in metrics:
        ci = metrics["accuracy_ci"]
        print(f"  Accuracy 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
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
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune per-class emotion thresholds on val set")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap confidence intervals")
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
                compute_bootstrap=args.bootstrap,
            )
        else:
            print("Warning: summarization validation data not found, skipping")
    
    # Evaluate emotion
    if eval_all or args.emotion_only:
        val_path = args.data_dir / "emotion" / "validation.jsonl"
        if not val_path.exists():
            val_path = args.data_dir / "emotion" / "val.jsonl"
        if val_path.exists():
            results["emotion"] = evaluate_emotion(
                pipeline, val_path,
                max_samples=args.max_samples,
                tune_thresholds=args.tune_thresholds,
                compute_bootstrap=args.bootstrap,
            )
        else:
            print("Warning: emotion validation data not found, skipping")
    
    # Evaluate topic
    if eval_all or args.topic_only:
        val_path = args.data_dir / "topic" / "validation.jsonl"
        if not val_path.exists():
            val_path = args.data_dir / "topic" / "val.jsonl"
        if val_path.exists():
            results["topic"] = evaluate_topic(
                pipeline, val_path,
                max_samples=args.max_samples,
                compute_bootstrap=args.bootstrap,
            )
        else:
            print("Warning: topic validation data not found, skipping")
    
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
        print("\n  Summarization:")
        print(f"    ROUGE-1: {s['rouge1']:.4f}")
        print(f"    ROUGE-L: {s['rougeL']:.4f}")
        if "bertscore_f1" in s:
            print(f"    BERTScore F1: {s['bertscore_f1']:.4f}")
    
    if "emotion" in results:
        print("\n  Emotion:")
        print(f"    Multi-label F1: {results['emotion']['multilabel_f1']:.4f}")
    
    if "topic" in results:
        print("\n  Topic:")
        print(f"    Accuracy: {results['topic']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
