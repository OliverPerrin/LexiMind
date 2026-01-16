"""
Training and evaluation metrics for LexiMind.

Provides metric computation utilities for all task types: accuracy for topic
classification, multi-label F1 for emotion detection, and ROUGE/BLEU/BERTScore
for summarization quality assessment.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def accuracy(predictions: Sequence[int | str], targets: Sequence[int | str]) -> float:
    return cast(float, accuracy_score(targets, predictions))


def multilabel_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    preds = predictions.float()
    gold = targets.float()
    true_positive = (preds * gold).sum(dim=1)
    precision = true_positive / (preds.sum(dim=1).clamp(min=1.0))
    recall = true_positive / (gold.sum(dim=1).clamp(min=1.0))
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    return float(f1.mean().item())


def rouge_like(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions or not references:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not ref_tokens:
            scores.append(0.0)
            continue
        overlap = len(set(pred_tokens) & set(ref_tokens))
        scores.append(overlap / len(ref_tokens))
    return sum(scores) / len(scores)


def calculate_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Calculate BLEU-4 score."""
    if not predictions or not references:
        return 0.0

    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]  # BLEU expects list of references
        scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoother))

    return cast(float, sum(scores) / len(scores))


def calculate_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
    device: str | None = None,
) -> Dict[str, float]:
    """
    Calculate BERTScore for semantic similarity between predictions and references.
    
    BERTScore measures semantic similarity using contextual embeddings, making it
    more robust than n-gram based metrics like ROUGE for paraphrased content.
    
    Args:
        predictions: Generated summaries/descriptions
        references: Reference summaries/descriptions
        model_type: BERT model to use (default: deberta-xlarge-mnli for best quality)
        batch_size: Batch size for encoding
        device: Device to use (auto-detected if None)
    
    Returns:
        Dict with 'precision', 'recall', 'f1' BERTScore averages
    """
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Warning: bert-score not installed. Run: pip install bert-score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Calculate BERTScore
    P, R, F1 = bert_score(
        list(predictions),
        list(references),
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
    }


def calculate_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """
    Calculate proper ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
    
    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores
    """
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge-score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references, strict=False):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }


def calculate_all_summarization_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    include_bertscore: bool = True,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """
    Calculate comprehensive summarization metrics for research paper reporting.
    
    Includes:
    - ROUGE-1, ROUGE-2, ROUGE-L (lexical overlap)
    - BLEU-4 (n-gram precision)
    - BERTScore (semantic similarity)
    
    Args:
        predictions: Generated summaries/descriptions
        references: Reference summaries/descriptions
        include_bertscore: Whether to compute BERTScore (slower but valuable)
        bertscore_model: Model for BERTScore computation
    
    Returns:
        Dict with all metric scores
    """
    metrics: Dict[str, float] = {}
    
    # ROUGE scores
    rouge_scores = calculate_rouge(predictions, references)
    metrics.update({f"rouge_{k}": v for k, v in rouge_scores.items()})
    
    # BLEU score
    metrics["bleu4"] = calculate_bleu(predictions, references)
    
    # BERTScore (semantic similarity - important for back-cover style descriptions)
    if include_bertscore:
        bert_scores = calculate_bertscore(
            predictions, references, model_type=bertscore_model
        )
        metrics.update({f"bertscore_{k}": v for k, v in bert_scores.items()})
    
    return metrics


def classification_report_dict(
    predictions: Sequence[int | str], targets: Sequence[int | str], labels: List[str] | None = None
) -> Dict[str, Any]:
    """Generate a comprehensive classification report."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    # Type hint help for static analysis since average=None returns arrays
    precision = cast(np.ndarray, precision)
    recall = cast(np.ndarray, recall)
    f1 = cast(np.ndarray, f1)
    support = cast(np.ndarray, support)

    report = {}
    if labels:
        for i, label in enumerate(labels):
            report[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1-score": float(f1[i]),
                "support": int(support[i]),
            }

    # Macro average
    report["macro avg"] = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1-score": float(np.mean(f1)),
        "support": int(np.sum(support)),
    }

    return report


def get_confusion_matrix(
    predictions: Sequence[int | str], targets: Sequence[int | str], labels: List[str] | None = None
) -> np.ndarray:
    """Compute confusion matrix."""
    return cast(np.ndarray, confusion_matrix(targets, predictions, labels=labels))
