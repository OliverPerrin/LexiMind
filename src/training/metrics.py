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
    model_type: str = "roberta-large",  # Uses ~1.4GB VRAM vs ~6GB for deberta-xlarge
    batch_size: int = 16,
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
        "precision": float(P.mean().item()),  # type: ignore[union-attr]
        "recall": float(R.mean().item()),  # type: ignore[union-attr]
        "f1": float(F1.mean().item()),  # type: ignore[union-attr]
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


# --------------- Multi-label Emotion Metrics ---------------


def multilabel_macro_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute macro F1: average F1 per class (as in GoEmotions paper).
    
    This averages F1 across labels, giving equal weight to each emotion class 
    regardless of prevalence. Directly comparable to GoEmotions baselines.
    """
    preds = predictions.float()
    gold = targets.float()
    
    # Per-class TP, FP, FN
    tp = (preds * gold).sum(dim=0)
    fp = (preds * (1 - gold)).sum(dim=0)
    fn = ((1 - preds) * gold).sum(dim=0)
    
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    
    # Zero out F1 for classes with no support in either predictions or targets
    mask = (tp + fp + fn) > 0
    if mask.sum() == 0:
        return 0.0
    return float(f1[mask].mean().item())


def multilabel_micro_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute micro F1: aggregate TP/FP/FN across all classes.
    
    This gives more weight to frequent classes. Useful when class distribution matters.
    """
    preds = predictions.float()
    gold = targets.float()
    
    tp = (preds * gold).sum()
    fp = (preds * (1 - gold)).sum()
    fn = ((1 - preds) * gold).sum()
    
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    return float(f1.item())


def multilabel_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1 for multi-label classification.
    
    Returns a dict mapping class name/index to its metrics.
    """
    preds = predictions.float()
    gold = targets.float()
    num_classes = preds.shape[1]
    
    tp = (preds * gold).sum(dim=0)
    fp = (preds * (1 - gold)).sum(dim=0)
    fn = ((1 - preds) * gold).sum(dim=0)
    
    report: Dict[str, Dict[str, float]] = {}
    for i in range(num_classes):
        name = class_names[i] if class_names else str(i)
        p = (tp[i] / (tp[i] + fp[i]).clamp(min=1e-8)).item()
        r = (tp[i] / (tp[i] + fn[i]).clamp(min=1e-8)).item()
        f = (2 * p * r) / max(p + r, 1e-8)
        report[name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": int(gold[:, i].sum().item()),
        }
    return report


def tune_per_class_thresholds(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Sequence[float] | None = None,
) -> tuple[List[float], float]:
    """Tune per-class thresholds on validation set to maximize macro F1.
    
    For each class, tries multiple thresholds and selects the one that 
    maximizes that class's F1 score. This is standard practice for multi-label 
    classification (used in the original GoEmotions paper).
    
    Args:
        logits: Raw model logits (batch, num_classes)
        targets: Binary target labels (batch, num_classes)
        thresholds: Candidate thresholds to try (default: 0.1 to 0.9 by 0.05)
    
    Returns:
        Tuple of (best_thresholds_per_class, resulting_macro_f1)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.9, 0.05).tolist()]
    
    probs = torch.sigmoid(logits)
    num_classes = probs.shape[1]
    gold = targets.float()
    
    best_thresholds: List[float] = []
    for c in range(num_classes):
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            preds = (probs[:, c] >= t).float()
            tp = (preds * gold[:, c]).sum()
            fp = (preds * (1 - gold[:, c])).sum()
            fn = ((1 - preds) * gold[:, c]).sum()
            if tp + fp > 0 and tp + fn > 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = (2 * p * r / (p + r)).item()
            else:
                f1 = 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    
    # Compute resulting macro F1 with tuned thresholds
    tuned_preds = torch.zeros_like(probs)
    for c in range(num_classes):
        tuned_preds[:, c] = (probs[:, c] >= best_thresholds[c]).float()
    macro_f1 = multilabel_macro_f1(tuned_preds, targets)
    
    return best_thresholds, macro_f1


# --------------- Statistical Tests ---------------


def bootstrap_confidence_interval(
    scores: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Args:
        scores: Per-sample metric values
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 95%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    scores_arr = np.array(scores)
    n = len(scores_arr)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    
    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)
    
    return (
        float(np.mean(scores_arr)),
        bootstrap_means[lower_idx],
        bootstrap_means[min(upper_idx, n_bootstrap - 1)],
    )


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """Paired bootstrap significance test between two systems.
    
    Tests if system B is significantly better than system A.
    
    Args:
        scores_a: Per-sample scores from system A
        scores_b: Per-sample scores from system B
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
    
    Returns:
        p-value (probability that B is not better than A)
    """
    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Both score lists must have the same length"
    
    n = len(a)
    
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = float(np.mean(b[idx]) - np.mean(a[idx]))
        if diff <= 0:
            count += 1
    
    return count / n_bootstrap
