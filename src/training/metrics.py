"""Metric helpers used during training and evaluation."""

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
