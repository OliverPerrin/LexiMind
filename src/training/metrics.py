"""Metric helpers used during training and evaluation."""
from __future__ import annotations

from typing import Sequence

import torch


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    matches = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return matches / max(1, len(predictions))


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
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not ref_tokens:
            scores.append(0.0)
            continue
        overlap = len(set(pred_tokens) & set(ref_tokens))
        scores.append(overlap / len(ref_tokens))
    return sum(scores) / len(scores)
