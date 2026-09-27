"""Validate observed plot inputs; no model execution and no synthetic fallback."""

from __future__ import annotations

import json
from pathlib import Path


def load_confusion_report(path: Path, task: str) -> tuple[list[str], list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get(task) if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"Missing task report: {task}")
    labels, matrix = record.get("labels"), record.get("confusion_matrix")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Confusion matrix requires its own ordered, unique labels")
    if not isinstance(matrix, list) or len(matrix) != len(labels):
        raise ValueError("Confusion matrix shape does not match labels")
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(labels):
            raise ValueError("Confusion matrix must be square")
        if any(type(value) is not int or value < 0 for value in row):
            raise ValueError("Confusion matrix counts must be nonnegative integers")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("Confusion matrix has no observed samples")
    if "num_samples" in record and (
        type(record["num_samples"]) is not int or record["num_samples"] != total
    ):
        raise ValueError("Confusion matrix total disagrees with the report's sample count")
    return labels, matrix
