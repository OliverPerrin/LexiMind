"""
Label metadata utilities for LexiMind.

Manages persistence and loading of emotion and topic label vocabularies
for multitask inference.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .atomic import atomic_write


def _validate_labels(labels: object, task: str) -> list[str]:
    # An explicit empty list preserves metadata for an absent single-task head.
    if not isinstance(labels, list) or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError(f"Label metadata requires a '{task}' list of nonblank strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Label metadata has duplicate '{task}' labels")
    return list(labels)


@dataclass
class LabelMetadata:
    """Container for label vocabularies persisted after training."""

    emotion: List[str]
    topic: List[str]

    def __post_init__(self) -> None:
        self.emotion = _validate_labels(self.emotion, "emotion")
        self.topic = _validate_labels(self.topic, "topic")

    @property
    def emotion_size(self) -> int:
        return len(self.emotion)

    @property
    def topic_size(self) -> int:
        return len(self.topic)

    @property
    def num_emotions(self) -> int:
        """Compatibility with the original core.LabelMetadata API."""
        return self.emotion_size

    @property
    def num_topics(self) -> int:
        return self.topic_size


def load_label_metadata(path: str | Path) -> LabelMetadata:
    """Load label vocabularies from a JSON file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Label metadata must be a JSON object")
    emotion = payload.get("emotion") if "emotion" in payload else payload.get("emotions")
    topic = payload.get("topic") if "topic" in payload else payload.get("topics")
    return LabelMetadata(
        emotion=_validate_labels(emotion, "emotion"), topic=_validate_labels(topic, "topic")
    )


def save_label_metadata(metadata: LabelMetadata, path: str | Path) -> None:
    """Persist label vocabularies to JSON."""

    if not isinstance(metadata, LabelMetadata):
        raise ValueError("Expected LabelMetadata when saving label vocabularies")
    # Revalidate mutable lists before opening any destination, preserving order.
    payload = {
        "emotion": _validate_labels(metadata.emotion, "emotion"),
        "topic": _validate_labels(metadata.topic, "topic"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(encoded))
