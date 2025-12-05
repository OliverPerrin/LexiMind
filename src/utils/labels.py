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


@dataclass
class LabelMetadata:
    """Container for label vocabularies persisted after training."""

    emotion: List[str]
    topic: List[str]

    @property
    def emotion_size(self) -> int:
        return len(self.emotion)

    @property
    def topic_size(self) -> int:
        return len(self.topic)


def load_label_metadata(path: str | Path) -> LabelMetadata:
    """Load label vocabularies from a JSON file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    emotion = payload.get("emotion") if "emotion" in payload else payload.get("emotions")
    topic = payload.get("topic") if "topic" in payload else payload.get("topics")
    if not isinstance(emotion, list) or not all(isinstance(item, str) for item in emotion):
        raise ValueError("Label metadata missing 'emotion' list of strings")
    if not isinstance(topic, list) or not all(isinstance(item, str) for item in topic):
        raise ValueError("Label metadata missing 'topic' list of strings")

    return LabelMetadata(emotion=emotion, topic=topic)


def save_label_metadata(metadata: LabelMetadata, path: str | Path) -> None:
    """Persist label vocabularies to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "emotion": metadata.emotion,
        "topic": metadata.topic,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
