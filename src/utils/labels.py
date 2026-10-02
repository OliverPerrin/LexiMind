"""
Label metadata utilities for LexiMind.

Manages persistence and loading of emotion and topic label vocabularies
for multitask inference.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

from .atomic import atomic_write

BOOK_INPUT_FORMAT = "book_title_description_v1"
BOOK_FIELDS = ("genre", "topic", "form", "audience")


def format_book_input(title: str, description: str) -> str:
    """One training/inference format containing only the two allowed input fields."""
    if not isinstance(title, str) or not isinstance(description, str):
        raise ValueError("Book title and description must be strings")
    if not title.strip() and not description.strip():
        raise ValueError("Book input must contain a title or description")
    return f"Title: {title}\nDescription: {description}"


def validate_book_field_labels(labels: object) -> list[str]:
    values = _validate_labels(labels, "book fields")
    if not values or any(
        re.fullmatch(r"(genre|topic|form|audience):[a-z][a-z_]*", value) is None for value in values
    ):
        raise ValueError("Book field columns require nonempty facet:label identifiers")
    return values


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
    topic_problem_type: Literal["single_label", "multi_label"] = "single_label"
    topic_input_format: str = "text"
    topic_mapping_sha256: str | None = None

    def __post_init__(self) -> None:
        self.emotion = _validate_labels(self.emotion, "emotion")
        self.topic = _validate_labels(self.topic, "topic")
        if self.topic_problem_type == "multi_label":
            self.topic = validate_book_field_labels(self.topic)
            if (
                self.topic_input_format != BOOK_INPUT_FORMAT
                or not isinstance(self.topic_mapping_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.topic_mapping_sha256) is None
            ):
                raise ValueError(
                    "Book field metadata requires its input format and mapping SHA-256"
                )
        elif (
            self.topic_problem_type != "single_label"
            or self.topic_input_format != "text"
            or self.topic_mapping_sha256 is not None
        ):
            raise ValueError("Invalid topic problem type or input/mapping metadata")

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
        emotion=_validate_labels(emotion, "emotion"),
        topic=_validate_labels(topic, "topic"),
        topic_problem_type=payload.get("topic_problem_type", "single_label"),
        topic_input_format=payload.get("topic_input_format", "text"),
        topic_mapping_sha256=payload.get("topic_mapping_sha256"),
    )


def save_label_metadata(metadata: LabelMetadata, path: str | Path) -> None:
    """Persist label vocabularies to JSON."""

    if not isinstance(metadata, LabelMetadata):
        raise ValueError("Expected LabelMetadata when saving label vocabularies")
    metadata.__post_init__()
    # Revalidate mutable lists before opening any destination, preserving order.
    payload: dict[str, object] = {
        "emotion": _validate_labels(metadata.emotion, "emotion"),
        "topic": _validate_labels(metadata.topic, "topic"),
    }
    if metadata.topic_problem_type == "multi_label":
        payload.update(
            topic_problem_type=metadata.topic_problem_type,
            topic_input_format=metadata.topic_input_format,
            topic_mapping_sha256=metadata.topic_mapping_sha256,
        )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(encoded))
