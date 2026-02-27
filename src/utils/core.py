"""
Utility functions for LexiMind.

Consolidated utilities including:
- Model checkpoint I/O
- Label metadata handling
- Seed management for reproducibility

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

# --------------- Checkpoint I/O ---------------


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Save model state dict, handling torch.compile artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Strip '_orig_mod.' prefix from compiled models
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save(state_dict, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Load model state dict, handling torch.compile artifacts."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)


# --------------- Label Metadata ---------------


@dataclass
class LabelMetadata:
    """Container for emotion and topic label vocabularies."""

    emotion: List[str]
    topic: List[str]

    @property
    def num_emotions(self) -> int:
        return len(self.emotion)

    @property
    def num_topics(self) -> int:
        return len(self.topic)


def load_labels(path: str | Path) -> LabelMetadata:
    """Load label metadata from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    emotion = data.get("emotion") or data.get("emotions", [])
    topic = data.get("topic") or data.get("topics", [])

    if not emotion or not topic:
        raise ValueError("Labels file must contain 'emotion' and 'topic' lists")

    return LabelMetadata(emotion=emotion, topic=topic)


def save_labels(labels: LabelMetadata, path: str | Path) -> None:
    """Save label metadata to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump({"emotion": labels.emotion, "topic": labels.topic}, f, indent=2)


# --------------- Reproducibility ---------------


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------- Config Loading ---------------


@dataclass
class Config:
    """Simple config wrapper."""

    data: dict


def load_yaml(path: str | Path) -> Config:
    """Load YAML configuration file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"YAML '{path}' must contain a mapping")
    return Config(data=content)
