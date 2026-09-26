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

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Preserve the original public entry points while sharing validation and I/O.
from .io import load_state, save_state
from .labels import LabelMetadata, load_label_metadata, save_label_metadata


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    save_state(model, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    load_state(model, path)


def load_labels(path: str | Path) -> LabelMetadata:
    return load_label_metadata(path)


def save_labels(labels: LabelMetadata, path: str | Path) -> None:
    save_label_metadata(labels, path)


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
