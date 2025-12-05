"""
Configuration utilities for LexiMind.

Provides YAML configuration loading with validation.

Author: Oliver Perrin
Date: December 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    data: Dict[str, Any]


def load_yaml(path: str) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"YAML configuration '{path}' must contain a mapping at the root")
    return Config(data=content)
