"""YAML config loader."""

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
