"""
Checkpoint I/O utilities for LexiMind.

Handles model state serialization with support for torch.compile artifacts.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .atomic import atomic_write


def _clean_key(key: str) -> str:
    """Remove compiler wrapper segments, preserving ordinary module names."""
    return ".".join(part for part in key.split(".") if part != "_orig_mod")


def _wrapper_child(parent: str, child: str) -> bool:
    """Is child the same module reached through additional wrapper layers?"""
    prefix = parent + "." if parent else ""
    if not child.startswith(prefix):
        return False
    suffix = child[len(prefix) :].split(".")
    return bool(suffix) and all(part == "_orig_mod" for part in suffix)


def normalize_state_dict(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Normalize compile wrappers without dropping tensors or version metadata.

    Wrapper modules and their wrapped modules normally both have metadata. The
    wrapped module's version wins over its wrapper's version. All other key
    collisions fail instead of silently selecting a tensor or metadata record.
    """
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must be a state-dict mapping")
    cleaned: OrderedDict[str, Any] = OrderedDict()
    originals: dict[str, str] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Checkpoint state keys must be nonempty strings")
        normalized = _clean_key(key)
        if not normalized:
            raise ValueError(f"Checkpoint key contains only compiler wrappers: {key!r}")
        if normalized in cleaned:
            raise ValueError(
                f"Checkpoint key collision after wrapper cleanup: {originals[normalized]!r}, {key!r}"
            )
        cleaned[normalized] = value
        originals[normalized] = key

    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("Checkpoint version metadata must be a mapping")
        clean_metadata: OrderedDict[str, Any] = OrderedDict()
        metadata_originals: dict[str, str] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise ValueError("Checkpoint version metadata must map module names to mappings")
            normalized = _clean_key(key)
            previous = metadata_originals.get(normalized)
            if previous is not None:
                if _wrapper_child(key, previous):
                    continue  # Keep the already-seen wrapped module's metadata.
                if not _wrapper_child(previous, key):
                    raise ValueError(
                        f"Checkpoint metadata collision after wrapper cleanup: {previous!r}, {key!r}"
                    )
            clean_metadata[normalized] = dict(value)
            metadata_originals[normalized] = key
        cleaned._metadata = clean_metadata  # type: ignore[attr-defined]
    return cleaned


def save_state(model: torch.nn.Module, path: str | Path) -> None:
    cleaned = normalize_state_dict(model.state_dict())
    atomic_write(path, lambda stream: torch.save(cleaned, stream))


def load_state(model: torch.nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    clean_state = normalize_state_dict(state)

    model.load_state_dict(clean_state)
