"""General utilities for LexiMind, loaded only when their public names are used."""

from importlib import import_module
from typing import Any

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_state",
    "load_state",
    "LabelMetadata",
    "load_labels",
    "save_labels",
    "load_label_metadata",
    "save_label_metadata",
    "set_seed",
    "Config",
    "load_yaml",
]

_MODULES = {
    "save_checkpoint": ".core",
    "load_checkpoint": ".core",
    "save_state": ".io",
    "load_state": ".io",
    "LabelMetadata": ".labels",
    "load_labels": ".core",
    "save_labels": ".core",
    "load_label_metadata": ".labels",
    "save_label_metadata": ".labels",
    "set_seed": ".core",
    "Config": ".core",
    "load_yaml": ".core",
}


def __getattr__(name: str) -> Any:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
