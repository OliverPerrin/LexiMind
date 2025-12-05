"""
Checkpoint I/O utilities for LexiMind.

Handles model state serialization with support for torch.compile artifacts.

Author: Oliver Perrin
Date: December 2025
"""

from pathlib import Path

import torch


def save_state(model: torch.nn.Module, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Handle torch.compile artifacts: strip '_orig_mod.' prefix
    state_dict = model.state_dict()
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        clean_state_dict[new_k] = v

    torch.save(clean_state_dict, destination)


def load_state(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)

    # Handle torch.compile artifacts in loaded checkpoints
    clean_state = {}
    for k, v in state.items():
        new_k = k.replace("_orig_mod.", "")
        clean_state[new_k] = v

    model.load_state_dict(clean_state)
