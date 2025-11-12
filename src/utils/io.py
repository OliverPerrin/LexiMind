"""Checkpoint IO helpers."""
from pathlib import Path

import torch


def save_state(model: torch.nn.Module, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)


def load_state(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)