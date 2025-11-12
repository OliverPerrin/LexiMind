"""Callback hooks for training."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_path: str,
    *,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist model and optimizer state for resuming training."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
    }
    if metrics:
        checkpoint["metrics"] = metrics

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.parent / f"{target.name}.tmp"
    try:
        torch.save(checkpoint, temp_path)
        temp_path.replace(target)
    except Exception:
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
