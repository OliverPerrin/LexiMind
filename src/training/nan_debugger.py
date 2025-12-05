"""
NaN debugging utilities for training.

Helps identify where NaNs originate in the model during training.

Author: Oliver Perrin
Date: December 2025
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class NaNDetector:
    """Detect and log NaNs in model parameters and gradients."""

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.nan_count = 0
        self.max_nans = 10

    def check_forward(self, outputs: torch.Tensor, loss: torch.Tensor, step: int) -> bool:
        """Check for NaNs in forward pass. Returns True if NaN found."""
        if not self.enabled:
            return False

        has_nan = False

        if torch.isnan(outputs).any():
            print(f"\n{'=' * 60}")
            print(f"⚠ NaN detected in MODEL OUTPUTS at step {step}")
            print(f"Output shape: {outputs.shape}")
            print(f"NaN count: {torch.isnan(outputs).sum().item()}")
            print(f"{'=' * 60}\n")
            has_nan = True

        if torch.isnan(loss):
            print(f"\n{'=' * 60}")
            print(f"⚠ NaN detected in LOSS at step {step}")
            print(f"Loss value: {loss.item()}")
            print(f"{'=' * 60}\n")
            has_nan = True

        if has_nan:
            self.nan_count += 1
            if self.nan_count >= self.max_nans:
                print(f"\n⚠ Too many NaNs ({self.nan_count}), stopping training")

        return has_nan

    def check_gradients(self, step: int) -> Optional[Tuple[str, torch.Tensor]]:
        """Check gradients for NaNs/Infs after backward pass."""
        if not self.enabled:
            return None

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"\n{'=' * 60}")
                    print(f"⚠ NaN in GRADIENT: {name}")
                    print(f"  Step: {step}")
                    print(f"  Grad shape: {param.grad.shape}")
                    print(f"  NaN count: {torch.isnan(param.grad).sum().item()}")
                    print(f"{'=' * 60}\n")
                    return (name, param.grad)

                if torch.isinf(param.grad).any():
                    print(f"\n{'=' * 60}")
                    print(f"⚠ Inf in GRADIENT: {name}")
                    print(f"  Step: {step}")
                    print(f"  Inf count: {torch.isinf(param.grad).sum().item()}")
                    print(f"{'=' * 60}\n")
                    return (name, param.grad)

        return None

    def check_parameters(self, step: int) -> Optional[str]:
        """Check parameters for NaNs/Infs."""
        if not self.enabled:
            return None

        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"\n{'=' * 60}")
                print(f"⚠ NaN in PARAMETER: {name}")
                print(f"  Step: {step}")
                print(f"{'=' * 60}\n")
                return str(name)

            if torch.isinf(param).any():
                print(f"\n{'=' * 60}")
                print(f"⚠ Inf in PARAMETER: {name}")
                print(f"  Step: {step}")
                print(f"{'=' * 60}\n")
                return str(name)

        return None


def gradient_stats(model: nn.Module) -> dict:
    """Get gradient statistics for debugging."""
    stats = {
        "max_grad": 0.0,
        "min_grad": float("inf"),
        "mean_grad": 0.0,
        "num_grads": 0,
    }

    grad_norms = []
    for _name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            stats["max_grad"] = max(stats["max_grad"], param.grad.abs().max().item())
            stats["min_grad"] = min(stats["min_grad"], param.grad.abs().min().item())
            stats["num_grads"] += 1

    if grad_norms:
        stats["mean_grad"] = sum(grad_norms) / len(grad_norms)

    return stats
