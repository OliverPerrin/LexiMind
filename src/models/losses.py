"""Observed-label losses shared by the model and training loop."""

import torch
import torch.nn.functional as F


def observed_label_mask(
    values: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Validate a B x C binary target contract without inspecting unknown values."""
    if values.ndim != 2 or values.shape != labels.shape or not values.shape[1]:
        raise ValueError("Multi-label logits and labels must have the same B x C shape")
    if not labels.is_floating_point():
        raise ValueError("Multi-label labels must be floating point")
    if labels.device != values.device:
        raise ValueError("Multi-label labels and logits must be on the same device")
    if label_mask is None:
        label_mask = torch.ones_like(labels, dtype=torch.bool)
    elif label_mask.dtype != torch.bool or label_mask.shape != labels.shape:
        raise ValueError("label_mask must be bool with the same B x C shape as labels")
    elif label_mask.device != labels.device:
        raise ValueError("label_mask and labels must be on the same device")
    known = labels[label_mask]
    if not torch.all(torch.isfinite(known) & ((known == 0) | (known == 1))):
        raise ValueError("Observed multi-label targets must be finite binary values (0 or 1)")
    return label_mask


def masked_binary_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean BCE over observed labels; unknown cells have exactly zero gradient.

    A missing mask means fully observed labels. Selecting before BCE also
    excludes NaN/Inf placeholders in unknown cells. An empty selection sums to
    a finite, differentiable zero, including when unknown logits are nonfinite.
    """
    if not logits.is_floating_point():
        raise ValueError("Multi-label logits must be floating point")
    mask = observed_label_mask(logits, labels, label_mask)
    known_logits = logits[mask]
    if known_logits.numel() == 0:
        return known_logits.sum()
    return F.binary_cross_entropy_with_logits(known_logits, labels[mask])
