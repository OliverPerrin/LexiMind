"""
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.

Implements the gradient surgery algorithm from:
    Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)

When task gradients conflict (negative cosine similarity), PCGrad projects
each gradient onto the normal plane of the conflicting gradient, reducing
negative transfer between tasks.

Key equation: if ⟨g_i, g_j⟩ < 0, replace g_i with g_i − (⟨g_i, g_j⟩ / ‖g_j‖²) g_j

Implementation notes:
- Per-task gradients are computed in a single ``torch.autograd.grad`` call
  over both shared (encoder) and head/decoder parameters, avoiding a
  second full backward pass through the graph.
- Projection is applied only to the shared-parameter portion; head grads
  are passed through unchanged (a head only receives gradient from its
  own task so there is nothing to project).

Usage:
    pcgrad = PCGrad()
    task_losses = {"summ": loss1, "emotion": loss2, "topic": loss3}
    stats = pcgrad.backward(
        task_losses, shared_params, head_params,
        task_weights=weights, gradient_accumulation_steps=accum,
    )

Author: Oliver Perrin
Date: April 2026
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


class PCGrad:
    """PCGrad optimizer wrapper for multi-task gradient conflict resolution.

    Computes per-task gradients independently, detects conflicts via cosine
    similarity, and projects conflicting gradients onto compatible directions.
    """

    def __init__(self, reduction: str = "sum"):
        """
        Args:
            reduction: How to combine projected gradients ("sum" or "mean")
        """
        if reduction not in {"sum", "mean"}:
            raise ValueError("PCGrad reduction must be sum or mean")
        self.reduction = reduction
        self._conflict_count = 0
        self._total_pairs = 0

    def backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: Sequence[torch.nn.Parameter],
        head_params: Optional[Sequence[torch.nn.Parameter]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """Compute PCGrad-projected gradients for shared params and standard
        gradients for head params, then accumulate into ``.grad``.

        A single ``torch.autograd.grad`` call per task computes gradients for
        shared + head parameters jointly, so we never pay for a second
        backward pass through the graph. Projection is applied only to the
        shared-parameter portion of each task's gradient vector.

        Args:
            task_losses: Dict mapping task name -> scalar loss tensor.
            shared_params: Parameters subject to PCGrad projection
                (typically the encoder).
            head_params: Parameters that are task-specific (decoder + heads).
                Their grads are summed across tasks without projection.
                Defaults to an empty list.
            task_weights: Optional per-task loss weights.
            gradient_accumulation_steps: Divide gradients by this factor.

        Returns:
            Dict with per-pair cosine similarity and conflict statistics.
        """
        if not task_losses:
            return {}

        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        task_weights = task_weights or {}
        shared = [p for p in shared_params if p.requires_grad]
        heads = [p for p in (head_params or []) if p.requires_grad]
        all_params = shared + heads
        if not all_params:
            return {}
        if len({id(p) for p in all_params}) != len(all_params):
            raise ValueError("Shared and private parameter lists must be disjoint and unique")
        n_shared = len(shared)
        # Keep original shared gradients for projection, and sum private grads
        # directly. An unused private head needs neither a dense zero allocation
        # per task nor a zero .grad that would trigger AdamW weight decay.
        task_grads: Dict[str, List[torch.Tensor]] = {}
        private_totals: List[torch.Tensor | None] = [None] * len(heads)
        shared_used = [False] * n_shared
        for task_index, (task_name, loss) in enumerate(task_losses.items()):
            scaled_loss = loss * task_weights.get(task_name, 1.0) / gradient_accumulation_steps
            grads = torch.autograd.grad(
                scaled_loss,
                all_params,
                retain_graph=task_index + 1 < len(task_losses),
                allow_unused=True,
            )
            task_grads[task_name] = []
            for i, (param, grad) in enumerate(zip(shared, grads[:n_shared], strict=True)):
                shared_used[i] |= grad is not None
                task_grads[task_name].append(grad if grad is not None else torch.zeros_like(param))
            for i, grad in enumerate(grads[n_shared:]):
                if grad is not None:
                    private_totals[i] = (
                        grad if private_totals[i] is None else private_totals[i] + grad
                    )

        stats = self._project_conflicting_gradients(task_grads, list(task_losses)) if shared else {}
        divisor = len(task_losses) if self.reduction == "mean" else 1
        combined: List[torch.Tensor | None] = [
            sum((grads[i] for grads in task_grads.values()), torch.zeros_like(param))
            if shared_used[i]
            else None
            for i, param in enumerate(shared)
        ] + private_totals
        for param, grad in zip(all_params, combined, strict=True):
            if grad is None:
                continue
            grad = grad / divisor
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad)
        return stats

    def _project_conflicting_gradients(
        self,
        task_grads: Dict[str, List[torch.Tensor]],
        task_names: List[str],
    ) -> Dict[str, float]:
        """Project each gradient against fixed, original other-task gradients.

        The progressively projected gradient belongs only to the current task;
        using it as the next task's reference changes the PCGrad algorithm.
        """
        stats: Dict[str, float] = {}
        originals = {
            name: torch.cat([g.flatten() for g in task_grads[name]]) for name in task_names
        }
        # Report symmetric conflicts from original gradients, not from whichever
        # partially projected pair happened to be visited first.
        for i, first in enumerate(task_names):
            for second in task_names[i + 1 :]:
                cosine = F.cosine_similarity(originals[first][None], originals[second][None]).item()
                key = "_".join(sorted([first, second]))
                stats[f"cos_sim_{key}"] = cosine
                stats[f"conflict_{key}"] = float(cosine < 0)
                self._total_pairs += 1
                self._conflict_count += int(cosine < 0)

        for name in task_names:
            projected = originals[name].clone()
            others = [other for other in task_names if other != name]
            random.shuffle(others)
            for other in others:
                reference = originals[other]
                dot = torch.dot(projected, reference)
                norm_sq = torch.dot(reference, reference)
                if dot < 0 and norm_sq > 1e-12:
                    projected = projected - dot / norm_sq * reference
            offset = 0
            for i, grad in enumerate(task_grads[name]):
                size = grad.numel()
                task_grads[name][i] = projected[offset : offset + size].reshape(grad.shape)
                offset += size
        if self._total_pairs:
            stats["conflict_rate"] = self._conflict_count / self._total_pairs
        return stats

    def reset_stats(self) -> None:
        """Reset running conflict statistics (call at epoch boundaries)."""
        self._conflict_count = 0
        self._total_pairs = 0
