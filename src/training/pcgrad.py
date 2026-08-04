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

        task_weights = task_weights or {}
        accum = gradient_accumulation_steps

        shared = [p for p in shared_params if p.requires_grad]
        heads = [p for p in (head_params or []) if p.requires_grad]
        if not shared and not heads:
            return {}

        all_params = shared + heads
        n_shared = len(shared)

        # Step 1: Per-task gradients over all params (single autograd pass per task).
        task_grads: Dict[str, List[torch.Tensor]] = {}
        for task_name, loss in task_losses.items():
            weight = task_weights.get(task_name, 1.0)
            scaled_loss = (loss * weight) / accum
            grads = torch.autograd.grad(
                scaled_loss,
                all_params,
                retain_graph=True,
                allow_unused=True,
            )
            task_grads[task_name] = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, all_params)
            ]

        # Step 2: Project conflicting shared-param gradients in place.
        shared_only = {name: grads[:n_shared] for name, grads in task_grads.items()}
        stats = self._project_conflicting_gradients(
            shared_only, list(task_losses.keys())
        )
        for name in task_grads:
            task_grads[name] = shared_only[name] + task_grads[name][n_shared:]

        # Step 3: Sum per-task grads (projected for shared, plain for heads)
        # and accumulate into .grad so gradient accumulation still works.
        n_tasks = len(task_grads)
        for i, p in enumerate(all_params):
            combined = torch.zeros_like(p)
            for name in task_grads:
                combined = combined + task_grads[name][i]
            if self.reduction == "mean" and n_tasks > 0:
                combined = combined / n_tasks
            if p.grad is None:
                p.grad = combined
            else:
                p.grad = p.grad + combined

        return stats

    def _project_conflicting_gradients(
        self,
        task_grads: Dict[str, List[torch.Tensor]],
        task_names: List[str],
    ) -> Dict[str, float]:
        """Project conflicting gradient pairs using PCGrad algorithm.

        Modifies task_grads in-place. For each pair (i, j), if cosine similarity
        is negative, projects g_i onto the normal plane of g_j.

        Returns cosine similarity stats for logging.
        """
        stats: Dict[str, float] = {}

        flat_grads: Dict[str, torch.Tensor] = {}
        for name in task_names:
            flat_grads[name] = torch.cat([g.flatten() for g in task_grads[name]])

        order = list(range(len(task_names)))
        random.shuffle(order)

        for idx_i in order:
            name_i = task_names[idx_i]
            for idx_j in order:
                if idx_i == idx_j:
                    continue
                name_j = task_names[idx_j]

                g_i = flat_grads[name_i]
                g_j = flat_grads[name_j]

                cos_sim = F.cosine_similarity(g_i.unsqueeze(0), g_j.unsqueeze(0)).item()

                pair_key = "_".join(sorted([name_i, name_j]))
                if f"cos_sim_{pair_key}" not in stats:
                    stats[f"cos_sim_{pair_key}"] = cos_sim
                    stats[f"conflict_{pair_key}"] = 1.0 if cos_sim < 0 else 0.0

                if cos_sim < 0:
                    self._conflict_count += 1
                    dot = torch.dot(g_i, g_j)
                    g_j_norm_sq = torch.dot(g_j, g_j)
                    if g_j_norm_sq > 1e-12:
                        proj_coeff = dot / g_j_norm_sq
                        g_i_projected = g_i - proj_coeff * g_j

                        offset = 0
                        for k, grad in enumerate(task_grads[name_i]):
                            numel = grad.numel()
                            task_grads[name_i][k] = g_i_projected[
                                offset : offset + numel
                            ].reshape(grad.shape)
                            offset += numel

                        flat_grads[name_i] = g_i_projected

                self._total_pairs += 1

        if self._total_pairs > 0:
            stats["conflict_rate"] = self._conflict_count / self._total_pairs

        return stats

    def reset_stats(self) -> None:
        """Reset running conflict statistics (call at epoch boundaries)."""
        self._conflict_count = 0
        self._total_pairs = 0