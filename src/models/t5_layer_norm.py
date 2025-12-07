"""T5-style Layer Normalization (RMSNorm without mean centering).

T5 uses a variant of RMSNorm that does NOT subtract the mean.
This is critical for matching T5's behavior.
"""

import torch
import torch.nn as nn


class T5LayerNorm(nn.Module):
    """
    T5-style layer normalization without mean centering.

    This is similar to RMSNorm but does NOT subtract the mean from x.
    Formula: output = x / sqrt(mean(x^2) + eps) * weight

    Args:
        normalized_shape: Input shape (typically d_model)
        eps: Small constant for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (*, normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        # T5 uses variance = mean(x^2), does NOT subtract mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Scale by learned weight (no bias in T5)
        return self.weight * hidden_states
