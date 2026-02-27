"""Prediction heads for Transformer models.

This module provides task-specific output heads:
- ClassificationHead: Sequence-level classification with pooling (mean/cls/max/attention)
- TokenClassificationHead: Per-token classification (NER, POS tagging)
- LMHead: Language modeling logits with optional weight tying
- ProjectionHead: MLP for representation learning / contrastive tasks

Author: Oliver Perrin
Date: 2025-10-23
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence positions.

    Computes a weighted sum of hidden states using a learned query vector,
    producing a single fixed-size representation. This is generally superior
    to mean pooling for classification tasks on encoder-decoder models where
    hidden states are optimized for cross-attention rather than pooling.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) - True for valid tokens, False for padding
        returns: (batch, d_model)
        """
        # Compute attention scores: (batch, seq_len, 1)
        scores = self.query(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        # Weighted sum: (batch, d_model)
        return (weights * x).sum(dim=1)


class ClassificationHead(nn.Module):
    """
    Sequence-level classification head.

    Args:
        d_model: hidden size from encoder/decoder
        num_labels: number of output classes
        pooler: one of 'mean', 'cls', 'max', 'attention' - how to pool the sequence
        dropout: dropout probability before final linear layer
        hidden_dim: optional intermediate dimension for 2-layer MLP (improves capacity)
    """

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        pooler: Literal["mean", "cls", "max", "attention"] = "mean",
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        assert pooler in ("mean", "cls", "max", "attention"), "pooler must be 'mean'|'cls'|'max'|'attention'"
        self.pooler = pooler
        self.dropout = nn.Dropout(dropout)

        if pooler == "attention":
            self.attn_pool = AttentionPooling(d_model)
        
        # Optional 2-layer MLP for more capacity (useful for multi-label)
        if hidden_dim is not None:
            self.out_proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.out_proj = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) - True for valid tokens, False for padding
        returns: (batch, num_labels)
        """
        if self.pooler == "attention":
            pooled = self.attn_pool(x, mask)
        elif self.pooler == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x = x * mask_expanded
                sum_embeddings = x.sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = x.mean(dim=1)
        elif self.pooler == "cls":
            pooled = x[:, 0, :]
        else:  # max
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                x = x.masked_fill(~mask_expanded, float("-inf"))
            pooled, _ = x.max(dim=1)
        pooled = self.dropout(pooled)
        return self.out_proj(pooled)


class TokenClassificationHead(nn.Module):
    """
    Per-token classification head. Useful for NER, POS, etc.

    Args:
        d_model: hidden size
        num_labels: number of per-token classes
        dropout: dropout probability applied before the linear layer
    """

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, num_labels)
        """
        x = self.dropout(x)
        return self.out_proj(x)


class LMHead(nn.Module):
    """
    Language modeling head: maps hidden states to logits over vocabulary.

    Args:
        d_model: hidden size
        vocab_size: vocabulary size
        tie_embedding: optional nn.Embedding instance to tie weights with
    """

    def __init__(self, d_model: int, vocab_size: int, tie_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.proj = nn.Linear(d_model, vocab_size, bias=True)

        if tie_embedding is not None:
            # Validate sizes
            assert tie_embedding.num_embeddings == vocab_size, (
                "vocab size mismatch for weight tying"
            )
            assert tie_embedding.embedding_dim == d_model, (
                "embedding dim must match d_model for weight tying"
            )
            # Tie weights: point the projection weight to the embedding weight Tensor
            # Remove the existing projection parameter in favor of the embedding weight
            # This keeps the same Parameter object, so updates affect both modules.
            self.proj.weight = tie_embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, d_model)
        returns logits: (batch, seq_len, vocab_size)
        """
        return self.proj(hidden_states)


class ProjectionHead(nn.Module):
    """
    Simple projection head for representation learning.

    Args:
        d_model: input dimension
        proj_dim: output projection dimension
        hidden_dim: intermediate dimension (optional)
        dropout: dropout probability
    """

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(d_model, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_model) or (batch, seq_len, d_model) - both supported.
        Returns:
            If input is 3D: (batch, seq_len, proj_dim)
            If input is 2D: (batch, proj_dim)
        """
        orig_dim = x.dim()
        if orig_dim == 3:
            B, T, D = x.shape
            out = self.net(x.view(B * T, D))
            return out.view(B, T, -1)
        elif orig_dim == 2:
            return self.net(x)
        else:
            raise ValueError("Input must be 2D or 3D tensor")
