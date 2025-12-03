"""
Prediction heads for Transformer models.

Includes:
- ClassificationHead: sequence-level classification with simple pooling (mean/cls/max).
- TokenClassificationHead: per-token classification (e.g., NER).
- LMHead: language-modeling head mapping hidden states to vocabulary logits. Optional weight tying to an Embedding.
- ProjectionHead: small projection MLP for representation learning / contrastive heads.

Keep these heads minimal, well-tested, and easy to compose on top of encoder/decoder outputs.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Sequence-level classification head.

    Args:
        d_model: hidden size from encoder/decoder
        num_labels: number of output classes
        pooler: one of 'mean', 'cls', 'max' - how to pool the sequence
        dropout: dropout probability before final linear layer
    """

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        pooler: Literal["mean", "cls", "max"] = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert pooler in ("mean", "cls", "max"), "pooler must be 'mean'|'cls'|'max'"
        self.pooler = pooler
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, num_labels)
        """
        if self.pooler == "mean":
            pooled = x.mean(dim=1)
        elif self.pooler == "cls":
            pooled = x[:, 0, :]
        else:  # max
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
