"""
LexiMind custom transformer models.

This package provides a from-scratch transformer implementation with:
- TransformerEncoder/TransformerDecoder
- MultiHeadAttention, FeedForward, PositionalEncoding
- Task heads: ClassificationHead, TokenClassificationHead, LMHead
- MultiTaskModel: composable wrapper for encoder/decoder + task heads
"""

from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoder import TransformerDecoder, TransformerDecoderLayer, create_causal_mask
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding
from .heads import ClassificationHead, TokenClassificationHead, LMHead, ProjectionHead
from .multitask import MultiTaskModel

__all__ = [
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "create_causal_mask",
    "MultiHeadAttention",
    "FeedForward",
    "PositionalEncoding",
    "ClassificationHead",
    "TokenClassificationHead",
    "LMHead",
    "ProjectionHead",
    "MultiTaskModel",
]
