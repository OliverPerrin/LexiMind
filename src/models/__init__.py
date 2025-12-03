"""
LexiMind custom transformer models.

This package provides a from-scratch transformer implementation with:
- TransformerEncoder/TransformerDecoder
- MultiHeadAttention, FeedForward, PositionalEncoding
- Task heads: ClassificationHead, TokenClassificationHead, LMHead
- MultiTaskModel: composable wrapper for encoder/decoder + task heads
"""

from .attention import MultiHeadAttention
from .decoder import TransformerDecoder, TransformerDecoderLayer, create_causal_mask
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .feedforward import FeedForward
from .heads import ClassificationHead, LMHead, ProjectionHead, TokenClassificationHead
from .multitask import MultiTaskModel
from .positional_encoding import PositionalEncoding

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
