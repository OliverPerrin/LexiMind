"""
Attention mechanisms for Transformer architecture.

This module implements the core attention mechanisms used in the Transformer model:
- ScaledDotProductAttention: Fundamental attention operation
- MultiHeadAttention: Parallel attention with learned projections

Author: Oliver Perrin
Date: 2025-10-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in "Attention Is All You Need".
    
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    The scaling factor (1/sqrt(d_k)) prevents the dot products from growing too large,
    which would push the softmax into regions with extremely small gradients.
    
    Args:
        None - this module has no learnable parameters
        
    Forward Args:
        query: Query tensor of shape (batch, seq_len, d_k)
        key: Key tensor of shape (batch, seq_len, d_k)
        value: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor of shape (batch, seq_len, seq_len)
              True/1 values indicate positions to attend to, False/0 to mask
              
    Returns:
        output: Attention output of shape (batch, seq_len, d_v)
        attention_weights: Attention probability matrix (batch, seq_len, seq_len)
    
    TODO: Implement the forward method below
    Research questions to answer:
    1. Why divide by sqrt(d_k)? What happens without it?
    2. How does masking work? When do we need it?
    3. What's the computational complexity?
    """
    
    def __init__(self):
        super().__init__()
        # TODO: Do you need any parameters here?
        pass
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Implement this method
        
        Steps:
        1. Compute attention scores: scores = query @ key.transpose(-2, -1)
        2. Scale by sqrt(d_k)
        3. Apply mask if provided (set masked positions to -inf before softmax)
        4. Apply softmax to get attention weights
        5. Compute output: output = attention_weights @ value
        6. Return both output and attention_weights
        """
        pass


# TODO: After you implement ScaledDotProductAttention, we'll add MultiHeadAttention