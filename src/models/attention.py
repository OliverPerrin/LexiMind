"""
Attention mechanisms for Transformer architecture.

This module implements the core attention mechanisms used in the Transformer model:
- ScaledDotProductAttention: Fundamental attention operation
- MultiHeadAttention: Parallel attention with learned projections

Doing this first for Bottom-Up implementation of the Transformer

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
        # Params not needed here.
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
        # Getting Dimension for Scaling
        d_k = query.size(-1)
        
        # Compute Attention Scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Applying Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        return torch.matmul(attention_weights, value), attention_weights
    
# --------------- Multi-Head Attention ---------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    
    Transforming the input into query, key, and value representations
    
    Args:
        d_model: Dimension of model (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Assert that d_model is divisible by num_heads
        # Why? Because d_k = d_model // num_heads must be an integer
        assert d_model % num_heads == 0
        
        # Assume d_v always equals d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Create 4 linear layers (W_Q, W_K, W_V, W_O)
        # All should be nn.Linear(d_model, d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        # Create ScaledDotProductAttention instance
        self.attention = ScaledDotProductAttention()
        # Create dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # Split into heads
        # Reshape from (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_k), Apply to Q, K, V
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, d_k)
        # Now all are: (batch=2, num_heads=8, seq_len=10, d_k=64)
        
        # Handle mask broadcasting for multi-head attention
        if mask is not None:
            # If mask is 3D (batch, seq, seq), add head dimension
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
        # Now mask broadcasts across all heads: (batch, 1, seq, seq) → (batch, 8, seq, seq)
        
        # Apply attention
        output, attn_weights = self.attention(Q, K, V, mask)
        # output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        
        # Concatenate heads
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model) # -1 in view means 'infer this dimension'
        # After transpose, the tensor's memory layout
        # is "scattered", contiguous() just reorganizes it in memory
        
        # Final linear projection
        output = self.W_O(output)
        # Apply dropout
        output = self.dropout(output)
        
        return output, attn_weights