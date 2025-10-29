# src/models/positional_encoding.py

"""
Positional Encoding for Transformer models.

Injects information about the position of tokens in a sequence, since
self-attention has no inherent notion of token order.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from "Attention Is All You Need".
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        pos: position in sequence (0 to max_len-1)
        i: dimension index (0 to d_model/2)
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout probability to apply after adding positional encoding
    
    Shape:
        Input: (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
    
    Example:
        >>> pos_enc = PositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq, d_model)
        >>> output = pos_enc(x)
        >>> output.shape
        torch.Size([32, 100, 512])
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a tensor of positions: [0, 1, 2, ..., max_len-1]
        # Create a tensor of dimension indices: [0, 1, 2, ..., d_model-1]
        # Compute the division term: 10000^(2i/d_model)
        # Apply sin to even indices, cos to odd indices
        # Register as buffer (not a parameter, but part of state_dict)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            
        Returns:
            x with positional encoding added (batch, seq_len, d_model)
        """
        # Get sequence length from input
        # Add the appropriate slice of positional encoding
        # Apply dropout
        # Return result
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # self.pe contains pre-computed encodings for all positions
        # just need to add the first seq_len positions to x
        return self.dropout(x)