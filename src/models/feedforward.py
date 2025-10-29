"""
Position-wise Feed-Forward Network.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Literal

class FeedForward(nn.Module):
    """
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Or with GELU: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: Literal["gelu", "relu"] = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # w_1
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # w_2
        
        # Weight Initialization
        init.xavier_uniform_(self.linear1.weight)
        init.zeros_(self.linear1.bias)
        init.xavier_uniform_(self.linear2.weight)
        init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        x = self.linear1(x)       # (batch, seq_len, d_ff)
        x = self.activation(x)    # activation
        x = self.dropout(x)       # dropout
        x = self.linear2(x)       # (batch, seq_len, d_model)
        return x
    