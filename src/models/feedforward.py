"""Position-wise Feed-Forward Network.

This module implements the FFN sublayer used in Transformer blocks:
- Standard FFN: Two linear layers with activation (GELU/ReLU)
- Gated FFN: SwiGLU (LLaMA-style) or Gated-GELU (T5/FLAN-T5 style)

Author: Oliver Perrin
Date: 2025-10-23
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class FeedForward(nn.Module):
    """
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    Or with GELU: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    Or with SwiGLU: FFN(x) = (Swish(xW_gate) * xW_up)W_down
    Or with gated-gelu: FFN(x) = (GELU(xW_gate) * xW_up)W_down  (T5/FLAN-T5 style)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "swiglu", "gated-gelu"] = "gelu",
        quantization: Optional[str] = None,
    ):
        super().__init__()
        self.activation_type = activation

        # Select Linear layer type based on quantization
        Linear = nn.Linear
        kwargs = {}
        if quantization == "4bit":
            try:
                import bitsandbytes as bnb

                Linear = bnb.nn.Linear4bit  # type: ignore
                kwargs = {"compute_dtype": torch.bfloat16, "quant_type": "nf4"}
            except (ImportError, AttributeError):
                print("bitsandbytes not installed or incompatible, falling back to nn.Linear")
        elif quantization == "8bit":
            try:
                import bitsandbytes as bnb

                Linear = bnb.nn.Linear8bitLt  # type: ignore
            except (ImportError, AttributeError):
                print("bitsandbytes not installed or incompatible, falling back to nn.Linear")

        if activation in ("swiglu", "gated-gelu"):
            # Gated FFN requires 3 linear layers: Gate, Up, Down
            # - swiglu uses SiLU (Swish) activation (LLaMA style)
            # - gated-gelu uses GELU activation (T5/FLAN-T5 style)
            self.linear_gate = Linear(d_model, d_ff, **kwargs)  # Gate projection (wi_0)
            self.linear1 = Linear(d_model, d_ff, **kwargs)  # Up projection (wi_1)
            self.linear2 = Linear(d_ff, d_model, **kwargs)  # Down projection (wo)

            if activation == "swiglu":
                self.activation = nn.SiLU()  # Swish activation
            else:  # gated-gelu
                self.activation = (
                    nn.GELU()
                )  # GELU activation (T5 uses gelu_new which is very close)

            # Init gate
            if not quantization:
                init.xavier_uniform_(self.linear_gate.weight)
                init.zeros_(self.linear_gate.bias)
        else:
            self.linear1 = Linear(d_model, d_ff, **kwargs)  # w_1
            self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.linear2 = Linear(d_ff, d_model, **kwargs)  # w_2

        self.dropout = nn.Dropout(dropout)

        # Weight Initialization
        if not quantization:
            init.xavier_uniform_(self.linear1.weight)
            init.zeros_(self.linear1.bias)
            init.xavier_uniform_(self.linear2.weight)
            init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        if self.activation_type in ("swiglu", "gated-gelu"):
            # Gated FFN: (activation(xW_gate) * xW_up) W_down
            gate = self.activation(self.linear_gate(x))
            up = self.linear1(x)
            x = gate * up
            x = self.dropout(x)
            x = self.linear2(x)
        else:
            x = self.linear1(x)  # (batch, seq_len, d_ff)
            x = self.activation(x)  # activation
            x = self.dropout(x)  # dropout
            x = self.linear2(x)  # (batch, seq_len, d_model)
        return x
