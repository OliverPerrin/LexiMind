"""
Transformer encoder implementation (Pre-LN).

Contains:
- TransformerEncoderLayer: one encoder block (self-attention + FFN with residuals + LayerNorm)
- TransformerEncoder: embedding + positional encoding + stack of encoder layers

Design choices:
- Pre-LN (LayerNorm before each sublayer) for stable training.
- The FeedForward module is position-wise and does NOT include residuals or normalization.
- MultiHeadAttention handles mask broadcasting from (B, S, S) -> (B, 1, S, S) internally.
- The encoder accepts either token ids (LongTensor) or precomputed embeddings (FloatTensor).
  If you pass token ids, provide vocab_size when constructing the encoder and optionally pad_token_id.
- Optionally collect attention weights by passing collect_attn=True to forward().
"""

from typing import Optional, Tuple, List, Union

import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer (Pre-LN).

    Args:
        d_model: model hidden size
        num_heads: number of attention heads
        d_ff: hidden dimension of the position-wise feed-forward network
        dropout: dropout probability applied to sublayer outputs
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        # set MHA internal dropout to 0.0 and use dropout1/dropout2 in the layer
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the encoder layer.

        Args:
            x: (batch, seq_len, d_model) - input embeddings / representations
            mask: optional attention mask, shape either (batch, seq_q, seq_k) or (batch, 1, seq_q, seq_k)

        Returns:
            x: (batch, seq_len, d_model)
            If you want attention weights, set collect_attn externally (the encoder stack can collect them).
        """
        # Self-attention sublayer (Pre-LN)
        x_norm = self.norm1(x)  # Pre-LN
        # self_attn expects query, key, value; for encoder they are the same
        attn_out, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward sublayer (Pre-LN)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        # Return output (and optionally attn_weights if caller wants to collect them)
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full encoder: token embedding + positional encoding + N encoder layers.

    Args:
        vocab_size: vocabulary size (ignored if you always pass embeddings)
        d_model: model hidden size
        num_layers: number of encoder layers to stack
        num_heads: number of attention heads
        d_ff: hidden dimension in FFN
        dropout: dropout probability (applied in positional encoding & residuals)
        max_len: maximum sequence length for positional encoding
        pad_token_id: optional token id for padding; if provided and input is token ids,
                      a padding mask will be constructed automatically
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Token embedding (only used if forward receives token ids)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (adds dropout internally)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Encoder layers stack
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
             for _ in range(num_layers)]
        )

        # Final LayerNorm for Pre-LN stacks (recommended)
        self.final_norm = nn.LayerNorm(d_model)

        # Dropout applied after embedding + positional encoding (paper uses this)
        self.input_dropout = nn.Dropout(dropout)

    def _build_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a 3D attention mask (batch, seq, seq) from input_ids and pad_token_id.
        True indicates valid positions; False indicates masked (pad).
        """
        assert self.pad_token_id is not None, "pad_token_id must be set to build padding mask from ids."
        # mask shape: (batch, seq) where True = token kept (non-pad)
        pad_mask = (input_ids != self.pad_token_id)
        # Convert to (batch, seq_q, seq_k) by outer product broadcasting
        # We want positions that are valid as both query and key
        attn_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
        # attn_mask dtype should be bool
        return attn_mask

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward through the encoder.

        Args:
            inputs: either
                - token ids: LongTensor of shape (batch, seq)
                - embeddings: FloatTensor of shape (batch, seq, d_model)
            mask: optional attention mask. If None and pad_token_id is set and inputs are token ids,
                  a padding mask will be created automatically with shape (batch, seq, seq).
                  The mask should be boolean where True indicates allowed attention.
            collect_attn: if True, returns (output, [attn_weights_per_layer]) where each entry is (batch, num_heads, seq, seq)

        Returns:
            output: (batch, seq, d_model)
            or (output, attn_list) if collect_attn True
        """
        # If inputs are token ids, embed them; otherwise assume they are embeddings
        if inputs.dim() == 2:  # token ids
            if self.embedding is None:
                raise ValueError("Encoder was not constructed with an embedding layer.")
            x = self.embedding(inputs) * math.sqrt(self.d_model)
        elif inputs.dim() == 3:  # already embeddings
            x = inputs
        else:
            raise ValueError("inputs must be (batch, seq) token ids or (batch, seq, d_model) embeddings")

        # Positional encoding + dropout
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # Build mask if needed
        if mask is None and inputs.dim() == 2 and self.pad_token_id is not None:
            mask = self._build_padding_mask(inputs)

        # Ensure mask is boolean and on the same device
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)

        attn_weights_per_layer: List[torch.Tensor] = []

        # Pass through each encoder layer (optionally collect attn)
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            if collect_attn:
                attn_weights_per_layer.append(attn)

        # Final normalization (Pre-LN stack)
        x = self.final_norm(x)

        if collect_attn:
            return x, attn_weights_per_layer
        return x