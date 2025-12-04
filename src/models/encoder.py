"""
Transformer encoder implementation (Pre-LN).

Contains:
- TransformerEncoderLayer: one encoder block (self-attention + FFN with residuals + LayerNorm (RMSNorm - modern convention))
- TransformerEncoder: embedding + positional encoding + stack of encoder layers

Design choices:
- Pre-LN (RMSNorm before each sublayer) for stable training.
- The FeedForward module is position-wise and does NOT include residuals or normalization.
- MultiHeadAttention handles mask broadcasting from (B, S, S) -> (B, 1, S, S) internally.
- The encoder accepts either token ids (LongTensor) or precomputed embeddings (FloatTensor).
  If you pass token ids, provide vocab_size when constructing the encoder and optionally pad_token_id.
- Optionally collect attention weights by passing collect_attn=True to forward().
"""

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

# Encoder implementation
from .attention import MultiHeadAttention, T5RelativePositionBias
from .feedforward import FeedForward
from .positional_encoding import LearnedPositionalEncoding, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer (Pre-LN).

    Args:
        d_model: model hidden size
        num_heads: number of attention heads
        d_ff: hidden dimension of the position-wise feed-forward network
        dropout: dropout probability applied to sublayer outputs
        quantization: optional quantization mode ("4bit", "8bit")
        activation: activation function for FFN ("gelu", "relu", or "swiglu")
        scale_attn_scores: Whether to scale attention scores by sqrt(d_k). T5 does NOT scale.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        quantization: Optional[str] = None,
        activation: Literal["gelu", "relu", "swiglu", "gated-gelu"] = "gated-gelu",
        scale_attn_scores: bool = True,  # T5 uses False
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.0,
            quantization=quantization,
            scale_scores=scale_attn_scores,
        )
        # set MHA internal dropout to 0.0 and use dropout1/dropout2 in the layer
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            quantization=quantization,
        )

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for the encoder layer.

        Args:
            x: (batch, seq_len, d_model) - input embeddings / representations
            mask: optional attention mask, shape either (batch, seq_q, seq_k) or (batch, 1, seq_q, seq_k)
            collect_attn: whether to return attention weights
            position_bias: optional (1, num_heads, seq_q, seq_k) T5-style relative position bias

        Returns:
            x: (batch, seq_len, d_model)
            If you want attention weights, set collect_attn externally (the encoder stack can collect them).
        """
        # Self-attention sublayer (Pre-LN)
        x_norm = self.norm1(x)  # Pre-LN
        # self_attn expects query, key, value; for encoder they are the same
        attn_out, attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            mask,
            return_attn_weights=collect_attn,
            position_bias=position_bias,
        )
        x = x + self.dropout1(attn_out)

        # Clamp inf values for fp16/bf16 training stability (like HuggingFace T5)
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)

        # Feed-forward sublayer (Pre-LN)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        # Clamp inf values for fp16/bf16 training stability
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)

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
        quantization: Optional[str] = None,
        use_learned_pos_enc: bool = False,
        activation: Literal["gelu", "relu", "swiglu", "gated-gelu"] = "gated-gelu",
        use_relative_position_bias: bool = False,  # T5-style relative position bias
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.use_relative_position_bias = use_relative_position_bias

        # Token embedding (only used if forward receives token ids)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding (disabled when using relative position bias for T5)
        self.relative_position_bias: Optional[T5RelativePositionBias] = None
        if use_relative_position_bias:
            # T5 uses relative position bias instead of absolute positional embeddings
            self.pos_encoder = None
            self.relative_position_bias = T5RelativePositionBias(
                num_heads=num_heads,
                num_buckets=32,
                max_distance=128,
                is_decoder=False,
            )
        elif use_learned_pos_enc:
            # T5 uses max_len=512 by default; we add buffer for special tokens
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=d_model, max_len=max_len + 2, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # T5 does NOT scale attention scores by sqrt(d_k), others do
        scale_attn_scores = not use_relative_position_bias

        # Encoder layers stack
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    quantization=quantization,
                    activation=activation,
                    scale_attn_scores=scale_attn_scores,
                )
                for _ in range(num_layers)
            ]
        )

        # Final RMSNorm for Pre-LN stacks (recommended)
        self.final_norm = nn.RMSNorm(d_model)

        # Dropout applied after embedding + positional encoding (paper uses this)
        self.input_dropout = nn.Dropout(dropout)

    def _build_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a 3D attention mask (batch, seq, seq) from input_ids and pad_token_id.
        True indicates valid positions; False indicates masked (pad).
        """
        assert (
            self.pad_token_id is not None
        ), "pad_token_id must be set to build padding mask from ids."
        # mask shape: (batch, seq) where True = token kept (non-pad)
        pad_mask = input_ids != self.pad_token_id
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
            # T5/FLAN-T5 does NOT scale embeddings by sqrt(d_model)
            x = self.embedding(inputs)
            seq_len = inputs.size(1)
        elif inputs.dim() == 3:  # already embeddings
            x = inputs
            seq_len = inputs.size(1)
        else:
            raise ValueError(
                "inputs must be (batch, seq) token ids or (batch, seq, d_model) embeddings"
            )

        # Positional encoding + dropout (only if not using relative position bias)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # Build mask if needed
        if mask is None and inputs.dim() == 2 and self.pad_token_id is not None:
            mask = self._build_padding_mask(inputs)

        # Ensure mask is boolean and on the same device
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)

        # Compute relative position bias if using T5-style
        position_bias = None
        if self.relative_position_bias is not None:
            position_bias = self.relative_position_bias(seq_len, seq_len, x.device)

        attn_weights_per_layer: List[torch.Tensor] = []

        # Pass through each encoder layer (optionally collect attn)
        for layer in self.layers:
            x, attn = layer(x, mask=mask, collect_attn=collect_attn, position_bias=position_bias)
            if collect_attn:
                attn_weights_per_layer.append(attn)

        # Final normalization (Pre-LN stack)
        x = self.final_norm(x)

        if collect_attn:
            return x, attn_weights_per_layer
        return x
