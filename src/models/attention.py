"""
Attention mechanisms for Transformer architecture.

This module implements the core attention mechanisms used in the Transformer model:
- ScaledDotProductAttention: Fundamental attention operation
- MultiHeadAttention: Parallel attention with learned projections
- T5RelativePositionBias: Relative position bias for T5-style attention

Doing this first for Bottom-Up implementation of the Transformer

Author: Oliver Perrin
Date: 2025-10-23
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class T5RelativePositionBias(nn.Module):
    """
    T5-style relative position bias for attention.

    T5 uses a learned embedding table to encode relative positions between tokens.
    Positions are bucketed to handle arbitrary sequence lengths efficiently.

    This is added to attention scores BEFORE softmax, not to embeddings.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.is_decoder = is_decoder

        # Learned embedding table: (num_buckets, num_heads)
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """
        Translate relative position to a bucket index.

        T5 uses a combination of exact positions (for nearby tokens) and
        logarithmically-spaced buckets (for distant tokens).
        """
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)

        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # Half buckets for exact positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half for logarithmically-spaced buckets
        relative_position_if_large = (
            max_exact
            + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
        query_position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute relative position bias for attention.

        Args:
            query_length: Number of query positions
            key_length: Number of key positions
            device: Device to create tensors on
            query_position_offset: Offset for query positions (for incremental decoding)
                                   When decoding step-by-step, query_length=1 but the actual
                                   position is past_len, so query_position_offset=past_len.

        Returns: (1, num_heads, query_length, key_length)
        """
        # Create position indices
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        context_position = (
            context_position + query_position_offset
        )  # Apply offset for incremental decoding
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        # Relative position: (query_length, key_length)
        relative_position = memory_position - context_position

        # Convert to bucket indices
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # Look up bias values: (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)

        # Reshape to (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values

    def forward(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
        query_position_offset: int = 0,
    ) -> torch.Tensor:
        return self.compute_bias(query_length, key_length, device, query_position_offset)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention using PyTorch's optimized backend.

    Uses F.scaled_dot_product_attention which automatically selects the best
    available kernel (FlashAttention v2, Memory-Efficient Attention, or math fallback)
    based on hardware and input shapes. On CUDA GPUs with appropriate compute capability,
    this will use FlashAttention for significantly improved speed and memory efficiency.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """

    def __init__(self, scale_scores: bool = True):
        """
        Args:
            scale_scores: Whether to scale attention scores by sqrt(d_k).
                          T5 does NOT scale scores, so set this to False for T5.
                          Standard transformers (BERT, GPT, etc.) use scaling.
        """
        super().__init__()
        self.scale_scores = scale_scores

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, num_heads, seq_q, d_k)
            key: (batch, num_heads, seq_k, d_k)
            value: (batch, num_heads, seq_k, d_v)
            mask: Optional boolean mask, True = attend, False = mask
            position_bias: Optional (1, num_heads, seq_q, seq_k) T5-style relative position bias

        Returns:
            output: (batch, num_heads, seq_q, d_v)
            attention_weights: Optional (batch, num_heads, seq_q, seq_k)
        """
        d_k = query.size(-1)
        scale_factor = 1.0 / math.sqrt(d_k) if self.scale_scores else 1.0

        # If we need attention weights, must use manual path
        if return_attn_weights:
            # Manual implementation with float32 softmax for numerical stability
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            if position_bias is not None:
                scores = scores + position_bias
            if mask is not None:
                mask_bool = mask.to(dtype=torch.bool, device=scores.device)
                if mask_bool.dim() == 2:
                    mask_bool = mask_bool.unsqueeze(1).unsqueeze(2)
                elif mask_bool.dim() == 3:
                    mask_bool = mask_bool.unsqueeze(1)
                scores = scores.masked_fill(~mask_bool, -1e4)
            p_attn = F.softmax(scores.float(), dim=-1).type_as(scores)
            p_attn = torch.nan_to_num(p_attn, nan=0.0, posinf=0.0, neginf=0.0)
            output = torch.matmul(p_attn, value)
            return output, p_attn

        # Use optimized SDPA path - torch.compile friendly version
        # Pre-scale query instead of using SDPA's scale parameter for better compile compatibility
        # This avoids issues with inductor and custom scale values
        if self.scale_scores:
            query = query * scale_factor

        # Build combined attention mask (float tensor added to scores)
        attn_mask = None

        if position_bias is not None or mask is not None:
            # Start with position bias if provided
            if position_bias is not None:
                # Clamp position bias to prevent overflow
                attn_mask = position_bias.to(dtype=query.dtype).clamp(-100, 100)

            # Add mask (convert bool mask to additive float mask)
            if mask is not None:
                mask_bool = mask.to(dtype=torch.bool, device=query.device)
                if mask_bool.dim() == 2:
                    mask_bool = mask_bool.unsqueeze(1).unsqueeze(2)
                elif mask_bool.dim() == 3:
                    mask_bool = mask_bool.unsqueeze(1)

                mask_float = torch.zeros(mask_bool.shape, dtype=query.dtype, device=query.device)
                mask_float = mask_float.masked_fill(~mask_bool, -1e4)

                if attn_mask is not None:
                    attn_mask = attn_mask + mask_float
                else:
                    attn_mask = mask_float

        # Use SDPA without custom scale (scale=None uses default 1/sqrt(d_k))
        # For T5 (scale_scores=False), we already didn't scale query above, so default scale is wrong
        # But we pre-scaled query for scaled attention, so we need scale=1.0 here
        # Actually simpler: always use scale=1.0 since we handle scaling ourselves
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,  # We handle scaling manually above
        )
        return output, None


# --------------- Rotary Positional Embeddings ---------------


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Encodes relative positions by rotating the query and key vectors.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, dim)
        seq_len = x.shape[2]
        # Slice cos/sin to current sequence length
        # unsqueeze to broadcast over batch and heads: (1, 1, seq_len, dim)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


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
        use_rope: Whether to use Rotary Positional Embeddings (default: False)
        max_len: Maximum sequence length for RoPE (default: 2048)
        use_lora: Whether to use LoRA (Low-Rank Adaptation) (default: False)
        lora_rank: Rank of LoRA matrices (default: 8)
        lora_alpha: Scaling factor for LoRA (default: 16)
        lora_dropout: Dropout probability for LoRA (default: 0.1)
        scale_scores: Whether to scale attention scores by sqrt(d_k). T5 does NOT scale.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_len: int = 2048,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        quantization: Optional[str] = None,
        scale_scores: bool = True,  # T5 uses scale_scores=False
    ):
        super().__init__()

        # Assert that d_model is divisible by num_heads
        # Why? Because d_k = d_model // num_heads must be an integer
        assert d_model % num_heads == 0

        # Assume d_v always equals d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

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

        # Create 4 linear layers (W_Q, W_K, W_V, W_O)
        # All should be nn.Linear(d_model, d_model)
        self.W_Q = Linear(d_model, d_model, **kwargs)
        self.W_K = Linear(d_model, d_model, **kwargs)
        self.W_V = Linear(d_model, d_model, **kwargs)
        self.W_O = Linear(d_model, d_model, **kwargs)
        # Create ScaledDotProductAttention instance
        # Note: T5 does NOT scale attention scores by sqrt(d_k)
        self.attention = ScaledDotProductAttention(scale_scores=scale_scores)
        # Create dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_len)

        # LoRA (Low-Rank Adaptation)
        self.use_lora = use_lora
        if use_lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_scaling = lora_alpha / lora_rank
            self.lora_dropout = nn.Dropout(p=lora_dropout)

            # LoRA for Query: W_Q' = W_Q + B_q @ A_q * scaling
            self.lora_q_A = nn.Linear(d_model, lora_rank, bias=False)
            self.lora_q_B = nn.Linear(lora_rank, d_model, bias=False)

            # LoRA for Value: W_V' = W_V + B_v @ A_v * scaling
            self.lora_v_A = nn.Linear(d_model, lora_rank, bias=False)
            self.lora_v_B = nn.Linear(lora_rank, d_model, bias=False)

            # Initialize LoRA parameters
            # A: Kaiming uniform, B: Zeros (so training starts with original behavior)
            nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_q_B.weight)
            nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_v_B.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            position_bias: Optional (1, num_heads, seq_q, seq_k) T5-style relative position bias

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # Apply LoRA if enabled
        if self.use_lora:
            # Q += (query @ A^T @ B^T) * scaling
            # Note: nn.Linear(x) computes x @ weight.T
            # So lora_q_A(x) is x @ A.T
            # lora_q_B(lora_q_A(x)) is (x @ A.T) @ B.T = x @ A.T @ B.T
            lora_q = self.lora_q_B(self.lora_q_A(self.lora_dropout(query))) * self.lora_scaling
            Q = Q + lora_q

            # V += (value @ A^T @ B^T) * scaling
            lora_v = self.lora_v_B(self.lora_v_A(self.lora_dropout(value))) * self.lora_scaling
            V = V + lora_v

        # Split into heads
        # Reshape from (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_k), Apply to Q, K, V
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, d_k)
        # Now all are: (batch=2, num_heads=8, seq_len=10, d_k=64)

        # Apply RoPE if enabled
        if self.use_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        # Handle mask broadcasting for multi-head attention
        if mask is not None:
            # If mask is 3D (batch, seq, seq), add head dimension
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
        # Now mask broadcasts across all heads: (batch, 1, seq, seq) → (batch, 8, seq, seq)

        # Apply attention with optional position bias
        output, attn_weights = self.attention(
            Q, K, V, mask, return_attn_weights=return_attn_weights, position_bias=position_bias
        )
        # output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        # Concatenate heads
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, -1, self.d_model
        )  # -1 in view means 'infer this dimension'
        # After transpose, the tensor's memory layout
        # is "scattered", contiguous() just reorganizes it in memory

        # Final linear projection
        output = self.W_O(output)
        # Apply dropout
        output = self.dropout(output)

        return output, attn_weights
