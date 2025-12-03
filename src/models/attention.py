"""
Attention mechanisms for Transformer architecture.

This module implements the core attention mechanisms used in the Transformer model:
- ScaledDotProductAttention: Fundamental attention operation
- MultiHeadAttention: Parallel attention with learned projections

Doing this first for Bottom-Up implementation of the Transformer

Author: Oliver Perrin
Date: 2025-10-23
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention using PyTorch's optimized backend.

    Uses F.scaled_dot_product_attention which automatically selects the best
    available kernel (FlashAttention v2, Memory-Efficient Attention, or math fallback)
    based on hardware and input shapes. On CUDA GPUs with appropriate compute capability,
    this will use FlashAttention for significantly improved speed and memory efficiency.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
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
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Steps:
        1. Compute attention scores: scores = query @ key.transpose(-2, -1)
        2. Scale by sqrt(d_k)
        3. Apply mask if provided (set masked positions to -inf before softmax)
        4. Apply softmax to get attention weights
        5. Compute output: output = attention_weights @ value
        6. Return both output and attention_weights
        """
        # NEW: FlashAttention implementation using PyTorch 2.0+ SDPA
        # This automatically selects the best kernel (FlashAttention, EfficientAttention, etc.)

        # Handle mask for SDPA
        # User mask: 1/True = attend, 0/False = mask
        # SDPA boolean mask: True = mask out, False = attend
        # So I invert the user mask if it's provided
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.to(dtype=torch.bool, device=query.device)

        # Call SDPA
        # Note: I don't apply dropout here as my original implementation doesn't
        # If we wanted to, I'd pass dropout_p to this method
        if not return_attn_weights:
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            # SDPA doesn't return attention weights by default for efficiency
            # I return None for weights when using the optimized kernel
            return output, None

        # --------- OLD: Manual implementation (Fallback when weights are needed) ---------------
        # Scaled Dot-Product Attention as described in "Attention Is All You Need" 2017.
        # Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        # The scaling factor (1/sqrt(d_k)) prevents the dot products from growing too large,
        # which would push the softmax into regions with extremely small gradients.
        # Args:
        #     None - this module has no learnable parameters
        # Forward Args:
        #     query: Query tensor of shape (batch, seq_len, d_k)
        #     key: Key tensor of shape (batch, seq_len, d_k)
        #     value: Value tensor of shape (batch, seq_len, d_v)
        #     mask: Optional mask tensor of shape (batch, seq_len, seq_len)
        #      True/1 values indicate positions to attend to, False/0 to mask
        # Returns:
        #     output: Attention output of shape (batch, seq_len, d_v)
        # attention_weights: Attention probability matrix (batch, seq_len, seq_len)
        # Getting Dimension for Scaling
        d_k = query.size(-1)

        # Compute Attention Scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask if provided
        if mask is not None:
            # Ensure mask is boolean and on same device as scores
            mask_bool = mask.to(dtype=torch.bool, device=scores.device)
            # masked_fill expects broadcastable mask: True means keep, False means mask out
            scores = scores.masked_fill(~mask_bool, float("-1e9"))

        # Softmax to get attention probabilities
        p_attn = F.softmax(scores, dim=-1)

        # If mask was provided, ensure masked positions are exactly zero (and handle all-masked rows)
        if mask is not None:
            # Convert mask to same dtype as p_attn for multiplication
            mask_float = mask.to(dtype=p_attn.dtype, device=p_attn.device)
            # Broadcast-multiply (zero out masked key positions)
            p_attn = p_attn * mask_float
            # Replace any NaNs (can occur when a row was entirely -inf prior to softmax) with 0.0
            # torch.nan_to_num is efficient and handles negative/positive inf as well
            p_attn = torch.nan_to_num(p_attn, nan=0.0, posinf=0.0, neginf=0.0)

            # re-normalize rows that still have non-zero sum, this is not strictly necessary
            # if mask is correct, but safe to avoid tiny numerical issues:
            row_sums = p_attn.sum(dim=-1, keepdim=True)
            # Avoid division by zero; only divide where row_sums > 0
            nonzero_rows = row_sums > 0
            p_attn = torch.where(nonzero_rows, p_attn / (row_sums + 1e-12), p_attn)

        output = torch.matmul(p_attn, value)
        return output, p_attn
        # ---------------------------------------------------


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
        self.attention = ScaledDotProductAttention()
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        # Apply attention
        output, attn_weights = self.attention(
            Q, K, V, mask, return_attn_weights=return_attn_weights
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
