"""Transformer Decoder implementation (Pre-LN).

This module implements the decoder component of the Transformer architecture:
- create_causal_mask: Generate causal attention masks
- TransformerDecoderLayer: Single decoder block with self-attn + cross-attn + FFN
- TransformerDecoder: Full stack with embeddings, positional encoding, and generation

Design notes:
- Pre-LN with RMSNorm for training stability
- Masks are boolean: True = attend, False = mask
- Supports T5-style relative position bias

Author: Oliver Perrin
Date: 2025-10-23
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, T5RelativePositionBias
from .feedforward import FeedForward
from .positional_encoding import LearnedPositionalEncoding, PositionalEncoding


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a (seq_len, seq_len) causal mask where entry (i, j) is True iff
    j <= i (query at i may attend to keys up to i).
    """
    # torch.triu(..., diagonal=1) is True above the diagonal. Invert to get allowed positions.
    mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask  # shape: (T, T)


class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer (Pre-LN):
      1) Masked self-attention
      2) Cross-attention (encoder -> decoder)
      3) Feed-forward
    Returns the updated tgt and a dict of attention maps.
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
        # use internal MHA dropout = 0.0; the layer handles dropout after sublayers
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.0,
            quantization=quantization,
            scale_scores=scale_attn_scores,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.0,
            quantization=quantization,
            scale_scores=scale_attn_scores,
        )
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            quantization=quantization,
        )

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
        self_attn_position_bias: Optional[torch.Tensor] = None,
        cross_attn_position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Args:
            tgt: (B, T, d_model)
            memory: (B, S, d_model)
            tgt_mask: optional mask for self-attn - shape (B, T, T) or (B, 1, T, T)
            memory_mask: optional mask for cross-attn - shape (B, S) or (B, 1, S) or (B, 1, T, S)
            collect_attn: whether to return attention weights
            self_attn_position_bias: optional T5 relative position bias for self-attention
            cross_attn_position_bias: optional T5 relative position bias for cross-attention

        Returns:
            (tgt_out, {"self": self_attn_weights, "cross": cross_attn_weights})
        """
        # Ensure masks are on same device and boolean
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(dtype=torch.bool, device=tgt.device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(dtype=torch.bool, device=tgt.device)
            # If memory_mask is provided as (B, S) (per-key padding), expand to (B, 1, 1, S)
            if memory_mask.dim() == 2:
                memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,S)
            # If it's (B, S, S) or (B, 1, S, S) leave as-is; if (B, T, S) convert to (B,1,T,S)
            elif memory_mask.dim() == 3 and memory_mask.shape[1] != 1:
                # assume (B, T, S) -> make (B, 1, T, S)
                memory_mask = memory_mask.unsqueeze(1)

        # --- Masked self-attention (Pre-LN) ---
        x_norm = self.norm1(tgt)
        self_out, self_attn = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            tgt_mask,
            return_attn_weights=collect_attn,
            position_bias=self_attn_position_bias,
        )
        tgt = tgt + self.dropout1(self_out)

        # Clamp inf values for fp16/bf16 training stability (like HuggingFace T5)
        if tgt.dtype == torch.float16 or tgt.dtype == torch.bfloat16:
            clamp_value = torch.finfo(tgt.dtype).max - 1000
            tgt = torch.clamp(tgt, min=-clamp_value, max=clamp_value)

        # --- Cross-attention (Pre-LN) ---
        x_norm = self.norm2(tgt)
        cross_out, cross_attn = self.cross_attn(
            x_norm,
            memory,
            memory,
            memory_mask,
            return_attn_weights=collect_attn,
            position_bias=cross_attn_position_bias,
        )
        tgt = tgt + self.dropout2(cross_out)

        # Clamp inf values for fp16/bf16 training stability
        if tgt.dtype == torch.float16 or tgt.dtype == torch.bfloat16:
            clamp_value = torch.finfo(tgt.dtype).max - 1000
            tgt = torch.clamp(tgt, min=-clamp_value, max=clamp_value)

        # --- Feed-forward (Pre-LN) ---
        x_norm = self.norm3(tgt)
        ffn_out = self.ffn(x_norm)
        tgt = tgt + self.dropout3(ffn_out)

        # Clamp inf values for fp16/bf16 training stability
        if tgt.dtype == torch.float16 or tgt.dtype == torch.bfloat16:
            clamp_value = torch.finfo(tgt.dtype).max - 1000
            tgt = torch.clamp(tgt, min=-clamp_value, max=clamp_value)

        return tgt, {"self": self_attn, "cross": cross_attn}


class TransformerDecoder(nn.Module):
    """
    Decoder stack with token embeddings and positional encoding.

    Forward returns logits (B, T, vocab_size) by default; if collect_attn=True returns (logits, attn_list).
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
        self.num_heads = num_heads
        self.use_relative_position_bias = use_relative_position_bias

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding (disabled when using relative position bias for T5)
        self.self_relative_position_bias: Optional[T5RelativePositionBias] = None
        self.cross_relative_position_bias: Optional[T5RelativePositionBias] = None
        if use_relative_position_bias:
            # T5 uses relative position bias instead of absolute positional embeddings
            self.pos_encoder = None
            # Self-attention position bias (decoder is causal, so is_decoder=True)
            self.self_relative_position_bias = T5RelativePositionBias(
                num_heads=num_heads,
                num_buckets=32,
                max_distance=128,
                is_decoder=True,
            )
            # T5 cross-attention does NOT use position bias
        elif use_learned_pos_enc:
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=d_model, max_len=max_len + 2, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # T5 does NOT scale attention scores by sqrt(d_k), others do
        scale_attn_scores = not use_relative_position_bias

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
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

        self.final_norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.input_dropout = nn.Dropout(dropout)

    def _build_padding_mask_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input ids to (B, T, T) boolean mask where True = allowed.

        Note: For T5, pad_token_id=0 is also used as decoder_start_token_id.
        During generation, we should NOT mask the start token. The caller should
        provide an explicit mask or set tgt_mask to avoid this issue.
        """
        assert self.pad_token_id is not None, "pad_token_id must be set to build mask from ids"
        pad_mask = input_ids != self.pad_token_id  # (B, T)
        attn_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B, T, T)
        return attn_mask

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
        skip_padding_mask: bool = False,  # Set True during generation to avoid masking start token
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        """
        Args:
            inputs: (B, T) token ids or (B, T, d_model) embeddings
            memory: (B, S, d_model)
            tgt_mask: optional; if None, will create (causal [+ padding if ids available])
            memory_mask: optional; if provided as (B, S) will be expanded to (B, 1, 1, S)
            skip_padding_mask: if True, only use causal mask (for generation where start_token=pad_token)
        """
        # Prepare embeddings
        if inputs.dim() == 2:  # token ids
            # T5/FLAN-T5 does NOT scale embeddings by sqrt(d_model)
            x = self.embedding(inputs)
        elif inputs.dim() == 3:
            x = inputs
        else:
            raise ValueError("inputs must be (B, T) token ids or (B, T, d_model) embeddings")

        # Apply positional encoding if not using relative position bias
        # (T5 uses relative position bias in attention instead of absolute positional embeddings)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.input_dropout(x)

        B, T, _ = x.shape

        # Build target mask if not provided: combine causal + padding (if available)
        if tgt_mask is None:
            causal = create_causal_mask(T, device=x.device)  # (T, T)
            if inputs.dim() == 2 and self.pad_token_id is not None and not skip_padding_mask:
                # During training: combine causal mask with padding mask
                pad_pairwise = self._build_padding_mask_from_ids(inputs)  # (B, T, T)
                combined = pad_pairwise & causal.unsqueeze(0)  # (B, T, T)
                tgt_mask = combined.unsqueeze(1)  # (B, 1, T, T) -> broadcast to heads
            else:
                # During generation (skip_padding_mask=True) or no padding info:
                # Use only causal mask - don't mask based on token values
                tgt_mask = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)
        else:
            # Ensure boolean and device alignment; accept (B, T, T) or (B,1,T,T) or (1,1,T,T)
            tgt_mask = tgt_mask.to(dtype=torch.bool, device=x.device)

        # Normalize memory_mask dtype/device and expand simple shapes
        if memory_mask is not None:
            memory_mask = memory_mask.to(dtype=torch.bool, device=x.device)
            if memory_mask.dim() == 2:  # (B, S) -> (B, 1, 1, S)
                memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)
            elif memory_mask.dim() == 3:  # (B, T, S) -> (B, 1, T, S)
                memory_mask = memory_mask.unsqueeze(1)

        attn_list: List[Dict[str, torch.Tensor]] = []

        # Compute relative position biases (T5-style)
        # Note: T5 uses relative position bias for self-attention but NOT for cross-attention
        if self.use_relative_position_bias and self.self_relative_position_bias is not None:
            self_position_bias = self.self_relative_position_bias(
                T, T, x.device
            )  # (1, num_heads, T, T)
        else:
            self_position_bias = None
        # Cross-attention position bias is None for T5 (see T5 paper/implementation)
        cross_position_bias = None

        # Pass through decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                collect_attn=collect_attn,
                self_attn_position_bias=self_position_bias,
                cross_attn_position_bias=cross_position_bias,
            )
            if collect_attn:
                attn_list.append(attn)

        x = self.final_norm(x)
        logits = self.output_projection(x)  # (B, T, vocab)

        if collect_attn:
            return logits, attn_list
        return logits

    def greedy_decode_naive(
        self,
        memory: torch.Tensor,
        max_len: int,
        start_token_id: int,
        end_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Naive greedy decoding using full forward pass (O(N^2) but simpler).
        Used for debugging to verify step() correctness.
        """
        if device is None:
            device = memory.device
        B = memory.size(0)

        # Initialize with start token
        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Full forward pass on entire generated sequence
            # skip_padding_mask=True because start_token=pad_token for T5
            logits = self.forward(
                generated, memory, memory_mask=memory_mask, skip_padding_mask=True
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits: (B, T, vocab)

            # Get logits for last position
            next_logits = logits[:, -1, :]  # (B, vocab)

            # Greedy: pick highest probability token
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if end_token_id is not None and (next_token == end_token_id).all():
                break

        return generated

    def greedy_decode(
        self,
        memory: torch.Tensor,
        max_len: int,
        start_token_id: int,
        end_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        *,
        min_len: Optional[int] = None,
        ban_token_ids: Optional[List[int]] = None,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding with KV caching for O(N) complexity.
        """
        if device is None:
            device = memory.device
        B = memory.size(0)

        # Initialize generated sequence with start token
        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        # Initialize cache
        cache: Dict[str, Any] = {"past_length": 0}
        if memory_mask is not None:
            cache["memory_mask"] = memory_mask

        min_len = 0 if min_len is None else max(0, min_len)

        # Keep track of finished sequences
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Use the last generated token for the next step
            last_token = generated[:, -1:]  # (B, 1)

            # Run one step of the decoder
            logits, cache = self.step(last_token, memory, cache)
            # logits: (B, vocab_size)

            next_step_logits = logits.clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    if finished[b]:
                        continue
                    gen_seq = generated[b]
                    unique_tokens = torch.unique(gen_seq)
                    current_logits = next_step_logits[b, unique_tokens]
                    next_step_logits[b, unique_tokens] = torch.where(
                        current_logits < 0,
                        current_logits * repetition_penalty,
                        current_logits / repetition_penalty,
                    )

            # Apply constraints
            if end_token_id is not None and generated.size(1) < max(1, min_len):
                next_step_logits[:, end_token_id] = float("-inf")

            if ban_token_ids:
                next_step_logits[:, ban_token_ids] = float("-inf")

            # N-gram repetition blocking
            if no_repeat_ngram_size > 0:
                for b in range(B):
                    if finished[b]:
                        continue
                    gen_seq = generated[b].tolist()
                    if len(gen_seq) < no_repeat_ngram_size - 1:
                        continue

                    prefix = tuple(gen_seq[-(no_repeat_ngram_size - 1) :])
                    banned_for_this_batch = set()

                    for i in range(len(gen_seq) - no_repeat_ngram_size + 1):
                        window = tuple(gen_seq[i : i + no_repeat_ngram_size - 1])
                        if window == prefix:
                            if i + no_repeat_ngram_size - 1 < len(gen_seq):
                                banned_for_this_batch.add(gen_seq[i + no_repeat_ngram_size - 1])

                    if banned_for_this_batch:
                        next_step_logits[b, list(banned_for_this_batch)] = float("-inf")

            # Greedy selection
            next_token = next_step_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for completion
            if end_token_id is not None:
                is_end = next_token.squeeze(-1) == end_token_id
                finished = finished | is_end
                if finished.all() and generated.size(1) >= max(1, min_len):
                    break

        return generated

    # -----------------------------
    # Incremental single-step API
    # -----------------------------
    def step(
        self,
        last_token_ids: torch.Tensor,
        memory: torch.Tensor,
        cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run one autoregressive step.

        Args:
            last_token_ids: (B, 1) last generated token ids
            memory: encoder outputs (B, S, d_model)
            cache: optional dict with previous cached keys/values and 'past_length'.

        Returns:
            logits: (B, vocab_size) logits for the next-token prediction
            new_cache: updated cache dictionary
        """
        device = memory.device
        B = last_token_ids.size(0)

        if cache is None:
            cache = {}
        past_len = int(cache.get("past_length", 0))

        # 1) Embed last token and add positional encoding for position `past_len`
        # T5/FLAN-T5 does NOT scale embeddings by sqrt(d_model)
        x = self.embedding(last_token_ids)  # (B,1,d)

        # Handle positional encoding for single step
        # Note: When using relative position bias (T5-style), pos_encoder is None
        if self.pos_encoder is not None:
            if hasattr(self.pos_encoder, "pe"):
                # Sinusoidal: use buffer directly
                pe: torch.Tensor = self.pos_encoder.pe  # type: ignore[union-attr]
                pos_idx = past_len
                if pos_idx >= pe.size(1):
                    raise RuntimeError(f"pos_idx {pos_idx} exceeds max_len {pe.size(1)}")
                x = x + pe[:, pos_idx : pos_idx + 1, :].to(device)
            elif hasattr(self.pos_encoder, "embeddings"):
                # Learned: lookup specific position
                # Create position ids: [past_len]
                pos_idx_t = torch.tensor([past_len], dtype=torch.long, device=device)
                # Lookup embedding: (1, d_model)
                pos_emb = self.pos_encoder.embeddings(pos_idx_t)  # type: ignore[union-attr]
                # Add to input: (B, 1, d_model) + (1, 1, d_model) broadcast
                x = x + pos_emb.unsqueeze(0)
                x = self.pos_encoder.dropout(x)  # type: ignore[union-attr]
            else:
                # fallback: call pos_encoder (likely incorrect for step-by-step if it assumes pos 0)
                x = self.pos_encoder(x)
        # When pos_encoder is None (relative position bias mode), we skip positional encoding
        # The position information is provided via relative_position_bias in attention

        # We will update new_cache incrementally
        new_cache = dict(cache)  # shallow copy
        new_cache["past_length"] = past_len + 1

        # Optional: memory_mask could be supplied in cache under 'memory_mask'
        memory_mask = new_cache.get("memory_mask", None)
        if memory_mask is not None:
            memory_mask = memory_mask.to(dtype=torch.bool, device=device)
            # expand (B, S) -> (B,1,1,S) if necessary
            if memory_mask.dim() == 2:
                memory_mask = memory_mask.unsqueeze(1).unsqueeze(1)
            elif memory_mask.dim() == 3:
                memory_mask = memory_mask.unsqueeze(1)

        # Compute position biases for incremental step (T5-style)
        # For step mode: query_length=1, but actual position is past_len
        # Self-attention: query at position past_len attends to keys at positions 0..past_len
        # Note: T5 uses relative position bias for self-attention but NOT for cross-attention
        if self.use_relative_position_bias and self.self_relative_position_bias is not None:
            # Self-attention bias: query_length=1, key_length=past_len+1, offset=past_len
            self_position_bias = self.self_relative_position_bias(
                query_length=1,
                key_length=past_len + 1,
                device=device,
                query_position_offset=past_len,
            )  # (1, num_heads, 1, past_len+1)
        else:
            self_position_bias = None
        # Cross-attention position bias is None for T5 (see T5 paper/implementation)
        cross_position_bias = None

        # Iterate layers, updating caches and computing output for current token only
        layer_input = x  # (B,1,d_model)
        for i, layer_module in enumerate(self.layers):
            layer = cast(TransformerDecoderLayer, layer_module)
            # -------------------
            # 1) Self-attention (incremental)
            # -------------------
            # Normalize input for pre-LN
            x_norm = layer.norm1(layer_input)  # (B,1,d)

            # Project Q,K,V for the new token
            Q_new = layer.self_attn.W_Q(x_norm)  # (B,1,d_model)
            K_new = layer.self_attn.W_K(x_norm)
            V_new = layer.self_attn.W_V(x_norm)

            # Reshape into heads: (B, num_heads, 1, d_k)
            B_, Lq, _ = Q_new.shape
            num_heads = layer.self_attn.num_heads
            d_k = layer.self_attn.d_k
            Qh = Q_new.view(B_, Lq, num_heads, d_k).transpose(1, 2)  # (B, num_heads, 1, d_k)
            Kh = K_new.view(B_, Lq, num_heads, d_k).transpose(1, 2)
            Vh = V_new.view(B_, Lq, num_heads, d_k).transpose(1, 2)

            # Retrieve cached keys/values for self-attn (if exist)
            cache_k = cache.get(f"self_k_{i}", None)
            cache_v = cache.get(f"self_v_{i}", None)
            if cache_k is None or cache_v is None:
                K_all = Kh  # (B, H, 1, d_k)
                V_all = Vh
            else:
                # concat along sequence dim (dim=2)
                K_all = torch.cat([cache_k.to(device), Kh], dim=2)
                V_all = torch.cat([cache_v.to(device), Vh], dim=2)

            # Store updated caches
            new_cache[f"self_k_{i}"] = K_all
            new_cache[f"self_v_{i}"] = V_all

            # Compute attention for the new token: Query length = 1, Key length = K_all.size(2)
            # Explicitly create mask for consistency with forward pass (though None should work)
            # mask=True means attend.
            step_mask = torch.ones(B_, 1, 1, K_all.size(2), dtype=torch.bool, device=device)
            attn_out_heads, self_attn_w = layer.self_attn.attention(
                Qh, K_all, V_all, mask=step_mask, position_bias=self_position_bias
            )
            # attn_out_heads: (B, H, 1, d_k)
            # concat heads, project out
            attn_out = attn_out_heads.transpose(1, 2).contiguous().view(B_, 1, num_heads * d_k)
            attn_out = layer.self_attn.W_O(attn_out)  # (B,1,d_model)
            attn_out = layer.self_attn.dropout(attn_out)
            layer_output = layer_input + layer.dropout1(attn_out)

            # -------------------
            # 2) Cross-attention (use cached memory projections if available)
            # -------------------
            x_norm2 = layer.norm2(layer_output)  # (B,1,d)
            # Ensure memory K/V are cached per layer
            mem_k = cache.get(f"mem_k_{i}", None)
            mem_v = cache.get(f"mem_v_{i}", None)
            if mem_k is None or mem_v is None:
                # project memory once for this layer and cache it
                # memory: (B, S, d_model)
                MK = layer.cross_attn.W_K(memory)  # (B, S, d_model)
                MV = layer.cross_attn.W_V(memory)
                Bm, S, _ = MK.shape
                MKh = MK.view(Bm, S, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(
                    1, 2
                )  # (B,H,S,d_k)
                MVh = MV.view(Bm, S, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(
                    1, 2
                )
                mem_k = MKh
                mem_v = MVh
                new_cache[f"mem_k_{i}"] = mem_k
                new_cache[f"mem_v_{i}"] = mem_v
            else:
                mem_k = mem_k.to(device)
                mem_v = mem_v.to(device)

            Qc = layer.cross_attn.W_Q(x_norm2)  # (B,1,d_model)
            Qch = Qc.view(B, 1, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(
                1, 2
            )  # (B,H,1,d_k)

            cross_out_heads, cross_attn_w = layer.cross_attn.attention(
                Qch, mem_k, mem_v, mask=memory_mask, position_bias=cross_position_bias
            )
            cross_out = (
                cross_out_heads.transpose(1, 2)
                .contiguous()
                .view(B, 1, layer.cross_attn.num_heads * layer.cross_attn.d_k)
            )
            cross_out = layer.cross_attn.W_O(cross_out)  # (B,1,d_model)
            cross_out = layer.cross_attn.dropout(cross_out)
            layer_output = layer_output + layer.dropout2(cross_out)

            # -------------------
            # 3) Feed-forward (incremental)
            # -------------------
            x_norm3 = layer.norm3(layer_output)
            ffn_out = layer.ffn(x_norm3)  # (B,1,d_model)
            layer_output = layer_output + layer.dropout3(ffn_out)

            # Prepare for next layer
            layer_input = layer_output

        # Final norm + output projection (for this single time step)
        out_norm = self.final_norm(layer_input)  # (B,1,d_model)
        logits = self.output_projection(out_norm)  # (B,1,vocab)
        logits = logits.squeeze(1)  # (B, vocab)

        return logits, new_cache
