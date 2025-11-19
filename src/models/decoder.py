"""
Transformer Decoder (Pre-LN) - implementation.

Implements:
- create_causal_mask
- TransformerDecoderLayer
- TransformerDecoder (stack + naive greedy decoding)

Conventions:
- Masks are boolean: True = allowed, False = masked.
- MultiHeadAttention expects masks broadcastable to (B, num_heads, T_q, T_k).
- This decoder uses Pre-LN (LayerNorm before each sublayer).
"""
from typing import Optional, Tuple, List, Union, Dict
import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding


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

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # use internal MHA dropout = 0.0; the layer handles dropout after sublayers
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            tgt: (B, T, d_model)
            memory: (B, S, d_model)
            tgt_mask: optional mask for self-attn - shape (B, T, T) or (B, 1, T, T)
            memory_mask: optional mask for cross-attn - shape (B, S) or (B, 1, S) or (B, 1, T, S)

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
        self_out, self_attn = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        tgt = tgt + self.dropout1(self_out)

        # --- Cross-attention (Pre-LN) ---
        x_norm = self.norm2(tgt)
        cross_out, cross_attn = self.cross_attn(x_norm, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(cross_out)

        # --- Feed-forward (Pre-LN) ---
        x_norm = self.norm3(tgt)
        ffn_out = self.ffn(x_norm)
        tgt = tgt + self.dropout3(ffn_out)

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
             for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.input_dropout = nn.Dropout(dropout)

    def _build_padding_mask_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input ids to (B, T, T) boolean mask where True = allowed.
        """
        assert self.pad_token_id is not None, "pad_token_id must be set to build mask from ids"
        pad_mask = (input_ids != self.pad_token_id)  # (B, T)
        attn_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B, T, T)
        return attn_mask

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        """
        Args:
            inputs: (B, T) token ids or (B, T, d_model) embeddings
            memory: (B, S, d_model)
            tgt_mask: optional; if None, will create (causal [+ padding if ids available])
            memory_mask: optional; if provided as (B, S) will be expanded to (B, 1, 1, S)
        """
        # Prepare embeddings
        if inputs.dim() == 2:  # token ids
            x = self.embedding(inputs) * math.sqrt(self.d_model)
        elif inputs.dim() == 3:
            x = inputs
        else:
            raise ValueError("inputs must be (B, T) token ids or (B, T, d_model) embeddings")

        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        B, T, _ = x.shape

        # Build target mask if not provided: combine causal + padding (if available)
        if tgt_mask is None:
            causal = create_causal_mask(T, device=x.device)  # (T, T)
            if inputs.dim() == 2 and self.pad_token_id is not None:
                pad_pairwise = self._build_padding_mask_from_ids(inputs)  # (B, T, T)
                combined = pad_pairwise & causal.unsqueeze(0)  # (B, T, T)
                tgt_mask = combined.unsqueeze(1)  # (B, 1, T, T) -> broadcast to heads
            else:
                # No per-batch padding info: broadcast causal to (1, 1, T, T)
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

        # Pass through decoder layers
        for layer in self.layers:
            x, attn = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if collect_attn:
                attn_list.append(attn)

        x = self.final_norm(x)
        logits = self.output_projection(x)  # (B, T, vocab)

        if collect_attn:
            return logits, attn_list
        return logits

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
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Naive greedy decoding: repeatedly run the decoder on the growing prefix.
        Not optimized (recomputes full decoder each step) but simple and correct.
        """
        if device is None:
            device = memory.device
        B = memory.size(0)
        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        min_len = 0 if min_len is None else max(0, min_len)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory, collect_attn=False, memory_mask=memory_mask)  # (B, L, V)
            assert isinstance(logits, torch.Tensor)  # type narrowing
            next_step_logits = logits[:, -1, :]

            # Apply constraints (min_len or ban_token_ids)
            should_clone = False
            if end_token_id is not None and generated.size(1) < max(1, min_len):
                should_clone = True
            if ban_token_ids:
                should_clone = True
            
            # Check for n-gram repetition
            if no_repeat_ngram_size > 0:
                # We might need to clone if we find something to ban
                pass 

            if should_clone:
                next_step_logits = next_step_logits.clone()

            if end_token_id is not None and generated.size(1) < max(1, min_len):
                next_step_logits[:, end_token_id] = float("-inf")
            
            if ban_token_ids:
                next_step_logits[:, ban_token_ids] = float("-inf")

            if no_repeat_ngram_size > 0:
                # Calculate banned tokens based on n-grams
                for b in range(B):
                    gen_seq = generated[b].tolist()
                    if len(gen_seq) < no_repeat_ngram_size - 1:
                        continue
                        
                    prefix = tuple(gen_seq[-(no_repeat_ngram_size - 1):])
                    banned_for_this_batch = set()
                    
                    # Scan history for prefix
                    for i in range(len(gen_seq) - no_repeat_ngram_size + 1):
                        window = tuple(gen_seq[i : i + no_repeat_ngram_size - 1])
                        if window == prefix:
                            # The token that followed this instance of prefix
                            if i + no_repeat_ngram_size - 1 < len(gen_seq):
                                banned_for_this_batch.add(gen_seq[i + no_repeat_ngram_size - 1])
                    
                    if banned_for_this_batch:
                        if not should_clone:
                             next_step_logits = next_step_logits.clone()
                             should_clone = True
                        next_step_logits[b, list(banned_for_this_batch)] = float("-inf")

            next_token = next_step_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if end_token_id is not None:
                # stop if all sequences ended
                if generated.size(1) >= max(1, min_len):
                    if (generated[:, -1] == end_token_id).all():
                        break

        return generated

    # -----------------------------
    # Incremental single-step API
    # -----------------------------
    def step(
        self,
        last_token_ids: torch.LongTensor,
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
        x = self.embedding(last_token_ids) * math.sqrt(self.d_model)  # (B,1,d)
        # Use positional encoding buffer directly (avoid dropout in pos_encoder)
        # pos_encoder.pe expected shape (1, max_len, d_model)
        if hasattr(self.pos_encoder, "pe"):
            pe = self.pos_encoder.pe  # (1, max_len, d_model)
            pos_idx = past_len
            if pos_idx >= pe.size(1):
                raise RuntimeError(f"pos_idx {pos_idx} exceeds max_len {pe.size(1)}")
            x = x + pe[:, pos_idx:pos_idx + 1, :].to(device)
        else:
            # fallback: call pos_encoder and rely on its dropout (less ideal)
            x = self.pos_encoder(x)

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

        # Iterate layers, updating caches and computing output for current token only
        layer_input = x  # (B,1,d_model)
        for i, layer in enumerate(self.layers):
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
            attn_out_heads, self_attn_w = layer.self_attn.attention(Qh, K_all, V_all, mask=None)
            # attn_out_heads: (B, H, 1, d_k)
            # concat heads, project out
            attn_out = attn_out_heads.transpose(1, 2).contiguous().view(B_, 1, num_heads * d_k)
            attn_out = layer.self_attn.W_O(attn_out)  # (B,1,d_model)
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
                MKh = MK.view(Bm, S, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(1, 2)  # (B,H,S,d_k)
                MVh = MV.view(Bm, S, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(1, 2)
                mem_k = MKh
                mem_v = MVh
                new_cache[f"mem_k_{i}"] = mem_k
                new_cache[f"mem_v_{i}"] = mem_v
            else:
                mem_k = mem_k.to(device)
                mem_v = mem_v.to(device)

            Qc = layer.cross_attn.W_Q(x_norm2)  # (B,1,d_model)
            Qch = Qc.view(B, 1, layer.cross_attn.num_heads, layer.cross_attn.d_k).transpose(1, 2)  # (B,H,1,d_k)

            cross_out_heads, cross_attn_w = layer.cross_attn.attention(Qch, mem_k, mem_v, mask=memory_mask)
            cross_out = cross_out_heads.transpose(1, 2).contiguous().view(B, 1, layer.cross_attn.num_heads * layer.cross_attn.d_k)
            cross_out = layer.cross_attn.W_O(cross_out)  # (B,1,d_model)
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