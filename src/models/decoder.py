"""
Transformer Decoder layout (Pre-LN)

Contents:
- create_causal_mask: utility to build a causal (subsequent) mask
- TransformerDecoderLayer: one decoder block (masked self-attn, cross-attn, FFN)
- TransformerDecoder: embedding/pos-encoding + stack of decoder layers + generation helpers

Notes / conventions:
- Pre-LN (LayerNorm before each sublayer) is assumed for stability (consistent with your encoder).
- MultiHeadAttention, FeedForward, PositionalEncoding are expected to live in src/models
  (you already implemented them).
- Masks use boolean semantics: True = allowed, False = masked.
- The decoder API supports:
    - inputs: token ids (LongTensor, (B, T)) or embeddings ((B, T, d_model))
    - memory: encoder outputs (B, S, d_model)
    - mask arguments: tgt_mask (causal/padding), memory_mask (encoder padding)
    - collect_attn: return attention maps per layer if requested
- Generation helpers (greedy) are skeletons that you can extend to beam search or caching.

TODO status keys:
- [IMPLEMENT] : core implementation required
- [OPTIONAL]  : useful enhancement (caching, beam search, advanced scheduling)
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
    Create a square causal mask of shape (seq_len, seq_len).
    True indicates allowed positions; False indicates masked (future) positions.

    Returns:
        mask: torch.BoolTensor of shape (seq_len, seq_len)
    """
    # return a mask with True on and below diagonal, False above diagonal 
    # The torch.trui function does masking, which is the idea of zeroing all the values in a matrix below a certain diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    # mask has True above diagonal (to be masked). Want True for allowed, so invert:
    return ~mask # shape (seq_len, seq_len) or (T, T)


class TransformerDecoderLayer(nn.Module):
    """
    One decoder layer with:
      - Masked self-attention (query/key/value = tgt)
      - Encoder-Decoder cross-attention (query = tgt, key/value = memory)
      - Position-wise FeedForward
      - Pre-LN + residuals + dropout

    Args:
      d_model: model hidden size
      num_heads: number of attention heads
      d_ff: ff intermediate size
      dropout: dropout for residuals / FFN
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # NOTE: instantiate internal MHA with dropout=0.0 and manage dropout at layer-level
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # LayerNorms (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropouts applied after sublayers (on sublayer outputs before residual add)
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
        Forward pass for one decoder layer.

        Args:
            tgt: (batch, tgt_len, d_model)
            memory: (batch, src_len, d_model)  -- encoder outputs
            tgt_mask: optional (batch, tgt_len, tgt_len) or (batch, 1, tgt_len, tgt_len)
            memory_mask: optional (batch, src_len, src_len) or (batch, 1, tgt_len, src_len)

        Returns:
            output: (batch, tgt_len, d_model)
            attn_maps: dict with keys 'self' and 'cross' containing attention tensors
                       shapes: (batch, num_heads, tgt_len, tgt_len) and (batch, num_heads, tgt_len, src_len)
        """
        # TODO [IMPLEMENT] Self-attention (Pre-LN)
        # x_norm = self.norm1(tgt)
        # self_out, self_attn = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        # tgt = tgt + self.dropout1(self_out)

        # TODO [IMPLEMENT] Cross-attention (Pre-LN)
        # x_norm = self.norm2(tgt)
        # cross_out, cross_attn = self.cross_attn(x_norm, memory, memory, memory_mask)
        # tgt = tgt + self.dropout2(cross_out)

        # TODO [IMPLEMENT] Feed-forward (Pre-LN)
        # x_norm = self.norm3(tgt)
        # ffn_out = self.ffn(x_norm)
        # tgt = tgt + self.dropout3(ffn_out)

        # TODO [RETURN] Return (tgt, {"self": self_attn, "cross": cross_attn})
        raise NotImplementedError("TransformerDecoderLayer.forward: implement Pre-LN pipeline")


class TransformerDecoder(nn.Module):
    """
    Full decoder: token embedding + positional encoding + stack of decoder layers.
    Also supports simple greedy decoding.

    Args:
        vocab_size: for token embeddings (if using token ids)
        d_model, num_layers, num_heads, d_ff, dropout, max_len, pad_token_id: same semantics as encoder
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

        # Token embedding (used if inputs are token ids)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm for Pre-LN stacks
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary (logits)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Input dropout (after pos encoding)
        self.input_dropout = nn.Dropout(dropout)

    def _build_padding_mask_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build (batch, seq, seq) boolean mask from input ids and pad_token_id.
        True = allowed, False = masked.
        """
        assert self.pad_token_id is not None, "pad_token_id must be set to build mask from ids"
        pad_mask = (input_ids != self.pad_token_id)  # (B, S)
        attn_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B, S, S)
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
        Forward pass for the decoder stack.

        Args:
            inputs: token ids (B, T) or embeddings (B, T, d_model)
            memory: encoder outputs (B, S, d_model)
            tgt_mask: optional mask for decoder self-attention. If None, a causal mask will be created.
                      Mask shapes: (B, T, T) or (B, 1, T, T)
            memory_mask: optional mask over memory (B, S, S) or (B, 1, T, S)
            collect_attn: if True returns (logits/outputs, [per-layer attn dicts])

        Returns:
            logits: (B, T, vocab_size) or (B, T, d_model) if you prefer returning hidden states
            or (logits, attn_list) if collect_attn True
        """
        # Inputs: if token ids, embed and scale; else assume embeddings
        if inputs.dim() == 2:  # token ids
            x = self.embedding(inputs) * math.sqrt(self.d_model)
        elif inputs.dim() == 3:
            x = inputs
        else:
            raise ValueError("inputs must be (B, T) token ids or (B, T, d_model) embeddings")

        # Positional encoding + dropout
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # Build tgt_mask if not provided: combine causal mask and padding mask if available
        seq_len = x.size(1)
        if tgt_mask is None:
            # base causal mask (T, T)
            causal = create_causal_mask(seq_len, device=x.device)  # [TODO implement]
            # expand to batch dim later if padding present
            if inputs.dim() == 2 and self.pad_token_id is not None:
                padding_mask = self._build_padding_mask_from_ids(inputs)  # (B, T, T)
                # combine: True only where both causal and padding allow attention
                # TODO: ensure shapes align; broadcast causal to (1, T, T) then & with padding_mask
                raise NotImplementedError("tgt_mask construction: combine causal + padding_mask")
            else:
                # TODO: Broadcast causal to (1, T, T) or (B, 1, T, T) depending on downstream expectations
                raise NotImplementedError("tgt_mask construction: broadcast causal mask for batch")

        # Ensure memory_mask is boolean on correct device if provided
        if memory_mask is not None:
            memory_mask = memory_mask.to(dtype=torch.bool, device=x.device)

        attn_list: List[Dict[str, torch.Tensor]] = []

        # Pass through layers
        for layer in self.layers:
            x, attn = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if collect_attn:
                attn_list.append(attn)

        x = self.final_norm(x)  # Pre-LN final normalization

        logits = self.output_projection(x)  # (B, T, vocab)
        if collect_attn:
            return logits, attn_list
        return logits

    # ---------------------------------------------------------------------
    # Generation / inference helpers (skeletons)
    # ---------------------------------------------------------------------
    def greedy_decode(
        self,
        memory: torch.Tensor,
        max_len: int,
        start_token_id: int,
        end_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """
        Greedy autoregressive decoding using the decoder stack.

        Args:
            memory: encoder outputs (B, S, d_model)
            max_len: maximum target length to generate
            start_token_id: BOS token id
            end_token_id: optional EOS token id to stop early
        Returns:
            generated: (B, T_out) long tensor of token ids
        """
        # TODO [IMPLEMENT]:
        #  - Start with tensor of shape (B, 1) filled with start_token_id
        #  - Repeatedly call decoder.forward in incremental mode (or full forward with causal mask)
        #  - At each step pick argmax over logits and append to sequence
        #  - Stop if all sequences produced end_token_id or max_len reached
        raise NotImplementedError("greedy_decode: implement autoregressive generation loop")

    # Optional: incremental step method with caching of past keys/values for speed
    def step(
        self,
        last_token_ids: torch.LongTensor,
        memory: torch.Tensor,
        cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single-step decoder that returns logits for the next token given last_token_ids.

        Args:
            last_token_ids: (B, 1) tokens at current time step
            memory: encoder outputs
            cache: optional dict storing per-layer cached keys/values

        Returns:
            logits: (B, vocab_size)
            new_cache: updated cache
        """
        # TODO [OPTIONAL]: implement fast incremental decoding caching keys/values per layer
        raise NotImplementedError("step: incremental decoding (optional optimization)")