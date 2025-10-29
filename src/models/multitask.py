"""
Multitask model composition utilities.

Provides:
- MultiTaskModel: lightweight wrapper to compose an encoder and/or decoder with
  multiple task heads (classification, token classification, LM head, etc.)
- add_head / remove_head helpers
- forward(task_name, ...) that routes inputs to the correct sub-modules
- compute_loss helper that uses common losses and ignore_index support

Design goals:
- Keep composition simple and explicit (use named heads per task)
- Support encoder-only tasks (classification, token classification) and
  seq2seq tasks (encoder -> decoder -> LMHead)
- Minimal dependencies on training loop; return logits and (optionally) loss
"""
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your components
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .heads import ClassificationHead, TokenClassificationHead, LMHead


class MultiTaskModel(nn.Module):
    """
    Compose encoder/decoder and task heads.

    Usage patterns:
    - Encoder-only classification:
        mt = MultiTaskModel(encoder=enc)
        mt.add_head("sentiment", ClassificationHead(...))
        logits = mt.forward("sentiment", {"input_ids": src_ids})
    - Seq2seq LM:
        mt = MultiTaskModel(encoder=enc, decoder=dec)
        mt.add_head("summarize", LMHead(...))
        logits = mt.forward("summarize", {"src_ids": src_ids, "tgt_ids": tgt_ids})
    """

    def __init__(
        self,
        encoder: Optional[TransformerEncoder] = None,
        decoder: Optional[TransformerDecoder] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads: Dict[str, nn.Module] = {}

    def add_head(self, name: str, module: nn.Module) -> None:
        """Register a head under a task name."""
        if name in self.heads:
            raise ValueError(f"Head '{name}' already exists")
        self.heads[name] = module
        self.add_module(f"head_{name}", module)

    def remove_head(self, name: str) -> None:
        """Remove a registered head."""
        if name not in self.heads:
            raise KeyError(name)
        del self._modules[f"head_{name}"]
        del self.heads[name]

    def forward(
        self,
        task: str,
        inputs: Dict[str, torch.Tensor],
        return_loss: bool = False,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Route inputs to appropriate model components and head.

        Args:
            task: registered head name
            inputs: dictionary; common keys:
                - For encoder tasks: "input_ids" or "embeddings" (B, S) or (B, S, d)
                - For seq2seq: "src_ids" (B,S) or "src_embeddings", and "tgt_ids" (B,T) or "tgt_embeddings"
                            when computing training loss, pass "labels" (B,T) for LM
            return_loss: if True and labels provided, returns (loss, logits)
            loss_kwargs: forwarded to compute_loss (e.g., ignore_index)

        Returns:
            logits (or (loss, logits) if return_loss True)
        """
        if task not in self.heads:
            raise KeyError(f"Unknown task/head '{task}'")

        head = self.heads[task]
        loss_kwargs = loss_kwargs or {}

        # Encoder-only heads expect encoder outputs
        if isinstance(head, (ClassificationHead, TokenClassificationHead)):
            if self.encoder is None:
                raise RuntimeError("Encoder is required for encoder-side heads")
            # accept either input_ids or embeddings
            if "input_ids" in inputs:
                enc_out = self.encoder(inputs["input_ids"])
            elif "embeddings" in inputs:
                enc_out = self.encoder(inputs["embeddings"])
            else:
                raise ValueError("inputs must contain 'input_ids' or 'embeddings' for encoder tasks")
            logits = head(enc_out)

            if return_loss:
                labels = inputs.get("labels", None)
                if labels is None:
                    raise ValueError("return_loss=True requires 'labels' in inputs")
                loss = self.compute_loss_for_head(head, logits, labels, **loss_kwargs)
                return loss, logits
            return logits

        # LM/seq2seq head: run encoder -> decoder -> lm head
        if isinstance(head, LMHead):
            if self.encoder is None or self.decoder is None:
                raise RuntimeError("Both encoder and decoder are required for LM-style heads")

            # Build encoder memory
            if "src_ids" in inputs:
                memory = self.encoder(inputs["src_ids"])
            elif "src_embeddings" in inputs:
                memory = self.encoder(inputs["src_embeddings"])
            else:
                raise ValueError("inputs must contain 'src_ids' or 'src_embeddings' for seq2seq tasks")

            # If training / teacher forcing: expect tgt_ids (shifted by caller) or embeddings
            if "tgt_ids" in inputs:
                decoder_inputs = inputs["tgt_ids"]
            elif "tgt_embeddings" in inputs:
                decoder_inputs = inputs["tgt_embeddings"]
            else:
                # For generation time you may call decoder.greedy_decode separately.
                # Here we don't attempt to generate when labels not provided.
                raise ValueError("Seq2seq tasks require 'tgt_ids' or 'tgt_embeddings' for training forward")

            # Run decoder. Decoder returns logits shaped (B, T, vocab) in this codebase.
            decoder_out = self.decoder(decoder_inputs, memory)

            # If decoder already returned logits matching the head vocab size, use them directly.
            # Otherwise, assume decoder returned hidden states and let the head project them.
            if isinstance(decoder_out, torch.Tensor) and decoder_out.shape[-1] == head.vocab_size:
                logits = decoder_out
            else:
                logits = head(decoder_out)

            if return_loss:
                labels = inputs.get("labels", None)
                if labels is None:
                    raise ValueError("return_loss=True requires 'labels' in inputs for seq2seq")
                loss = self.compute_loss_for_head(head, logits, labels, **loss_kwargs)
                return loss, logits
            return logits

        # Otherwise unsupported head type
        raise RuntimeError(f"Unsupported head type: {type(head)}")

    def compute_loss_for_head(
        self,
        head: nn.Module,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Default loss dispatch:
         - ClassificationHead: CrossEntropy on (B, num_labels)
         - TokenClassificationHead: CrossEntropy per token (flattened)
         - LMHead: CrossEntropy per token (flattened), ignore_index supported

        Returns scalar loss.
        """
        if isinstance(head, ClassificationHead):
            # logits: (B, num_labels) or (B, num_labels) direct
            loss = F.cross_entropy(logits, labels.long())
            return loss

        if isinstance(head, TokenClassificationHead):
            # logits: (B, T, C), labels: (B, T)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T).long(), ignore_index=ignore_index)
            return loss

        if isinstance(head, LMHead):
            # logits: (B, T, V), labels: (B, T)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T).long(), ignore_index=ignore_index)
            return loss

        # Generic fall-back: try CrossEntropy on final dim
        if logits.dim() == 2:
            return F.cross_entropy(logits, labels.long())

        # If we can't determine, raise
        raise RuntimeError("Cannot compute loss for unknown head type")