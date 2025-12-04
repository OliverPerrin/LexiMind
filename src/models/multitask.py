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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import TransformerDecoder

# Import your components
from .encoder import TransformerEncoder
from .heads import ClassificationHead, LMHead, TokenClassificationHead


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

    Args:
        encoder: optional encoder backbone.
        decoder: optional decoder backbone.
        decoder_outputs_logits: set True when ``decoder.forward`` already returns vocabulary logits;
            set False if the decoder produces hidden states that must be projected by the LM head.
    """

    def __init__(
        self,
        encoder: Optional[TransformerEncoder] = None,
        decoder: Optional[TransformerDecoder] = None,
        *,
        decoder_outputs_logits: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads: Dict[str, nn.Module] = {}
        # When True, decoder.forward(...) is expected to return logits already projected to the vocabulary space.
        # When False, decoder outputs hidden states that must be passed through the registered LM head.
        self.decoder_outputs_logits = decoder_outputs_logits

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
        # Unwrap for type checking if compiled
        check_head = head
        if hasattr(head, "_orig_mod"):
            check_head = head._orig_mod

        loss_kwargs = loss_kwargs or {}

        # Encoder-only heads expect encoder outputs
        if isinstance(check_head, (ClassificationHead, TokenClassificationHead)):
            if self.encoder is None:
                raise RuntimeError("Encoder is required for encoder-side heads")
            # accept either input_ids or embeddings
            if "input_ids" in inputs:
                encoder_mask = None
                if "attention_mask" in inputs:
                    encoder_mask = self._expand_attention_mask(
                        inputs["attention_mask"], inputs["input_ids"].device
                    )
                enc_out = self.encoder(inputs["input_ids"], mask=encoder_mask)
            elif "embeddings" in inputs:
                encoder_mask = inputs.get("attention_mask")
                if encoder_mask is not None:
                    encoder_mask = self._expand_attention_mask(
                        encoder_mask, inputs["embeddings"].device
                    )
                enc_out = self.encoder(inputs["embeddings"], mask=encoder_mask)
            else:
                raise ValueError(
                    "inputs must contain 'input_ids' or 'embeddings' for encoder tasks"
                )

            # Pass attention_mask to head if available (needed for mean pooling to ignore padding)
            if isinstance(check_head, ClassificationHead):
                logits = head(enc_out, mask=inputs.get("attention_mask"))
            else:
                logits = head(enc_out)

            if return_loss:
                labels = inputs.get("labels", None)
                if labels is None:
                    raise ValueError("return_loss=True requires 'labels' in inputs")
                loss = self.compute_loss_for_head(check_head, logits, labels, **loss_kwargs)
                return loss, logits
            return logits

        # LM/seq2seq head: run encoder -> decoder -> lm head
        if isinstance(check_head, LMHead):
            if self.encoder is None or self.decoder is None:
                raise RuntimeError("Both encoder and decoder are required for LM-style heads")

            # Build encoder memory
            src_mask = inputs.get("src_mask")
            if src_mask is None:
                src_mask = inputs.get("attention_mask")
            encoder_mask = None
            reference_tensor = inputs.get("src_ids")
            if reference_tensor is None:
                reference_tensor = inputs.get("src_embeddings")
            if src_mask is not None and reference_tensor is not None:
                encoder_mask = self._expand_attention_mask(src_mask, reference_tensor.device)

            if "src_ids" in inputs:
                memory = self.encoder(inputs["src_ids"], mask=encoder_mask)
            elif "src_embeddings" in inputs:
                memory = self.encoder(inputs["src_embeddings"], mask=encoder_mask)
            else:
                raise ValueError(
                    "inputs must contain 'src_ids' or 'src_embeddings' for seq2seq tasks"
                )

            # Clone memory to prevent CUDA Graph buffer overwrites when passing between compiled graphs
            # This fixes "accessing tensor output of CUDAGraphs that has been overwritten" error
            if isinstance(memory, torch.Tensor):
                memory = memory.clone()

            # If training / teacher forcing: expect tgt_ids (shifted by caller) or embeddings
            if "tgt_ids" in inputs:
                decoder_inputs = inputs["tgt_ids"]
            elif "tgt_embeddings" in inputs:
                decoder_inputs = inputs["tgt_embeddings"]
            else:
                # For generation time you may call decoder.greedy_decode separately.
                # Here we don't attempt to generate when labels not provided.
                raise ValueError(
                    "Seq2seq tasks require 'tgt_ids' or 'tgt_embeddings' for training forward"
                )

            decoder_out = self.decoder(decoder_inputs, memory, memory_mask=src_mask)

            if self.decoder_outputs_logits:
                if not isinstance(decoder_out, torch.Tensor):
                    raise TypeError(
                        "Decoder is configured to return logits, but forward returned a non-tensor value."
                    )
                logits = decoder_out
            else:
                logits = head(decoder_out)

            if return_loss:
                labels = inputs.get("labels", None)
                if labels is None:
                    raise ValueError("return_loss=True requires 'labels' in inputs for seq2seq")
                loss = self.compute_loss_for_head(check_head, logits, labels, **loss_kwargs)
                return loss, logits
            return logits

        # Otherwise unsupported head type
        raise RuntimeError(f"Unsupported head type: {type(check_head)}")

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
            loss = F.cross_entropy(
                logits.view(B * T, C), labels.view(B * T).long(), ignore_index=ignore_index
            )
            return loss

        if isinstance(head, LMHead):
            # logits: (B, T, V), labels: (B, T)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V), labels.view(B * T).long(), ignore_index=ignore_index
            )
            return loss

        # Generic fall-back: try CrossEntropy on final dim
        if logits.dim() == 2:
            return F.cross_entropy(logits, labels.long())

        # If we can't determine, raise
        raise RuntimeError("Cannot compute loss for unknown head type")

    @staticmethod
    def _expand_attention_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        if mask is None:
            return None  # type: ignore[return-value]
        bool_mask = mask.to(device=device, dtype=torch.bool)
        if bool_mask.dim() == 2:
            return bool_mask.unsqueeze(1) & bool_mask.unsqueeze(2)
        if bool_mask.dim() in (3, 4):
            return bool_mask
        raise ValueError("Attention mask must be 2D, 3D, or 4D tensor")
