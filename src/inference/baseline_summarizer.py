"""Thin wrapper around the custom transformer summarizer."""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch
from ..api.inference import load_models


class TransformerSummarizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        models = load_models(config or {})
        if not models.get("loaded"):
            raise RuntimeError("load_models returned an unloaded model; check configuration")
        self.model = models["mt"]
        self.preprocessor = models["preprocessor"]
        self.device = models["device"]

    def summarize(
        self,
        text: str,
        compression: float = 0.25,
        collect_attn: bool = False,
    ) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
        batch = self.preprocessor.batch_encode([text])
        tokenizer = self.preprocessor.tokenizer
        encoder = self.model.encoder
        decoder = self.model.decoder
        if tokenizer is None or encoder is None or decoder is None:
            raise RuntimeError("Model components are missing; ensure encoder, decoder, and tokenizer are set")
        input_ids = batch.input_ids.to(self.device)
        memory = encoder(input_ids)
        src_len = batch.lengths[0]
        target_len = max(4, int(src_len * compression))
        generated = decoder.greedy_decode(
            memory,
            max_len=min(self.preprocessor.max_length, target_len),
            start_token_id=tokenizer.bos_id,
            end_token_id=tokenizer.eos_id,
        )
        summary = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        return summary.strip(), None if not collect_attn else {}
