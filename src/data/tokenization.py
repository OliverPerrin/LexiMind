"""
Tokenizer facade for LexiMind.

Wraps HuggingFace tokenizers with a simplified interface that handles
special token management, batch encoding, and T5-specific conventions
for decoder input preparation.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenizerConfig:
    pretrained_model_name: str = "google/flan-t5-base"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    lower: bool = False


class Tokenizer:
    """Lightweight façade over a HuggingFace tokenizer."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        cfg = config or TokenizerConfig()
        self.config = cfg
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name
        )
        self._pad_token_id = self._resolve_id(self._tokenizer.pad_token_id)

        # T5 uses different special tokens than BART:
        # T5: pad=0, eos=1, no explicit bos (uses pad or eos as decoder start)
        # BART: bos=0, pad=1, eos=2
        # We use eos_token_id as bos for T5 decoder start (common practice)
        eos_id = self._tokenizer.eos_token_id
        bos_id = self._tokenizer.bos_token_id

        # For T5, decoder_start_token_id is typically pad_token_id (0)
        # But we'll use a sensible default based on what's available
        if bos_id is not None:
            self._bos_token_id = self._resolve_id(bos_id)
        elif (
            hasattr(self._tokenizer, "decoder_start_token_id")
            and self._tokenizer.decoder_start_token_id is not None
        ):
            self._bos_token_id = self._resolve_id(self._tokenizer.decoder_start_token_id)
        else:
            # T5 convention: use pad_token_id as decoder start
            self._bos_token_id = self._pad_token_id

        self._eos_token_id = self._resolve_id(
            eos_id if eos_id is not None else self._tokenizer.sep_token_id
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        vocab = getattr(self._tokenizer, "vocab_size", None)
        if vocab is None:
            raise RuntimeError("Tokenizer must expose vocab_size")
        return int(vocab)

    @staticmethod
    def _resolve_id(value) -> int:
        if value is None:
            raise ValueError("Tokenizer is missing required special token ids")
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)

    def encode(self, text: str) -> List[int]:
        content = text.lower() if self.config.lower else text
        return cast(
            List[int],
            self._tokenizer.encode(
                content,
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
            ),
        )

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        normalized = (text.lower() if self.config.lower else text for text in texts)
        encoded = self._tokenizer.batch_encode_plus(
            list(normalized),
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_attention_mask=False,
            return_tensors=None,
        )
        return cast(List[List[int]], encoded["input_ids"])

    def batch_encode(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        normalized = [text.lower() if self.config.lower else text for text in texts]
        encoded = self._tokenizer(
            normalized,
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=max_length or self.config.max_length,
            return_tensors="pt",
        )
        input_ids = cast(torch.Tensor, encoded["input_ids"])
        attention_mask = cast(torch.Tensor, encoded["attention_mask"])
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids: Iterable[int]) -> str:
        return cast(str, self._tokenizer.decode(list(token_ids), skip_special_tokens=True))

    def decode_batch(self, sequences: Sequence[Sequence[int]]) -> List[str]:
        prepared = [list(seq) for seq in sequences]
        return cast(List[str], self._tokenizer.batch_decode(prepared, skip_special_tokens=True))

    def prepare_decoder_inputs(self, labels: torch.Tensor) -> torch.Tensor:
        """Shift decoder labels to create input ids prefixed by BOS."""

        bos = self.bos_token_id
        pad = self.pad_token_id
        decoder_inputs = torch.full_like(labels, pad)
        decoder_inputs[:, 0] = bos
        decoder_inputs[:, 1:] = labels[:, :-1]
        return decoder_inputs
