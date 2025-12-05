"""
Text preprocessing for LexiMind.

Lightweight text cleaning and tokenization pipeline for model input preparation.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

import torch

from .tokenization import Tokenizer, TokenizerConfig

# --------------- Text Cleaning ---------------


class TextCleaner:
    """Basic text normalization."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """Strip, normalize whitespace, optionally lowercase."""
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        return " ".join(text.split())

    def clean_batch(self, texts: Sequence[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(t) for t in texts]

    # Backwards compatibility alias
    def transform(self, texts: Sequence[str]) -> List[str]:
        """Alias for clean_batch (sklearn-style interface)."""
        return self.clean_batch(texts)


# --------------- Batch Output ---------------


@dataclass
class Batch:
    """Tokenized batch ready for model consumption."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: List[int]


# --------------- Preprocessor ---------------


class TextPreprocessor:
    """Combines text cleaning with tokenization."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        tokenizer_config: TokenizerConfig | None = None,
        tokenizer_name: str = "google/flan-t5-base",
        max_length: int | None = None,
        lowercase: bool = True,
    ) -> None:
        self.cleaner = TextCleaner(lowercase=lowercase)

        # Initialize or validate tokenizer
        if tokenizer is None:
            cfg = tokenizer_config or TokenizerConfig(pretrained_model_name=tokenizer_name)
            if max_length is not None:
                cfg = replace(cfg, max_length=max_length)
            self.tokenizer = Tokenizer(cfg)
        else:
            self.tokenizer = tokenizer
            if max_length is not None and max_length != tokenizer.config.max_length:
                raise ValueError(
                    "max_length conflicts with tokenizer config - "
                    "initialize tokenizer with desired settings"
                )

        self.max_length = max_length or self.tokenizer.config.max_length

    def clean_text(self, text: str) -> str:
        """Clean a single text."""
        return self.cleaner.clean(text)

    def batch_encode(self, texts: Sequence[str]) -> Batch:
        """Clean and tokenize texts into a batch."""
        cleaned = self.cleaner.clean_batch(texts)
        encoded = self.tokenizer.batch_encode(cleaned, max_length=self.max_length)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
        lengths = attention_mask.sum(dim=1).tolist()

        return Batch(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths)

    def __call__(self, texts: Sequence[str]) -> Batch:
        """Alias for batch_encode."""
        return self.batch_encode(texts)


# --------------- Backwards Compatibility ---------------

# Keep old name for any imports
BasicTextCleaner = TextCleaner
