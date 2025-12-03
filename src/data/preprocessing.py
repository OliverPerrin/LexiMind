"""Text preprocessing utilities built around Hugging Face tokenizers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence

import torch
from sklearn.base import BaseEstimator, TransformerMixin

from .tokenization import Tokenizer, TokenizerConfig


class BasicTextCleaner(BaseEstimator, TransformerMixin):
    """Minimal text cleaner following scikit-learn conventions."""

    def __init__(self, lowercase: bool = True, strip: bool = True) -> None:
        self.lowercase = lowercase
        self.strip = strip

    def fit(self, texts: Iterable[str], y: Iterable[str] | None = None):
        return self

    def transform(self, texts: Iterable[str]) -> List[str]:
        return [self._clean_text(text) for text in texts]

    def _clean_text(self, text: str) -> str:
        item = text.strip() if self.strip else text
        if self.lowercase:
            item = item.lower()
        return " ".join(item.split())


@dataclass(slots=True)
class Batch:
    """Bundle of tensors returned by the text preprocessor."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: List[int]


class TextPreprocessor:
    """Coordinate lightweight text cleaning and tokenization.

    When supplying an already-initialized tokenizer instance, its configuration is left
    untouched. If a differing ``max_length`` is requested, a ``ValueError`` is raised to
    avoid mutating shared tokenizer state.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        tokenizer_config: TokenizerConfig | None = None,
        tokenizer_name: str = "facebook/bart-base",
        max_length: int | None = None,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        sklearn_transformer: TransformerMixin | None = None,
    ) -> None:
        self.cleaner = BasicTextCleaner(lowercase=lowercase, strip=True)
        self.lowercase = lowercase
        if remove_stopwords:
            raise ValueError(
                "Stop-word removal is not supported because it conflicts with subword tokenizers; "
                "clean the text externally before initializing TextPreprocessor."
            )
        self._stop_words = None
        self._sklearn_transformer = sklearn_transformer

        if tokenizer is None:
            cfg = tokenizer_config or TokenizerConfig(pretrained_model_name=tokenizer_name)
            if max_length is not None:
                cfg = replace(cfg, max_length=max_length)
            self.tokenizer = Tokenizer(cfg)
        else:
            self.tokenizer = tokenizer
            if max_length is not None and max_length != tokenizer.config.max_length:
                raise ValueError(
                    "Provided tokenizer config.max_length does not match requested max_length; "
                    "initialise the tokenizer with desired settings before passing it in."
                )

        self.max_length = max_length or self.tokenizer.config.max_length

    def clean_text(self, text: str) -> str:
        item = self.cleaner.transform([text])[0]
        return self._normalize_tokens(item)

    def _normalize_tokens(self, text: str) -> str:
        """Apply token-level normalization and optional stop-word filtering."""
        # Note: Pre-tokenization word-splitting is incompatible with subword tokenizers.
        # Stop-word filtering should be done post-tokenization or not at all for transformers.
        return text

    def _apply_sklearn_transform(self, texts: List[str]) -> List[str]:
        if self._sklearn_transformer is None:
            return texts

        transform = getattr(self._sklearn_transformer, "transform", None)
        if transform is None:
            raise AttributeError("Provided sklearn transformer must implement a 'transform' method")
        transformed = transform(texts)
        if isinstance(transformed, list):
            return transformed  # assume downstream type is already list[str]
        if hasattr(transformed, "tolist"):
            transformed = transformed.tolist()

        result = list(transformed)
        if not all(isinstance(item, str) for item in result):
            result = [str(item) for item in result]
        return result

    def _prepare_texts(self, texts: Sequence[str]) -> List[str]:
        cleaned = self.cleaner.transform(texts)
        normalized = [self._normalize_tokens(text) for text in cleaned]
        return self._apply_sklearn_transform(normalized)

    def batch_encode(self, texts: Sequence[str]) -> Batch:
        cleaned = self._prepare_texts(texts)
        encoded = self.tokenizer.batch_encode(cleaned, max_length=self.max_length)
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"].to(dtype=torch.bool)
        lengths = attention_mask.sum(dim=1).tolist()
        return Batch(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths)

    def __call__(self, texts: Sequence[str]) -> Batch:
        return self.batch_encode(texts)
