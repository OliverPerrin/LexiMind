"""
DataLoader builders for LexiMind.

Task-specific collators and factory functions for summarization, emotion, and topic.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .dataset import (
    EmotionDataset,
    EmotionExample,
    SummarizationDataset,
    SummarizationExample,
    TopicDataset,
    TopicExample,
)
from .tokenization import Tokenizer

# --------------- Collators ---------------


class SummarizationCollator:
    """Prepare encoder-decoder batches for seq2seq summarization."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_source_length: int | None = None,
        max_target_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: List[SummarizationExample]) -> Dict[str, torch.Tensor]:
        sources = [ex.source for ex in batch]
        targets = [ex.summary for ex in batch]

        src_enc = self.tokenizer.batch_encode(sources, max_length=self.max_source_length)
        tgt_enc = self.tokenizer.batch_encode(targets, max_length=self.max_target_length)

        # Shift targets: tgt_ids = [BOS, A, B], labels = [A, B, EOS]
        ids = tgt_enc["input_ids"]
        mask = tgt_enc["attention_mask"]

        tgt_ids = ids[:, :-1]
        labels = ids[:, 1:].clone()
        labels[mask[:, 1:] == 0] = -100  # Mask padding in loss

        return {
            "src_ids": src_enc["input_ids"],
            "src_mask": src_enc["attention_mask"],
            "tgt_ids": tgt_ids,
            "labels": labels,
        }


class EmotionCollator:
    """Prepare batches for multi-label emotion classification."""

    def __init__(
        self, tokenizer: Tokenizer, dataset: EmotionDataset, *, max_length: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.binarizer = dataset.binarizer
        self.max_length = max_length

    def __call__(self, batch: List[EmotionExample]) -> Dict[str, torch.Tensor]:
        texts = [ex.text for ex in batch]
        encoded = self.tokenizer.batch_encode(texts, max_length=self.max_length)
        labels = torch.as_tensor(
            self.binarizer.transform([ex.emotions for ex in batch]),
            dtype=torch.float32,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class TopicCollator:
    """Prepare batches for single-label topic classification."""

    def __init__(
        self, tokenizer: Tokenizer, dataset: TopicDataset, *, max_length: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = dataset.encoder
        self.max_length = max_length

    def __call__(self, batch: List[TopicExample]) -> Dict[str, torch.Tensor]:
        texts = [ex.text for ex in batch]
        encoded = self.tokenizer.batch_encode(texts, max_length=self.max_length)
        labels = torch.as_tensor(
            self.encoder.transform([ex.topic for ex in batch]),
            dtype=torch.long,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


# --------------- Factory Functions ---------------


def build_summarization_dataloader(
    dataset: SummarizationDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    max_source_length: int | None = None,
    max_target_length: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create dataloader for summarization task."""
    collator = SummarizationCollator(
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )


def build_emotion_dataloader(
    dataset: EmotionDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    max_length: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create dataloader for emotion classification task."""
    collator = EmotionCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_topic_dataloader(
    dataset: TopicDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    max_length: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create dataloader for topic classification task."""
    collator = TopicCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
