"""Task-aware DataLoader builders for the LexiMind multitask suite."""

from __future__ import annotations

from typing import List

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


class SummarizationCollator:
    """Prepare encoder-decoder batches for abstractive summarization."""

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

    def __call__(self, batch: List[SummarizationExample]) -> dict[str, torch.Tensor]:
        sources = [example.source for example in batch]
        targets = [example.summary for example in batch]

        source_enc = self.tokenizer.batch_encode(sources, max_length=self.max_source_length)
        target_enc = self.tokenizer.batch_encode(targets, max_length=self.max_target_length)

        # target_enc["input_ids"] is [BOS, A, B, EOS, PAD...]
        # We want:
        # tgt_ids (decoder input): [BOS, A, B, EOS] (drop last PAD or EOS if full)
        # labels (target): [A, B, EOS, PAD] (drop first BOS)

        ids = target_enc["input_ids"]
        mask = target_enc["attention_mask"]

        # Slice to create shifted inputs/targets
        # tgt_ids: everything except the last token
        tgt_ids = ids[:, :-1]

        # labels: everything except the first token (BOS)
        labels = ids[:, 1:].clone()

        # Adjust mask for labels to ignore padding
        # The mask corresponds to the original ids. We slice it to match labels.
        labels_mask = mask[:, 1:]
        labels[labels_mask == 0] = -100

        return {
            "src_ids": source_enc["input_ids"],
            "src_mask": source_enc["attention_mask"],
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

    def __call__(self, batch: List[EmotionExample]) -> dict[str, torch.Tensor]:
        texts = [example.text for example in batch]
        encoded = self.tokenizer.batch_encode(texts, max_length=self.max_length)
        label_array = self.binarizer.transform([example.emotions for example in batch])
        labels = torch.as_tensor(label_array, dtype=torch.float32)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class TopicCollator:
    """Prepare batches for topic classification using the projection head."""

    def __init__(
        self, tokenizer: Tokenizer, dataset: TopicDataset, *, max_length: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = dataset.encoder
        self.max_length = max_length

    def __call__(self, batch: List[TopicExample]) -> dict[str, torch.Tensor]:
        texts = [example.text for example in batch]
        encoded = self.tokenizer.batch_encode(texts, max_length=self.max_length)
        labels = torch.as_tensor(
            self.encoder.transform([example.topic for example in batch]), dtype=torch.long
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


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
    collator = EmotionCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    collator = TopicCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
