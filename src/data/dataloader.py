"""Task-aware DataLoader builders for the LexiMind multitask suite."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from .dataset import EmotionDataset, EmotionExample, SummarizationDataset, SummarizationExample, TopicDataset, TopicExample
from .tokenization import Tokenizer


class SummarizationCollator:
    """Prepare encoder-decoder batches for abstractive summarization."""

    def __init__(self, tokenizer: Tokenizer, *, max_source_length: int | None = None, max_target_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: List[SummarizationExample]) -> dict[str, torch.Tensor]:
        sources = [example.source for example in batch]
        targets = [example.summary for example in batch]

        source_enc = self.tokenizer.batch_encode(sources, max_length=self.max_source_length)
        target_enc = self.tokenizer.batch_encode(targets, max_length=self.max_target_length)

        labels = target_enc["input_ids"].clone()
        decoder_input_ids = self.tokenizer.prepare_decoder_inputs(target_enc["input_ids"])
        labels[target_enc["attention_mask"] == 0] = -100

        return {
            "src_ids": source_enc["input_ids"],
            "src_mask": source_enc["attention_mask"],
            "tgt_ids": decoder_input_ids,
            "labels": labels,
        }


class EmotionCollator:
    """Prepare batches for multi-label emotion classification."""

    def __init__(self, tokenizer: Tokenizer, dataset: EmotionDataset, *, max_length: int | None = None) -> None:
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

    def __init__(self, tokenizer: Tokenizer, dataset: TopicDataset, *, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.encoder = dataset.encoder
        self.max_length = max_length

    def __call__(self, batch: List[TopicExample]) -> dict[str, torch.Tensor]:
        texts = [example.text for example in batch]
        encoded = self.tokenizer.batch_encode(texts, max_length=self.max_length)
        labels = torch.as_tensor(self.encoder.transform([example.topic for example in batch]), dtype=torch.long)
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
) -> DataLoader:
    collator = SummarizationCollator(
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def build_emotion_dataloader(
    dataset: EmotionDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    max_length: int | None = None,
) -> DataLoader:
    collator = EmotionCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def build_topic_dataloader(
    dataset: TopicDataset,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool = True,
    max_length: int | None = None,
) -> DataLoader:
    collator = TopicCollator(tokenizer, dataset, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
