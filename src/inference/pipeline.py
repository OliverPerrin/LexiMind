"""
Inference pipeline for LexiMind.

Unified interface for summarization, emotion detection, and topic classification
with batched processing and device management.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, cast

import torch
import torch.nn.functional as F

from ..data.tokenization import Tokenizer

# --------------- Text Formatting ---------------


def _format_summary(text: str) -> str:
    """Clean and format generated summary text.

    - Capitalize first letter
    - Fix period spacing (". " not " .")
    - Remove extra whitespace
    - Ensure proper sentence endings
    """
    if not text:
        return text

    # Strip and normalize whitespace
    text = " ".join(text.split())

    # Remove leading punctuation/special chars
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)  # Remove space before punctuation
    text = re.sub(
        r"([.!?])([A-Za-z])", r"\1 \2", text
    )  # Add space after sentence-ending punctuation

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Capitalize after sentence-ending punctuation
    text = re.sub(r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), text)

    # Ensure ends with punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


# --------------- Configuration ---------------


@dataclass
class InferenceConfig:
    """Pipeline settings."""

    summary_max_length: int = 128
    summary_repetition_penalty: float = 1.2  # Penalize repeated tokens
    summary_length_penalty: float = 1.5  # Encourage EOS token as length increases (>1 = shorter)
    summary_formatting: bool = True  # Apply text cleanup/formatting to generated summaries
    emotion_threshold: float = 0.5
    device: str | None = None


@dataclass
class EmotionPrediction:
    labels: List[str]
    scores: List[float]


@dataclass
class TopicPrediction:
    label: str
    confidence: float


# --------------- Pipeline ---------------


class InferencePipeline:
    """Multi-task inference with batched processing."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        *,
        emotion_labels: Sequence[str] | None = None,
        topic_labels: Sequence[str] | None = None,
        config: InferenceConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        # Resolve device
        chosen = device or self.config.device
        if chosen is None:
            param = next(model.parameters(), None)
            buffer = next(model.buffers(), None)
            chosen = (
                param.device
                if param is not None
                else buffer.device
                if buffer is not None
                else "cpu"
            )
        self.device = torch.device(chosen)

        self.model.to(self.device)
        self.model.eval()

        self.emotion_labels = list(emotion_labels) if emotion_labels else None
        self.topic_labels = list(topic_labels) if topic_labels else None

    # --------------- Summarization ---------------

    def summarize(self, texts: Sequence[str], *, max_length: int | None = None) -> List[str]:
        """Generate summaries for input texts."""
        if not texts:
            return []

        encoded = self.tokenizer.batch_encode(list(texts))
        src_ids = encoded["input_ids"].to(self.device)
        src_mask = encoded["attention_mask"].to(self.device)
        max_len = self.config.summary_max_length if max_length is None else max_length
        if max_len < 1:
            raise ValueError("max_length must be positive")

        model = cast(Any, self.model)
        if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
            raise RuntimeError("Model must have encoder and decoder for summarization")

        with torch.inference_mode():
            # Encode
            enc_mask = (
                src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
            )
            memory = model.encoder(src_ids, mask=enc_mask)

            return self._summarize_memory(memory, src_mask, max_len)

    def _summarize_memory(
        self, memory: torch.Tensor, src_mask: torch.Tensor, max_len: int
    ) -> List[str]:
        """Decode already-encoded memory; caller holds inference_mode."""
        model = cast(Any, self.model)

        # Keep the established decoding settings unchanged.
        ban_ids = [self.tokenizer.bos_token_id, self.tokenizer.pad_token_id]
        unk = getattr(self.tokenizer._tokenizer, "unk_token_id", None)
        if isinstance(unk, int):
            ban_ids.append(unk)

        generated = model.decoder.greedy_decode(
            memory=memory,
            max_len=max_len,
            start_token_id=self.tokenizer.bos_token_id,
            end_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            min_len=10,
            ban_token_ids=[i for i in ban_ids if i is not None],
            no_repeat_ngram_size=3,
            repetition_penalty=self.config.summary_repetition_penalty,
            length_penalty=self.config.summary_length_penalty,
            memory_mask=src_mask,
        )

        # Decode and format summaries
        raw_summaries = self.tokenizer.decode_batch(generated.tolist())
        if not self.config.summary_formatting:
            return raw_summaries
        return [_format_summary(s) for s in raw_summaries]

    # --------------- Emotion ---------------

    def predict_emotions(
        self,
        texts: Sequence[str],
        *,
        threshold: float | None = None,
    ) -> List[EmotionPrediction]:
        """Predict emotions for input texts."""
        if not texts:
            return []
        if not self.emotion_labels:
            raise RuntimeError("emotion_labels required for emotion prediction")

        encoded = self.tokenizer.batch_encode(list(texts))
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        thresh = self.config.emotion_threshold if threshold is None else threshold
        if not 0 <= thresh <= 1:
            raise ValueError("Emotion threshold must be in [0, 1]")

        with torch.inference_mode():
            logits = self.model.forward("emotion", inputs)
            return self._emotion_predictions(logits, thresh)

    def _emotion_predictions(
        self, logits: torch.Tensor, threshold: float
    ) -> List[EmotionPrediction]:
        if logits.ndim != 2 or logits.shape[-1] != len(self.emotion_labels or []):
            raise ValueError("Emotion logits do not match the supplied label order")
        probs = torch.sigmoid(logits)

        results = []
        for row in probs.cpu():
            pairs = [
                (label, score)
                for label, score in zip(self.emotion_labels or [], row.tolist(), strict=True)
                if score >= threshold
            ]
            results.append(
                EmotionPrediction(
                    labels=[label for label, _ in pairs],
                    scores=[score for _, score in pairs],
                )
            )
        return results

    # --------------- Topic ---------------

    def predict_topics(self, texts: Sequence[str]) -> List[TopicPrediction]:
        """Predict topic for input texts."""
        if not texts:
            return []
        if not self.topic_labels:
            raise RuntimeError("topic_labels required for topic prediction")

        encoded = self.tokenizer.batch_encode(list(texts))
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.inference_mode():
            logits = self.model.forward("topic", inputs)
            return self._topic_predictions(logits)

    def _topic_predictions(self, logits: torch.Tensor) -> List[TopicPrediction]:
        if logits.ndim != 2 or logits.shape[-1] != len(self.topic_labels or []):
            raise ValueError("Topic logits do not match the supplied label order")
        probs = F.softmax(logits, dim=-1)

        results = []
        for row in probs.cpu():
            idx = int(row.argmax().item())
            results.append(
                TopicPrediction(
                    label=(self.topic_labels or [])[idx],
                    confidence=row[idx].item(),
                )
            )
        return results

    # --------------- Batch Prediction ---------------

    def batch_predict(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Run all three tasks on input texts."""
        if not self.emotion_labels or not self.topic_labels:
            raise RuntimeError("Both emotion_labels and topic_labels required")

        text_list = list(texts)
        if not text_list:
            return {"summaries": [], "emotion": [], "topic": []}
        model = cast(Any, self.model)
        # Older/custom model implementations retain the existing task-forward
        # fallback. LexiMind's model can reuse one encoder pass for all tasks.
        if callable(getattr(model, "classify_encoded", None)):
            if not 0 <= self.config.emotion_threshold <= 1:
                raise ValueError("Emotion threshold must be in [0, 1]")
            if self.config.summary_max_length < 1:
                raise ValueError("summary_max_length must be positive")
            encoded = self.tokenizer.batch_encode(text_list)
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            with torch.inference_mode():
                memory = model.encoder(ids, mask=mask.unsqueeze(1) & mask.unsqueeze(2))
                return {
                    "summaries": self._summarize_memory(
                        memory, mask, self.config.summary_max_length
                    ),
                    "emotion": self._emotion_predictions(
                        model.classify_encoded("emotion", memory, mask),
                        self.config.emotion_threshold,
                    ),
                    "topic": self._topic_predictions(model.classify_encoded("topic", memory, mask)),
                }
        return {
            "summaries": self.summarize(text_list),
            "emotion": self.predict_emotions(text_list),
            "topic": self.predict_topics(text_list),
        }

    # --------------- Helpers ---------------
    # (helper methods removed - encoding now happens inline)
