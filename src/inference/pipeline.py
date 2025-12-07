"""
Inference pipeline for LexiMind.

Unified interface for summarization, emotion detection, and topic classification
with batched processing and device management.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, List, Sequence, cast

import torch
import torch.nn.functional as F

from ..data.preprocessing import Batch, TextPreprocessor
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
        preprocessor: TextPreprocessor | None = None,
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
            chosen = param.device if param else "cpu"
        self.device = torch.device(chosen)

        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = preprocessor or TextPreprocessor(tokenizer=tokenizer)
        self.emotion_labels = list(emotion_labels) if emotion_labels else None
        self.topic_labels = list(topic_labels) if topic_labels else None

    # --------------- Summarization ---------------

    def summarize(self, texts: Sequence[str], *, max_length: int | None = None) -> List[str]:
        """Generate summaries for input texts."""
        if not texts:
            return []

        batch = self._to_device(self.preprocessor.batch_encode(texts))
        src_ids = batch.input_ids
        src_mask = batch.attention_mask
        max_len = max_length or self.config.summary_max_length

        model = cast(Any, self.model)
        if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
            raise RuntimeError("Model must have encoder and decoder for summarization")

        with torch.inference_mode():
            # Encode
            enc_mask = (
                src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
            )
            memory = model.encoder(src_ids, mask=enc_mask)

            # Decode with constraints to improve quality
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

        batch = self._to_device(self.preprocessor.batch_encode(texts))
        inputs = self._model_inputs(batch)
        thresh = threshold or self.config.emotion_threshold

        with torch.inference_mode():
            logits = self.model.forward("emotion", inputs)
            probs = torch.sigmoid(logits)

        results = []
        for row in probs.cpu():
            pairs = [
                (label, score)
                for label, score in zip(self.emotion_labels, row.tolist(), strict=False)
                if score >= thresh
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

        batch = self._to_device(self.preprocessor.batch_encode(texts))
        inputs = self._model_inputs(batch)

        with torch.inference_mode():
            logits = self.model.forward("topic", inputs)
            probs = F.softmax(logits, dim=-1)

        results = []
        for row in probs.cpu():
            idx = int(row.argmax().item())
            results.append(
                TopicPrediction(
                    label=self.topic_labels[idx],
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
        return {
            "summaries": self.summarize(text_list),
            "emotion": self.predict_emotions(text_list),
            "topic": self.predict_topics(text_list),
        }

    # --------------- Helpers ---------------

    def _to_device(self, batch: Batch) -> Batch:
        """Move batch tensors to device with non_blocking for speed."""
        updates = {}
        for f in fields(batch):
            val = getattr(batch, f.name)
            if torch.is_tensor(val):
                updates[f.name] = val.to(self.device, non_blocking=True)
        return replace(batch, **updates) if updates else batch

    @staticmethod
    def _model_inputs(batch: Batch) -> Dict[str, torch.Tensor]:
        """Extract model inputs from batch."""
        inputs = {"input_ids": batch.input_ids}
        if batch.attention_mask is not None:
            inputs["attention_mask"] = batch.attention_mask
        return inputs
