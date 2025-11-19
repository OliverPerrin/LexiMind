"""Inference helpers for multitask LexiMind models."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F

from ..data.preprocessing import Batch, TextPreprocessor
from ..data.tokenization import Tokenizer


@dataclass(slots=True)
class InferenceConfig:
    """Configuration knobs for the inference pipeline."""

    summary_max_length: int = 128
    emotion_threshold: float = 0.5
    device: str | None = None


@dataclass(slots=True)
class EmotionPrediction:
    labels: List[str]
    scores: List[float]


@dataclass(slots=True)
class TopicPrediction:
    label: str
    confidence: float


class InferencePipeline:
    """Run summarization, emotion, and topic heads through a unified interface."""

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
        chosen_device = device or self.config.device
        if chosen_device is None:
            first_param = next(model.parameters(), None)
            chosen_device = first_param.device if first_param is not None else "cpu"
        self.device = torch.device(chosen_device)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = preprocessor or TextPreprocessor(tokenizer=tokenizer)
        self.emotion_labels = list(emotion_labels) if emotion_labels is not None else None
        self.topic_labels = list(topic_labels) if topic_labels is not None else None

    def summarize(self, texts: Sequence[str], *, max_length: int | None = None) -> List[str]:
        if not texts:
            return []
        batch = self._batch_to_device(self.preprocessor.batch_encode(texts))
        src_ids = batch.input_ids
        src_mask = batch.attention_mask
        max_len = max_length or self.config.summary_max_length

        if not hasattr(self.model, "encoder") or not hasattr(self.model, "decoder"):
            raise RuntimeError("Model must expose encoder and decoder attributes for summarization.")

        with torch.inference_mode():
            encoder_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
            memory = self.model.encoder(src_ids, mask=encoder_mask)
            # Relax min_len to avoid forcing repetition if the model wants to stop
            min_len = 0
            generated = self.model.decoder.greedy_decode(
                memory=memory,
                max_len=max_len,
                start_token_id=self.tokenizer.eos_token_id,
                end_token_id=self.tokenizer.eos_token_id,
                device=self.device,
                min_len=min_len,
            )
            
            # If the first token is EOS, it means empty generation.
            # Try forcing a different start token if that happens, or just accept it.
            # For now, we just decode.
            
        return self.tokenizer.decode_batch(generated.tolist())

    def predict_emotions(
        self,
        texts: Sequence[str],
        *,
        threshold: float | None = None,
    ) -> List[EmotionPrediction]:
        if not texts:
            return []
        if self.emotion_labels is None or not self.emotion_labels:
            raise RuntimeError("emotion_labels must be provided to decode emotion predictions")

        batch = self._batch_to_device(self.preprocessor.batch_encode(texts))
        model_inputs = self._batch_to_model_inputs(batch)
        decision_threshold = threshold or self.config.emotion_threshold

        with torch.inference_mode():
            logits = self.model.forward("emotion", model_inputs)
            probs = torch.sigmoid(logits)

        predictions: List[EmotionPrediction] = []
        for row in probs.cpu():
            pairs = [
                (label, score)
                for label, score in zip(self.emotion_labels, row.tolist())
                if score >= decision_threshold
            ]
            labels = [label for label, _ in pairs]
            scores = [score for _, score in pairs]
            predictions.append(EmotionPrediction(labels=labels, scores=scores))
        return predictions

    def predict_topics(self, texts: Sequence[str]) -> List[TopicPrediction]:
        if not texts:
            return []
        if self.topic_labels is None or not self.topic_labels:
            raise RuntimeError("topic_labels must be provided to decode topic predictions")

        batch = self._batch_to_device(self.preprocessor.batch_encode(texts))
        model_inputs = self._batch_to_model_inputs(batch)

        with torch.inference_mode():
            logits = self.model.forward("topic", model_inputs)
            probs = F.softmax(logits, dim=-1)

        results: List[TopicPrediction] = []
        for row in probs.cpu():
            scores = row.tolist()
            best_index = int(row.argmax().item())
            results.append(TopicPrediction(label=self.topic_labels[best_index], confidence=scores[best_index]))
        return results

    def batch_predict(self, texts: Iterable[str]) -> dict[str, object]:
        text_list = list(texts)
        if self.emotion_labels is None or not self.emotion_labels:
            raise RuntimeError("emotion_labels must be provided for batch predictions")
        if self.topic_labels is None or not self.topic_labels:
            raise RuntimeError("topic_labels must be provided for batch predictions")
        return {
            "summaries": self.summarize(text_list),
            "emotion": self.predict_emotions(text_list),
            "topic": self.predict_topics(text_list),
        }

    def _batch_to_device(self, batch: Batch) -> Batch:
        tensor_updates: dict[str, torch.Tensor] = {}
        for item in fields(batch):
            value = getattr(batch, item.name)
            if torch.is_tensor(value):
                tensor_updates[item.name] = value.to(self.device)
        if not tensor_updates:
            return batch
        return replace(batch, **tensor_updates)

    @staticmethod
    def _batch_to_model_inputs(batch: Batch) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = {"input_ids": batch.input_ids}
        if batch.attention_mask is not None:
            inputs["attention_mask"] = batch.attention_mask
        return inputs
