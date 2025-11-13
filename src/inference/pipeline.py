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
            generated = self._constrained_greedy_decode(memory, max_len, memory_mask=src_mask)

        trimmed_sequences: List[List[int]] = []
        for row in generated.cpu().tolist():
            trimmed_sequences.append(self._trim_special_tokens(row))

        return self.tokenizer.decode_batch(trimmed_sequences)

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
        decision_threshold = self.config.emotion_threshold if threshold is None else float(threshold)

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

    def _constrained_greedy_decode(
        self,
        memory: torch.Tensor,
        max_len: int,
        *,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run greedy decoding while banning BOS/PAD tokens from the generated sequence."""

        device = memory.device
        batch_size = memory.size(0)
        bos = self.tokenizer.bos_token_id
        pad = getattr(self.tokenizer, "pad_token_id", None)
        eos = getattr(self.tokenizer, "eos_token_id", None)

        generated = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)
        expanded_memory_mask = None
        if memory_mask is not None:
            expanded_memory_mask = memory_mask.to(device=device, dtype=torch.bool)

        for _ in range(max(1, max_len) - 1):
            decoder_out = self.model.decoder(generated, memory, memory_mask=expanded_memory_mask)
            logits = decoder_out if isinstance(decoder_out, torch.Tensor) else decoder_out[0]

            step_logits = logits[:, -1, :].clone()
            if bos is not None and bos < step_logits.size(-1):
                step_logits[:, bos] = float("-inf")
            if pad is not None and pad < step_logits.size(-1):
                step_logits[:, pad] = float("-inf")

            next_token = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos is not None and torch.all(next_token.squeeze(-1) == eos):
                break

        return generated

    def _trim_special_tokens(self, sequence: Sequence[int]) -> List[int]:
        """Remove leading BOS and trailing PAD/EOS tokens from a generated sequence."""

        bos = self.tokenizer.bos_token_id
        pad = getattr(self.tokenizer, "pad_token_id", None)
        eos = getattr(self.tokenizer, "eos_token_id", None)

        trimmed: List[int] = []
        for idx, token in enumerate(sequence):
            if idx == 0 and bos is not None and token == bos:
                continue
            if pad is not None and token == pad:
                continue
            if eos is not None and token == eos:
                break
            trimmed.append(int(token))
        return trimmed

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
