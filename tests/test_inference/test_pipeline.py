"""Integration tests for the inference pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from src.data.tokenization import Tokenizer, TokenizerConfig
from src.inference.pipeline import EmotionPrediction, InferenceConfig, InferencePipeline, TopicPrediction
from src.utils.labels import LabelMetadata


def _local_tokenizer_config() -> TokenizerConfig:
    root = Path(__file__).resolve().parents[2]
    hf_path = root / "artifacts" / "hf_tokenizer"
    return TokenizerConfig(pretrained_model_name=str(hf_path))


class DummyEncoder(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, 8, device=input_ids.device)


class DummyDecoder(torch.nn.Module):
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        tokens = tokenizer.tokenizer.encode("dummy summary", add_special_tokens=False)
        sequence = [tokenizer.bos_token_id, *tokens, tokenizer.eos_token_id]
        self.register_buffer("sequence", torch.tensor(sequence, dtype=torch.long))

    def greedy_decode(
        self,
        *,
        memory: torch.Tensor,
        max_len: int,
        start_token_id: int,
        end_token_id: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        seq = self.sequence.to(device)
        if seq.numel() > max_len:
            seq = seq[:max_len]
        batch = memory.size(0)
        return seq.unsqueeze(0).repeat(batch, 1)


class DummyModel(torch.nn.Module):
    def __init__(self, tokenizer: Tokenizer, metadata: LabelMetadata) -> None:
        super().__init__()
        self.encoder = DummyEncoder()
        self.decoder = DummyDecoder(tokenizer)
        emotion_logits = torch.tensor([-2.0, 3.0, -1.0], dtype=torch.float32)
        topic_logits = torch.tensor([0.25, 2.5, 0.1], dtype=torch.float32)
        self.register_buffer("_emotion_logits", emotion_logits)
        self.register_buffer("_topic_logits", topic_logits)

    def forward(self, task: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover - simple dispatch
        batch = inputs["input_ids"].size(0)
        if task == "emotion":
            return self._emotion_logits.unsqueeze(0).repeat(batch, 1)
        if task == "topic":
            return self._topic_logits.unsqueeze(0).repeat(batch, 1)
        raise KeyError(task)


def _build_pipeline() -> InferencePipeline:
    tokenizer = Tokenizer(_local_tokenizer_config())
    metadata = LabelMetadata(emotion=["anger", "joy", "sadness"], topic=["news", "sports", "tech"])
    model = DummyModel(tokenizer, metadata)
    return InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        emotion_labels=metadata.emotion,
        topic_labels=metadata.topic,
        config=InferenceConfig(summary_max_length=12),
    )


def test_pipeline_predictions_across_tasks() -> None:
    pipeline = _build_pipeline()
    text = "A quick unit test input."

    summaries = pipeline.summarize([text])
    assert summaries == ["dummy summary"], "Summaries should be decoded from dummy decoder sequence"

    emotions = pipeline.predict_emotions([text])
    assert len(emotions) == 1
    emotion = emotions[0]
    assert isinstance(emotion, EmotionPrediction)
    assert emotion.labels == ["joy"], "Only the positive logit should pass the threshold"

    topics = pipeline.predict_topics([text])
    assert len(topics) == 1
    topic = topics[0]
    assert isinstance(topic, TopicPrediction)
    assert topic.label == "sports"
    assert topic.confidence > 0.0

    combined = pipeline.batch_predict([text])
    assert combined["summaries"] == summaries
    combined_emotions = cast(list[EmotionPrediction], combined["emotion"])
    combined_topics = cast(list[TopicPrediction], combined["topic"])
    assert combined_emotions[0].labels == emotion.labels
    assert combined_topics[0].label == topic.label