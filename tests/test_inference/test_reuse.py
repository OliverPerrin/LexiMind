"""Inference API parity on tiny deterministic fixtures; no model downloads."""

from types import SimpleNamespace

import pytest
import torch

from src.inference.pipeline import InferenceConfig, InferencePipeline
from src.models.heads import ClassificationHead
from src.models.multitask import MultiTaskModel


class TinyTokenizer:
    bos_token_id = pad_token_id = 0
    eos_token_id = 1
    _tokenizer = SimpleNamespace(unk_token_id=2)

    def __init__(self):
        self.calls = 0

    def batch_encode(self, texts):
        self.calls += 1
        ids = torch.tensor([[3, 4, 0] if i % 2 == 0 else [5, 6, 7] for i in range(len(texts))])
        return {"input_ids": ids, "attention_mask": ids != 0}

    def decode_batch(self, sequences):
        return ["synthetic summary" for _ in sequences]


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.calls = 0

    def forward(self, ids, mask=None):
        self.calls += 1
        return self.embedding(ids)


class TinyDecoder(torch.nn.Module):
    def greedy_decode(self, *, memory, **kwargs):
        return torch.ones(memory.shape[0], 2, dtype=torch.long, device=memory.device)


def pipeline():
    torch.manual_seed(9)
    tokenizer = TinyTokenizer()
    model = MultiTaskModel(encoder=TinyEncoder(), decoder=TinyDecoder())
    model.add_head("emotion", ClassificationHead(4, 3, pooler="attention", dropout=0.0))
    model.add_head("topic", ClassificationHead(4, 2, pooler="mean", dropout=0.0))
    return InferencePipeline(
        model,
        tokenizer,
        emotion_labels=["z", "a", "m"],
        topic_labels=["second", "first"],
        config=InferenceConfig(summary_formatting=False),
    )


def test_shared_encoding_matches_separate_tasks_with_padding_and_keeps_checkpoint_keys():
    p = pipeline()
    texts = ["first", "second"]
    state_keys = set(p.model.state_dict())
    expected = {
        "summaries": p.summarize(texts),
        "emotion": p.predict_emotions(texts),
        "topic": p.predict_topics(texts),
    }
    assert p.model.encoder.calls == 3
    assert p.tokenizer.calls == 3
    p.model.encoder.calls = p.tokenizer.calls = 0
    actual = p.batch_predict(texts)
    assert actual == expected
    assert p.model.encoder.calls == 1
    assert p.tokenizer.calls == 1
    assert set(p.model.state_dict()) == state_keys
    assert all(parameter.grad is None for parameter in p.model.parameters())


def test_parameter_device_detection_does_not_boolean_test_a_tensor():
    p = pipeline()
    assert next(p.model.parameters()).numel() > 1
    assert p.device == torch.device("cpu")


def test_zero_threshold_returns_all_labels_and_keeps_order():
    p = pipeline()
    assert p.predict_emotions(["example"], threshold=0)[0].labels == ["z", "a", "m"]
    for threshold in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="threshold"):
            p.predict_emotions(["example"], threshold=threshold)


def test_label_shape_mismatch_is_not_silently_truncated():
    p = pipeline()
    p.emotion_labels = ["only one"]
    with pytest.raises(ValueError, match="label order"):
        p.predict_emotions(["example"])


def test_empty_batch_never_tokenizes_or_encodes():
    p = pipeline()
    assert p.batch_predict([]) == {"summaries": [], "emotion": [], "topic": []}
    assert p.tokenizer.calls == p.model.encoder.calls == 0


@pytest.mark.parametrize("relative_bias", [False, True])
def test_reuse_parity_with_tiny_untrained_encoder_decoder(relative_bias):
    from src.models.decoder import TransformerDecoder
    from src.models.encoder import TransformerEncoder

    torch.manual_seed(23)
    tokenizer = TinyTokenizer()
    tokenizer.decode_batch = lambda sequences: [" ".join(map(str, row)) for row in sequences]
    encoder = TransformerEncoder(
        vocab_size=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
        max_len=16,
        pad_token_id=0,
        use_relative_position_bias=relative_bias,
    )
    decoder = TransformerDecoder(
        vocab_size=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
        max_len=16,
        pad_token_id=0,
        use_relative_position_bias=relative_bias,
    )
    model = MultiTaskModel(encoder=encoder, decoder=decoder)
    model.add_head("emotion", ClassificationHead(8, 3, pooler="attention", dropout=0.0))
    model.add_head("topic", ClassificationHead(8, 2, pooler="mean", dropout=0.0))
    p = InferencePipeline(
        model,
        tokenizer,
        emotion_labels=["z", "a", "m"],
        topic_labels=["second", "first"],
        config=InferenceConfig(summary_max_length=12, summary_formatting=False),
    )
    calls = []
    hook = encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        expected = {
            "summaries": p.summarize(["a", "b"]),
            "emotion": p.predict_emotions(["a", "b"]),
            "topic": p.predict_topics(["a", "b"]),
        }
        assert len(calls) == 3
        calls.clear()
        actual = p.batch_predict(["a", "b"])
        assert actual == expected
        assert len(calls) == 1
    finally:
        hook.remove()
