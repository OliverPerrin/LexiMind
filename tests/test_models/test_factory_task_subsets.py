"""Tiny synthetic factory compatibility checks; pretrained loading is disabled."""

from types import SimpleNamespace

import pytest
import torch

from src.inference.pipeline import InferencePipeline
from src.models.factory import ModelConfig, build_multitask_model


def tiny_model(emotions=3, topics=2):
    tokenizer = SimpleNamespace(vocab_size=19, pad_token_id=0, config=SimpleNamespace(max_length=8))
    model = build_multitask_model(
        tokenizer,
        num_emotions=emotions,
        num_topics=topics,
        config=ModelConfig(
            d_model=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_attention_heads=2,
            ffn_dim=16,
            dropout=0.0,
        ),
        load_pretrained=False,
    )
    return model, tokenizer


def test_positive_count_head_names_and_shapes_keep_existing_checkpoint_interface():
    model, _ = tiny_model()
    state = model.state_dict()
    expected_head_shapes = {
        "head_summarization.proj.weight": (19, 8),
        "head_summarization.proj.bias": (19,),
        "head_emotion.attn_pool.query.weight": (1, 8),
        "head_emotion.out_proj.0.weight": (4, 8),
        "head_emotion.out_proj.0.bias": (4,),
        "head_emotion.out_proj.3.weight": (3, 4),
        "head_emotion.out_proj.3.bias": (3,),
        "head_topic.out_proj.weight": (2, 8),
        "head_topic.out_proj.bias": (2,),
    }
    assert {
        name: tuple(value.shape) for name, value in state.items() if name.startswith("head_")
    } == expected_head_shapes
    assert set(model.heads) == {"summarization", "emotion", "topic"}


@pytest.mark.parametrize("emotions,topics", [(0, 0), (3, 0), (0, 2)])
def test_zero_counts_remove_only_inactive_heads_and_common_state_loads_strictly(emotions, topics):
    full, _ = tiny_model()
    subset, _ = tiny_model(emotions, topics)
    removed = tuple(
        f"head_{task}." for task, count in (("emotion", emotions), ("topic", topics)) if count == 0
    )
    common_state = {
        name: value for name, value in full.state_dict().items() if not name.startswith(removed)
    }
    assert set(subset.state_dict()) == set(common_state)
    subset.load_state_dict(common_state, strict=True)
    for task, count in (("emotion", emotions), ("topic", topics)):
        if not count:
            with pytest.raises(KeyError, match=f"Unknown task/head '{task}'"):
                subset.forward(task, {})
        else:
            # Same existing active head and encoder produce identical tiny-fixture logits.
            inputs = {
                "input_ids": torch.tensor([[2, 3, 0]]),
                "attention_mask": torch.tensor([[True, True, False]]),
            }
            full.eval()
            subset.eval()
            with torch.no_grad():
                assert torch.equal(full.forward(task, inputs), subset.forward(task, inputs))


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_head_counts_fail_before_model_construction(count):
    with pytest.raises(ValueError, match="num_emotions must be a nonnegative integer"):
        tiny_model(emotions=count)


def test_subset_inference_rejects_missing_labels_before_tokenization():
    model, tokenizer = tiny_model(0, 2)
    pipeline = InferencePipeline(
        model, tokenizer, emotion_labels=[], topic_labels=["a", "b"], device="cpu"
    )
    with pytest.raises(RuntimeError, match="emotion_labels required"):
        pipeline.predict_emotions(["synthetic text"])
    with pytest.raises(RuntimeError, match="Both emotion_labels and topic_labels required"):
        pipeline.batch_predict(["synthetic text"])


def test_subset_checkpoint_and_empty_labels_reconstruct_through_inference_factory(
    tmp_path, monkeypatch
):
    import json
    from dataclasses import asdict

    from src.data.tokenization import TokenizerConfig
    from src.inference import factory
    from src.utils.io import save_state
    from src.utils.labels import LabelMetadata, load_label_metadata, save_label_metadata

    original, tokenizer = tiny_model(0, 2)
    checkpoint = tmp_path / "synthetic.pt"
    labels_path = tmp_path / "labels.json"
    model_path = tmp_path / "model.json"
    save_state(original, checkpoint)
    save_label_metadata(LabelMetadata(emotion=[], topic=["a", "b"]), labels_path)
    assert load_label_metadata(labels_path).emotion_size == 0
    config = ModelConfig(
        d_model=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_attention_heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    model_path.write_text(json.dumps(asdict(config)))
    monkeypatch.setattr(factory, "Tokenizer", lambda config: tokenizer)
    pipeline, labels = factory.create_inference_pipeline(
        checkpoint,
        labels_path,
        model_config_path=model_path,
        tokenizer_config=TokenizerConfig(pretrained_model_name="synthetic-no-download"),
        device="cpu",
    )
    assert labels.emotion == []
    assert set(pipeline.model.heads) == {"summarization", "topic"}
    expected, actual = original.state_dict(), pipeline.model.state_dict()
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    with pytest.raises(RuntimeError, match="emotion_labels required"):
        pipeline.predict_emotions(["synthetic text"])
