"""Book field output contracts using fixed synthetic logits only."""

import json
from types import SimpleNamespace

import pytest
import torch

from src.inference.pipeline import InferencePipeline
from src.utils.labels import (
    BOOK_INPUT_FORMAT,
    LabelMetadata,
    format_book_input,
    load_label_metadata,
    save_label_metadata,
)

LABELS = ["genre:fantasy", "genre:romance", "topic:science"]


class Tokenizer:
    def batch_encode(self, texts, **kwargs):
        self.texts = texts
        return {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }


class Model(torch.nn.Module):
    def __init__(self, mode="multi_label"):
        super().__init__()
        self.heads = {"topic": SimpleNamespace(problem_type=mode)}
        self.register_buffer("logits", torch.tensor([2.0, 1.0, -2.0]))

    def forward(self, task, inputs):
        assert task == "topic" and set(inputs) == {"input_ids", "attention_mask"}
        return self.logits.repeat(len(inputs["input_ids"]), 1)


def pipeline():
    return InferencePipeline(
        Model(),
        Tokenizer(),
        topic_labels=LABELS,
        topic_problem_type="multi_label",
        topic_input_format=BOOK_INPUT_FORMAT,
    )


def test_book_fields_use_independent_scores_and_never_force_an_argmax():
    instance = pipeline()
    book = {"title": "Synthetic", "description": "Narrative only"}
    result = instance.predict_book_fields([book], thresholds=0.7)[0]
    assert result.fields == {
        "genre": ["fantasy", "romance"],
        "topic": [],
        "form": [],
        "audience": [],
    }
    assert result.scores["genre:fantasy"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert instance.tokenizer.texts == [format_book_input(**book)]
    result = instance.predict_book_fields([book], thresholds=0.99)[0]
    assert all(not labels for labels in result.fields.values())
    with pytest.raises(ValueError, match="predict_book_fields"):
        instance.predict_topics(["text"])
    with pytest.raises(ValueError, match="predict_book_fields"):
        instance.batch_predict(["text"])


@pytest.mark.parametrize(
    "threshold", [True, float("nan"), float("inf"), -0.1, 1.1, {}, {LABELS[0]: 0.5}]
)
def test_threshold_policy_is_explicit_complete_and_finite(threshold):
    with pytest.raises(ValueError, match="threshold"):
        pipeline().predict_book_fields([], thresholds=threshold)


def test_metadata_cannot_be_passed_as_classifier_input():
    with pytest.raises(ValueError, match="only title and description"):
        pipeline().predict_book_fields(
            [{"title": "T", "description": "D", "source_labels": "fantasy"}], thresholds=0.5
        )


def test_head_and_saved_loss_modes_must_match():
    with pytest.raises(ValueError, match="head mode"):
        InferencePipeline(Model(), Tokenizer(), topic_labels=LABELS)


def test_label_metadata_preserves_legacy_and_round_trips_book_semantics(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"emotions": [], "topics": ["news"]}))
    legacy = load_label_metadata(path)
    assert legacy.topic_problem_type == "single_label"
    save_label_metadata(legacy, path)
    assert json.loads(path.read_text()) == {"emotion": [], "topic": ["news"]}
    meta = LabelMetadata([], LABELS, "multi_label", BOOK_INPUT_FORMAT, "a" * 64)
    save_label_metadata(meta, path)
    assert load_label_metadata(path) == meta
    meta.topic_mapping_sha256 = None
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_label_metadata(meta, path)
    assert path.read_bytes() == before


def test_resume_rejects_same_columns_with_different_loss_or_mapping(tmp_path):
    from omegaconf import OmegaConf

    from scripts.train import validate_resume_labels

    path = tmp_path / "labels.json"
    save_label_metadata(LabelMetadata([], LABELS, "multi_label", BOOK_INPUT_FORMAT, "a" * 64), path)
    cfg = OmegaConf.create({"resume_from": "not-executed.pt", "resume_labels": str(path)})
    with pytest.raises(ValueError, match="loss mode"):
        validate_resume_labels(cfg, emotion=[], topic=LABELS)
    with pytest.raises(ValueError, match="mapping"):
        validate_resume_labels(
            cfg,
            emotion=[],
            topic=LABELS,
            topic_problem_type="multi_label",
            topic_input_format=BOOK_INPUT_FORMAT,
            topic_mapping_sha256="b" * 64,
        )
    validate_resume_labels(
        cfg,
        emotion=[],
        topic=LABELS,
        topic_problem_type="multi_label",
        topic_input_format=BOOK_INPUT_FORMAT,
        topic_mapping_sha256="a" * 64,
    )


def test_factory_checks_paired_label_contract_before_tokenizer_or_weights(tmp_path, monkeypatch):
    from src.inference import factory

    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"not real weights")
    metadata = LabelMetadata([], LABELS, "multi_label", BOOK_INPUT_FORMAT, "a" * 64)
    save_label_metadata(metadata, tmp_path / "labels.json")
    alternative = tmp_path / "supplied.json"
    save_label_metadata(
        LabelMetadata([], LABELS, "multi_label", BOOK_INPUT_FORMAT, "b" * 64), alternative
    )
    monkeypatch.setattr(factory, "Tokenizer", lambda *args: pytest.fail("Tokenizer created"))
    with pytest.raises(ValueError, match="checkpoint directory"):
        factory.create_inference_pipeline(checkpoint, alternative)
