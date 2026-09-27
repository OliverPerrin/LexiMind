"""Synthetic partial book supervision; no source corpus or model training."""

import json

import pytest
import torch

from src.data.dataloader import PartialTopicCollator, build_task_dataloaders
from src.data.dataset import (
    IndexedJsonl,
    PartialTopicDataset,
    PartialTopicExample,
    load_training_datasets,
)
from src.utils.labels import BOOK_INPUT_FORMAT, format_book_input

LABELS = ["topic:science", "genre:fantasy", "audience:children"]


class Tokenizer:
    def batch_encode(self, texts, **kwargs):
        self.texts = texts
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }


def splits(tmp_path, *, sidecar=None):
    schema = (
        sidecar
        if sidecar is not None
        else {
            "schema_version": 1,
            "problem_type": "multi_label",
            "input_format": BOOK_INPUT_FORMAT,
            "mapping_sha256": "a" * 64,
            "labels": LABELS,
        }
    )
    (tmp_path / "labels.json").write_text(json.dumps(schema))
    row = {
        "title": "Synthetic book",
        "description": "Only narrative input.",
        "positive": ["genre:fantasy"],
        "negative": ["topic:science"],
        "source_labels": ["DO NOT TOKENIZE"],
        "provider_url": "https://example.test/private-metadata",
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "val.jsonl").write_text(json.dumps({**row, "positive": [], "negative": []}) + "\n")
    (tmp_path / "test.jsonl").write_text("must never be read")
    return row


def test_partial_loader_keeps_lazy_text_ordered_columns_and_unknown_states(tmp_path):
    row = splits(tmp_path)
    train, val = load_training_datasets(
        {"topic": str(tmp_path)}, ["topic"], topic_problem_type="multi_label"
    )
    dataset = train["topic"]
    assert isinstance(dataset, PartialTopicDataset)
    assert isinstance(dataset._examples, IndexedJsonl)
    assert dataset.topic_classes == LABELS
    assert dataset.topic_mapping_sha256 == "a" * 64
    tokenizer = Tokenizer()
    loader = build_task_dataloaders(
        train, tokenizer, batch_size=1, shuffle=False, max_length=32, classification_max_length=32
    )["topic"]
    batch = next(iter(loader))
    assert batch["labels"].dtype == torch.float32 and batch["label_mask"].dtype == torch.bool
    assert batch["labels"].tolist() == [[0.0, 1.0, 0.0]]
    assert batch["label_mask"].tolist() == [[True, True, False]]
    assert tokenizer.texts == [format_book_input(row["title"], row["description"])]
    assert "DO NOT TOKENIZE" not in tokenizer.texts[0]
    unknown = PartialTopicCollator(tokenizer, val["topic"])([val["topic"][0]])
    assert not unknown["label_mask"].any()


@pytest.mark.parametrize(
    "sidecar",
    [
        LABELS,
        {},
        {"schema_version": True},
        {
            "schema_version": 1,
            "problem_type": "multi_label",
            "input_format": BOOK_INPUT_FORMAT,
            "labels": LABELS,
            "mapping_sha256": None,
        },
    ],
)
def test_partial_mode_never_discovers_columns_from_incomplete_targets(tmp_path, sidecar):
    splits(tmp_path, sidecar=sidecar)
    with pytest.raises(ValueError):
        load_training_datasets(
            {"topic": str(tmp_path)}, ["topic"], topic_problem_type="multi_label"
        )


@pytest.mark.parametrize(
    "positive,negative",
    [(["genre:missing"], []), (["genre:fantasy"], ["genre:fantasy"]), (["genre:fantasy"] * 2, [])],
)
def test_collator_refuses_unknown_repeated_or_conflicting_targets(positive, negative):
    example = PartialTopicExample("Synthetic", "Body", positive, negative)
    dataset = PartialTopicDataset([example], labels=LABELS, mapping_sha256="a" * 64)
    with pytest.raises(ValueError):
        PartialTopicCollator(Tokenizer(), dataset)([example])


def test_partial_shape_requires_explicit_mode_and_does_not_replace_legacy_topics(tmp_path):
    splits(tmp_path)
    with pytest.raises((ValueError, KeyError)):
        train, _ = load_training_datasets({"topic": str(tmp_path)}, ["topic"])
        _ = train["topic"][0]
