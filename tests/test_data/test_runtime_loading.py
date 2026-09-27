"""Synthetic storage/collation contracts; no corpus, tokenizer, or model downloads."""

import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader

from src.data.dataloader import build_task_dataloaders
from src.data.dataset import (
    EmotionDataset,
    IndexedJsonl,
    SummarizationDataset,
    TopicDataset,
    load_emotion_jsonl,
    load_splits,
    load_summarization_jsonl,
    load_topic_jsonl,
    load_training_datasets,
    split_emotion_val,
    validate_task_directories,
)


def write_rows(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def identity_batch(examples):
    return examples


class SyntheticTokenizer:
    """Small explicit token map for checking collator masks, not model quality."""

    def batch_encode(self, texts, *, max_length=None, padding=None, pad_to_multiple_of=None):
        assert padding == "longest"
        assert pad_to_multiple_of == 8
        tokens = [([len(word) + 2 for word in text.split()] + [1])[:max_length] for text in texts]
        width = ((max(map(len, tokens)) + 7) // 8) * 8
        ids = torch.zeros((len(tokens), width), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(tokens):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = True
        return {"input_ids": ids, "attention_mask": mask}

    def prepare_decoder_inputs(self, ids):
        shifted = torch.zeros_like(ids)
        shifted[:, 1:] = ids[:, :-1]
        return shifted


@pytest.mark.parametrize(
    "loader,rows",
    [
        (
            load_summarization_jsonl,
            [
                {"source": "é short", "summary": "story", "domain": "book"},
                {"source": "long text", "summary": "other", "type": "paper"},
            ],
        ),
        (
            load_emotion_jsonl,
            [{"text": "é short", "emotions": ["joy", "sad"]}, {"text": "long text"}],
        ),
        (
            load_topic_jsonl,
            [{"text": "é short", "topic": "z"}, {"text": "long text", "topic": "a"}],
        ),
    ],
)
def test_indexed_order_blank_lines_slices_and_pickle_match_eager(tmp_path, loader, rows):
    path = write_rows(tmp_path / "train.jsonl", rows)
    path.write_bytes(b"\n" + path.read_bytes().replace(b"\n", b"\r\n\n"))
    eager = loader(path)
    indexed = loader(path, lazy=True)
    assert isinstance(eager, list)
    assert isinstance(indexed, IndexedJsonl)
    assert list(indexed) == eager
    assert indexed[-1] == eager[-1]
    assert indexed[::-1] == eager[::-1]
    assert indexed.read_many([1, 0, 1]) == [eager[1], eager[0], eager[1]]
    assert list(pickle.loads(pickle.dumps(indexed))) == eager
    with pytest.raises(IndexError):
        indexed[2]
    with pytest.raises(TypeError):
        indexed[1.5]


@pytest.mark.parametrize("lazy", [False, True])
def test_jsonl_limit_stops_before_invalid_tail(tmp_path, lazy):
    path = write_rows(tmp_path / "train.jsonl", [{"text": "first", "topic": "a"}])
    with path.open("a") as handle:
        handle.write("invalid JSON tail\n")
    assert load_topic_jsonl(path, limit=1, lazy=lazy)[0].text == "first"
    assert load_topic_jsonl(path, limit=0, lazy=lazy) == []
    with pytest.raises(ValueError, match="line 2"):
        list(load_topic_jsonl(path, lazy=lazy))


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_limits_fail_before_loading(tmp_path, limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_topic_jsonl(tmp_path / "missing.jsonl", limit=limit)


def test_json_array_compatibility_and_domain_metadata(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            [{"source": "a", "summary": "b", "domain": "book"}, {"source": "c", "summary": "d"}]
        )
    )
    result = load_summarization_jsonl(path, lazy=True, limit=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].domain == "book"


def test_index_contains_no_corpus_text_and_decode_occurs_only_on_access(tmp_path):
    path = write_rows(
        tmp_path / "train.jsonl", [{"source": "x" * 5000, "summary": "short"} for _ in range(20)]
    )
    indexed = load_summarization_jsonl(path, lazy=True)
    # Only path/config and offsets are sent to spawned workers, not the 100KB text.
    assert len(pickle.dumps(indexed)) < 2000
    constructor = Mock(wraps=indexed.constructor)
    indexed.constructor = constructor
    dataset = SummarizationDataset(indexed)
    assert dataset._examples is indexed
    constructor.assert_not_called()
    assert len(dataset.__getitems__([3, 1, 3])) == 3
    assert constructor.call_count == 3


def test_index_refuses_stale_offsets(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", [{"text": "first", "topic": "a"}])
    indexed = load_topic_jsonl(path, lazy=True)
    path.write_text('{"text":"replacement with different size","topic":"a"}\n')
    with pytest.raises(RuntimeError, match="changed after indexing"):
        indexed[0]


def test_provided_label_tools_do_not_decode_validation_rows(tmp_path):
    path = write_rows(
        tmp_path / "validation.jsonl", [{"text": "first", "topic": "a", "emotions": ["joy"]}]
    )
    emotion = load_emotion_jsonl(path, lazy=True)
    topic = load_topic_jsonl(path, lazy=True)
    emotion.constructor = Mock(side_effect=AssertionError("unnecessary emotion read"))
    topic.constructor = Mock(side_effect=AssertionError("unnecessary topic read"))
    assert EmotionDataset(
        emotion, binarizer=MultiLabelBinarizer().fit([["joy"]])
    ).emotion_classes == ["joy"]
    assert TopicDataset(topic, encoder=LabelEncoder().fit(["a"])).topic_classes == ["a"]


def test_spawned_worker_reads_index_without_shared_seek_state(tmp_path):
    path = write_rows(
        tmp_path / "train.jsonl", [{"text": f"row {i}", "topic": "a"} for i in range(5)]
    )
    dataset = TopicDataset(load_topic_jsonl(path, lazy=True))
    loader = DataLoader(
        dataset,
        batch_sampler=[[4, 0, 2], [1, 3]],
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=identity_batch,
    )
    assert [[ex.text for ex in batch] for batch in loader] == [
        ["row 4", "row 0", "row 2"],
        ["row 1", "row 3"],
    ]
    assert dataset[0].text == "row 0"


def test_active_tasks_limits_label_order_and_full_calibration_partition(tmp_path):
    emotion_dir = tmp_path / "emotion"
    emotion_dir.mkdir()
    write_rows(
        emotion_dir / "train.jsonl",
        [{"text": "one", "emotions": ["sad", "joy"]}, {"text": "two", "emotions": ["anger"]}],
    )
    validation = [{"text": f"row {i}", "emotions": ["joy"]} for i in range(12)]
    val_path = write_rows(emotion_dir / "validation.jsonl", validation)
    (emotion_dir / "test.jsonl").write_text("never parse held-out test data")
    expected_selection, _ = split_emotion_val(load_emotion_jsonl(val_path))
    train, val = load_training_datasets(
        {"emotion": str(emotion_dir), "topic": None, "summarization": None},
        ["emotion"],
        max_train_samples=1,
        max_val_samples=2,
    )
    assert set(train) == set(val) == {"emotion"}
    assert len(train["emotion"]) == 1
    assert train["emotion"].emotion_classes == ["anger", "joy", "sad"]
    assert list(val["emotion"]) == expected_selection[:2]
    assert val["emotion"].binarizer is train["emotion"].binarizer
    with pytest.raises(ValueError, match="explicit reviewed directory.*topic"):
        load_training_datasets({"emotion": str(emotion_dir), "topic": None}, ["emotion", "topic"])


def test_profile_split_selection_does_not_parse_validation_or_test(tmp_path):
    write_rows(
        tmp_path / "train.jsonl",
        [
            {"text": "one", "topic": "z"},
            {"text": "two", "topic": "a"},
            {"text": "three", "topic": "z"},
        ],
    )
    (tmp_path / "validation.jsonl").write_text("not for profiling")
    (tmp_path / "test.jsonl").write_text("never read")
    train, val = load_training_datasets(
        {"topic": tmp_path}, ["topic"], max_train_samples=2, include_validation=False
    )
    assert train["topic"].topic_classes == ["a", "z"]
    assert len(train["topic"]) == 2
    assert val == {}


def test_split_loader_preserves_legacy_one_argument_callback(tmp_path):
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text("fixture")
    calls = []
    splits = load_splits(tmp_path, lambda path: calls.append(Path(path).stem) or [])
    assert set(splits) == {"train", "val"}
    assert calls == ["train", "validation"]


@pytest.mark.parametrize("tasks", [[], ["unknown"], ["topic", "topic"]])
def test_invalid_task_selection_is_actionable(tasks):
    with pytest.raises(ValueError, match="nonempty, unique subset"):
        validate_task_directories({}, tasks)


def test_dynamic_padding_and_decoder_labels_match_eager_batches(tmp_path):
    rows = [
        {"source": "one two", "summary": "story"},
        {"source": "a", "summary": "end"},
        {"source": " ".join(["word"] * 20), "summary": "last"},
    ]
    path = write_rows(tmp_path / "train.jsonl", rows)
    outputs = []
    for lazy in (False, True):
        dataset = SummarizationDataset(load_summarization_jsonl(path, lazy=lazy))
        loader = build_task_dataloaders(
            {"summarization": dataset},
            SyntheticTokenizer(),
            batch_size=2,
            shuffle=False,
            max_length=64,
            classification_max_length=32,
        )["summarization"]
        outputs.append(list(loader))
    for eager, indexed in zip(*outputs, strict=True):
        assert eager.keys() == indexed.keys()
        for key in eager:
            assert torch.equal(eager[key], indexed[key])
    first, last = outputs[1]
    assert first["src_ids"].shape == (2, 8)
    assert last["src_ids"].shape == (1, 24)
    assert first["src_ids"][0].tolist() == [5, 5, 1, 0, 0, 0, 0, 0]
    assert first["src_mask"][0].tolist() == [True, True, True, False, False, False, False, False]
    assert first["labels"][0].tolist() == [7, 1, -100, -100, -100, -100, -100, -100]
    assert first["tgt_ids"][0].tolist() == [0, 7, 1, 0, 0, 0, 0, 0]


def test_entrypoints_reject_unset_data_before_tokenizers_or_gpu(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from scripts import profile_training, train

    cfg = OmegaConf.create(
        {
            "seed": 17,
            "device": "cuda",
            "data": {"processed": {"topic": None}},
            "training": {"trainer": {"tasks": ["topic"]}},
        }
    )
    for entrypoint in (train, profile_training):
        monkeypatch.setattr(
            entrypoint, "Tokenizer", Mock(side_effect=AssertionError("tokenizer constructed"))
        )
        monkeypatch.setattr(
            entrypoint.torch.cuda,
            "get_device_capability",
            Mock(side_effect=AssertionError("GPU accessed")),
        )
        monkeypatch.setattr(
            entrypoint.torch.cuda,
            "get_device_name",
            Mock(side_effect=AssertionError("GPU accessed")),
        )
        with pytest.raises(ValueError, match="explicit reviewed directory.*topic"):
            entrypoint.main.__wrapped__(cfg)


def test_evaluate_checks_selected_directories_before_pipeline(tmp_path, monkeypatch):
    import sys

    from scripts import evaluate

    monkeypatch.setattr(
        evaluate, "create_inference_pipeline", Mock(side_effect=AssertionError("model loaded"))
    )
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--data-dir", str(tmp_path), "--topic-only"])
    with pytest.raises(ValueError, match="directory for 'topic'.*validation.jsonl"):
        evaluate.main()


def test_index_iteration_does_not_read_excluded_large_tail(tmp_path, monkeypatch):
    import io

    first = b'{"text":"first","topic":"a"}\n'
    tail = b'{"text":"' + b"x" * 1000000 + b'","topic":"b"}\n'
    path = tmp_path / "train.jsonl"
    path.write_bytes(first + tail)
    indexed = load_topic_jsonl(path, limit=1, lazy=True)

    class CountedStream(io.BytesIO):
        def __init__(self, raw):
            super().__init__(raw)
            self.line_sizes = []

        def readline(self, *args):
            raw = super().readline(*args)
            self.line_sizes.append(len(raw))
            return raw

        def __iter__(self):
            return self

        def __next__(self):
            raw = self.readline()
            if not raw:
                raise StopIteration
            return raw

    stream = CountedStream(first + tail)
    original_open = Path.open
    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *args, **kwargs: (
            stream if self == indexed.path else original_open(self, *args, **kwargs)
        ),
    )
    assert [row.text for row in indexed] == ["first"]
    assert stream.line_sizes == [len(first)]


@pytest.mark.parametrize("task", ["emotion", "topic"])
@pytest.mark.parametrize("explicit_order", [False, True])
def test_caps_preserve_complete_training_vocabulary_and_explicit_column_order(
    tmp_path, task, explicit_order
):
    train_rows = [
        {"text": "first", "emotions": ["joy"], "topic": "A"},
        {"text": "second", "emotions": ["sadness"], "topic": "B"},
    ]
    write_rows(tmp_path / "train.jsonl", train_rows)
    write_rows(
        tmp_path / "validation.jsonl",
        [{"text": "validation", "emotions": ["sadness"], "topic": "B"} for _ in range(4)],
    )
    expected = ["joy", "sadness"] if task == "emotion" else ["A", "B"]
    if explicit_order:
        expected.reverse()
        (tmp_path / "labels.json").write_text(json.dumps(expected))
    train, val = load_training_datasets({task: tmp_path}, [task], max_train_samples=1)
    assert len(train[task]) == 1
    assert getattr(train[task], f"{task}_classes") == expected
    loader = build_task_dataloaders(
        val,
        SyntheticTokenizer(),
        batch_size=2,
        shuffle=False,
        max_length=32,
        classification_max_length=32,
    )[task]
    batch = next(iter(loader))
    if task == "emotion":
        wanted = [float(label == "sadness") for label in expected]
        assert batch["labels"].tolist() == [wanted, wanted]
    else:
        assert batch["labels"].tolist() == [expected.index("B"), expected.index("B")]
    # Vocabulary discovery retains only the capped examples in the training dataset.
    assert isinstance(train[task]._examples, IndexedJsonl)
    assert len(train[task]._examples) == 1


@pytest.mark.parametrize("task", ["emotion", "topic"])
@pytest.mark.parametrize("labels", [[], ["a", "a"], ["a", " "], ["a", 1], {"labels": ["a"]}])
def test_invalid_explicit_label_vocabulary_is_rejected(tmp_path, task, labels):
    write_rows(tmp_path / "train.jsonl", [{"text": "first", "emotions": ["a"], "topic": "a"}])
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="nonempty JSON array of unique nonblank"):
        load_training_datasets({task: tmp_path}, [task], max_train_samples=1)


@pytest.mark.parametrize("task", ["emotion", "topic"])
@pytest.mark.parametrize("explicit_order", [False, True])
def test_unknown_validation_labels_fail_without_expanding_vocab_or_tokenizing(
    tmp_path, task, explicit_order
):
    write_rows(
        tmp_path / "train.jsonl", [{"text": "training", "emotions": ["known"], "topic": "known"}]
    )
    write_rows(
        tmp_path / "validation.jsonl",
        [
            {"text": "validation", "emotions": ["validation-only"], "topic": "validation-only"}
            for _ in range(4)
        ],
    )
    if explicit_order:
        (tmp_path / "labels.json").write_text('["known"]')
    train, val = load_training_datasets({task: tmp_path}, [task], max_train_samples=1)
    assert getattr(train[task], f"{task}_classes") == ["known"]
    tokenizer = Mock()
    tokenizer.batch_encode.side_effect = AssertionError(
        "unknown labels must fail before tokenization"
    )
    loader = build_task_dataloaders(
        val, tokenizer, batch_size=2, shuffle=False, max_length=32, classification_max_length=32
    )[task]
    with pytest.raises(ValueError, match=f"Unknown {task} labels.*validation-only"):
        next(iter(loader))


@pytest.mark.parametrize("task", ["emotion", "topic"])
def test_explicit_vocabulary_cannot_silently_drop_consumed_training_labels(tmp_path, task):
    write_rows(
        tmp_path / "train.jsonl",
        [{"text": "training", "emotions": ["not-declared"], "topic": "not-declared"}],
    )
    (tmp_path / "labels.json").write_text('["declared"]')
    train, _ = load_training_datasets({task: tmp_path}, [task], max_train_samples=1)
    loader = build_task_dataloaders(
        train,
        SyntheticTokenizer(),
        batch_size=1,
        shuffle=False,
        max_length=32,
        classification_max_length=32,
    )[task]
    with pytest.raises(ValueError, match=f"Unknown {task} labels.*not-declared"):
        next(iter(loader))


def test_resume_rejects_missing_or_different_label_order_before_gpu_or_tokenizer(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    from scripts import train

    write_rows(tmp_path / "train.jsonl", [{"text": "training", "topic": "B"}])
    (tmp_path / "labels.json").write_text('["B", "A"]')
    cfg = OmegaConf.create(
        {
            "seed": 17,
            "device": "cuda",
            "data": {"processed": {"topic": str(tmp_path)}},
            "training": {"trainer": {"tasks": ["topic"]}},
            "resume_from": "synthetic.pt",
            "resume_labels": None,
        }
    )
    monkeypatch.setattr(
        train, "Tokenizer", Mock(side_effect=AssertionError("tokenizer constructed"))
    )
    monkeypatch.setattr(
        train.torch.cuda, "get_device_capability", Mock(side_effect=AssertionError("GPU accessed"))
    )
    with pytest.raises(ValueError, match="requires explicit resume_labels"):
        train.main.__wrapped__(cfg)
    saved = tmp_path / "checkpoint-labels.json"
    saved.write_text('{"emotion": [], "topic": ["A", "B"]}')
    cfg.resume_labels = str(saved)
    with pytest.raises(ValueError, match="vocabularies/order differ"):
        train.main.__wrapped__(cfg)
    # The exact same order is accepted by the guard without loading any checkpoint.
    saved.write_text('{"emotion": [], "topic": ["B", "A"]}')
    train.validate_resume_labels(cfg, emotion=[], topic=["B", "A"])
    # Inactive labels also bind; same active dimensions cannot bypass their mismatch.
    saved.write_text('{"emotion": ["unexpected"], "topic": ["B", "A"]}')
    with pytest.raises(ValueError, match="vocabularies/order differ"):
        train.validate_resume_labels(cfg, emotion=[], topic=["B", "A"])
