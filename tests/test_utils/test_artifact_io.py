"""Synthetic artifact round trips only; no historical checkpoint or model runs."""

import json
from collections import OrderedDict

import pytest
import torch

from src.utils import core
from src.utils.atomic import atomic_write
from src.utils.io import load_state, normalize_state_dict, save_state
from src.utils.labels import LabelMetadata, load_label_metadata, save_label_metadata


class VersionedModule(torch.nn.Module):
    _version = 7

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.loaded_version = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        self.loaded_version = local_metadata.get("version")
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)


class Wrapper(torch.nn.Module):
    """Reproduce torch.compile wrapper names without invoking compilation."""

    def __init__(self, module):
        super().__init__()
        self._orig_mod = module


def test_plain_and_compiled_state_roundtrip_preserves_tensor_and_metadata(tmp_path):
    for name, source in (("plain", VersionedModule()), ("wrapped", Wrapper(VersionedModule()))):
        path = tmp_path / f"{name}.pt"
        save_state(source, path)
        destination = VersionedModule()
        with torch.no_grad():
            destination.weight.zero_()
        load_state(destination, path)
        torch.testing.assert_close(destination.weight, torch.tensor([1.0, 2.0]))
        assert destination.loaded_version == 7
        saved = torch.load(path, weights_only=True)
        assert isinstance(saved, OrderedDict)
        assert saved._metadata[""]["version"] == 7


def test_existing_compiled_state_loading_and_legacy_api_are_compatible(tmp_path):
    source = Wrapper(VersionedModule())
    path = tmp_path / "fixture.pt"
    torch.save(source.state_dict(), path)
    destination = VersionedModule()
    core.load_checkpoint(destination, path)
    assert destination.loaded_version == 7
    path2 = tmp_path / "legacy.pt"
    core.save_checkpoint(destination, path2)
    assert torch.load(path2, weights_only=True)._metadata[""]["version"] == 7


def test_nested_wrapper_metadata_and_order_survive_without_mutating_source():
    model = torch.nn.Module()
    model.child = Wrapper(VersionedModule())
    source = Wrapper(model).state_dict()
    original_keys, original_metadata = list(source), dict(source._metadata)
    normalized = normalize_state_dict(source)
    assert list(normalized) == ["child.weight"]
    assert normalized._metadata["child"]["version"] == 7
    assert list(source) == original_keys
    assert dict(source._metadata) == original_metadata
    # Metadata order is not relied upon to select the wrapped module's version.
    source._metadata = OrderedDict(reversed(list(source._metadata.items())))
    assert normalize_state_dict(source)._metadata["child"]["version"] == 7


def test_cleanup_preserves_literal_similar_names_and_rejects_parameter_collisions():
    value = torch.tensor([1.0])
    source = OrderedDict(
        [("layer_orig_mod.weight", value), ("_orig_mod.child._orig_mod.weight", value)]
    )
    assert list(normalize_state_dict(source)) == ["layer_orig_mod.weight", "child.weight"]
    with pytest.raises(ValueError, match="key collision"):
        normalize_state_dict({"weight": value, "_orig_mod.weight": value})


def test_ambiguous_metadata_collisions_are_rejected():
    source = OrderedDict([("weight", torch.tensor([1.0]))])
    source._metadata = OrderedDict(
        [("child._orig_mod", {"version": 1}), ("_orig_mod.child", {"version": 2})]
    )
    with pytest.raises(ValueError, match="metadata collision"):
        normalize_state_dict(source)


def test_save_collision_cannot_replace_an_existing_checkpoint(tmp_path):
    model = VersionedModule()
    model._orig_mod = VersionedModule()
    path = tmp_path / "weights.pt"
    path.write_bytes(b"previous checkpoint")
    with pytest.raises(ValueError, match="key collision"):
        save_state(model, path)
    assert path.read_bytes() == b"previous checkpoint"


def test_load_collision_fails_before_mutating_destination(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.zeros(2), "_orig_mod.weight": torch.ones(2)}, path)
    destination = VersionedModule()
    with pytest.raises(ValueError, match="key collision"):
        load_state(destination, path)
    torch.testing.assert_close(destination.weight, torch.tensor([1.0, 2.0]))
    assert destination.loaded_version is None


def test_checkpoint_serialization_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"previous checkpoint")

    def fail_save(state, stream):
        stream.write(b"incomplete checkpoint")
        raise OSError("disk exhausted")

    monkeypatch.setattr("src.utils.io.torch.save", fail_save)
    with pytest.raises(OSError, match="disk exhausted"):
        save_state(VersionedModule(), path)
    assert path.read_bytes() == b"previous checkpoint"
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_atomic_replacement_preserves_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / "labels.json"
    path.write_bytes(b"previous labels")

    def fail_replace(*args):
        raise PermissionError("read-only destination")

    monkeypatch.setattr("src.utils.atomic.os.replace", fail_replace)
    with pytest.raises(PermissionError):
        atomic_write(path, lambda stream: stream.write(b"new labels"))
    assert path.read_bytes() == b"previous labels"
    assert not list(tmp_path.glob("*.tmp"))


def test_explicit_empty_task_vocabularies_and_legacy_properties_roundtrip(tmp_path):
    metadata = LabelMetadata(emotion=[], topic=["Science", "Fiction"])
    assert core.LabelMetadata is LabelMetadata
    assert metadata.emotion_size == metadata.num_emotions == 0
    assert metadata.topic_size == metadata.num_topics == 2
    path = tmp_path / "labels.json"
    core.save_labels(metadata, path)
    loaded = load_label_metadata(path)
    assert loaded == metadata == core.load_labels(path)
    assert path.read_text() == json.dumps(
        {"emotion": [], "topic": ["Science", "Fiction"]}, ensure_ascii=False, indent=2
    )


def test_plural_labels_remain_supported_and_explicit_empty_singular_wins(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"emotions": ["joy"], "topics": []}))
    assert load_label_metadata(path) == LabelMetadata(emotion=["joy"], topic=[])
    path.write_text(json.dumps({"emotion": [], "emotions": ["joy"], "topic": []}))
    assert load_label_metadata(path) == LabelMetadata(emotion=[], topic=[])


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"emotion": ["joy"]},
        {"emotion": ["joy", "joy"], "topic": []},
        {"emotion": [""], "topic": []},
        {"emotion": [], "topic": [" "]},
        {"emotion": [], "topic": [1]},
        {"emotion": "joy", "topic": []},
    ],
)
def test_invalid_label_payloads_fail_before_model_construction(tmp_path, payload):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_label_metadata(path)
    with pytest.raises(ValueError):
        core.load_labels(path)


def test_mutated_label_metadata_cannot_overwrite_previous_labels(tmp_path):
    path = tmp_path / "labels.json"
    metadata = LabelMetadata(emotion=["joy"], topic=[])
    save_label_metadata(metadata, path)
    previous = path.read_bytes()
    metadata.emotion.append("joy")
    with pytest.raises(ValueError, match="duplicate"):
        save_label_metadata(metadata, path)
    assert path.read_bytes() == previous


def test_label_collections_are_copied_without_reordering_or_renaming():
    emotions = ["sadness", "joy"]
    metadata = LabelMetadata(emotion=emotions, topic=[])
    emotions.append("anger")
    assert metadata.emotion == ["sadness", "joy"]
