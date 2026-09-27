"""Synthetic contracts for the retained training entry point; no experiment runs."""


def test_weights_only_resume_does_not_apply_last_epoch_to_best(tmp_path):
    from scripts.train import resume_start_epoch

    (tmp_path / "last_epoch.json").write_text('{"epoch": 8}')
    assert resume_start_epoch(tmp_path / "last.pt") == 9
    assert resume_start_epoch(tmp_path / "best.pt") == 1
    assert resume_start_epoch(tmp_path / "epoch_3.pt") == 4
    assert resume_start_epoch(tmp_path / "seed_17_best.pt") == 1


def test_training_split_loader_does_not_open_test_data(tmp_path):
    from scripts.train import load_splits

    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text("synthetic fixture")
    calls = []
    result = load_splits(tmp_path, lambda path: calls.append(path) or [])
    assert set(result) == {"train", "val"}
    assert all(not path.endswith("/test.jsonl") for path in calls)


def test_checkpoint_contract_is_available_before_weights_and_cannot_be_replaced(tmp_path):
    import pytest

    from scripts.train import prepare_checkpoint_labels
    from src.utils.labels import BOOK_INPUT_FORMAT, LabelMetadata, load_label_metadata

    metadata = LabelMetadata([], ["genre:fantasy"], "multi_label", BOOK_INPUT_FORMAT, "a" * 64)
    labels = tmp_path / "artifacts" / "labels.json"
    checkpoint_dir = tmp_path / "checkpoints"
    prepare_checkpoint_labels(metadata, labels, checkpoint_dir)
    assert load_label_metadata(labels) == metadata
    assert load_label_metadata(checkpoint_dir / "labels.json") == metadata
    assert not list(checkpoint_dir.glob("*.pt"))
    (checkpoint_dir / "last.pt").write_bytes(b"synthetic weight placeholder")
    prepare_checkpoint_labels(metadata, labels, checkpoint_dir)
    before = labels.read_bytes()
    changed = LabelMetadata([], ["genre:fantasy"], "multi_label", BOOK_INPUT_FORMAT, "b" * 64)
    with pytest.raises(ValueError, match="Different label metadata"):
        prepare_checkpoint_labels(changed, labels, checkpoint_dir)
    assert labels.read_bytes() == before


def test_book_mode_refuses_unbound_existing_checkpoint_directory(tmp_path):
    import pytest

    from scripts.train import prepare_checkpoint_labels
    from src.utils.labels import BOOK_INPUT_FORMAT, LabelMetadata

    (tmp_path / "last.pt").write_bytes(b"legacy weights")
    metadata = LabelMetadata([], ["genre:fantasy"], "multi_label", BOOK_INPUT_FORMAT, "a" * 64)
    with pytest.raises(ValueError, match="fresh directory"):
        prepare_checkpoint_labels(metadata, tmp_path / "other" / "labels.json", tmp_path)
    assert not (tmp_path / "other").exists()


def test_profiler_rejects_book_mode_before_cuda_data_or_model_setup(monkeypatch):
    import pytest
    from omegaconf import OmegaConf

    from scripts import profile_training

    monkeypatch.setattr(
        profile_training, "validate_task_directories", lambda *args: pytest.fail("Data opened")
    )
    monkeypatch.setattr(
        profile_training.torch.cuda, "get_device_name", lambda: pytest.fail("CUDA queried")
    )
    cfg = OmegaConf.create({"data": {"topic_problem_type": "multi_label"}})
    with pytest.raises(ValueError, match="no profile was started"):
        profile_training.main.__wrapped__(cfg)
