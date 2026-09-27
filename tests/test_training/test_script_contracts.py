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
