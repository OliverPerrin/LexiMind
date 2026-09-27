"""Synthetic candidate contracts only; no network or model execution."""

import json

import pytest

from scripts import prepare_ag_news_candidate as prep
from src.research.candidate_io import create_or_verify


def test_source_row_identity_is_stable_and_is_not_an_article_id():
    raw = {"text": "  Exact fixture text\n", "label": 3}
    row = prep.convert_record(raw, "train", 7, "a" * 64)
    assert row == prep.convert_record(raw, "train", 7, "a" * 64)
    assert row["text"] == raw["text"]
    assert row["topic"] == "Sci/Tech" and row["label_id"] == 3
    assert row["source_row"] == 7 and row["provider_split"] == "train"
    assert "sha256:" + "a" * 64 + ":row:7" in row["record_id"]
    assert not {"document_id", "article_id", "url", "provider_article_id"} & row.keys()
    assert prep.convert_record(raw, "train", 8, "a" * 64)["record_id"] != row["record_id"]
    assert prep.convert_record(raw, "train", 7, "b" * 64)["record_id"] != row["record_id"]


@pytest.mark.parametrize(
    "raw",
    [
        {"text": "text", "label": True},
        {"text": "text", "label": 4},
        {"text": "text", "label": 1.0},
        {"text": "", "label": 1},
        {"text": "text", "label": 1, "article_id": "unexpected"},
    ],
)
def test_malformed_or_changed_source_schema_is_rejected(raw):
    with pytest.raises(ValueError):
        prep.convert_record(raw, "train", 1, "a" * 64)


def test_reconstruction_retains_splits_duplicates_and_unknown_rights(tmp_path, monkeypatch):
    candidate, legacy = tmp_path / "candidate", tmp_path / "legacy"
    (candidate / "raw").mkdir(parents=True)
    legacy.mkdir()
    sources = {
        "train": [
            {"text": "private repeated fixture", "label": 0},
            {"text": "private repeated fixture", "label": 2},
        ],
        "test": [
            {"text": "private repeated fixture", "label": 0},
            {"text": "unshared text", "label": 1},
        ],
    }
    receipt = {
        "files": {
            split: {
                "path": f"data/{split}.parquet",
                "sha256": "a" * 64 if split == "train" else "b" * 64,
                "rows": len(rows),
            }
            for split, rows in sources.items()
        }
    }
    (candidate / "raw/acquisition.json").write_text(json.dumps(receipt))
    old = legacy / "train.jsonl"
    old.write_text(json.dumps({"text": "private repeated fixture", "topic": "Legacy label"}) + "\n")
    before = old.read_bytes()
    monkeypatch.setattr(prep, "parquet_rows", lambda path, columns: iter(sources[path.stem]))
    monkeypatch.setattr(prep, "version", lambda package: "synthetic decoder")
    report = prep.prepare_candidate(candidate, receipt, legacy)
    assert old.read_bytes() == before
    assert report["status"] == "candidate_prepared_not_admitted"
    assert report["training_authorized"] is False
    assert report["identity_scope"] == "provider_source_row"
    assert report["upstream_article_ids_available"] is False
    assert report["provider_validation_split_available"] is False
    assert report["duplicate_review"]["exact_text"]["groups"] == 1
    assert report["duplicate_review"]["exact_text"]["records_in_groups"] == 3
    assert report["duplicate_review"]["exact_text"]["cross_split_groups"] == 1
    assert report["duplicate_review"]["exact_text"]["different_label_groups"] == 1
    assert report["rights_review"]["status"] == "unresolved"
    assert (
        report["legacy_comparison"]["files"]["train.jsonl"]["exact_text_matches_to_candidate"] == 1
    )
    assert "private repeated fixture" not in json.dumps(report)
    for split, rows in sources.items():
        actual = [
            json.loads(line)
            for line in (candidate / "prepared" / f"{split}.jsonl").read_text().splitlines()
        ]
        assert [row["text"] for row in actual] == [row["text"] for row in rows]
        assert len({row["record_id"] for row in actual}) == len(rows)
    assert report == prep.prepare_candidate(candidate, receipt, legacy)


def test_source_count_mismatch_does_not_publish_partial_prepared_file(tmp_path, monkeypatch):
    receipt = {
        "files": {
            "train": {"path": "data/train.parquet", "sha256": "a" * 64, "rows": 2},
            "test": {"path": "data/test.parquet", "sha256": "b" * 64, "rows": 0},
        }
    }
    monkeypatch.setattr(prep, "parquet_rows", lambda *args: iter([{"text": "fixture", "label": 0}]))
    with pytest.raises(ValueError, match="row count"):
        prep.prepare_candidate(tmp_path, receipt, tmp_path / "absent-legacy")
    assert not (tmp_path / "prepared/train.jsonl").exists()


def test_shared_writer_preserves_existing_files_and_cleans_failed_stage(tmp_path):
    path = tmp_path / "candidate.jsonl"
    create_or_verify(path, [b"old bytes"])
    with pytest.raises(ValueError, match="differs"):
        create_or_verify(path, [b"new bytes"])

    def interrupted():
        yield b"partial"
        raise OSError("fixture interrupted serialization")

    with pytest.raises(OSError):
        create_or_verify(tmp_path / "new.jsonl", interrupted())
    assert path.read_bytes() == b"old bytes"
    assert not (tmp_path / "new.jsonl").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["candidate.jsonl"]


def test_missing_cache_cannot_implicitly_acquire_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="--fetch"):
        prep.acquire_sources(tmp_path, fetch=False)


def test_missing_original_split_cannot_be_reported_as_complete_candidate(tmp_path):
    with pytest.raises(ValueError, match="every original source split"):
        prep.prepare_candidate(tmp_path, {"files": {}}, tmp_path / "legacy")
    assert not (tmp_path / "prepared").exists()


def test_pinned_release_has_only_two_small_source_files_and_four_labels():
    assert set(prep.FILES) == {"train", "test"}
    assert sum(file["bytes"] for file in prep.FILES.values()) == 19820267
    assert sum(file["bytes"] for file in prep.FILES.values()) < prep.MAX_SOURCE_BYTES
    assert prep.LABELS == ["World", "Sports", "Business", "Sci/Tech"]
