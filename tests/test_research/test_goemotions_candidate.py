"""Synthetic reconstruction checks; no network, models, or original rows needed."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_goemotions_candidate as prep


def test_converter_preserves_text_label_order_comment_identity_and_original_split():
    source = {"id": "opaque-comment-fixture", "text": "  Fixture text.\n", "labels": [25, 17]}
    output = prep.convert_record(source, "validation", 7, "a" * 64)
    assert output["text"] == source["text"]
    assert output["label_ids"] == [25, 17]
    assert output["emotions"] == ["sadness", "joy"]
    assert output["provider_comment_id"] == source["id"]
    assert output["provider_split"] == output["split"] == "validation"
    assert output["provider_revision"] == prep.REVISION
    assert output["document_id"].endswith(":comment:opaque-comment-fixture")
    assert output["source_provenance"] == {"file_sha256": "a" * 64, "row": 7}
    assert output["candidate_status"] == "not_admitted"
    assert source == {
        "id": "opaque-comment-fixture",
        "text": "  Fixture text.\n",
        "labels": [25, 17],
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"id": ""},
        {"text": None},
        {"labels": [True]},
        {"labels": [28]},
        {"labels": [-1]},
        {"labels": []},
    ],
)
def test_invalid_source_shapes_fail_before_admission(patch):
    with pytest.raises(ValueError, match="Invalid simplified"):
        prep.convert_record(
            {"id": "fixture", "text": "text", "labels": [17], **patch}, "train", 1, "a" * 64
        )


def test_exact_matching_does_not_normalize_text_or_reorder_labels():
    key = prep.exact_match_key("text", ["joy", "sadness"])
    assert key != prep.exact_match_key("text ", ["joy", "sadness"])
    assert key != prep.exact_match_key("text", ["sadness", "joy"])


def test_candidate_retains_duplicates_and_annotation_differences_without_touching_legacy(
    tmp_path, monkeypatch
):
    candidate, legacy = tmp_path / "candidate", tmp_path / "legacy"
    (candidate / "raw").mkdir(parents=True)
    legacy.mkdir()
    sources = {
        "train": [
            {"id": "private-fixture-a", "text": "private fixture repeated content", "labels": [17]},
            {"id": "private-fixture-b", "text": "private fixture repeated content", "labels": [17]},
            {"id": "private-fixture-c", "text": "private fixture repeated content", "labels": [25]},
            {
                "id": "private-fixture-d",
                "text": "private fixture unique content",
                "labels": [17, 25],
            },
        ],
        "validation": [
            {"id": "private-fixture-e", "text": "private fixture repeated content", "labels": [17]}
        ],
        "test": [{"id": "private-fixture-f", "text": "different private fixture", "labels": [17]}],
    }
    receipt = {
        "files": {split: {"sha256": "a" * 64, "rows": len(rows)} for split, rows in sources.items()}
    }
    (candidate / "raw/acquisition.json").write_text(json.dumps(receipt))
    for split, rows in sources.items():
        (legacy / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "text": row["text"],
                        "emotions": [prep.LABELS[index] for index in row["labels"]],
                    }
                )
                + "\n"
                for row in rows
            )
        )
    before = {path: path.read_bytes() for path in legacy.iterdir()}
    monkeypatch.setattr(
        prep, "iter_parquet_rows", lambda path: iter(sources[path.name.split("-")[0]])
    )
    monkeypatch.setattr(prep, "version", lambda package: "fixture-decoder")
    report = prep.prepare_candidate(candidate, receipt, legacy)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert report["status"] == "candidate_prepared_not_admitted"
    assert report["training_authorized"] is False
    review = report["duplicate_review"]
    assert review["provider_comment_id"]["groups"] == 0
    assert review["exact_text"]["records_in_groups"] == 4
    assert review["exact_text"]["cross_split_groups"] == 1
    assert review["exact_text_groups_with_different_label_sets"] == 1
    assert review["exact_text_and_ordered_labels"]["records_in_groups"] == 3
    counts = report["legacy_comparison"]["splits"]["train"]["counts"]
    assert counts["all_splits_ambiguous_candidates"] == 2
    assert counts["all_splits_unique_exact_candidate"] == 2
    assert counts["all_splits_unmatched"] == 0
    for split, rows in sources.items():
        actual = [
            json.loads(line)
            for line in (candidate / "prepared" / f"{split}.jsonl").read_text().splitlines()
        ]
        assert len(actual) == len(rows)
        assert [row["text"] for row in actual] == [row["text"] for row in rows]
    serialized = json.dumps(report)
    assert "private fixture repeated content" not in serialized
    assert "private-fixture-a" not in serialized
    assert prep.prepare_candidate(candidate, receipt, legacy) == report


def test_candidate_artifact_is_create_only_or_exactly_verified(tmp_path):
    path = tmp_path / "data.jsonl"
    first = prep._create_or_verify(path, [b"first\n"])
    assert prep._create_or_verify(path, [b"first\n"]) == first
    with pytest.raises(ValueError, match="differs"):
        prep._create_or_verify(path, [b"second\n"])
    assert path.read_bytes() == b"first\n"


def test_missing_cache_does_not_trigger_implicit_network_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="--fetch"):
        prep.acquire_sources(tmp_path, fetch=False)


def test_cached_sources_and_document_receipts_fail_closed_when_corrupted(tmp_path, monkeypatch):
    expected = {}
    files = {}
    for split in prep.SPLITS:
        payload = ("synthetic source " + split).encode()
        path = tmp_path / "raw" / prep.source_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[split] = {"bytes": len(payload), "sha256": prep.sha(payload), "rows": 1}
        files[split] = {**expected[split], "url": prep.source_url(split)}
    monkeypatch.setattr(prep, "PINNED_FILES", expected)
    documents = []
    urls = {
        **prep.DOCUMENTS,
        "README.md": f"https://huggingface.co/datasets/{prep.REPO}/resolve/{prep.REVISION}/README.md",
    }
    for name, url in urls.items():
        path = tmp_path / "raw/documents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic provenance fixture")
        documents.append({"name": name, "url": url, "sha256": prep.file_hash(path)})
    receipt = {
        "schema_version": 1,
        "repo": prep.REPO,
        "revision": prep.REVISION,
        "config": "simplified",
        "label_names": prep.LABELS,
        "public": True,
        "gated": False,
        "provider_license_declaration": ["apache-2.0"],
        "files": files,
        "documents": documents,
    }
    prep._check_cached_sources(tmp_path, receipt)
    with pytest.raises(ValueError, match="provenance documents"):
        prep._check_cached_sources(tmp_path, {**receipt, "documents": []})
    (tmp_path / "raw" / prep.source_path("train")).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="corrupt pinned source"):
        prep._check_cached_sources(tmp_path, receipt)


def test_cli_refuses_to_write_candidate_or_manifest_into_legacy_data(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    before = legacy / "train.jsonl"
    before.write_text("legacy fixture")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(prep.__file__)),
            "--candidate-dir",
            str(legacy),
            "--legacy-dir",
            str(legacy),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "read-only legacy" in result.stderr
    assert before.read_text() == "legacy fixture"


def test_pinned_payload_metadata_is_bounded_and_not_the_raw_release():
    assert sum(item["bytes"] for item in prep.PINNED_FILES.values()) == 3464371
    assert sum(item["bytes"] for item in prep.PINNED_FILES.values()) < prep.MAX_DOWNLOAD_BYTES
    assert all(prep.source_path(split).startswith("simplified/") for split in prep.SPLITS)
    assert len(prep.LABELS) == len(set(prep.LABELS)) == 28


def test_committed_manifest_exposes_no_row_text_and_remains_unadmitted():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "research/preparation/goemotions_candidate_manifest.json").read_text()
    )
    assert manifest["revision"] == prep.REVISION
    assert manifest["status"] == "candidate_prepared_not_admitted"
    assert manifest["training_authorized"] is False
    assert sum(item["rows"] for item in manifest["prepared_files"].values()) == 54263
    assert manifest["duplicate_review"]["provider_comment_id"]["groups"] == 0
    for split in prep.SPLITS:
        assert (
            manifest["acquisition"]["files"][split]["sha256"] == prep.PINNED_FILES[split]["sha256"]
        )
