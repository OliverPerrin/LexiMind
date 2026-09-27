"""Synthetic source audits only; no model loading or research evaluation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_research_data as auditor


def write_task(root, task, rows, labels=None):
    folder = root / task
    folder.mkdir(parents=True)
    for split, records in rows.items():
        (folder / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    if labels is not None:
        (folder / "labels.json").write_text(json.dumps(labels), encoding="utf-8")


def complete_emotion(root):
    write_task(
        root,
        "emotion",
        {
            split: [
                {
                    "text": f"unique comment {split}",
                    "emotions": ["joy"],
                    "document_id": f"hf:fixture:{split}",
                    "identity_scope": "provider_document",
                    "work_identity_status": "unresolved",
                    "identity_source": "https://example.org/fixture",
                }
            ]
            for split in ("train", "validation", "test")
        },
        labels=["joy"],
    )


def test_streamed_inventory_is_deterministic_read_only_and_does_not_copy_text(tmp_path):
    root = tmp_path / "input"
    secret = "Private source content that must remain inside this synthetic fixture."
    write_task(
        root,
        "books",
        {
            "train": [
                {"text": secret, "title": "Private title", "author": "", "type": "gutenberg"}
            ],
            "validation": [
                {"text": secret, "title": "Private title", "author": "", "type": "gutenberg"}
            ],
            "test": [
                {
                    "text": "Different input",
                    "title": "Different title",
                    "author": "",
                    "type": "gutenberg",
                }
            ],
        },
    )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    inventory, audit = auditor.audit_data(root, sample_limit=2)
    assert (inventory, audit) == auditor.audit_data(root, sample_limit=2)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert inventory["records"] == 3
    assert audit["blocking_findings"]["missing_parent_identity"] == 3
    assert audit["duplicate_groups"]["input_normalized"]["cross_split_groups"] == 1
    assert audit["duplicate_groups"]["declared_work"]["groups"] == 0
    assert audit["duplicate_groups"]["title_only_candidate"]["groups"] == 1
    encoded = json.dumps([inventory, audit])
    assert secret not in encoded and "Private title" not in encoded
    for entry in inventory["files"]:
        assert entry["sha256"] == auditor.sha((root / entry["path"]).read_bytes())
        for reference in entry["bounded_anomalies"]:
            raw_line = (
                (root / reference["file"])
                .read_bytes()
                .splitlines(keepends=True)[reference["line"] - 1]
            )
            assert reference["record_sha256"] == auditor.sha(raw_line)


def test_normalized_equality_preserves_case_and_punctuation(tmp_path):
    root = tmp_path / "input"
    write_task(
        root,
        "emotion",
        {
            "train": [{"text": "Keep  Café.\n", "emotions": ["joy"]}],
            "validation": [{"text": "Keep Cafe\u0301.", "emotions": ["joy"]}],
            "test": [
                {"text": "keep Café.", "emotions": ["joy"]},
                {"text": "Keep Café!", "emotions": ["joy"]},
            ],
        },
        labels=["joy"],
    )
    _, audit = auditor.audit_data(root)
    assert audit["duplicate_groups"]["input_raw"]["groups"] == 0
    assert audit["duplicate_groups"]["input_normalized"]["groups"] == 1
    pair = next(
        pair for pair in audit["pairwise_intersections"] if pair["kind"] == "input_normalized"
    )
    assert {pair["left"], pair["right"]} == {"emotion/train.jsonl", "emotion/validation.jsonl"}


def test_document_group_overlap_and_title_candidates_are_separate_evidence(tmp_path):
    root = tmp_path / "input"
    common = {
        "type": "literary",
        "document_id": "hf:kmfoda/booksum:bid:123",
        "identity_scope": "provider_document",
        "work_identity_status": "unresolved",
        "identity_source": "https://example.org/source",
    }
    write_task(
        root,
        "summarization",
        {
            "train": [
                {**common, "source": "chapter one", "summary": "Same target", "title": "Title one"}
            ],
            "test": [
                {**common, "source": "chapter two", "summary": "Same target", "title": "Title two"}
            ],
            "validation": [
                {
                    **common,
                    "document_id": "hf:kmfoda/booksum:bid:999",
                    "source": "other chapter",
                    "summary": "Other target",
                    "title": "Other title",
                }
            ],
        },
    )
    inventory, audit = auditor.audit_data(root)
    assert (
        sum(
            entry["identity_and_provenance"]["explicit_provider_document_only"]
            for entry in inventory["files"]
        )
        == 3
    )
    assert audit["duplicate_groups"]["declared_document"]["cross_split_groups"] == 1
    assert audit["duplicate_groups"]["declared_work"]["groups"] == 0
    assert audit["identical_targets_across_distinct_titles"]["groups"] == 1
    assert audit["blocking_findings"]["unresolved_literary_work_identity"] == 3


def test_historical_comparison_extracts_counts_without_metric_values(tmp_path):
    root = tmp_path / "input"
    complete_emotion(root)
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"emotion": {"num_samples": 4, "macro_f1": 0.123456789}}))
    _, audit = auditor.audit_data(root, historical_report=history)
    comparison = audit["historical_count_comparison"]["comparisons"][0]
    assert comparison["row_count_difference"] == -3
    assert "macro_f1" not in json.dumps(audit)
    assert "0.123456789" not in json.dumps(audit)


def test_malformed_records_and_unknown_labels_fail_closed(tmp_path):
    root = tmp_path / "input"
    write_task(
        root,
        "emotion",
        {"train": [{"text": "comment", "emotions": ["unlisted private label"]}]},
        labels=["joy"],
    )
    with (root / "emotion/train.jsonl").open("a") as stream:
        stream.write('{"text":"one","text":"two"}\n["not an object"]\n')
    inventory, audit = auditor.audit_data(root)
    assert inventory["files"][0]["malformed_records"] == 2
    assert audit["blocking_findings"]["unknown_label_occurrences"] == 1
    assert audit["blocking_findings"]["missing_task_split_files"] == 2
    assert "unlisted private label" not in json.dumps([inventory, audit])


def test_even_clean_mechanical_checks_cannot_authorize_training(tmp_path):
    root = tmp_path / "input"
    complete_emotion(root)
    _, audit = auditor.audit_data(root)
    assert audit["mechanical_checks_passed"] is True
    assert audit["status"] == "requires_protocol_review"
    assert audit["training_authorized"] is False
    command = [
        sys.executable,
        str(Path(auditor.__file__)),
        "--input-dir",
        str(root),
        "--inventory",
        str(tmp_path / "inventory.json"),
        "--audit",
        str(tmp_path / "audit.json"),
        "--historical-report",
        str(tmp_path / "absent.json"),
        "--require-ready",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads((tmp_path / "audit.json").read_text())["training_authorized"] is False


def test_report_cli_cannot_overwrite_read_only_inputs(tmp_path):
    root = tmp_path / "input"
    complete_emotion(root)
    protected = root / "emotion/train.jsonl"
    before = protected.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(auditor.__file__)),
            "--input-dir",
            str(root),
            "--inventory",
            str(protected),
            "--audit",
            str(tmp_path / "audit.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "outside the read-only input" in result.stderr
    assert protected.read_bytes() == before


def test_corpus_change_after_a_file_is_scanned_aborts_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "input"
    complete_emotion(root)
    original = auditor.json.loads

    def mutate_earlier_file(raw, *args, **kwargs):
        row = original(raw, *args, **kwargs)
        if isinstance(row, dict) and row.get("text") == "unique comment validation":
            with (root / "emotion/train.jsonl").open("a") as stream:
                stream.write("\n")
        return row

    monkeypatch.setattr(auditor.json, "loads", mutate_earlier_file)
    with pytest.raises(RuntimeError, match="corpus changed"):
        auditor.audit_data(root)
