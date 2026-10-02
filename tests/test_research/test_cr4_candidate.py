"""Synthetic CR4 preservation/identity contracts; no external data or models."""

import csv
import gzip
import hashlib
import json
from io import BytesIO

import pytest

from src.research import cr4
from src.research.builders import cr4 as prep


def row(**changes):
    return {
        "file_id": "book-a",
        "classification_id": "101",
        "user_id": "reader-a",
        "workflow_name": "Contemporary Literature",
        "created_at": "2024-10-01 00:00:00 UTC",
        "subject_ids": "501",
        "passage": "Ada felt nervous, then relieved.",
        "highlighted_char": "Ada",
        "Category": "FIC",
        "Genre": "BS",
        "Code": "NA",
        "PUBL_DATE": "2010",
        "t0": "Yes",
        "t1": "nervous, relieved",
        **changes,
    }


def source(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cr4.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_exact_unicode_multiline_response_and_character_context_survive(tmp_path):
    original = row(t1='café\n"relieved", nervous  ', highlighted_char="Ada")
    path = source(tmp_path / "source.csv", [original])
    (item,) = cr4.iter_annotations(path)
    assert item["condition"] == {"passage": original["passage"], "highlighted_character": "Ada"}
    assert item["human_response"] == {"t0": "Yes", "t1": original["t1"]}
    assert item["source"]["source_row"] == 1
    assert item["source"]["classification_id"] == "101"
    assert item["validity"]["unambiguous_input_condition"]
    assert "labels" not in item
    assert cr4.normalized("Cafe\u0301  \n WORRIED") == "café worried"


@pytest.mark.parametrize("response,state", [("", "blank"), (" \n ", "blank"), ("NA", "na_marker")])
def test_missing_response_never_becomes_neutral_or_negative(response, state):
    result = cr4.annotation(1, row(t1=response))
    assert result["human_response"]["t1"] == response
    assert result["validity"]["response_state"] == state
    assert "response_" + state in result["validity"]["flags"]
    assert "neutral" not in str(result)


def test_rejected_character_is_distinct_from_missing_response_and_invalid_input():
    result = cr4.validity(row(t0=cr4.NO_CHARACTER, t1="NA"))
    assert result["character_decision"] == "no"
    assert result["unambiguous_input_condition"]
    assert "character_rejected_or_passage_error" in result["flags"]
    assert "response_with_rejected_character" in cr4.validity(row(t0=cr4.NO_CHARACTER))["flags"]
    assert "unknown_t0" in cr4.validity(row(t0="Maybe"))["flags"]
    assert "highlight_not_found" in cr4.validity(row(highlighted_char="Other"))["flags"]
    assert not cr4.validity(row(highlighted_char=""))["unambiguous_input_condition"]
    duplicate = cr4.validity(row(passage="Ada told Ada the news."))
    assert not duplicate["unambiguous_input_condition"]
    assert "highlight_span_ambiguous" in duplicate["flags"]


def test_long_and_copied_responses_are_preserved_but_flagged(tmp_path):
    long = row(t1="x" * 222345)
    path = source(tmp_path / "long.csv", [long])
    (result,) = cr4.iter_annotations(path)
    assert result["human_response"]["t1"] == long["t1"]
    assert "long_response_requires_review" in result["validity"]["flags"]
    assert "response_copies_passage" in cr4.validity(row(t1=row()["passage"].upper()))["flags"]


@pytest.mark.parametrize(
    "text",
    [
        "wrong,header\n",
        ",".join(cr4.COLUMNS) + "\none,column\n",
        ",".join(cr4.COLUMNS) + '\n"unterminated',
    ],
)
def test_malformed_csv_is_rejected(tmp_path, text):
    path = tmp_path / "invalid.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match="CSV|field count|schema"):
        list(cr4.source_rows(path))


def test_record_bound_covers_multiline_fields_and_restores_csv_setting(tmp_path, monkeypatch):
    path = source(tmp_path / "large.csv", [row(t1="short line\n" * 1000)])
    before = csv.field_size_limit()
    monkeypatch.setattr(cr4, "MAX_RECORD_BYTES", 500)
    with pytest.raises(ValueError, match="bound"):
        list(cr4.source_rows(path))
    assert csv.field_size_limit() == before


def test_audit_retains_annotation_references_and_reports_overlap_and_disagreement(tmp_path):
    rows = [
        row(),
        row(classification_id="102", user_id="reader-b", t1="WORRIED"),
        row(classification_id="103", t1="worried "),
        row(
            classification_id="104",
            file_id="book-b",
            subject_ids="502",
            t0=cr4.NO_CHARACTER,
            t1="NA",
        ),
        row(classification_id="105", workflow_name="World Literature", subject_ids="503", t1=""),
    ]
    path = source(tmp_path / "source.csv", rows)
    before = path.read_bytes()
    index = tmp_path / "index.gz"
    observed = prep.audit_source(path, index)
    assert observed["annotation_rows"] == 5
    assert observed["identity"]["source_document_ids"] == 2
    assert observed["identity"]["namespaced_source_documents"] == 3
    assert observed["identity"]["file_ids_reused_across_workflows"] == 1
    assert observed["identity"]["repeated_annotator_subject_pairs"] == 1
    assert observed["identity"]["conditions_across_documents"] == 1
    assert observed["disagreement"]["conditions_with_mixed_character_decisions"] == 1
    assert observed["disagreement"]["conditions_with_multiple_raw_responses"] == 1
    assert observed["responses"]["unique_raw_responses"] == 3
    assert observed["responses"]["unique_comparison_normalized_responses"] == 2
    assert observed["responses"]["t0_no_with_nonblank_t1_rows"] == 1
    assert observed["responses"]["t0_no_with_nonblank_non_NA_t1_rows"] == 0
    with gzip.open(index, "rt") as handle:
        indexed = [json.loads(line) for line in handle]
    assert len(indexed) == 3
    assert sorted(n for item in indexed for n in item["source_rows"]) == [1, 2, 3, 4, 5]
    assert len({item["document_namespace"] for item in indexed}) == 3
    assert "Ada" not in str(indexed) and "nervous" not in str(indexed)
    assert path.read_bytes() == before
    assert prep.audit_source(path, index) == observed


def test_no_silent_annotation_deduplication(tmp_path):
    path = source(tmp_path / "source.csv", [row(), row()])
    result = prep.audit_source(path, tmp_path / "index.gz")
    assert result["annotation_rows"] == 2
    assert result["identity"]["duplicate_classification_ids"] == 1


def test_download_is_bounded_verified_and_create_only(tmp_path, monkeypatch):
    data = b"synthetic bytes"
    spec = {
        "url": "https://borealisdata.ca/test",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "provider_md5": hashlib.md5(data).hexdigest(),
    }

    class Response(BytesIO):
        def geturl(self):
            return spec["url"]

    monkeypatch.setattr(prep, "urlopen", lambda *a, **kw: Response(data))
    destination = tmp_path / "verified.csv"
    prep.download_file(destination, spec)
    assert destination.read_bytes() == data
    prep.download_file(destination, spec)
    for invalid in (
        {**spec, "bytes": 2},
        {**spec, "sha256": "0" * 64},
        {**spec, "provider_md5": "0" * 32},
    ):
        with pytest.raises(ValueError, match="pinned|checksum"):
            prep.download_file(tmp_path / "bad.csv", invalid)
        assert not (tmp_path / "bad.csv").exists()
    destination.write_bytes(b"existing different contents")
    with pytest.raises(ValueError, match="Existing"):
        prep.download_file(destination, spec)
    assert destination.read_bytes() == b"existing different contents"


def test_local_first_preparation_never_downloads_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "urlopen", lambda *a, **kw: pytest.fail("Network must be opt-in"))
    with pytest.raises(ValueError, match="Missing file"):
        prep.prepare_candidate(tmp_path / "data/research_candidates/cr4/v1.0")
    with pytest.raises(ValueError, match="cache must remain"):
        prep.prepare_candidate(tmp_path / "outside")
