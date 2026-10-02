"""Book field mapping/reference preparation without private archives or model calls."""

from __future__ import annotations

import copy
import gzip
import io
import json
import sqlite3
from pathlib import Path

import pytest

from src.research.book_fields import (
    FACETS,
    input_hash,
    input_payload,
    label_state,
    map_source_labels,
    validate_mapping,
    validate_states,
)
from src.research.builders import bgc_source as bgc
from src.research.builders import book_fields as prep
from src.research.candidate_io import helper_hashes
from src.research.io import file_hash, read_json
from tests.test_research.test_bgc_candidate import archive, book

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def mapping():
    return read_json(ROOT / "research/preparation/book_field_mapping.json")


def validate(mapping):
    audit = read_json(ROOT / "research/preparation/bgc_candidate_manifest.json")
    validate_mapping(
        mapping,
        source_labels=set(audit["observed"]["label_record_counts"]),
        archive_sha256=audit["archive"]["sha256"],
        hierarchy_sha256=audit["observed"]["members"]["hierarchy.txt"]["sha256"],
    )


def test_all_observed_labels_are_explicitly_mapped_or_declined(mapping):
    validate(mapping)
    assert len(mapping["mappings"]) == 146
    assert set(mapping["facets"]) == set(FACETS)
    assert set(value["decision"] for value in mapping["mappings"].values()) == {
        "mapped",
        "ambiguous",
        "unsupported",
    }
    assert "mood" not in mapping["facets"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "wrong_source",
        "unknown_target",
        "implicit_negative",
        "bool_schema",
        "ambiguous_positive",
    ],
)
def test_invalid_mapping_fails_instead_of_silently_skipping_labels(mapping, mutation):
    if mutation == "missing":
        del mapping["mappings"]["Fantasy"]
    elif mutation == "extra":
        mapping["mappings"]["unseen"] = copy.deepcopy(mapping["mappings"]["Fantasy"])
    elif mutation == "wrong_source":
        mapping["source_hierarchy_sha256"] = "0" * 64
    elif mutation == "unknown_target":
        mapping["mappings"]["Fantasy"]["positive"]["genre"] = ["invented"]
    elif mutation == "implicit_negative":
        mapping["semantics"]["source_omissions"] = "negative"
    elif mutation == "bool_schema":
        mapping["schema_version"] = True
    else:
        mapping["mappings"]["Fantasy"]["decision"] = "ambiguous"
    with pytest.raises(ValueError):
        validate(mapping)


def test_union_categories_are_not_split_and_unknown_is_not_negative(mapping):
    fields = map_source_labels([(1, "Mystery & Suspense"), (1, "Gothic & Horror")], mapping)
    assert fields["genre"]["positive"] == ["gothic_horror", "mystery_suspense"]
    assert label_state(fields, "genre", "mystery", mapping) == "unknown"
    assert label_state(fields, "form", "fiction", mapping) == "unknown"  # No source-parent closure.
    assert all(not value["negative"] for value in fields.values())
    fields["genre"]["negative"] = ["romance"]
    assert label_state(fields, "genre", "romance", mapping) == "negative"
    fields["genre"]["positive"].append("romance")
    with pytest.raises(ValueError, match="both"):
        validate_states(fields, mapping)


def test_form_audience_and_topics_do_not_become_genres(mapping):
    fields = map_source_labels([(0, "Children’s Books"), (1, "Poetry"), (1, "History")], mapping)
    assert fields["genre"]["positive"] == []
    assert fields["audience"]["positive"] == ["children"]
    assert fields["form"]["positive"] == ["poetry"]
    assert fields["topic"]["positive"] == ["history"]
    assert all(
        not value["positive"]
        for value in map_source_labels([(1, "Women’s Fiction")], mapping).values()
    )
    with pytest.raises(ValueError, match="Unmapped"):
        map_source_labels([(0, "Unknown provider label")], mapping)


def test_input_hash_excludes_all_target_and_identity_metadata():
    (row,) = bgc.records(io.BytesIO(book(body="Literal &amp; café <em>text</em>")))
    payload = input_payload(row)
    assert set(payload) == {"title", "description"}
    assert payload["description"] == "Literal &amp; café <em>text</em>"
    before = input_hash(payload)
    row.update(labels=[(0, "Romance")], isbn="other", author="Other", copyright="changed")
    assert input_hash(input_payload(row)) == before
    with pytest.raises(ValueError, match="only"):
        input_hash({**payload, "genre": "Fantasy"})
    assert input_hash({**payload, "title": "different"}) != before


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared_source(tmp_path, mapping, monkeypatch):
    # A complete synthetic audit and group manifest; no real source bytes needed in CI.
    for name in [
        "src/research/candidate_io.py",
        "src/research/io.py",
        "src/research/book_fields.py",
        "src/research/book_groups.py",
        "src/catalog/storage.py",
        "src/research/builders/bgc_source.py",
        "src/research/builders/bgc_groups.py",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic dependency: " + name)
    source = tmp_path / "data/research_candidates/bgc/source.zip"
    source.parent.mkdir(parents=True)
    archive(source)
    digest = file_hash(source)
    monkeypatch.setattr(prep, "SOURCE_SHA256", digest)
    audit = {
        "status": "source_audited_not_admitted",
        "training_authorized": False,
        "archive": prep.reference(tmp_path, source),
        "observed": bgc.audit_archive(source),
        "source_declared_license": "CC BY-NC 4.0; synthetic fixture attribution",
        "preparation_script_sha256": file_hash(tmp_path / "src/research/builders/bgc_source.py"),
        "preparation_helper_sha256": helper_hashes(tmp_path),
    }
    audit_path = tmp_path / "research/preparation/bgc_candidate_manifest.json"
    dump(audit_path, audit)
    mapping["source_archive_sha256"] = digest
    mapping["source_hierarchy_sha256"] = audit["observed"]["members"]["hierarchy.txt"]["sha256"]
    mapping["mappings"] = {
        label: mapping["mappings"][label] for label in audit["observed"]["label_record_counts"]
    }
    mapping_path = tmp_path / "research/preparation/book_field_mapping.json"
    dump(mapping_path, mapping)
    assignments = source.parent / "assignments.jsonl"
    rows = [
        {
            "record_id": f"bgc:{split}:1",
            "source_split": split,
            "source_row": 1,
            "group_id": "bgc-group:" + str(number) * 64,
            "proposed_split": split,
            "review_required": False,
        }
        for number, split in enumerate(bgc.MEMBERS)
    ]
    assignments.write_text("".join(json.dumps(row) + "\n" for row in rows))
    groups = {
        "status": "candidate_leakage_groups_not_admitted",
        "training_authorized": False,
        "archive": audit["archive"],
        "candidate_manifest": prep.reference(tmp_path, audit_path),
        "assignments": prep.reference(tmp_path, assignments),
        "preparation_script_sha256": file_hash(tmp_path / "src/research/builders/bgc_groups.py"),
        "preparation_helper_sha256": {
            **helper_hashes(tmp_path),
            "src/research/book_groups.py": file_hash(tmp_path / "src/research/book_groups.py"),
            "src/research/builders/bgc_source.py": file_hash(
                tmp_path / "src/research/builders/bgc_source.py"
            ),
        },
    }
    group_path = tmp_path / "research/preparation/bgc_group_manifest.json"
    dump(group_path, groups)
    return tmp_path, mapping_path, group_path, source.parent / "fields-v1"


def test_references_replay_without_copying_text_and_resolve_clean_inputs(prepared_source):
    root, mapping_path, group_path, output = prepared_source
    first = prep.prepare_fields(*prepared_source)
    path = root / first["field_references"]["path"]
    before = path.read_bytes()
    plain = gzip.decompress(before)
    assert b"made-up blurb" not in plain and b"Synthetic Title" not in plain
    assert b"synthetic fixture" in plain  # Attribution preserved with row/edition references.
    assert first["observed"]["uncompressed_reference_bytes"] == len(plain)
    assert before[4:8] == b"\x00" * 4  # No current-time gzip header.
    assert prep.prepare_fields(*prepared_source) == first
    assert path.read_bytes() == before
    rows = list(prep.iter_resolved_candidates(root, first))
    assert len(rows) == 3 and set(rows[0]["input"]) == {"title", "description"}
    assert rows[0]["candidate"]["source"]["isbn13"] == "9780451457998"
    assert rows[0]["candidate"]["source"]["source_row"] == 1
    assert rows[0]["candidate"]["fields"]["genre"] == {"positive": ["fantasy"], "negative": []}
    assert first["training_authorized"] is False and first["human_gold"] is False
    assert first["observed"]["negative_labels"] == 0
    coverage = first["observed"]["proposed_split_label_coverage"]["train"]
    assert coverage["genre"]["present_labels"] == 1
    assert "romance" in coverage["genre"]["missing_labels"]
    assert "fantasy" not in coverage["genre"]["missing_labels"]
    assert list(output.iterdir()) == [path]
    path.write_bytes(before + b"changed")
    with pytest.raises(ValueError, match="differs"):
        prep.prepare_fields(*prepared_source)
    assert path.read_bytes().endswith(b"changed")


@pytest.mark.parametrize(
    "what",
    [
        "audit",
        "group_binding",
        "group_builder",
        "group_helper",
        "missing_helper",
        "missing_row",
        "extra_row",
        "duplicate_row",
        "group_split",
    ],
)
def test_stale_or_incomplete_grouping_never_publishes(prepared_source, what):
    root, _, group_path, output = prepared_source
    groups = read_json(group_path)
    if what == "audit":
        (root / "src/research/builders/bgc_source.py").write_text("changed")
    elif what == "group_binding":
        groups["candidate_manifest"]["sha256"] = "0" * 64
    elif what == "group_builder":
        groups["preparation_script_sha256"] = "0" * 64
    elif what == "group_helper":
        groups["preparation_helper_sha256"]["src/research/io.py"] = "0" * 64
    elif what == "missing_helper":
        del groups["preparation_helper_sha256"]["src/research/book_groups.py"]
    else:
        path = root / groups["assignments"]["path"]
        rows = list(prep.jsonl_rows(path))
        if what == "missing_row":
            rows.pop()
        elif what == "extra_row":
            rows.append({**rows[0], "record_id": "bgc:train:2", "source_row": 2})
        elif what == "duplicate_row":
            rows.append(rows[0])
        else:
            rows[1]["group_id"] = rows[0]["group_id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        groups["assignments"] = prep.reference(root, path)
    dump(group_path, groups)
    with pytest.raises(ValueError):
        prep.prepare_fields(*prepared_source)
    assert not (output / "field_references.jsonl.gz").exists()


def test_input_hash_and_row_alignment_are_checked_even_after_reference_repin(prepared_source):
    root, _, _, _ = prepared_source
    manifest = prep.prepare_fields(*prepared_source)
    path = root / manifest["field_references"]["path"]
    rows = list(prep.jsonl_rows(path))
    rows[0]["input_sha256"] = "0" * 64
    path.write_bytes(
        gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0)
    )
    manifest["field_references"] = prep.reference(root, path)
    with pytest.raises(ValueError, match="input hash"):
        list(prep.iter_resolved_candidates(root, manifest))
    rows[0]["record_id"] = "bgc:train:2"
    path.write_bytes(
        gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0)
    )
    manifest["field_references"] = prep.reference(root, path)
    with pytest.raises(ValueError, match="alignment"):
        list(prep.iter_resolved_candidates(root, manifest))


def test_bound_and_output_directory_guards(prepared_source, tmp_path):
    root, mapping_path, group_path, _ = prepared_source
    with pytest.raises(ValueError, match="ignored"):
        prep.prepare_fields(root, mapping_path, group_path, root / "tracked-output")
    path = tmp_path / "oversized.jsonl"
    path.write_bytes(b"x" * (prep.MAX_REFERENCE_LINE + 1))
    with pytest.raises(ValueError, match="bound"):
        list(prep.jsonl_rows(path))
    with sqlite3.connect(":memory:") as db:
        path.write_text(json.dumps({"source_row": True}) + "\n")
        with pytest.raises(ValueError, match="Invalid"):
            prep.load_assignments(db, path)
