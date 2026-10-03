"""Adversarial synthetic source boundaries for the weak-field diagnostic."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from copy import deepcopy

import pytest

from src.research import field_baseline_data as data
from src.research.book_fields import FACETS, input_hash, input_payload, map_source_labels
from src.research.builders.bgc_source import MEMBERS, isbn13, provider_id, records
from tests.test_research.test_bgc_candidate import book


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def group_id(identifier):
    return "bgc-group:" + hashlib.sha256(identifier.encode()).hexdigest()


@pytest.fixture
def packet(tmp_path, monkeypatch):
    """Small fixed cohort; all content is invented here, never copied from BGC."""
    monkeypatch.setattr(data, "SAMPLE_COUNTS", {"train": 2, "dev": 1})
    mapping = {
        "facets": {facet: {"labels": ["fantasy"]} for facet in FACETS},
        "mappings": {"Fantasy": {"positive": {"genre": ["fantasy"]}}},
    }
    assignments, candidates, source = [], [], {role: [] for role in MEMBERS}
    for split, count in (("train", 3), ("dev", 2), ("test", 1)):
        for number in range(1, count + 1):
            identifier = f"bgc:{split}:{number}"
            raw = book(
                title=f"Invented {split} {number}",
                body=f"Synthetic passage {split} {number}.",
                labels=((0, "Fantasy"),),
            )
            source[split].append(raw)
            (parsed,) = records(io.BytesIO(raw))
            effective = "test" if (split, number) in {("train", 3), ("dev", 2)} else split
            assignment = {
                "record_id": identifier,
                "source_split": split,
                "source_row": number,
                "group_id": group_id(identifier),
                "proposed_split": effective,
                "review_required": False,
            }
            assignments.append(assignment)
            candidates.append(
                {
                    "record_id": identifier,
                    "source": {
                        "source_split": split,
                        "source_row": number,
                        "isbn13": isbn13(parsed["isbn"]),
                        "provider_book_id": provider_id(parsed["url"]),
                        "provider_url": parsed["url"],
                        "attribution": parsed["copyright"],
                        "language": parsed["language"],
                    },
                    "input_sha256": input_hash(input_payload(parsed)),
                    "source_labels": [[0, "Fantasy"]],
                    "fields": map_source_labels(parsed["labels"], mapping),
                    "group": {
                        key: assignment[key]
                        for key in ("group_id", "proposed_split", "review_required")
                    },
                }
            )
    fields = {
        "assignments": {"path": "assignments.jsonl"},
        "field_references": {"path": "references.jsonl"},
        "archive": {"path": "source.zip"},
        "observed": {"records": len(assignments)},
    }
    partition = {
        "components": {"path": "components.jsonl"},
        "counts": {"overlay_components": 0, "overridden_bgc_groups": 0},
    }
    groups = {"counts": {"records": len(assignments), "groups": len(assignments)}}
    fixture = {
        "root": tmp_path,
        "mapping": mapping,
        "assignments": assignments,
        "candidates": candidates,
        "source": source,
        "fields": fields,
        "partition": partition,
        "groups": groups,
        "components": [],
    }
    save_metadata(fixture)
    return fixture


def save_metadata(packet):
    for filename, key in (
        ("assignments.jsonl", "assignments"),
        ("references.jsonl", "candidates"),
        ("components.jsonl", "components"),
    ):
        write_jsonl(packet["root"] / filename, packet[key])


def select(packet):
    return data._metadata_selection(
        packet["root"], packet["fields"], packet["partition"], packet["groups"], packet["mapping"]
    )


def write_archive(packet):
    members = {MEMBERS[role]: b"".join(rows) for role, rows in packet["source"].items()}
    with zipfile.ZipFile(packet["root"] / "source.zip", "w") as archive:
        for member, raw in members.items():
            archive.writestr(member, raw)
    return {
        "source_declared_license": "Synthetic fixture only",
        "observed": {
            "splits": {role: {"rows": len(rows)} for role, rows in packet["source"].items()},
            "members": {
                member: {"sha256": hashlib.sha256(raw).hexdigest()}
                for member, raw in members.items()
            },
        },
    }


def mock_bindings(monkeypatch, packet, audit):
    monkeypatch.setattr(
        data,
        "_load_bindings",
        lambda root, config: (
            packet["fields"],
            packet["partition"],
            packet["groups"],
            audit,
            packet["mapping"],
            {},
        ),
    )


def test_cohort_keeps_original_and_effective_reserves_out_before_text_resolution(packet):
    chosen, selected, counts = select(packet)
    assert set(chosen["train"]) == {"bgc:train:1", "bgc:train:2"}
    assert chosen["dev"] == ["bgc:dev:1"]
    assert set(selected) == {*chosen["train"], *chosen["dev"]}
    assert counts["excluded_by_first_reason"] == {"effective_reserved": 2, "source_reserved": 1}
    assert counts["record_overlap"] == counts["group_overlap"] == 0
    # Reversing metadata order cannot change the predeclared cohort.
    packet["assignments"].reverse()
    packet["candidates"].reverse()
    save_metadata(packet)
    assert select(packet) == (chosen, selected, counts)


def test_invalid_utf8_in_unselected_effective_test_frames_is_never_decoded(packet, monkeypatch):
    poison = b"<book>\n\xff\xfe intentionally invalid frame\n</book>\n"
    packet["source"]["train"][2] = poison
    packet["source"]["dev"][1] = poison
    packet["source"]["test"][0] = b"This original test member must never be opened.\xff"
    audit = write_archive(packet)
    mock_bindings(monkeypatch, packet, audit)
    opened = []
    original_open = zipfile.ZipFile.open

    def guarded_open(archive, name, *args, **kwargs):
        assert name != MEMBERS["test"], "Original test text was opened"
        opened.append(name)
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    result = data.prepare_field_baseline_data(packet["root"], {})
    assert opened == [MEMBERS["train"], MEMBERS["dev"]]
    assert len(result["train"]) == 2 and len(result["dev"]) == 1
    assert result["provenance"]["decoded_records"] == 3
    assert result["provenance"]["test_records_decoded"] == 0
    for row in result["train"] + result["dev"]:
        assert row["input"]["description"].startswith("Synthetic passage")
        assert row["source_split"] == row["effective_split"]
        assert all(not states["negative"] for states in row["fields"].values())


@pytest.mark.parametrize(
    "mutation", ["role", "source_row", "group", "missing", "duplicate", "states"]
)
def test_candidate_join_corruption_fails_before_source_can_be_opened(packet, mutation):
    candidate = packet["candidates"][0]
    if mutation == "role":
        candidate["source"]["source_split"] = "dev"
    elif mutation == "source_row":
        candidate["source"]["source_row"] += 1
    elif mutation == "group":
        candidate["group"]["group_id"] = group_id("invented group")
    elif mutation == "missing":
        packet["candidates"].pop()
    elif mutation == "duplicate":
        packet["candidates"].append(deepcopy(candidate))
    else:
        candidate["fields"]["genre"]["positive"] = []
    save_metadata(packet)
    # No archive exists; metadata validation must fail without trying source reads.
    assert not (packet["root"] / "source.zip").exists()
    with pytest.raises(ValueError, match="assignment|reference|mapping"):
        select(packet)


@pytest.mark.parametrize("mutation", ["duplicate", "conflicting_group_roles", "boolean_row"])
def test_assignment_identity_and_group_roles_fail_closed(packet, mutation):
    if mutation == "duplicate":
        packet["assignments"].append(deepcopy(packet["assignments"][0]))
    elif mutation == "boolean_row":
        packet["assignments"][0]["source_row"] = True
    else:
        packet["assignments"][3]["group_id"] = packet["assignments"][0]["group_id"]
    save_metadata(packet)
    with pytest.raises(ValueError, match="assignment"):
        select(packet)


@pytest.mark.parametrize(
    "mutation", ["review_required", "non_singleton", "source_effective", "overlay"]
)
def test_exclusions_cause_shortfall_instead_of_resampling_reserved_books(packet, mutation):
    assignment, candidate = packet["assignments"][0], packet["candidates"][0]
    if mutation == "review_required":
        assignment["review_required"] = candidate["group"]["review_required"] = True
    elif mutation == "source_effective":
        assignment["proposed_split"] = candidate["group"]["proposed_split"] = "dev"
    elif mutation == "non_singleton":
        packet["assignments"][1]["group_id"] = assignment["group_id"]
        packet["candidates"][1]["group"]["group_id"] = assignment["group_id"]
        packet["groups"]["counts"]["groups"] -= 1
    else:
        nodes = sorted([assignment["group_id"], "other-source:synthetic"])
        packet["components"] = [
            {
                "component_id": "book-component:"
                + hashlib.sha256("\n".join(nodes).encode()).hexdigest(),
                "members": nodes,
                "proposed_split": "train",
                "status": "candidate_not_admitted",
                "training_eligible": False,
            }
        ]
        packet["partition"]["counts"] = {"overlay_components": 1, "overridden_bgc_groups": 1}
    save_metadata(packet)
    with pytest.raises(ValueError, match="shortfall.*no resampling"):
        select(packet)


@pytest.mark.parametrize("mutation", ["input_hash", "source_identity", "source_labels"])
def test_selected_source_must_match_exact_metadata_receipt(packet, monkeypatch, mutation):
    candidate = packet["candidates"][0]
    if mutation == "input_hash":
        candidate["input_sha256"] = "f" * 64
    elif mutation == "source_identity":
        candidate["source"]["provider_book_id"] = "another-book"
    else:
        # Identical mapping, different literal source depth must still fail.
        candidate["source_labels"] = [[1, "Fantasy"]]
    save_metadata(packet)
    audit = write_archive(packet)
    mock_bindings(monkeypatch, packet, audit)
    with pytest.raises(ValueError, match="failed verification.*no resampling"):
        data.prepare_field_baseline_data(packet["root"], {})


@pytest.mark.parametrize("mutation", ["hash", "count", "truncated"])
def test_selected_member_integrity_and_framing_are_checked(packet, mutation):
    _, selected, _ = select(packet)
    if mutation == "truncated":
        packet["source"]["train"][-1] = b"truncated unselected record\n"
    audit = write_archive(packet)
    if mutation == "hash":
        audit["observed"]["members"][MEMBERS["train"]]["sha256"] = "0" * 64
    elif mutation == "count":
        audit["observed"]["splits"]["train"]["rows"] += 1
    with pytest.raises(ValueError, match="changed|framing/count"):
        list(data._selected_source_rows(packet["root"] / "source.zip", selected, audit))


@pytest.mark.parametrize("mutation", ["sha256", "bytes", "escape"])
def test_corrupt_top_level_receipt_rejected_before_json_or_archive_loading(tmp_path, mutation):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    reference = {"path": path.name, "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    if mutation == "sha256":
        reference["sha256"] = "0" * 64
    elif mutation == "bytes":
        reference["bytes"] += 1
    else:
        reference["path"] = "../outside.json"
    with pytest.raises(ValueError, match="changed|relative"):
        data.prepare_field_baseline_data(
            tmp_path, {"field_manifest": reference, "partition_manifest": reference}
        )
