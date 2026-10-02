"""Candidate review provenance and source spans; no private archive or human-label fabrication."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.research.book_fields import FACETS, input_hash, label_state
from src.research.builders import book_fields as fields
from src.research.builders import field_review as prep
from src.research.candidate_io import json_bytes
from src.research.field_reviews import evidence_span, human_reviewed_states, validate_review_record
from src.research.io import read_json
from tests.test_research.test_book_fields import mapping as mapping
from tests.test_research.test_book_fields import prepared_source as prepared_source

ROOT = Path(__file__).resolve().parents[2]


def candidate_review(payload=None):
    payload = payload or {
        "title": "Synthetic & café",
        "description": "A story for adults; explicitly not for children.",
    }
    candidate = {
        "record_id": "bgc:train:1",
        "input_sha256": input_hash(payload),
        "source": {"provider_url": "https://example.com/book/1"},
        "fields": {field: {"positive": [], "negative": []} for field in FACETS},
    }
    candidate["fields"]["genre"]["positive"] = ["fantasy"]
    review = {
        "record_id": candidate["record_id"],
        "input_sha256": candidate["input_sha256"],
        "source_url": candidate["source"]["provider_url"],
        "selection": ["synthetic review case"],
        "agent_review": {
            "reviewer": {
                "kind": "agent",
                "id": "test-agent",
                "method": "assistant_source_inspection",
                "reviewed_at": "2026-09-27",
            },
            "decisions": [
                {
                    "field": "audience",
                    "label": "children",
                    "state": "negative",
                    "evidence": [
                        evidence_span(payload, "description", 0, len(payload["description"]))
                    ],
                    "rationale": "Synthetic explicit audience denial, not a missing source tag.",
                }
            ],
        },
        "human_review": None,
    }
    return review, candidate, payload


def test_agent_decisions_never_become_human_training_evidence(mapping):
    review, candidate, payload = candidate_review()
    validate_review_record(review, candidate, payload, mapping)
    states = human_reviewed_states(review, candidate, payload, mapping)
    assert label_state(states, "audience", "children", mapping) == "unknown"
    assert label_state(states, "genre", "fantasy", mapping) == "unknown"
    assert all(not labels for facet in states.values() for labels in facet.values())
    assert review["human_review"] is None


def test_only_explicit_human_decisions_are_extracted_without_source_or_agent_defaults(mapping):
    review, candidate, payload = candidate_review()
    review["human_review"] = copy.deepcopy(review["agent_review"])
    review["human_review"]["reviewer"].update(
        kind="human", id="synthetic-reviewer", method="direct_source_review"
    )
    states = human_reviewed_states(review, candidate, payload, mapping)
    assert states["audience"] == {"positive": [], "negative": ["children"]}
    assert label_state(states, "genre", "fantasy", mapping) == "unknown"
    assert label_state(states, "audience", "teen_young_adult", mapping) == "unknown"
    review["human_review"]["decisions"][0]["state"] = "unknown"
    assert human_reviewed_states(review, candidate, payload, mapping)["audience"]["negative"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "input_hash",
        "source_url",
        "input",
        "bool_offset",
        "invalid_input_field",
        "out_of_range",
        "empty_span",
        "span_hash",
        "copied_text",
        "no_evidence",
        "no_rationale",
        "unknown_label",
        "unknown_state",
        "duplicate",
        "contradiction",
        "source_positive_conflict",
        "source_negative_conflict",
        "agent_in_human_slot",
        "missing_reviewer",
        "bad_date",
        "wrong_method",
        "empty_review",
        "extra_text",
        "repeated_selection",
        "malformed_review",
    ],
)
def test_invalid_or_stale_review_is_rejected(mapping, mutation):
    review, candidate, payload = candidate_review()
    decision = review["agent_review"]["decisions"][0]
    span = decision["evidence"][0]
    if mutation == "id":
        review["record_id"] = "bgc:train:2"
    elif mutation == "input_hash":
        review["input_sha256"] = "0" * 64
    elif mutation == "source_url":
        review["source_url"] = "https://example.com/different"
    elif mutation == "input":
        payload["description"] += " changed"
    elif mutation == "invalid_input_field":
        span["input_field"] = []
    elif mutation == "bool_offset":
        span["start"] = False
    elif mutation == "out_of_range":
        span["end"] += 1
    elif mutation == "empty_span":
        span["end"] = span["start"]
    elif mutation == "span_hash":
        span["sha256"] = "0" * 64
    elif mutation == "copied_text":
        span["text"] = "unwanted source text"
    elif mutation == "no_evidence":
        decision["evidence"] = []
    elif mutation == "no_rationale":
        decision["rationale"] = " "
    elif mutation == "unknown_label":
        decision["label"] = "adult_default"
    elif mutation == "unknown_state":
        decision["state"] = "missing_means_negative"
    elif mutation in {"duplicate", "contradiction"}:
        review["agent_review"]["decisions"].append(
            {**decision, "state": "positive" if mutation == "contradiction" else "negative"}
        )
    elif mutation == "source_positive_conflict":
        candidate["fields"]["audience"]["positive"] = ["children"]
    elif mutation == "source_negative_conflict":
        candidate["fields"]["audience"]["negative"] = ["children"]
        decision["state"] = "positive"
    elif mutation == "agent_in_human_slot":
        review["human_review"] = review["agent_review"]
    elif mutation == "missing_reviewer":
        del review["agent_review"]["reviewer"]["id"]
    elif mutation == "bad_date":
        review["agent_review"]["reviewer"]["reviewed_at"] = "2026-02-30"
    elif mutation == "wrong_method":
        review["agent_review"]["reviewer"]["method"] = "paid_teacher"
    elif mutation == "empty_review":
        review["agent_review"]["decisions"] = []
    elif mutation == "extra_text":
        review["description"] = payload["description"]
    elif mutation == "repeated_selection":
        review["selection"] *= 2
    else:
        review["agent_review"] = []
    with pytest.raises(ValueError):
        validate_review_record(review, candidate, payload, mapping)


def test_unicode_character_offsets_are_not_byte_offsets(mapping):
    review, candidate, payload = candidate_review()
    position = payload["title"].index("é")
    decision = review["agent_review"]["decisions"][0]
    decision["state"] = "unknown"
    decision["evidence"] = [evidence_span(payload, "title", position, position + 1)]
    validate_review_record(review, candidate, payload, mapping)
    assert decision["evidence"][0]["end"] - decision["evidence"][0]["start"] == 1


def test_selection_is_bounded_deterministic_and_retains_empty_human_slots():
    candidates = [
        {
            "record_id": f"synthetic:{index}",
            "input_sha256": "a" * 64,
            "source_labels": [[0, label]],
            "source": {"provider_url": "https://example.com/book"},
        }
        for index, label in enumerate(prep.COVERAGE_LABELS)
    ]
    candidates.extend(
        {
            "record_id": key,
            "input_sha256": "b" * 64,
            "source_labels": [],
            "source": {"provider_url": "https://example.com/book"},
        }
        for key in prep.NEGATION_RECORDS
    )
    rows = prep.select_records(iter(candidates))
    assert len(rows) == 32
    assert rows == prep.select_records(iter(candidates))
    assert all(row["human_review"] is row["agent_review"] is None for row in rows)
    with pytest.raises(ValueError, match="every"):
        prep.select_records(iter(candidates[:-1]))


@pytest.fixture
def prepared_review(prepared_source):
    root, _, _, _ = prepared_source
    manifest = fields.prepare_fields(*prepared_source)
    manifest_path = root / "research/preparation/book_field_manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    for name in ("src/research/builders/book_fields.py", "src/research/field_reviews.py"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic: " + name)
    item = next(fields.iter_resolved_candidates(root, manifest))
    candidate, payload = item["candidate"], item["input"]
    review, _, _ = candidate_review(payload)
    review.update(
        record_id=candidate["record_id"],
        input_sha256=candidate["input_sha256"],
        source_url=candidate["source"]["provider_url"],
    )
    packet = {
        "schema_version": 1,
        "status": "candidate_field_reviews_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "bindings": prep._bindings(root, manifest_path, manifest),
        "contract": prep.CONTRACT,
        "selection": "Synthetic integration record",
        "records": [review],
    }
    packet_path = root / "research/preparation/book_field_review.json"
    packet_path.write_bytes(json_bytes(packet))
    return root, packet_path, manifest_path, root / "data/research_candidates/bgc/review"


def test_preparation_replays_locally_without_copying_source_text_to_report(prepared_review):
    root, packet_path, _, output = prepared_review
    report = prep.prepare_review(*prepared_review)
    assert report == prep.prepare_review(*prepared_review)
    assert report["observed"]["decisions"]["agent"]["negative"] == 1
    assert report["observed"]["explicit_human_labels"] == 0
    assert report["observed"]["reviewed_records"] == {"agent": 1, "human": 0}
    assert report["training_authorized"] is report["human_gold"] is False
    assert b"made-up blurb" not in json_bytes(report) + packet_path.read_bytes()
    worksheet = root / report["local_worksheet"]["path"]
    assert b"made-up blurb" in worksheet.read_bytes()
    assert list(output.iterdir()) == [worksheet]
    worksheet.write_bytes(b"preexisting work")
    with pytest.raises(ValueError, match="differs"):
        prep.prepare_review(*prepared_review)
    assert worksheet.read_bytes() == b"preexisting work"


@pytest.mark.parametrize(
    "mutation",
    [
        "stale_source",
        "stale_record",
        "duplicate",
        "too_many",
        "unknown_means_negative",
        "training",
        "gold",
        "extra_text",
        "empty",
        "unresolved",
    ],
)
def test_invalid_packet_never_publishes_worksheet(prepared_review, mutation):
    _, packet_path, _, output = prepared_review
    packet = read_json(packet_path)
    if mutation == "stale_source":
        packet["bindings"]["archive"]["sha256"] = "0" * 64
    elif mutation == "stale_record":
        packet["records"][0]["input_sha256"] = "0" * 64
    elif mutation == "duplicate":
        packet["records"] *= 2
    elif mutation == "too_many":
        packet["records"] *= 49
    elif mutation == "unknown_means_negative":
        packet["contract"]["default_state"] = "negative"
    elif mutation == "training":
        packet["training_authorized"] = True
    elif mutation == "gold":
        packet["human_gold"] = True
    elif mutation == "extra_text":
        packet["text"] = "unwanted"
    elif mutation == "empty":
        packet["records"] = []
    else:
        packet["records"][0]["record_id"] = "bgc:train:999"
    packet_path.write_bytes(json_bytes(packet))
    with pytest.raises(ValueError):
        prep.prepare_review(*prepared_review)
    assert not output.exists()


def test_resolved_text_cannot_be_published_in_tracked_directory(prepared_review):
    root, packet_path, manifest_path, _ = prepared_review
    with pytest.raises(ValueError, match="ignored"):
        prep.prepare_review(root, packet_path, manifest_path, root / "research/preparation/text")


def test_committed_packet_has_no_human_labels_or_training_permission():
    packet = read_json(ROOT / "research/preparation/book_field_review.json")
    assert packet["training_authorized"] is packet["human_gold"] is False
    assert len(packet["records"]) == 32
    assert all(row["human_review"] is None for row in packet["records"])
    decisions = [
        decision for row in packet["records"] for decision in row["agent_review"]["decisions"]
    ]
    assert sum(d["state"] == "negative" for d in decisions) == 1
    austen = next(row for row in packet["records"] if row["record_id"] == "bgc:train:6018")
    assert austen["agent_review"]["decisions"][0]["state"] == "unknown"


def test_packet_changed_during_resolution_never_publishes(prepared_review, monkeypatch):
    _, packet_path, _, output = prepared_review
    resolve = prep.iter_resolved_candidates

    def changed_packet(*args):
        for row in resolve(*args):
            packet_path.write_bytes(packet_path.read_bytes() + b"\n")
            yield row

    monkeypatch.setattr(prep, "iter_resolved_candidates", changed_packet)
    with pytest.raises(ValueError, match="changed"):
        prep.prepare_review(*prepared_review)
    assert not output.exists()
