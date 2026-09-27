"""Evidence-bound candidate reviews; missing decisions remain unknown, never negative."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from src.research.book_fields import FACETS, STATES, input_hash, validate_states


def evidence_span(payload: dict[str, str], field: str, start: int, end: int) -> dict[str, Any]:
    """Refer to literal input characters without redistributing the source passage."""
    if (
        not isinstance(field, str)
        or field not in {"title", "description"}
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(payload[field])
    ):
        raise ValueError("Evidence requires an in-range nonempty input character span")
    return {
        "input_field": field,
        "start": start,
        "end": end,
        "sha256": hashlib.sha256(payload[field][start:end].encode("utf-8")).hexdigest(),
    }


def _review(value: Any, kind: str, candidate: dict, payload: dict, mapping: dict) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"reviewer", "decisions"}:
        raise ValueError("Review must contain reviewer provenance and decisions")
    reviewer = value["reviewer"]
    if (
        not isinstance(reviewer, dict)
        or set(reviewer) != {"kind", "id", "method", "reviewed_at"}
        or reviewer["kind"] != kind
        or not isinstance(reviewer["id"], str)
        or not reviewer["id"].strip()
        or reviewer["method"]
        != {"agent": "assistant_source_inspection", "human": "direct_source_review"}[kind]
        or not isinstance(reviewer["reviewed_at"], str)
    ):
        raise ValueError("Review requires correctly separated agent/human provenance")
    try:
        reviewed_date = date.fromisoformat(reviewer["reviewed_at"])
    except ValueError as error:
        raise ValueError("Review date must be an ISO calendar date") from error
    if reviewed_date.isoformat() != reviewer["reviewed_at"]:
        raise ValueError("Review date must be an ISO calendar date")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("An unfilled review must be null, not an empty completed review")
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
            "field",
            "label",
            "state",
            "evidence",
            "rationale",
        }:
            raise ValueError("Decision requires label, state, evidence and rationale only")
        field, label, state = decision["field"], decision["label"], decision["state"]
        if (
            not isinstance(field, str)
            or field not in FACETS
            or not isinstance(label, str)
            or label not in mapping["facets"][field]["labels"]
            or not isinstance(state, str)
            or state not in STATES
        ):
            raise ValueError("Unknown review field, label or state")
        if (field, label) in seen:
            raise ValueError("Repeated or contradictory review decision")
        seen.add((field, label))
        if state == "negative" and label in candidate["fields"][field]["positive"]:
            raise ValueError("A reviewed negative cannot override a source positive")
        if state == "positive" and label in candidate["fields"][field]["negative"]:
            raise ValueError("A reviewed positive cannot override a source negative")
        if not isinstance(decision["rationale"], str) or not decision["rationale"].strip():
            raise ValueError("Every review decision requires a rationale")
        spans = decision["evidence"]
        if not isinstance(spans, list) or not spans:
            raise ValueError("Every decision, including abstentions, requires source evidence")
        for span in spans:
            if not isinstance(span, dict) or set(span) != {"input_field", "start", "end", "sha256"}:
                raise ValueError("Evidence stores offsets and digest only, never copied text")
            expected = evidence_span(payload, span["input_field"], span["start"], span["end"])
            if span != expected:
                raise ValueError("Evidence span hash changed")


def validate_review_record(review: dict, candidate: dict, payload: dict, mapping: dict) -> None:
    """Validate provenance and literal evidence; this cannot adjudicate semantic accuracy."""
    if not isinstance(review, dict) or set(review) != {
        "record_id",
        "input_sha256",
        "source_url",
        "selection",
        "agent_review",
        "human_review",
    }:
        raise ValueError("Review record requires references and separate agent/human slots only")
    if (
        review["record_id"] != candidate["record_id"]
        or review["source_url"] != candidate["source"]["provider_url"]
        or review["input_sha256"] != candidate["input_sha256"]
        or review["input_sha256"] != input_hash(payload)
    ):
        raise ValueError("Review record or input hash is stale")
    selection = review["selection"]
    if (
        not isinstance(selection, list)
        or not selection
        or any(not isinstance(value, str) or not value.strip() for value in selection)
        or len(set(selection)) != len(selection)
    ):
        raise ValueError("Review record requires unique selection reasons")
    validate_states(candidate["fields"], mapping)
    _review(review["agent_review"], "agent", candidate, payload, mapping)
    _review(review["human_review"], "human", candidate, payload, mapping)


def human_reviewed_states(review: dict, candidate: dict, payload: dict, mapping: dict) -> dict:
    """Extract explicit human candidate labels, without admitting data or copying agent labels."""
    validate_review_record(review, candidate, payload, mapping)
    states: dict[str, dict[str, list[str]]] = {
        field: {"positive": [], "negative": []} for field in FACETS
    }
    if review["human_review"] is not None:
        for decision in review["human_review"]["decisions"]:
            if decision["state"] != "unknown":
                states[decision["field"]][decision["state"]].append(decision["label"])
    for values in states.values():
        for labels in values.values():
            labels.sort()
    validate_states(states, mapping)
    return states
