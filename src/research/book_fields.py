"""Authored book-field mapping and sparse partial-label states, without model imports."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

FACETS = ("genre", "topic", "form", "audience")
STATES = {"positive", "negative", "unknown"}


def input_payload(row: dict[str, Any]) -> dict[str, str]:
    """The sole classifier input: literal source title and blurb, never metadata."""
    return {"title": row["title"], "description": row["body"]}


def input_hash(payload: dict[str, str]) -> str:
    if set(payload) != {"title", "description"} or any(
        not isinstance(value, str) for value in payload.values()
    ):
        raise ValueError("Book input must contain only string title and description")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_mapping(
    mapping: dict[str, Any], *, source_labels: set[str], archive_sha256: str, hierarchy_sha256: str
) -> None:
    """Bind the complete exact-label mapping to one audited source taxonomy."""
    if (
        type(mapping.get("schema_version")) is not int
        or mapping["schema_version"] != 1
        or mapping.get("status") != "authored_weak_candidate_not_human_gold"
        or mapping.get("source_id") != "bgc"
        or mapping.get("source_archive_sha256") != archive_sha256
        or mapping.get("source_hierarchy_sha256") != hierarchy_sha256
    ):
        raise ValueError("Mapping source/schema/status does not match the pinned BGC archive")
    semantics = mapping.get("semantics", {})
    if (
        semantics.get("default_state") != "unknown"
        or semantics.get("source_omissions") != "unknown"
        or set(semantics.get("allowed_states", [])) != STATES
        or semantics.get("preparation_only") is not True
    ):
        raise ValueError("Mapping must retain unknown omissions and preparation-only states")
    facets = mapping.get("facets", {})
    if set(facets) != set(FACETS):
        raise ValueError("Book mapping must separate genre, topic, form and audience")
    for facet in facets.values():
        labels = facet.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(
                not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z_]*", label)
                for label in labels
            )
            or len(set(labels)) != len(labels)
            or not isinstance(facet.get("definition"), str)
            or not facet["definition"].strip()
        ):
            raise ValueError("Field vocabularies require unique label IDs and a definition")
    decisions = mapping.get("mappings", {})
    if set(decisions) != source_labels:
        raise ValueError("Mapping must cover every observed source label exactly")
    for entry in decisions.values():
        if set(entry) != {"decision", "positive", "note"}:
            raise ValueError("Mapping entries require decision, positive targets and note only")
        positive = entry["positive"]
        if not isinstance(positive, dict) or set(positive) - set(FACETS):
            raise ValueError("Unknown mapped field")
        if entry["decision"] not in {"mapped", "ambiguous", "unsupported"}:
            raise ValueError("Unknown source-label decision")
        if (entry["decision"] == "mapped") != bool(positive):
            raise ValueError("Only mapped source labels can supply positives")
        if not isinstance(entry["note"], str) or not entry["note"].strip():
            raise ValueError("Every mapping decision needs its scope/rationale")
        for field, values in positive.items():
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) for value in values)
                or len(set(values)) != len(values)
                or set(values) - set(facets[field]["labels"])
            ):
                raise ValueError("Mapped targets must be unique members of their field vocabulary")


def validate_states(states: dict[str, Any], mapping: dict[str, Any]) -> None:
    """Sparse unlisted labels mean unknown; explicit negatives are never inferred."""
    if set(states) != set(FACETS):
        raise ValueError("Partial labels require all four independent fields")
    for field, values in states.items():
        if not isinstance(values, dict) or set(values) != {"positive", "negative"}:
            raise ValueError("Field states require positive and negative lists")
        vocabulary = set(mapping["facets"][field]["labels"])
        for entries in values.values():
            if (
                not isinstance(entries, list)
                or any(not isinstance(entry, str) for entry in entries)
                or len(set(entries)) != len(entries)
                or set(entries) - vocabulary
            ):
                raise ValueError("Unknown or repeated partial-label value")
        if set(values["positive"]) & set(values["negative"]):
            raise ValueError("A label cannot be both positive and negative")


def label_state(states: dict[str, Any], field: str, label: str, mapping: dict[str, Any]) -> str:
    validate_states(states, mapping)
    if field not in FACETS or label not in mapping["facets"][field]["labels"]:
        raise ValueError("Unknown book field or label")
    for state in ("positive", "negative"):
        if label in states[field][state]:
            return state
    return "unknown"


def map_source_labels(labels: list[tuple[int, str]], mapping: dict[str, Any]) -> dict[str, Any]:
    """Map only present exact source labels; no ancestor closure, mood or negatives."""
    positives: dict[str, set[str]] = {field: set() for field in FACETS}
    for _, label in labels:
        if label not in mapping["mappings"]:
            raise ValueError(f"Unmapped BGC source label: {label}")
        for field, targets in mapping["mappings"][label]["positive"].items():
            positives[field].update(targets)
    return {field: {"positive": sorted(positives[field]), "negative": []} for field in FACETS}
