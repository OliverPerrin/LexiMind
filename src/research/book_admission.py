"""Read-only admission of separate future book-relevance receipts.

An empty blocker list means mechanical consistency only, not verified human truth,
consent, research execution authorization, or an evaluation result.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .annotations import query_content_sha256, sha256, validate_future_records, validate_packet
from .io import parse_json, safe_path

_RECEIPTS = {
    "collection_manifest": "book_collection",
    "partition_manifest": "book_partitions",
    "eligibility_manifest": "book_eligibility",
    "rubric_review": "book_rubric_review",
}
_HEADER = {"schema_version", "kind", "purpose", "study_id", "catalog_sha256", "rubric_sha256"}
_PARTITIONS = {"pilot", "development", "test"}
_OTHER_EXCLUSIONS = {"insufficient_evidence", "explicit_query_constraint", "rights_unresolved"}


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name}: expected exactly {sorted(fields)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name}: expected nonempty trimmed text")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected array")
    return value


def _ids(value: Any, name: str) -> set[str]:
    rows = _list(value, name)
    for row in rows:
        _text(row, name)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name}: duplicate ID")
    return set(rows)


def _path(root: Path, value: Any) -> Path:
    return safe_path(root, _text(value, "artifact path"))


def _read_ref(root: Path, reference: Any, kind: str) -> dict[str, Any]:
    ref = _object(reference, {"kind", "path", "sha256", "bytes"}, "artifact reference")
    if ref["kind"] != kind:
        raise ValueError(f"Expected artifact kind {kind}")
    if (
        type(ref["bytes"]) is not int
        or ref["bytes"] < 1
        or not isinstance(ref["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
    ):
        raise ValueError("Artifact requires integer byte length and lowercase SHA-256")
    raw = _path(root, ref["path"]).read_bytes()
    if len(raw) != ref["bytes"] or sha256(raw) != ref["sha256"]:
        raise ValueError(f"Artifact bytes/hash changed: {ref['path']}")
    result = parse_json(raw)
    if not isinstance(result, dict):
        raise ValueError("Receipt must be a JSON object")
    return result


def _header(
    receipt: dict[str, Any],
    kind: str,
    fields: set[str],
    plan: dict[str, Any],
    packet: dict[str, Any],
) -> None:
    _object(receipt, _HEADER | fields, kind)
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("Unsupported book receipt schema")
    if receipt["kind"] != kind or receipt["purpose"] != "book_relevance_primary":
        raise ValueError("Receipt kind/purpose does not describe primary book relevance")
    if receipt["study_id"] != _text(plan.get("study_id"), "study ID"):
        raise ValueError("Receipt belongs to another study")
    if receipt["catalog_sha256"] != packet["catalog"]["sha256"]:
        raise ValueError("Receipt binds a different catalogue")
    expected = {name: row["sha256"] for name, row in packet["rubrics"].items()}
    if receipt["rubric_sha256"] != expected:
        raise ValueError("Receipt binds different rubric bytes")


def _assignments(value: Any, key: str, expected: set[str]) -> dict[str, str]:
    result = {}
    for row in _list(value, f"{key} assignments"):
        _object(row, {key, "partition"}, "partition assignment")
        identifier = _text(row[key], key)
        partition = _text(row["partition"], "partition")
        if identifier in result or partition not in _PARTITIONS:
            raise ValueError("Duplicate assignment or unsupported partition")
        result[identifier] = partition
    if set(result) != expected:
        raise ValueError(f"Assignments must map every known {key} exactly once")
    return result


def _check_admission(root: Path, plan: dict[str, Any], packet: dict[str, Any]) -> None:
    applied = plan["applied_study"]
    receipts = {key: _read_ref(root, applied[key], kind) for key, kind in _RECEIPTS.items()}
    collection = receipts["collection_manifest"]
    partitions = receipts["partition_manifest"]
    eligibility = receipts["eligibility_manifest"]
    review = receipts["rubric_review"]
    _header(
        collection, "book_collection", {"queries", "judgments", "primary_query_ids"}, plan, packet
    )
    _header(
        partitions,
        "book_partitions",
        {
            "generalization",
            "policy",
            "catalogue_work_ids",
            "seed_work_assignments",
            "query_assignments",
            "query_family_assignments",
        },
        plan,
        packet,
    )
    _header(eligibility, "book_eligibility", {"policy", "queries"}, plan, packet)
    _header(review, "book_rubric_review", {"status", "attestation"}, plan, packet)

    queries, judgments = collection["queries"], collection["judgments"]
    validate_future_records(queries, judgments, packet)
    if not queries or not judgments:
        raise ValueError("Separate collection has no independent queries/judgments")
    if any(row["kind"] != "recommendation" for row in judgments):
        raise ValueError("This admission contract covers relevance, not mood gold")
    work_ids = {row["work_id"] for row in packet["items"]}
    query_map = {row["query_id"]: row for row in queries}
    family_ids = {row["query_family_id"] for row in queries}
    seed_ids = {seed["work_id"] for query in queries for seed in query["seed_works"]}
    if (
        partitions["generalization"] != "new_query_families_fixed_catalogue"
        or applied.get("generalization") != partitions["generalization"]
    ):
        raise ValueError("Book admission requires an explicit fixed-catalogue new-query claim")
    if _ids(partitions["catalogue_work_ids"], "catalogue work IDs") != work_ids:
        raise ValueError("Partition receipt must track the complete fixed catalogue")
    policy = partitions["policy"]
    if policy != {
        "status": "frozen",
        "unit": "query_family_and_seed_work",
        "seed_work_overlap": "disjoint",
        "query_family_overlap": "disjoint",
        "seed_policy": "same_partition_as_query",
        "candidate_population": "fixed_catalogue",
    }:
        raise ValueError("Partition policy is not the reviewed seed/family-disjoint policy")
    seeds_by_partition = _assignments(partitions["seed_work_assignments"], "work_id", seed_ids)
    query_parts = _assignments(partitions["query_assignments"], "query_id", set(query_map))
    families = _assignments(partitions["query_family_assignments"], "query_family_id", family_ids)
    content_partitions: dict[str, str] = {}
    for query in queries:
        partition = query_parts[query["query_id"]]
        content = query_content_sha256(query)
        if content in content_partitions and content_partitions[content] != partition:
            raise ValueError(
                "Duplicate canonical query content crosses partitions despite different query/family IDs"
            )
        content_partitions[content] = partition
        if families[query["query_family_id"]] != partition:
            raise ValueError("Query family crosses partitions")
        if any(seeds_by_partition[seed["work_id"]] != partition for seed in query["seed_works"]):
            raise ValueError("Seed work crosses query partition")
    primary = _ids(collection["primary_query_ids"], "primary query IDs")
    if not primary or primary != {key for key, value in query_parts.items() if value == "test"}:
        raise ValueError("Primary queries must be exactly all test-partition queries")

    if eligibility["policy"] != {
        "status": "frozen",
        "population": "all_packet_works",
        "seed_policy": "exclude",
        "candidate_partition_policy": "shared_catalogue",
        "other_exclusions": "explicit_reason",
    }:
        raise ValueError("Eligibility policy is not frozen over the full catalogue")
    eligible: dict[str, set[str]] = {}
    for row in _list(eligibility["queries"], "query eligibility"):
        _object(row, {"query_id", "eligible_work_ids", "excluded_works"}, "query eligibility")
        query_id = _text(row["query_id"], "query ID")
        if query_id not in query_map or query_id in eligible:
            raise ValueError("Unknown or duplicate eligibility query")
        allowed = _ids(row["eligible_work_ids"], "eligible work IDs")
        excluded = {}
        for exclusion in _list(row["excluded_works"], "exclusions"):
            _object(exclusion, {"work_id", "reason", "detail"}, "work exclusion")
            work_id = _text(exclusion["work_id"], "excluded work ID")
            _text(exclusion["detail"], "exclusion detail")
            reason = _text(exclusion["reason"], "exclusion reason")
            if work_id in excluded:
                raise ValueError("Duplicate excluded work")
            excluded[work_id] = reason
        if allowed & excluded.keys() or allowed | excluded.keys() != work_ids:
            raise ValueError("Eligibility/exclusions must partition the complete catalogue")
        seeds = {seed["work_id"] for seed in query_map[query_id]["seed_works"]}
        for work_id in work_ids:
            exclusion_reason = excluded.get(work_id)
            if work_id in seeds:
                if exclusion_reason != "seed_work":
                    raise ValueError("Every seed work must be explicitly excluded")
            elif exclusion_reason is not None and exclusion_reason not in _OTHER_EXCLUSIONS:
                raise ValueError("Unsupported within-partition exclusion")
        if query_id in primary and not allowed:
            raise ValueError("A primary query has no eligible candidates")
        eligible[query_id] = allowed
    if set(eligible) != set(query_map):
        raise ValueError("Every query needs full eligibility accounting")
    for row in judgments:
        if row["work_id"] not in eligible[row["query_id"]]:
            raise ValueError("Judgment refers to an ineligible query/work pair")

    if review["status"] != "frozen":
        raise ValueError("Rubric review is not frozen")
    attestation = _read_ref(root, review["attestation"], "book_human_review")
    _header(
        attestation,
        "book_human_review",
        {
            "source_kind",
            "reviewer_id",
            "completed_at",
            "decision",
            "scope",
            "review_notes",
            "reviewed_artifacts",
            "reviewed_judgment_ids",
            "adjudications",
        },
        plan,
        packet,
    )
    if (
        attestation["source_kind"] != "human_review"
        or attestation["decision"] != "approved_for_primary_relevance"
        or attestation["scope"] != "metadata_supported_relevance_without_whole_work_mood"
    ):
        raise ValueError("Missing human review of the specified relevance scope")
    reviewer = _text(attestation["reviewer_id"], "reviewer ID")
    if not re.fullmatch(r"reviewer:[a-z0-9][a-z0-9._-]{0,95}", reviewer):
        raise ValueError("Malformed reviewer ID")
    _text(attestation["review_notes"], "human review notes")
    completed = datetime.fromisoformat(
        _text(attestation["completed_at"], "review time").replace("Z", "+00:00")
    )
    if completed.tzinfo is None:
        raise ValueError("Human review time needs a timezone")
    expected_refs = {key: applied[key] for key in _RECEIPTS if key != "rubric_review"}
    if attestation["reviewed_artifacts"] != expected_refs:
        raise ValueError("Human review does not bind this collection/partition/eligibility")
    judgment_map = {row["judgment_id"]: row for row in judgments}
    if _ids(attestation["reviewed_judgment_ids"], "reviewed judgment IDs") != set(judgment_map):
        raise ValueError(
            "Review must account for all judgments, including disagreements/abstentions"
        )
    target_pairs = {(query_id, work_id) for query_id in primary for work_id in eligible[query_id]}
    constraint_violations = {
        (row["query_id"], row["work_id"]) for row in judgments if row["constraint_violations"]
    }
    adjudicated = set()
    for row in _list(attestation["adjudications"], "adjudications"):
        _object(
            row,
            {"query_id", "work_id", "relevance", "basis_judgment_ids", "rationale"},
            "adjudication",
        )
        pair = (_text(row["query_id"], "query ID"), _text(row["work_id"], "work ID"))
        if pair not in target_pairs or pair in adjudicated:
            raise ValueError("Unknown or duplicate primary adjudication")
        if type(row["relevance"]) is not int or row["relevance"] not in range(4):
            raise ValueError("Primary adjudication needs an integer grade; null is not zero")
        if row["relevance"] > 0 and pair in constraint_violations:
            raise ValueError(
                "Positive adjudication cannot discard retained hard-constraint violations; explicit constraint-resolution evidence is required"
            )
        _text(row["rationale"], "adjudication rationale")
        raters = set()
        for identifier in _ids(row["basis_judgment_ids"], "adjudication evidence IDs"):
            evidence = judgment_map.get(identifier)
            if evidence is None or (evidence["query_id"], evidence["work_id"]) != pair:
                raise ValueError("Adjudication cites an unknown or different query/work judgment")
            if evidence["relevance"] is None:
                raise ValueError("An abstention cannot supply a primary relevance grade")
            raters.add(evidence["rater_id"])
        if len(raters) < 2:
            raise ValueError("Primary pair lacks two independent non-abstaining rater records")
        adjudicated.add(pair)
    if adjudicated != target_pairs:
        raise ValueError("Primary relevance coverage is incomplete for the eligible catalogue")


def validate_book_admission(root: Path, plan: dict, packet: dict) -> list[str]:
    """Return blockers for future applied receipts; never alter the empty packet."""
    try:
        validate_packet(
            packet,
            _path(root, packet["catalog"]["path"]),
            {name: _path(root, ref["path"]) for name, ref in packet["rubrics"].items()},
            root=root,
        )
        applied = plan.get("applied_study")
        if not isinstance(applied, dict):
            return ["Book admission: missing applied_study object"]
        missing = [
            f"Book admission: missing {key}" for key in _RECEIPTS if applied.get(key) is None
        ]
        if missing:
            return missing
        _check_admission(root, plan, packet)
        return []
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"Book admission: {exc}"]
