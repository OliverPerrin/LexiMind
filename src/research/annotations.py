"""Unlabelled annotation preparation and schema checks, using no model libraries.

Passing these checks proves structure/provenance consistency, never human truth,
consent, source rights, annotator competence, or eligibility as research gold.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .io import parse_json
from .io import read_json as read_json

RECORD_HASH_FORMAT = "sha256:json-sort-keys-utf8-ensure-ascii-false-compact-v1"
_WORK = re.compile(r"OL[1-9][0-9]*W\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = r"[a-z0-9][a-z0-9._-]{0,95}"
_MOODS = {
    "reflective",
    "playful",
    "tense",
    "unsettling",
    "melancholic",
    "hopeful",
    "bleak",
    "wonder-filled",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_record_bytes(record: dict[str, Any]) -> bytes:
    """Hash the whole metadata record, not just the displayed reference fields."""
    return json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _fields(value: Any, required: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name}: expected exactly fields {sorted(required)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name}: expected nonempty trimmed text")
    return value


def _identifier(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(prefix + ":" + _SLUG, value):
        raise ValueError(f"Malformed {prefix} ID")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name}: expected lowercase SHA-256")
    return value


def _work(value: Any) -> str:
    if not isinstance(value, str) or not _WORK.fullmatch(value):
        raise ValueError("Malformed Open Library work ID")
    return value


def _strings(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name}: expected {'nonempty ' if nonempty else ''}list")
    for entry in value:
        _text(entry, name)
    if len(value) != len(set(value)):
        raise ValueError(f"{name}: duplicate entries")
    return value


def _url(value: Any) -> str:
    url = _text(value, "source URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(character.isspace() for character in url)
    ):
        raise ValueError("Source URL must be HTTPS without credentials or whitespace")
    return url


def _reference(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_annotation_packet(
    catalog_path: Path, rubric_paths: dict[str, Path], *, root: Path
) -> dict[str, Any]:
    """Reference every work; do not select splits, write queries, or assign labels."""
    if set(rubric_paths) != {"recommendation", "mood"}:
        raise ValueError("Both recommendation and mood rubric paths are required")
    catalog_bytes = catalog_path.read_bytes()
    records = parse_json(catalog_bytes)
    if not isinstance(records, list) or not records:
        raise ValueError("Catalogue must be a nonempty JSON array")
    items = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Catalogue item must be an object")
        work_id = _work(record.get("id"))
        if work_id in seen:
            raise ValueError(f"Duplicate catalogue work: {work_id}")
        seen.add(work_id)
        identifiers = record.get("identifiers")
        if (
            not isinstance(identifiers, dict)
            or identifiers.get("openLibraryWork") != f"/works/{work_id}"
        ):
            raise ValueError(f"Inconsistent catalogue identity: {work_id}")
        source = record.get("source")
        if not isinstance(source, dict):
            raise ValueError("Catalogue source must be an object")
        urls = {_url(source.get("url"))}
        if record.get("descriptionSource") is not None:
            urls.add(_url(record["descriptionSource"]))
        revision = record.get("sourceRevision")
        if type(revision) is not int or revision < 1:
            raise ValueError("Source revision must be a positive integer")
        items.append(
            {
                "work_id": work_id,
                "record_sha256": sha256(canonical_record_bytes(record)),
                "source_urls": sorted(urls),
                "source_revision": revision,
                "source_content_sha256": _digest(record.get("sourceContentHash"), "source hash"),
            }
        )
    return {
        "schema_version": 1,
        "status": "unlabelled_preparation",
        "collection_authorized": False,
        "gold_verified": False,
        "human_review_required": True,
        "catalog": {
            "path": _reference(catalog_path, root),
            "sha256": sha256(catalog_bytes),
            "bytes": len(catalog_bytes),
            "record_count": len(records),
            "record_hash_format": RECORD_HASH_FORMAT,
        },
        "rubrics": {
            name: {
                "path": _reference(path, root),
                "sha256": sha256(path.read_bytes()),
                "status": "draft",
            }
            for name, path in sorted(rubric_paths.items())
        },
        "items": sorted(items, key=lambda item: item["work_id"]),
        "queries": [],
        "judgments": [],
    }


def validate_packet(
    packet: dict[str, Any], catalog_path: Path, rubric_paths: dict[str, Path], *, root: Path
) -> None:
    """Verify an empty preparation packet against the actual source bytes."""
    expected = build_annotation_packet(catalog_path, rubric_paths, root=root)
    # JSON serialization distinguishes bool/int and int/float, unlike Python equality.
    if canonical_record_bytes(packet) != canonical_record_bytes(expected):
        raise ValueError("Packet differs from its empty preparation schema or current source bytes")


def _context(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _fields(
        packet,
        {
            "schema_version",
            "status",
            "collection_authorized",
            "gold_verified",
            "human_review_required",
            "catalog",
            "rubrics",
            "items",
            "queries",
            "judgments",
        },
        "packet",
    )
    if (
        type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
        or packet.get("status") != "unlabelled_preparation"
        or packet.get("collection_authorized") is not False
        or packet.get("gold_verified") is not False
        or packet.get("human_review_required") is not True
        or packet["queries"] != []
        or packet["judgments"] != []
    ):
        raise ValueError("Expected an unlabelled preparation packet, not an approval receipt")
    catalog = _fields(
        packet["catalog"],
        {
            "path",
            "sha256",
            "bytes",
            "record_count",
            "record_hash_format",
        },
        "catalogue reference",
    )
    _text(catalog["path"], "catalogue path")
    _digest(catalog["sha256"], "catalogue hash")
    if (
        type(catalog["bytes"]) is not int
        or catalog["bytes"] < 1
        or type(catalog["record_count"]) is not int
        or catalog["record_count"] < 1
        or catalog["record_hash_format"] != RECORD_HASH_FORMAT
        or not isinstance(packet["items"], list)
        or len(packet["items"]) != catalog["record_count"]
    ):
        raise ValueError("Invalid catalogue size or hash format")
    _fields(packet["rubrics"], {"recommendation", "mood"}, "rubric references")
    for rubric in packet["rubrics"].values():
        _fields(rubric, {"path", "sha256", "status"}, "rubric reference")
        _text(rubric["path"], "rubric path")
        _digest(rubric["sha256"], "rubric hash")
        if rubric["status"] != "draft":
            raise ValueError("Preparation rubrics must remain draft")
    items = {}
    for item in packet["items"]:
        _fields(
            item,
            {
                "work_id",
                "record_sha256",
                "source_urls",
                "source_revision",
                "source_content_sha256",
            },
            "item reference",
        )
        work_id = _work(item["work_id"])
        _digest(item["record_sha256"], "record hash")
        _digest(item["source_content_sha256"], "source hash")
        if type(item["source_revision"]) is not int or item["source_revision"] < 1:
            raise ValueError("Invalid source revision")
        for url in _strings(item["source_urls"], "item source URLs", nonempty=True):
            _url(url)
        if work_id in items:
            raise ValueError("Duplicate packet work ID")
        items[work_id] = item
    return items


def _origin(record: dict[str, Any], allow_test_fixtures: bool) -> None:
    if record["source_kind"] == "human" and record["fixture_only"] is False:
        return
    if (
        allow_test_fixtures
        and record["source_kind"] == "test_fixture"
        and record["fixture_only"] is True
    ):
        return
    raise ValueError(
        "Only human-origin records are eligible; illustrative/generated data is not gold"
    )


def _pins(record: dict[str, Any], packet: dict[str, Any], rubric: str) -> None:
    for field, expected in (
        ("catalog_sha256", packet["catalog"]["sha256"]),
        ("rubric_sha256", packet["rubrics"][rubric]["sha256"]),
    ):
        if _digest(record[field], field) != expected:
            raise ValueError(f"Stale or incorrect {field}")


def _work_reference(reference: dict[str, Any], items: dict[str, Any]) -> None:
    work_id = _work(reference["work_id"])
    if work_id not in items:
        raise ValueError(f"Unknown work ID: {work_id}")
    if _digest(reference["record_sha256"], "record hash") != items[work_id]["record_sha256"]:
        raise ValueError("Stale or incorrect work record hash")


def query_content_sha256(record: dict[str, Any]) -> str:
    """Fingerprint validated query content independently of arbitrary record IDs.

    Preserve case and punctuation, normalize NFC/whitespace, and account for seed
    context. Constraint IDs and ordering cannot disguise the same requirements.
    This catches exact canonical duplicates, not semantic paraphrases/families.
    """

    def normalized(value: str) -> str:
        return " ".join(unicodedata.normalize("NFC", value).split())

    content = {
        "text": normalized(record["text"]),
        "intent": normalized(record["intent"]),
        "constraints": sorted({normalized(item["description"]) for item in record["constraints"]}),
        "seed_work_ids": sorted(seed["work_id"] for seed in record["seed_works"]),
    }
    return sha256(canonical_record_bytes(content))


def validate_query(
    record: dict[str, Any], packet: dict[str, Any], *, allow_test_fixtures: bool = False
) -> None:
    """Check a future query schema; does not authorize or verify its collection."""
    _fields(
        record,
        {
            "query_id",
            "query_family_id",
            "text",
            "intent",
            "constraints",
            "seed_works",
            "catalog_sha256",
            "rubric_sha256",
            "author_id",
            "source_kind",
            "fixture_only",
        },
        "query",
    )
    items = _context(packet)
    _origin(record, allow_test_fixtures)
    _pins(record, packet, "recommendation")
    _identifier(record["query_id"], "query")
    _identifier(record["query_family_id"], "family")
    _identifier(record["author_id"], "author")
    _text(record["text"], "query text")
    _text(record["intent"], "query intent")
    if not isinstance(record["constraints"], list) or not isinstance(record["seed_works"], list):
        raise ValueError("Constraints and seed works must be arrays")
    constraints = set()
    for constraint in record["constraints"]:
        _fields(constraint, {"constraint_id", "description"}, "constraint")
        constraint_id = _identifier(constraint["constraint_id"], "constraint")
        _text(constraint["description"], "constraint description")
        if constraint_id in constraints:
            raise ValueError("Duplicate constraint ID")
        constraints.add(constraint_id)
    seeds = set()
    for seed in record["seed_works"]:
        _fields(seed, {"work_id", "record_sha256"}, "seed work")
        _work_reference(seed, items)
        if seed["work_id"] in seeds:
            raise ValueError("Duplicate seed work")
        seeds.add(seed["work_id"])


def _evidence(value: Any) -> dict[str, Any]:
    evidence = _fields(
        value,
        {
            "scope",
            "reading_coverage",
            "source_urls",
            "locations",
            "edition_id",
            "language",
        },
        "evidence",
    )
    coverage = {
        "metadata": {"metadata_only"},
        "description": {"description_only"},
        "excerpt": {"partial"},
        "whole_work": {"complete"},
    }
    if (
        not isinstance(evidence["scope"], str)
        or not isinstance(evidence["reading_coverage"], str)
        or evidence["scope"] not in coverage
        or evidence["reading_coverage"] not in coverage[evidence["scope"]]
    ):
        raise ValueError("Evidence scope and reading coverage disagree")
    for url in _strings(evidence["source_urls"], "evidence URLs", nonempty=True):
        _url(url)
    _strings(
        evidence["locations"],
        "evidence locations",
        nonempty=evidence["scope"] in {"excerpt", "whole_work"},
    )
    language = evidence["language"]
    if not isinstance(language, str) or not re.fullmatch(
        r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language
    ):
        raise ValueError("Evidence language must be a language tag")
    edition = evidence["edition_id"]
    if edition is not None and (
        not isinstance(edition, str)
        or not re.fullmatch(
            r"(?:OL[1-9][0-9]*M|isbn:(?:[0-9]{10}|[0-9]{13})|edition:" + _SLUG + ")", edition
        )
    ):
        raise ValueError("Malformed edition identifier")
    if evidence["scope"] in {"excerpt", "whole_work"} and edition is None:
        raise ValueError("Reading evidence requires an edition/translation identifier")
    return evidence


def validate_judgment(
    record: dict[str, Any],
    packet: dict[str, Any],
    queries: list[dict[str, Any]],
    *,
    allow_test_fixtures: bool = False,
) -> None:
    """Validate future judgment structure, never certify its truth as gold."""
    if not isinstance(record, dict):
        raise ValueError("Judgment must be an object")
    common = {
        "judgment_id",
        "kind",
        "work_id",
        "record_sha256",
        "catalog_sha256",
        "rubric_sha256",
        "rater_id",
        "source_kind",
        "fixture_only",
        "rationale",
        "evidence",
    }
    kind = record.get("kind")
    if kind == "recommendation":
        extra = {"query_id", "relevance", "abstention_reason", "constraint_violations"}
    elif kind == "mood":
        extra = {"mood_observations", "evidence_status", "eligible_for_whole_work_gold"}
    else:
        raise ValueError("Unknown judgment kind")
    _fields(record, common | extra, "judgment")
    items = _context(packet)
    _origin(record, allow_test_fixtures)
    _pins(record, packet, kind)
    _work_reference(record, items)
    _identifier(record["judgment_id"], "judgment")
    _identifier(record["rater_id"], "rater")
    _text(record["rationale"], "rationale")
    evidence = _evidence(record["evidence"])
    if kind == "recommendation":
        query_map = {}
        for query in queries:
            validate_query(query, packet, allow_test_fixtures=allow_test_fixtures)
            if query["query_id"] in query_map:
                raise ValueError("Duplicate query ID")
            query_map[query["query_id"]] = query
        query_id = _identifier(record["query_id"], "query")
        if query_id not in query_map:
            raise ValueError("Unknown query ID")
        relevance = record["relevance"]
        if relevance is None:
            _text(record["abstention_reason"], "abstention reason")
        elif type(relevance) is not int or relevance not in range(4):
            raise ValueError("Relevance must be null or an integer from 0 to 3, never bool/float")
        elif record["abstention_reason"] is not None:
            raise ValueError("A scored judgment cannot also be an abstention")
        allowed = {c["constraint_id"] for c in query_map[query_id]["constraints"]}
        violations = _strings(record["constraint_violations"], "constraint violations")
        if not set(violations) <= allowed:
            raise ValueError("Unknown constraint violation ID")
        if violations and relevance is not None and relevance != 0:
            raise ValueError(
                "Known hard-constraint violations require relevance zero or explicit abstention"
            )
    else:
        observations = _strings(record["mood_observations"], "mood observations")
        if not set(observations) <= _MOODS:
            raise ValueError("Unknown draft mood term; revise the schema/rubric explicitly")
        if not isinstance(record["evidence_status"], str) or record["evidence_status"] not in {
            "supported",
            "uncertain",
            "insufficient_evidence",
        }:
            raise ValueError("Unknown mood evidence status")
        eligible = record["eligible_for_whole_work_gold"]
        if type(eligible) is not bool:
            raise ValueError("Gold eligibility assertion must be boolean")
        if evidence["scope"] not in {"excerpt", "whole_work"}:
            raise ValueError("Metadata/description evidence cannot establish book mood")
        if eligible and (
            evidence["scope"] != "whole_work"
            or record["evidence_status"] != "supported"
            or not observations
            or record["fixture_only"]
        ):
            raise ValueError("Unsupported assertion of whole-work mood eligibility")


def validate_future_records(
    queries: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    packet: dict[str, Any],
    *,
    allow_test_fixtures: bool = False,
) -> None:
    """Batch schema validation; does not modify the empty preparation packet."""
    if not isinstance(queries, list) or not isinstance(judgments, list):
        raise ValueError("Queries and judgments must be arrays")
    query_ids = set()
    for query in queries:
        validate_query(query, packet, allow_test_fixtures=allow_test_fixtures)
        if query["query_id"] in query_ids:
            raise ValueError("Duplicate query ID")
        query_ids.add(query["query_id"])
    judgment_ids = set()
    assignments = set()
    for judgment in judgments:
        validate_judgment(judgment, packet, queries, allow_test_fixtures=allow_test_fixtures)
        assignment = (
            judgment["kind"],
            judgment.get("query_id"),
            judgment["work_id"],
            judgment["rater_id"],
        )
        if judgment["judgment_id"] in judgment_ids or assignment in assignments:
            raise ValueError("Duplicate judgment ID or rater assignment")
        judgment_ids.add(judgment["judgment_id"])
        assignments.add(assignment)
