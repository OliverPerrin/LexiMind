"""Build an offline source-first worksheet, or import human drafts into new local candidates."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from src.research.book_fields import FACETS
from src.research.book_partitions import group_overrides
from src.research.builders.book_fields import checked, jsonl_rows, reference
from src.research.builders.field_review import CONTRACT, MAX_RECORDS
from src.research.candidate_io import create_or_verify, json_bytes, sha
from src.research.field_reviews import validate_review_record
from src.research.io import read_json
from src.utils.labels import BOOK_INPUT_FORMAT, LabelMetadata

ROOT = Path(__file__).resolve().parents[2]
DRAFT_KIND = "leximind_human_field_review_draft"


def _local_output(root: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve() / "data/research_candidates"):
        raise ValueError("Review output must remain in ignored data/research_candidates")
    return path


def _unadmitted(value: object) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("status") == "candidate_field_reviews_not_admitted"
        and value.get("training_authorized") is False
        and value.get("human_gold") is False
    )


def _load(root: Path, manifest_path: Path) -> dict:
    """Verify the pinned worksheet and all input bindings; never resolve external URLs."""
    manifest_ref = reference(root, manifest_path)
    manifest = read_json(manifest_path)
    if not _unadmitted(manifest):
        raise ValueError("Review manifest must remain an unadmitted candidate")
    bindings = {
        "manifest": manifest_ref,
        "packet": manifest["review_packet"],
        "worksheet": manifest["local_worksheet"],
        "sources": manifest["bindings"],
    }
    refs = [manifest_ref, bindings["packet"], bindings["worksheet"], *bindings["sources"].values()]
    for item in refs:
        checked(root, item)
    packet = read_json(checked(root, bindings["packet"]))
    worksheet = read_json(checked(root, bindings["worksheet"]))
    mapping = read_json(checked(root, bindings["sources"]["mapping"]))
    if (
        not _unadmitted(packet)
        or set(packet)
        != {
            "schema_version",
            "status",
            "training_authorized",
            "human_gold",
            "bindings",
            "contract",
            "selection",
            "records",
        }
        or packet.get("contract") != CONTRACT
        or packet.get("bindings") != bindings["sources"]
        or not isinstance(packet.get("records"), list)
        or not 1 <= len(packet["records"]) <= MAX_RECORDS
        or not isinstance(worksheet, list)
        or len(worksheet) != len(packet["records"])
    ):
        raise ValueError("Packet, worksheet and manifest do not share the review contract")
    records = {row["record_id"]: row for row in packet["records"]}
    if len(records) != len(packet["records"]):
        raise ValueError("Repeated packet record")
    resolved = {}
    for row in worksheet:
        review, payload = row["review"], row["input"]
        rid = review["record_id"]
        if (
            rid in resolved
            or records.get(rid) != review
            or set(payload) != {"title", "description"}
            or any(not isinstance(text, str) for text in payload.values())
        ):
            raise ValueError("Worksheet record or literal input is stale or repeated")
        candidate = {
            "record_id": rid,
            "input_sha256": review["input_sha256"],
            "source": row["source"],
            "fields": row["source_fields"],
        }
        validate_review_record(review, candidate, payload, mapping)
        resolved[rid] = {"candidate": candidate, "input": payload, "prior": review["agent_review"]}
    if set(resolved) != set(records) or set(mapping["facets"]) != set(FACETS):
        raise ValueError("Review records or field vocabulary are incomplete")
    return {
        "bindings": bindings,
        "refs": refs,
        "packet": packet,
        "rows": resolved,
        "mapping": mapping,
    }


def _check_sources(root: Path, bundle: dict) -> None:
    for item in bundle["refs"]:
        checked(root, item)


def build(root: Path, manifest_path: Path, output: Path) -> dict:
    output = _local_output(root, output)
    bundle = _load(root, manifest_path)
    data = {
        "bindings": bundle["bindings"],
        "facets": bundle["mapping"]["facets"],
        "rows": [
            {
                "record_id": row["record_id"],
                **bundle["rows"][row["record_id"]],
                "human_review": row["human_review"],
            }
            for row in bundle["packet"]["records"]
        ],
    }
    # JSON is inert data, including when a source contains a closing script tag.
    encoded = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    page = HTML.replace("__REVIEW_DATA__", encoded).encode()
    _check_sources(root, bundle)
    return {"path": str(output), **create_or_verify(output, [page])}


def import_draft(
    root: Path,
    manifest_path: Path,
    draft_path: Path,
    output: Path,
    *,
    preserve_conflicts: bool = False,
) -> dict:
    output = _local_output(root, output)
    if output == draft_path.resolve():
        raise ValueError("Output must be separate from the human draft")
    bundle = _load(root, manifest_path)
    if output in {(root / item["path"]).resolve() for item in bundle["refs"]}:
        raise ValueError("Output must be a new candidate, separate from source evidence")
    if draft_path.stat().st_size > 8_000_000:
        raise ValueError("Human draft exceeds the 8 MB bound")
    draft = read_json(draft_path)
    if (
        not isinstance(draft, dict)
        or set(draft) != {"schema_version", "kind", "bindings", "records"}
        or type(draft["schema_version"]) is not int
        or draft["schema_version"] != 1
        or draft["kind"] != DRAFT_KIND
        or draft["bindings"] != bundle["bindings"]
        or not isinstance(draft["records"], list)
        or not 1 <= len(draft["records"]) <= len(bundle["rows"])
    ):
        raise ValueError("Malformed or stale human draft; rebuild from current pinned evidence")
    packet = copy.deepcopy(bundle["packet"])
    by_id = {row["record_id"]: row for row in packet["records"]}
    seen = set()
    for entry in draft["records"]:
        if not isinstance(entry, dict) or set(entry) != {
            "record_id",
            "input_sha256",
            "human_review",
        }:
            raise ValueError(
                "Draft may supply only record references and explicit human review slots"
            )
        rid = entry["record_id"]
        if not isinstance(rid, str) or rid not in by_id or rid in seen:
            raise ValueError("Unknown or repeated draft record")
        seen.add(rid)
        review = by_id[rid]
        if entry["input_sha256"] != review["input_sha256"] or entry["human_review"] is None:
            raise ValueError(f"{rid}: stale input or empty human review")
        review["human_review"] = entry["human_review"]
        row = bundle["rows"][rid]
        try:
            validator = (
                _validate_independent_assertions if preserve_conflicts else validate_review_record
            )
            validator(review, row["candidate"], row["input"], bundle["mapping"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{rid}: {error}") from error
    if preserve_conflicts:
        packet = {
            "schema_version": 2,
            "kind": "leximind_field_review_candidate",
            "bindings": bundle["bindings"],
            "training_authorized": False,
            "human_gold": False,
            "records": packet["records"],
            "conflicts": _conflicts(packet["records"], bundle),
            "adjudications": [],
        }
    _check_sources(root, bundle)
    return {
        "path": str(output),
        "human_records": len(seen),
        "source_conflicts": len(packet["conflicts"]) if preserve_conflicts else 0,
        "training_authorized": False,
        **create_or_verify(output, [json_bytes(packet)]),
    }


def _validate_independent_assertions(
    review: dict, candidate: dict, payload: dict, mapping: dict
) -> None:
    """Keep source/agent validation strict; validate human assertions independently.

    An isolated copy with unknown comparison states permits a human assertion to
    disagree with source metadata without changing source evidence or weakening
    the legacy validator. Mandatory conflicts are derived from the real source.
    """
    validate_review_record({**review, "human_review": None}, candidate, payload, mapping)
    isolated = copy.deepcopy(candidate)
    isolated["fields"] = {field: {"positive": [], "negative": []} for field in FACETS}
    validate_review_record({**review, "agent_review": None}, isolated, payload, mapping)


def _conflicts(records: list[dict], bundle: dict) -> list[dict]:
    conflicts = []
    for review in records:
        rid = review["record_id"]
        source = bundle["rows"][rid]["candidate"]["fields"]
        for decision in (review["human_review"] or {}).get("decisions", []):
            opposite = {"positive": "negative", "negative": "positive"}.get(decision["state"])
            if opposite and decision["label"] in source[decision["field"]][opposite]:
                assertion = {
                    "record_id": rid,
                    "input_sha256": review["input_sha256"],
                    "field": decision["field"],
                    "label": decision["label"],
                    "source_state": opposite,
                    "human_review": review["human_review"]["reviewer"],
                    "human_decision": decision,
                }
                conflicts.append({**assertion, "assertion_sha256": sha(json_bytes(assertion))})
    return conflicts


def _candidate(root: Path, bundle: dict, path: Path) -> dict:
    packet = read_json(path)
    if (
        not isinstance(packet, dict)
        or set(packet)
        != {
            "schema_version",
            "kind",
            "bindings",
            "training_authorized",
            "human_gold",
            "records",
            "conflicts",
            "adjudications",
        }
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 2
        or packet["kind"] != "leximind_field_review_candidate"
        or packet["bindings"] != bundle["bindings"]
        or packet["training_authorized"] is not False
        or packet["human_gold"] is not False
        or not isinstance(packet["records"], list)
    ):
        raise ValueError("Expected a pinned v2 conflict-preserving candidate")
    prior = {r["record_id"]: r for r in bundle["packet"]["records"]}
    seen = set()
    for review in packet["records"]:
        rid = review["record_id"]
        if rid not in prior or rid in seen:
            raise ValueError("Unknown or repeated candidate identity")
        seen.add(rid)
        if {k: v for k, v in review.items() if k != "human_review"} != {
            k: v for k, v in prior[rid].items() if k != "human_review"
        }:
            raise ValueError("Candidate changed pinned agent/source references")
        row = bundle["rows"][rid]
        _validate_independent_assertions(review, row["candidate"], row["input"], bundle["mapping"])
    if seen != set(prior) or packet["conflicts"] != _conflicts(packet["records"], bundle):
        raise ValueError("Candidate conflicts or record coverage changed")
    _validate_adjudications(packet["adjudications"], packet["conflicts"], bundle)
    return packet


def _validate_adjudications(values: object, conflicts: list[dict], bundle: dict) -> None:
    if not isinstance(values, list):
        raise ValueError("Adjudications must be an explicit list")
    by_hash = {c["assertion_sha256"]: c for c in conflicts}
    seen = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "assertion_sha256",
            "choice",
            "reviewer",
            "evidence",
            "rationale",
        }:
            raise ValueError(
                "Adjudication requires assertion hash, choice, reviewer, evidence and rationale"
            )
        digest = value["assertion_sha256"]
        if not isinstance(digest, str) or digest not in by_hash or digest in seen:
            raise ValueError("Unknown or repeated adjudication assertion")
        seen.add(digest)
        conflict = by_hash[digest]
        if value["choice"] not in {"human", "source", "unknown"}:
            raise ValueError("Adjudication choice must be human, source or unknown")
        # Reuse strict human provenance/evidence validation without implying a source override.
        decision = {
            "field": conflict["field"],
            "label": conflict["label"],
            "state": "unknown",
            "evidence": value["evidence"],
            "rationale": value["rationale"],
        }
        prior = next(
            r for r in bundle["packet"]["records"] if r["record_id"] == conflict["record_id"]
        )
        review = {**prior, "human_review": {"reviewer": value["reviewer"], "decisions": [decision]}}
        row = bundle["rows"][conflict["record_id"]]
        _validate_independent_assertions(review, row["candidate"], row["input"], bundle["mapping"])


def adjudicate(
    root: Path, manifest_path: Path, candidate_path: Path, draft_path: Path, output: Path
) -> dict:
    output = _local_output(root, output)
    bundle = _load(root, manifest_path)
    candidate_ref = reference(root, candidate_path)
    packet = _candidate(root, bundle, candidate_path)
    if output in {
        candidate_path.resolve(),
        draft_path.resolve(),
        *((root / r["path"]).resolve() for r in bundle["refs"]),
    }:
        raise ValueError("Adjudication output must preserve all input evidence")
    if draft_path.stat().st_size > 8_000_000:
        raise ValueError("Adjudication draft exceeds the 8 MB bound")
    draft = read_json(draft_path)
    if (
        not isinstance(draft, dict)
        or set(draft) != {"schema_version", "kind", "candidate", "adjudications"}
        or type(draft["schema_version"]) is not int
        or draft["schema_version"] != 1
        or draft["kind"] != "leximind_field_adjudication_draft"
        or draft["candidate"] != candidate_ref
        or not isinstance(draft["adjudications"], list)
        or not draft["adjudications"]
    ):
        raise ValueError("Expected a nonempty adjudication draft bound to this exact candidate")
    packet["adjudications"] = [*packet["adjudications"], *draft["adjudications"]]
    _validate_adjudications(packet["adjudications"], packet["conflicts"], bundle)
    _check_sources(root, bundle)
    checked(root, candidate_ref)
    return {
        "path": str(output),
        "adjudications": len(packet["adjudications"]),
        "unresolved_conflicts": len(packet["conflicts"]) - len(packet["adjudications"]),
        "training_authorized": False,
        **create_or_verify(output, [json_bytes(packet)]),
    }


def _validate_development_separation(
    root: Path, train_manifest: dict, bindings: dict, assignments: dict, overrides: dict
) -> None:
    """Require common split evidence and disjoint resolved train/dev identities."""
    train_sources = train_manifest["bindings"]["sources"]
    keys = (
        "archive",
        "mapping",
        "assignments",
        "components",
        "partition_manifest",
        "field_manifest",
        "field_references",
    )
    if any(train_sources.get(key) != bindings[key] for key in keys):
        raise ValueError("Train/dev exports require identical source and partition bindings")
    records = train_manifest.get("records")
    if (
        not isinstance(records, list)
        or not records
        or type(train_manifest.get("rows")) is not int
        or train_manifest["rows"] != len(records)
    ):
        raise ValueError("Training export requires complete identity metadata")
    identities, groups = set(), set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "record_id",
                "input_sha256",
                "group_id",
                "effective_group_id",
                "source_split",
                "effective_split",
            }
            or not isinstance(record["record_id"], str)
            or record["record_id"] in identities
            or not isinstance(record["effective_group_id"], str)
            or record["source_split"] != "train"
            or record["effective_split"] != "train"
        ):
            raise ValueError("Malformed or repeated training export identity")
        identities.add(record["record_id"])
        groups.add(record["effective_group_id"])
    resolved = {}
    for assignment in jsonl_rows(checked(root, bindings["assignments"])):
        if assignment["record_id"] in identities:
            if assignment["record_id"] in resolved:
                raise ValueError("Repeated training source assignment")
            resolved[assignment["record_id"]] = assignment
    if set(resolved) != identities:
        raise ValueError("Training export identity missing from common assignments")
    for record in records:
        assignment = resolved[record["record_id"]]
        group = assignment["group_id"]
        overlay = overrides.get(group)
        if (
            assignment["source_split"] != "train"
            or assignment["review_required"]
            or record["group_id"] != group
            or record["effective_group_id"] != (overlay["component_id"] if overlay else group)
            or (overlay["proposed_split"] if overlay else assignment["proposed_split"]) != "train"
            or (overlay and overlay["status"] != "candidate_not_admitted")
        ):
            raise ValueError(
                "Training export identity disagrees with original/effective split evidence"
            )
    dev_groups = {
        overrides[a["group_id"]]["component_id"] if a["group_id"] in overrides else a["group_id"]
        for a in assignments.values()
    }
    if identities & set(assignments) or groups & dev_groups:
        raise ValueError("Train/dev exports overlap record or effective group identities")


def export_review(
    root: Path,
    manifest_path: Path,
    candidate_path: Path,
    output: Path,
    *,
    role: str,
    labels: list[str] | None = None,
    training_labels: Path | None = None,
) -> dict:
    """Export explicit reviewed supervision; source metadata never completes missing labels."""
    output = _local_output(root, output)
    if output.exists():
        raise FileExistsError("Export requires a new directory; preserve prior data")
    if role not in {"train", "dev"}:
        raise ValueError("Export role must be train or dev")
    bundle = _load(root, manifest_path)
    candidate_ref = reference(root, candidate_path)
    packet = _candidate(root, bundle, candidate_path)
    bindings = bundle["bindings"]["sources"]
    if not {"assignments", "components", "partition_manifest", "field_manifest"} <= set(bindings):
        raise ValueError(
            "Export requires pinned original assignments and effective partition components"
        )
    partition = read_json(checked(root, bindings["partition_manifest"]))
    fields = read_json(checked(root, bindings["field_manifest"]))
    if (
        partition["components"] != bindings["components"]
        or partition["inputs"]["fields"] != bindings["field_manifest"]
        or any(
            partition["inputs"][key] != bindings[key] or fields[key] != bindings[key]
            for key in ("archive", "assignments", "field_references")
        )
        or partition["inputs"]["field_mapping"] != bindings["mapping"]
        or fields["mapping"] != bindings["mapping"]
    ):
        raise ValueError("Export partition, fields and source bindings disagree")
    wanted = {r["record_id"] for r in packet["records"] if r["human_review"] is not None}
    assignments = {}
    counts: dict[str, int] = {}
    for assignment in jsonl_rows(checked(root, bindings["assignments"])):
        group = assignment["group_id"]
        counts[group] = counts.get(group, 0) + 1
        if assignment["record_id"] in wanted:
            if assignment["record_id"] in assignments:
                raise ValueError("Repeated source assignment")
            assignments[assignment["record_id"]] = assignment
    if set(assignments) != wanted:
        raise ValueError("Reviewed identities missing from source assignments")
    source_candidates = {}
    for candidate in jsonl_rows(checked(root, bindings["field_references"])):
        if candidate["record_id"] in wanted:
            if candidate["record_id"] in source_candidates:
                raise ValueError("Repeated source field reference")
            source_candidates[candidate["record_id"]] = candidate
    if set(source_candidates) != wanted:
        raise ValueError("Reviewed identities missing from source fields")
    overrides = group_overrides(jsonl_rows(checked(root, bindings["components"])))
    vocabulary = [
        f"{field}:{label}"
        for field in FACETS
        for label in bundle["mapping"]["facets"][field]["labels"]
    ]
    schema_ref = None
    if role == "dev":
        if labels is not None or training_labels is None:
            raise ValueError(
                "Dev export requires --training-labels from a reviewed train export, not --labels"
            )
        schema_ref = reference(root, training_labels)
        schema = read_json(checked(root, schema_ref))
        if (
            set(schema)
            != {"schema_version", "problem_type", "input_format", "labels", "mapping_sha256"}
            or type(schema["schema_version"]) is not int
            or schema["schema_version"] != 1
            or schema["problem_type"] != "multi_label"
            or schema["input_format"] != BOOK_INPUT_FORMAT
            or schema["mapping_sha256"] != bindings["mapping"]["sha256"]
        ):
            raise ValueError("Dev labels must match the exact training format and mapping")
        train_manifest_path = training_labels.parent / "manifest.json"
        train_manifest_ref = reference(root, train_manifest_path)
        train_manifest = read_json(checked(root, train_manifest_ref))
        if (
            train_manifest.get("kind") != "leximind_reviewed_field_export"
            or train_manifest.get("role") != "train"
            or train_manifest.get("files", {}).get("labels.json")
            != {"sha256": schema_ref["sha256"], "bytes": schema_ref["bytes"]}
            or set(train_manifest.get("coverage", {})) != set(schema["labels"])
            or any(
                counts.get("positive", 0) < 1 or counts.get("negative", 0) < 1
                for counts in train_manifest["coverage"].values()
            )
        ):
            raise ValueError(
                "Dev metadata must come from a reviewed train export with both-sign coverage"
            )
        train_file_ref = {
            "path": str(
                (training_labels.parent / "train.jsonl").resolve().relative_to(root.resolve())
            ),
            **train_manifest["files"]["train.jsonl"],
        }
        checked(root, train_file_ref)
        checked(root, train_manifest["candidate"])
        _validate_development_separation(root, train_manifest, bindings, assignments, overrides)
        labels = schema["labels"]
    elif training_labels is not None:
        raise ValueError("--training-labels applies to dev only")
    labels = vocabulary if labels is None else labels
    LabelMetadata([], labels, "multi_label", BOOK_INPUT_FORMAT, bindings["mapping"]["sha256"])
    if any(label not in vocabulary for label in labels):
        raise ValueError("Requested label outside the pinned mapping")
    coverage = {label: {"positive": 0, "negative": 0} for label in labels}
    excluded_decisions = 0
    resolutions = {a["assertion_sha256"]: a for a in packet["adjudications"]}
    conflicts = {(c["record_id"], c["field"], c["label"]): c for c in packet["conflicts"]}
    rows, provenance = [], []
    positive_count = negative_count = 0
    for review in packet["records"]:
        if review["human_review"] is None:
            continue
        rid = review["record_id"]
        assignment, candidate = assignments[rid], source_candidates[rid]
        row = bundle["rows"][rid]
        group = assignment["group_id"]
        overlay = overrides.get(group)
        effective = overlay["proposed_split"] if overlay else assignment["proposed_split"]
        if (
            assignment["source_split"] != role
            or effective != role
            or assignment["review_required"]
            or candidate["group"]["review_required"]
            or counts[group] != 1
            or candidate["group"]["group_id"] != group
            or candidate["source"]["source_split"] != role
            or candidate["group"]["proposed_split"] != assignment["proposed_split"]
            or candidate["source"] != row["candidate"]["source"]
            or candidate["fields"] != row["candidate"]["fields"]
            or candidate["input_sha256"] != review["input_sha256"]
            or (overlay and overlay["status"] != "candidate_not_admitted")
        ):
            raise ValueError(
                f"{rid}: original/effective role or unambiguous singleton group failed"
            )
        targets: dict[str, list[str]] = {"positive": [], "negative": []}
        for decision in review["human_review"]["decisions"]:
            state = decision["state"]
            conflict = conflicts.get((rid, decision["field"], decision["label"]))
            if conflict:
                resolution = resolutions.get(conflict["assertion_sha256"])
                if resolution is None:
                    raise ValueError(f"{rid}: resolve every source conflict before export")
                choice = resolution["choice"]
                state = (
                    decision["state"]
                    if choice == "human"
                    else conflict["source_state"]
                    if choice == "source"
                    else "unknown"
                )
            column = f"{decision['field']}:{decision['label']}"
            if column not in coverage:
                excluded_decisions += 1
                continue
            if state != "unknown":
                targets[state].append(column)
                coverage[column][state] += 1
        if not targets["positive"] and not targets["negative"]:
            continue
        for values in targets.values():
            values.sort()
        positive_count += len(targets["positive"])
        negative_count += len(targets["negative"])
        rows.append({**row["input"], **targets})
        provenance.append(
            {
                "record_id": rid,
                "input_sha256": review["input_sha256"],
                "group_id": group,
                "effective_group_id": overlay["component_id"] if overlay else group,
                "source_split": role,
                "effective_split": effective,
            }
        )
    if not rows:
        raise ValueError(
            "Export requires nonempty explicit supervision in the requested vocabulary"
        )
    if role == "train":
        missing = {
            label: counts
            for label, counts in coverage.items()
            if not counts["positive"] or not counts["negative"]
        }
        if missing:
            raise ValueError(
                "Each requested training label needs explicit positive AND negative supervision; "
                "use an explicitly reviewed --labels subset if appropriate: " + json.dumps(missing)
            )
    schema = {
        "schema_version": 1,
        "problem_type": "multi_label",
        "input_format": BOOK_INPUT_FORMAT,
        "labels": labels,
        "mapping_sha256": bindings["mapping"]["sha256"],
    }
    split = "train" if role == "train" else "val"
    # JSONL needs one compact object per physical line, unlike pretty packet JSON.
    raw_rows = b"".join(
        (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for r in rows
    )
    schema_bytes = json_bytes(schema)
    manifest = {
        "schema_version": 1,
        "kind": "leximind_reviewed_field_export",
        "role": role,
        "candidate": candidate_ref,
        "bindings": bundle["bindings"],
        "records": provenance,
        "rows": len(rows),
        "reviewed_records": len(wanted),
        "unknown_only_or_excluded_records": len(wanted) - len(rows),
        "positive_labels": positive_count,
        "negative_labels": negative_count,
        "human_gold": False,
        "formal_study_admitted": False,
        "training_authorized": False,
        "unknown_labels": "masked; no implicit source/agent completion",
        "coverage": {
            label: {**counts, "unknown": len(rows) - sum(counts.values())}
            for label, counts in coverage.items()
        },
        "excluded_decisions": excluded_decisions,
        "training_labels": schema_ref,
        "training_export_manifest": train_manifest_ref if role == "dev" else None,
        "readiness_scope": "At least one example per sign is feasibility only, not sufficient study quality",
        "files": {
            f"{split}.jsonl": {"sha256": sha(raw_rows), "bytes": len(raw_rows)},
            "labels.json": {"sha256": sha(schema_bytes), "bytes": len(schema_bytes)},
        },
    }
    _check_sources(root, bundle)
    checked(root, candidate_ref)
    if schema_ref is not None:
        checked(root, schema_ref)
        checked(root, train_manifest_ref)
        checked(root, train_file_ref)
    output.mkdir(parents=True, exist_ok=False)
    create_or_verify(output / f"{split}.jsonl", [raw_rows])
    create_or_verify(output / "labels.json", [schema_bytes])
    create_or_verify(output / "manifest.json", [json_bytes(manifest)])
    return {
        "path": str(output),
        "rows": len(rows),
        "role": role,
        "positive_labels": positive_count,
        "negative_labels": negative_count,
        "formal_study_admitted": False,
        "training_authorized": False,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mode", choices=("build", "import", "adjudicate", "export"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "research/preparation/book_field_review_manifest.json",
    )
    parser.add_argument("--candidate", type=Path, help="Pinned v2 candidate for adjudicate/export")
    parser.add_argument(
        "--preserve-conflicts",
        action="store_true",
        help="Import as v2 candidate; preserve opposing source/human assertions",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Explicit ordered facet:label training subset; default full mapping",
    )
    parser.add_argument(
        "--training-labels",
        type=Path,
        help="Exact train export labels.json; required for dev export",
    )
    parser.add_argument(
        "--role", choices=("train", "dev"), help="Explicit original/effective role for export"
    )
    parser.add_argument(
        "--draft",
        type=Path,
        help="Exported human/adjudication JSON; required for import/adjudicate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New candidate file, or new export directory, under data/research_candidates",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if (args.mode in {"import", "adjudicate"}) != (args.draft is not None):
        parser.error("Supply --draft for import or adjudicate only")
    if (args.mode in {"adjudicate", "export"}) != (args.candidate is not None):
        parser.error("Supply --candidate for adjudicate or export only")
    if args.preserve_conflicts and args.mode != "import":
        parser.error("--preserve-conflicts applies to import only")
    if (args.mode == "export") != (args.role is not None):
        parser.error("Supply --role for export only")
    if args.mode != "export" and (args.labels is not None or args.training_labels is not None):
        parser.error("Label selection applies to export only")
    try:
        if args.mode == "build":
            result = build(ROOT, args.manifest, args.output)
        elif args.mode == "import":
            result = import_draft(
                ROOT,
                args.manifest,
                args.draft,
                args.output,
                preserve_conflicts=args.preserve_conflicts,
            )
        elif args.mode == "adjudicate":
            result = adjudicate(ROOT, args.manifest, args.candidate, args.draft, args.output)
        else:
            result = export_review(
                ROOT,
                args.manifest,
                args.candidate,
                args.output,
                role=args.role,
                labels=args.labels,
                training_labels=args.training_labels,
            )
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>LexiMind · Field review</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#21372f;background:#f5f6f0}*{box-sizing:border-box}body{margin:0}header,main{max-width:1400px;margin:auto;padding:24px 36px}header{border-bottom:1px solid #d8dfd6}h1{font-size:27px;letter-spacing:-.8px;margin:0}h2{font-size:24px;line-height:1.3;margin:8px 0 24px}h3{font-size:17px;margin:0 0 10px}.muted,small{color:#607167}p{margin:10px 0}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.spread{justify-content:space-between}button,input,select,textarea{font:inherit}button,.upload{border:1px solid #bdcac0;border-radius:6px;padding:8px 12px;background:#fff;color:#214b38;cursor:pointer}button:hover,.upload:hover{background:#edf4ec}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #d0a547;outline-offset:2px}.primary{background:#285a42;color:white;border-color:#285a42}.primary:hover{background:#1e4632}input,select,textarea{padding:8px;border:1px solid #bdcac0;border-radius:5px;background:white;max-width:100%}input[type=date]{width:158px}textarea{width:100%;min-height:85px;resize:vertical}label{display:block}label>small{display:block}.upload input{display:none}main{display:grid;grid-template-columns:minmax(0,1fr) minmax(390px,1fr);gap:32px}.panel{background:white;padding:26px;border:1px solid #e0e5db;border-radius:10px;min-width:0}.source{white-space:pre-wrap;font:18px/1.75 Georgia,serif;overflow-wrap:anywhere;user-select:text}.toolbar{margin:20px 0}.facets{display:flex;gap:6px;flex-wrap:wrap}.facets button[aria-pressed=true]{background:#e1eddc;border-color:#5f886f}.labels{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:16px 0;max-height:340px;overflow:auto}.label{display:flex;justify-content:space-between;gap:8px;text-align:left;font-size:14px}.label[data-active=true]{outline:2px solid #5f886f;outline-offset:-2px}.label b{font-size:11px;text-transform:uppercase;letter-spacing:.4px;font-weight:500;align-self:center;color:#66766a}.label[data-state=positive]{background:#eaf3e8}.label[data-state=negative]{background:#f8eeee}.label[data-state=unknown][data-reviewed=true]{background:#f7f2e4}.editor{border-top:1px solid #dde4d9;padding-top:20px}.evidence{background:#f5f6f0;padding:12px;max-height:140px;overflow:auto;font-size:14px;margin:12px 0;white-space:pre-wrap;overflow-wrap:anywhere}.status{min-height:28px;margin:12px 0;color:#3e6350}.status.error,.conflict{color:#993c24}.conflict{font-size:14px;min-height:22px}details{margin-top:24px;border-top:1px solid #dde4d9;padding-top:16px}summary{cursor:pointer;color:#677269;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}#editor[hidden]{display:none}.steps{font-size:14px;margin:16px 0}button:disabled{opacity:.4;cursor:default}@media(max-width:850px){header,main{padding:18px}main{grid-template-columns:1fr;gap:18px}.panel{padding:20px}.source{font-size:17px}}
</style>
<header><div class="bar spread"><div><small>LEXIMIND / HUMAN ANNOTATION</small><h1>Read the evidence. Make the call.</h1></div><div class="bar"><label class="upload">Open saved draft<input id="resume" type="file" accept="application/json,.json"></label><button id="export" class="primary">Export JSON draft</button></div></div>
<p class="muted">A local worksheet. Untouched fields stay unknown. Saved decisions remain candidates for review.</p>
<div class="bar"><label><small>Your reviewer identity</small><input id="reviewer" placeholder="Name or reviewer ID" autocomplete="off"></label><label><small>Review date</small><input id="date" type="date"></label><span id="progress" class="muted"></span></div>
<div id="status" class="status" role="status" aria-live="polite"></div></header>
<main><section class="panel"><div class="bar spread"><small id="position"></small><div class="bar"><button id="previous">Previous</button><button id="next">Next</button></div></div>
<label class="toolbar"><small>Source record</small><select id="record"></select></label><h2 id="title" class="source" data-source-field="title"></h2><div id="description" class="source" data-source-field="description"></div>
<details id="prior"><summary>Reveal source metadata and earlier agent suggestions</summary><p class="muted">Weak metadata and agent suggestions are context. They do not fill your decisions.</p><pre id="prior-data"></pre></details></section>
<section class="panel"><h3>Field decisions</h3><p class="steps">Choose a label, select a passage in the title or description, then attach it and explain your decision. A negative needs explicit contrary evidence.</p>
<div id="facets" class="facets"></div><p id="definition" class="muted"></p><div id="labels" class="labels"></div>
<div id="editor" class="editor" hidden><h3 id="label-title"></h3><p id="label-note" class="muted"></p><label>State <select id="state"><option value="unknown">Unknown / abstain</option><option value="positive">Positive</option><option value="negative">Negative</option></select></label>
<div id="conflict" class="conflict" role="status"></div><div class="bar"><button id="attach">Attach selected evidence</button><button id="clear-evidence">Clear evidence</button></div><div id="evidence" class="evidence">No evidence attached.</div>
<label>Rationale <textarea id="rationale" placeholder="What does this passage support, deny, or leave uncertain?"></textarea></label><div class="bar toolbar"><button id="save" class="primary">Save decision</button><button id="remove">Remove decision</button></div><small>Removing a decision restores its untouched unknown state. Export regularly to keep your work.</small></div>
</section></main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent), $=id=>document.getElementById(id), rows=data.rows;
const states=new Set(['positive','negative','unknown']), decisions=new Map(), cp=s=>Array.from(s), nice=s=>s.replaceAll('_',' ');
let index=0,facet=Object.keys(data.facets)[0],label=null,evidence=[],editing=false,unexported=false,selection=null;
const canonical=v=>JSON.stringify(v,(_,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.entries(x).sort(([a],[b])=>a.localeCompare(b))):x);
const digest=async s=>{if(!globalThis.crypto?.subtle)throw Error('This browser cannot hash local evidence. Use a current browser with Web Crypto.');return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(s)))].map(x=>x.toString(16).padStart(2,'0')).join('');};
function message(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function row(){return rows[index];}
function saved(){return decisions.get(row().record_id);}
function decision(){return saved()?.decisions.find(d=>d.field===facet&&d.label===label);}
function progress(){let n=0;for(const r of decisions.values())n+=r.decisions.length;$('progress').textContent=`${n} explicit decisions across ${decisions.size} of ${rows.length} records`;}
function conflict(state){return state!=='unknown'&&row().candidate.fields[facet][state==='positive'?'negative':'positive'].includes(label);}
function showConflict(){const yes=label&&conflict($('state').value);$('conflict').textContent=yes?'Conflicts with a source label. Your draft can retain this decision; import with --preserve-conflicts retains it for separate adjudication.':'';}
function drawEvidence(){const text=row().input;$('evidence').textContent=evidence.length?evidence.map(e=>`${e.input_field} [${e.start}, ${e.end}): “${cp(text[e.input_field]).slice(e.start,e.end).join('')}”`).join('\n\n'):'No evidence attached.';}
function drawLabels(){const container=$('labels');container.replaceChildren();for(const name of data.facets[facet].labels){const d=saved()?.decisions.find(d=>d.field===facet&&d.label===name),b=document.createElement('button');b.className='label';b.dataset.state=d?.state||'unknown';b.dataset.reviewed=String(!!d);b.dataset.active=String(name===label);const text=document.createElement('span'),badge=document.createElement('b');text.textContent=nice(name);badge.textContent=d?.state||'unknown';b.append(text,badge);b.onclick=()=>openLabel(name);container.append(b);}}
function canLeave(){return !editing||confirm('Discard the unsaved edits for this label? Saved decisions will remain.');}
function openLabel(name){if(!canLeave())return;label=name;const d=decision();evidence=structuredClone(d?.evidence||[]);$('state').value=d?.state||'unknown';$('rationale').value=d?.rationale||'';$('label-title').textContent=nice(label);$('label-note').textContent=data.facets[facet].label_notes?.[label]||'';$('editor').hidden=false;editing=false;drawLabels();drawEvidence();showConflict();}
function drawFacets(){const container=$('facets');container.replaceChildren();for(const [name,spec]of Object.entries(data.facets)){const b=document.createElement('button');b.textContent=`${nice(name)} (${spec.labels.length})`;b.setAttribute('aria-pressed',String(name===facet));b.onclick=()=>{if(!canLeave())return;facet=name;label=null;editing=false;$('editor').hidden=true;drawFacets();drawLabels();};container.append(b);}$('definition').textContent=data.facets[facet].definition;}
function draw(){const r=row();$('position').textContent=`RECORD ${index+1} OF ${rows.length}`;$('record').value=String(index);$('title').textContent=r.input.title;$('description').textContent=r.input.description;$('previous').disabled=index===0;$('next').disabled=index===rows.length-1;$('prior').open=false;$('prior-data').textContent=JSON.stringify({source:r.candidate.source,source_fields:r.candidate.fields,agent_suggestions:r.prior},null,2);label=null;evidence=[];selection=null;editing=false;$('editor').hidden=true;drawFacets();drawLabels();progress();}
function go(n){if(canLeave()){index=n;draw();}else $('record').value=String(index);}
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.input.title}`;$('record').append(o);if(r.human_review)decisions.set(r.record_id,structuredClone(r.human_review));});
$('record').onchange=e=>go(Number(e.target.value));$('previous').onclick=()=>go(index-1);$('next').onclick=()=>go(index+1);
const today=new Date();$('date').value=`${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`;
document.addEventListener('selectionchange',()=>{const s=getSelection();if(!s?.rangeCount||s.isCollapsed)return;const range=s.getRangeAt(0),start=range.startContainer.nodeType===1?range.startContainer:range.startContainer.parentElement,end=range.endContainer.nodeType===1?range.endContainer:range.endContainer.parentElement,box=start?.closest('[data-source-field]');if(!box||box!==end?.closest('[data-source-field]')){selection=null;return;}const before=range.cloneRange();before.selectNodeContents(box);before.setEnd(range.startContainer,range.startOffset);const a=cp(before.toString()).length,b=a+cp(range.toString()).length;selection={record_id:row().record_id,input_field:box.dataset.sourceField,start:a,end:b};});
$('attach').onclick=async()=>{try{if(!selection||selection.record_id!==row().record_id)throw Error('Select a passage within the source title or description first.');const selected={...selection},text=cp(row().input[selected.input_field]).slice(selected.start,selected.end).join(''),key=`${row().record_id}/${facet}/${label}`;if(!text)throw Error('The selected passage is empty.');const span={input_field:selected.input_field,start:selected.start,end:selected.end,sha256:await digest(text)};if(key!==`${row().record_id}/${facet}/${label}`)throw Error('Selection changed; attach the evidence again.');if(!evidence.some(e=>canonical(e)===canonical(span)))evidence.push(span);editing=true;drawEvidence();message('Evidence attached. Add a rationale and save the decision.');}catch(e){message(e.message,true);}};
$('clear-evidence').onclick=()=>{evidence=[];editing=true;drawEvidence();};$('state').onchange=()=>{editing=true;showConflict();};$('rationale').oninput=()=>{editing=true;};
$('save').onclick=()=>{const id=$('reviewer').value.trim(),date=$('date').value,rationale=$('rationale').value.trim();if(!id||!date||!evidence.length||!rationale)return message('Add your identity, review date, selected evidence and a rationale before saving.',true);if(saved()&&saved().reviewer.id!==id)return message(`This record already has decisions by ${saved().reviewer.id}. Keep that identity or use a separate draft.`,true);const d={field:facet,label,state:$('state').value,evidence:structuredClone(evidence),rationale},review={reviewer:{kind:'human',id,method:'direct_source_review',reviewed_at:date},decisions:[...(saved()?.decisions||[]).filter(x=>x.field!==facet||x.label!==label),d]};decisions.set(row().record_id,review);editing=false;unexported=true;drawLabels();progress();message(conflict(d.state)?'Decision saved in draft with a source conflict. Use review import --preserve-conflicts, then adjudicate before export.':'Decision saved. Export your draft to keep it.',conflict(d.state));};
$('remove').onclick=()=>{if(saved()){saved().decisions=saved().decisions.filter(d=>d.field!==facet||d.label!==label);if(!saved().decisions.length)decisions.delete(row().record_id);unexported=true;}editing=false;openLabel(label);progress();message('Decision removed; the field is unknown.');};
function draft(){return{schema_version:1,kind:'leximind_human_field_review_draft',bindings:data.bindings,records:rows.filter(r=>decisions.has(r.record_id)).map(r=>({record_id:r.record_id,input_sha256:r.candidate.input_sha256,human_review:decisions.get(r.record_id)}))};}
$('export').onclick=()=>{if(editing)return message('Save or remove the current unsaved decision before exporting.',true);if(!decisions.size)return message('There are no explicit human decisions to export yet.',true);const blob=new Blob([JSON.stringify(draft(),null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='leximind-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);unexported=false;message('JSON draft exported. Use review import --preserve-conflicts to retain every decision, then adjudicate conflicts before training export.');};
$('resume').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>8000000)throw Error('Draft exceeds the 8 MB limit.');if((editing||unexported)&&!confirm('Replace this page’s work with the selected draft? Export first to retain it.'))return;const d=JSON.parse(await file.text()),pending=new Map();if(d.schema_version!==1||d.kind!=='leximind_human_field_review_draft'||canonical(d.bindings)!==canonical(data.bindings)||!Array.isArray(d.records)||!d.records.length||d.records.length>rows.length)throw Error('Draft does not match this pinned worksheet.');for(const item of d.records){const source=rows.find(r=>r.record_id===item.record_id),review=item.human_review;if(!source||pending.has(item.record_id)||item.input_sha256!==source.candidate.input_sha256||review?.reviewer?.kind!=='human'||review.reviewer.method!=='direct_source_review'||typeof review.reviewer.id!=='string'||!review.reviewer.id.trim()||!/^\d{4}-\d{2}-\d{2}$/.test(review.reviewer.reviewed_at)||!Array.isArray(review.decisions)||!review.decisions.length)throw Error('Malformed human review.');const seen=new Set();for(const x of review.decisions){const key=`${x.field}/${x.label}`;if(!data.facets[x.field]?.labels.includes(x.label)||!states.has(x.state)||seen.has(key)||typeof x.rationale!=='string'||!x.rationale.trim()||!Array.isArray(x.evidence)||!x.evidence.length)throw Error('Malformed or repeated field decision.');seen.add(key);for(const span of x.evidence){const text=source.input[span.input_field];if(typeof text!=='string'||!Number.isInteger(span.start)||!Number.isInteger(span.end)||span.start<0||span.end<=span.start||span.end>cp(text).length||await digest(cp(text).slice(span.start,span.end).join(''))!==span.sha256)throw Error('Evidence offsets or hash do not match the source.');}}pending.set(item.record_id,review);}decisions.clear();pending.forEach((v,k)=>decisions.set(k,v));$('reviewer').value=d.records[0].human_review.reviewer.id;editing=false;unexported=false;draw();message('Draft restored. Source conflicts are checked during import.');}catch(error){message(error.message,true);}finally{e.target.value='';}};
addEventListener('beforeunload',e=>{if(editing||unexported){e.preventDefault();e.returnValue='';}});draw();
</script></html>"""
