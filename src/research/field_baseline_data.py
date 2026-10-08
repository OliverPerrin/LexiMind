"""Pinned, metadata-selected inputs for the bounded BGC retrieval diagnostic.

This is separate from book-field admission. Only selected original train/dev
record frames are decoded; reserved text is excluded and the source test ZIP
member is never opened. Raw train/dev member bytes are streamed for integrity.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

from .book_fields import (
    FACETS,
    input_hash,
    input_payload,
    map_source_labels,
    validate_mapping,
    validate_states,
)
from .book_partitions import POLICY as PARTITION_POLICY
from .book_partitions import group_overrides
from .builders.bgc_source import (
    MAX_RECORD_BYTES,
    MEMBERS,
    SOURCE_BYTES,
    SOURCE_SHA256,
    SOURCE_URL,
    isbn13,
    provider_id,
    records,
)
from .builders.book_fields import checked, jsonl_rows
from .candidate_io import sha
from .io import file_hash, read_json, safe_path

POLICY = "bgc-field-retrieval-v1"
SAMPLE_COUNTS = {"train": 4096, "dev": 1024}
MAX_RECORDS = 100_000
_COMMON_HELPERS = {"src/research/candidate_io.py", "src/research/io.py"}


def _implementation(root: Path, document: dict, script: str, helpers: set[str]) -> None:
    expected = {name: file_hash(safe_path(root, name)) for name in sorted(helpers)}
    if (
        document.get("preparation_script_sha256") != file_hash(safe_path(root, script))
        or document.get("preparation_helper_sha256") != expected
    ):
        raise ValueError(f"Preparation implementation changed: {script}")


def _load_bindings(root: Path, config: dict) -> tuple[dict, dict, dict, dict, dict, dict]:
    if config.get("policy", POLICY) != POLICY:
        raise ValueError("Unknown field retrieval selection policy")
    bindings = {name: config[name] for name in ("field_manifest", "partition_manifest")}
    fields = read_json(checked(root, bindings["field_manifest"]))
    partition = read_json(checked(root, bindings["partition_manifest"]))
    if (
        fields.get("schema_version") != 1
        or fields.get("status") != "weak_field_candidates_not_admitted"
        or fields.get("training_authorized") is not False
        or fields.get("human_gold") is not False
        or partition.get("schema_version") != 1
        or partition.get("policy") != PARTITION_POLICY
        or partition.get("status") != "candidate_component_overlay_not_admitted"
        or partition.get("training_authorized") is not False
        or partition.get("training_performed") is not False
    ):
        raise ValueError("Expected unchanged, unadmitted field and partition preparation")
    for name in (
        "archive",
        "candidate_manifest",
        "grouping_manifest",
        "mapping",
        "assignments",
        "field_references",
    ):
        bindings[name] = fields[name]
    bindings["components"] = partition["components"]
    bindings["constraints"] = partition["constraints"]
    for name, reference in partition["inputs"].items():
        bindings[f"partition_input:{name}"] = reference
    for reference in bindings.values():
        checked(root, reference)
    required = {
        "fields": bindings["field_manifest"],
        "groups": fields["grouping_manifest"],
        "archive": fields["archive"],
        "candidate_manifest": fields["candidate_manifest"],
        "assignments": fields["assignments"],
        "field_mapping": fields["mapping"],
        "field_references": fields["field_references"],
    }
    if any(partition["inputs"].get(name) != ref for name, ref in required.items()):
        raise ValueError("Field and cross-source partition bindings disagree")
    groups = read_json(safe_path(root, fields["grouping_manifest"]["path"]))
    audit = read_json(safe_path(root, fields["candidate_manifest"]["path"]))
    mapping = read_json(safe_path(root, fields["mapping"]["path"]))
    if (
        groups.get("schema_version") != 1
        or groups.get("status") != "candidate_leakage_groups_not_admitted"
        or groups.get("policy") != "bgc-leakage-groups-v1"
        or groups.get("training_authorized") is not False
        or groups.get("archive") != fields["archive"]
        or groups.get("candidate_manifest") != fields["candidate_manifest"]
        or groups.get("assignments") != fields["assignments"]
        or audit.get("schema_version") != 1
        or audit.get("status") != "source_audited_not_admitted"
        or audit.get("training_authorized") is not False
        or audit.get("source_url") != SOURCE_URL
        or audit.get("archive") != fields["archive"]
        or fields["archive"]["sha256"] != SOURCE_SHA256
        or fields["archive"]["bytes"] != SOURCE_BYTES
    ):
        raise ValueError("Source, grouping and field preparation disagree")
    _implementation(root, audit, "src/research/builders/bgc_source.py", _COMMON_HELPERS)
    _implementation(
        root,
        groups,
        "src/research/builders/bgc_groups.py",
        _COMMON_HELPERS | {"src/research/book_groups.py", "src/research/builders/bgc_source.py"},
    )
    _implementation(
        root,
        fields,
        "src/research/builders/book_fields.py",
        _COMMON_HELPERS
        | {
            "src/research/builders/bgc_source.py",
            "src/research/book_fields.py",
            "src/catalog/storage.py",
        },
    )
    partition_implementation = _COMMON_HELPERS | {
        "src/research/builders/book_partitions.py",
        "src/research/builders/book_groups_review.py",
        "src/research/builders/bgc_source.py",
        "src/research/book_partitions.py",
        "src/catalog/storage.py",
    }
    if partition.get("implementation_sha256") != {
        name: file_hash(safe_path(root, name)) for name in sorted(partition_implementation)
    }:
        raise ValueError("Partition preparation implementation changed")
    validate_mapping(
        mapping,
        source_labels=set(audit["observed"]["label_record_counts"]),
        archive_sha256=fields["archive"]["sha256"],
        hierarchy_sha256=audit["observed"]["members"]["hierarchy.txt"]["sha256"],
    )
    if sum(len(mapping["facets"][facet]["labels"]) for facet in FACETS) != 48:
        raise ValueError("Diagnostic requires the frozen 48-label field vocabulary")
    return fields, partition, groups, audit, mapping, bindings


def _assignments(path: Path) -> tuple[dict, Counter, dict]:
    assignments: dict[str, dict] = {}
    group_splits: dict[str, str] = {}
    group_counts: Counter = Counter()
    for row in jsonl_rows(path):
        split, number, identifier = (
            row.get("source_split"),
            row.get("source_row"),
            row.get("record_id"),
        )
        group = row.get("group_id")
        if (
            len(assignments) >= MAX_RECORDS
            or split not in MEMBERS
            or type(number) is not int
            or number < 1
            or identifier != f"bgc:{split}:{number}"
            or identifier in assignments
            or not isinstance(group, str)
            or not re.fullmatch(r"bgc-group:[a-f0-9]{64}", group)
            or row.get("proposed_split") not in MEMBERS
            or type(row.get("review_required")) is not bool
            or (group in group_splits and group_splits[group] != row["proposed_split"])
        ):
            raise ValueError("Invalid, duplicate or conflicting BGC assignment")
        assignments[identifier] = row
        group_counts[group] += 1
        group_splits[group] = row["proposed_split"]
    return assignments, group_counts, group_splits


def _metadata_selection(
    root: Path, fields: dict, partition: dict, groups: dict, mapping: dict, *, sample_counts=None
):
    caps = dict(SAMPLE_COUNTS if sample_counts is None else sample_counts)
    if caps not in (SAMPLE_COUNTS, {"train": 16384, "dev": 1024}):
        raise ValueError("Unsupported fixed metadata sample counts")
    assignments, group_counts, group_splits = _assignments(
        safe_path(root, fields["assignments"]["path"])
    )
    if (
        len(assignments) != fields["observed"]["records"]
        or len(assignments) != groups["counts"]["records"]
        or len(group_counts) != groups["counts"]["groups"]
    ):
        raise ValueError("Assignment counts differ from pinned preparation")
    components = list(jsonl_rows(safe_path(root, partition["components"]["path"])))
    component_ids: set[str] = set()
    members: set[str] = set()
    for component in components:
        nodes, identifier = component.get("members"), component.get("component_id")
        if (
            not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(node, str) for node in nodes)
            or nodes != sorted(set(nodes))
            or identifier != "book-component:" + sha("\n".join(nodes))
            or identifier in component_ids
            or members.intersection(nodes)
            or component.get("training_eligible") is not False
            or component.get("proposed_split") not in {*MEMBERS, None}
            or component.get("status")
            != (
                "quarantined_title_only_ambiguity"
                if component["proposed_split"] is None
                else "candidate_not_admitted"
            )
            or any(node.startswith("bgc-group:") and node not in group_counts for node in nodes)
        ):
            raise ValueError("Invalid or overlapping cross-source partition component")
        component_ids.add(identifier)
        members.update(nodes)
    overrides = group_overrides(components)
    if (
        len(components) != partition["counts"]["overlay_components"]
        or len(overrides) != partition["counts"]["overridden_bgc_groups"]
    ):
        raise ValueError("Cross-source component counts changed")
    eligible: dict[str, list[tuple[str, str]]] = {role: [] for role in caps}
    exclusions: Counter = Counter()
    seen = set()
    for candidate in jsonl_rows(safe_path(root, fields["field_references"]["path"])):
        identifier = candidate.get("record_id")
        assignment = assignments.get(identifier)
        if identifier in seen or assignment is None:
            raise ValueError("Unknown or repeated field reference")
        seen.add(identifier)
        expected_group = {
            key: assignment[key] for key in ("group_id", "proposed_split", "review_required")
        }
        if (
            candidate.get("group") != expected_group
            or candidate.get("source", {}).get("source_split") != assignment["source_split"]
            or candidate["source"].get("source_row") != assignment["source_row"]
        ):
            raise ValueError("Field candidate differs from its pinned assignment")
        validate_states(candidate["fields"], mapping)
        if candidate["fields"] != map_source_labels(candidate["source_labels"], mapping):
            raise ValueError("Field candidate states differ from source mapping")
        group = assignment["group_id"]
        effective = (
            overrides[group]["proposed_split"] if group in overrides else group_splits[group]
        )
        # Mutually exclusive first-reason counts, in this declared policy order.
        reason = (
            "non_singleton_group"
            if group_counts[group] != 1
            else "cross_source_component"
            if group in overrides
            else "review_required"
            if assignment["review_required"]
            else "source_reserved"
            if assignment["source_split"] not in caps
            else "effective_reserved"
            if effective not in caps
            else "source_effective_disagreement"
            if assignment["source_split"] != effective
            else None
        )
        if reason:
            exclusions[reason] += 1
        else:
            eligible[effective].append((sha(f"{POLICY}:{effective}:{identifier}"), identifier))
    if seen != assignments.keys():
        raise ValueError("Field references do not cover every BGC assignment")
    chosen = {}
    for role, cap in caps.items():
        if len(eligible[role]) < cap:
            raise ValueError(
                f"Diagnostic {role} shortfall: {len(eligible[role])} eligible, {cap} required; no resampling"
            )
        chosen[role] = [identifier for _, identifier in sorted(eligible[role])[:cap]]
    selected_ids = {identifier for ids in chosen.values() for identifier in ids}
    selected = {
        row["record_id"]: row
        for row in jsonl_rows(safe_path(root, fields["field_references"]["path"]))
        if row["record_id"] in selected_ids
    }
    selected_groups = {
        role: {assignments[identifier]["group_id"] for identifier in identifiers}
        for role, identifiers in chosen.items()
    }
    if (
        len(selected_ids) != sum(caps.values())
        or set(chosen["train"]) & set(chosen["dev"])
        or selected_groups["train"] & selected_groups["dev"]
        or any(len(selected_groups[role]) != cap for role, cap in caps.items())
    ):
        raise ValueError("Diagnostic source IDs and singleton groups must be disjoint")
    counts = {
        "source_records": len(assignments),
        "source_groups": len(group_counts),
        "excluded_by_first_reason": dict(sorted(exclusions.items())),
        "eligible": {role: len(rows) for role, rows in eligible.items()},
        "selected": dict(caps),
        "eligible_not_selected": {role: len(rows) - caps[role] for role, rows in eligible.items()},
        "selected_groups": {role: len(values) for role, values in selected_groups.items()},
        "shortfall": {role: 0 for role in caps},
        "record_overlap": 0,
        "group_overlap": 0,
    }
    return chosen, selected, counts


def _selected_source_rows(archive_path: Path, selected: dict, audit: dict):
    """Count raw frames; never decode an unselected record or open a test member."""
    wanted = {
        role: {
            row["source"]["source_row"]
            for row in selected.values()
            if row["source"]["source_split"] == role
        }
        for role in SAMPLE_COUNTS
    }
    with zipfile.ZipFile(archive_path) as archive:
        for role in SAMPLE_COUNTS:
            count, size = 0, 0
            parts: list[bytes] = []
            digest = hashlib.sha256()
            with archive.open(MEMBERS[role]) as stream:
                while raw := stream.readline(MAX_RECORD_BYTES + 1):
                    digest.update(raw)
                    if len(raw) > MAX_RECORD_BYTES:
                        raise ValueError("BGC line exceeds record bound")
                    if size == 0 and not raw.strip():
                        continue
                    size += len(raw)
                    if size > MAX_RECORD_BYTES:
                        raise ValueError("BGC record exceeds memory bound")
                    chosen = count + 1 in wanted[role]
                    if chosen:
                        parts.append(raw)
                    if raw.strip() == b"</book>":
                        count += 1
                        if chosen:
                            decoded = list(records(io.BytesIO(b"".join(parts))))
                            if len(decoded) != 1:
                                raise ValueError("Selected BGC frame must contain one record")
                            yield f"bgc:{role}:{count}", decoded[0]
                        parts, size = [], 0
            if size or count != audit["observed"]["splits"][role]["rows"]:
                raise ValueError("Source framing/count differs from the pinned audit")
            if digest.hexdigest() != audit["observed"]["members"][MEMBERS[role]]["sha256"]:
                raise ValueError("Original train/dev source member changed")


def prepare_field_baseline_data(root: Path, config: dict) -> dict:
    """Return fixed diagnostic rows and source receipts without writing any files.

    ``config`` contains checked artifact references ``field_manifest`` and
    ``partition_manifest``. The optional ``policy`` must match :data:`POLICY`.
    All sample sizes and the selection salt are fixed by this implementation.
    """
    root = root.resolve()
    fields, partition, groups, audit, mapping, bindings = _load_bindings(root, config)
    caps = None
    if config.get("kind") in {
        "book_source_assignment_data_scaling",
        "book_source_assignment_loss_weighting",
    }:
        from .field_baseline import validate_source_recovery_config

        validate_source_recovery_config(config)
        caps = {"train": config["train_limit"], "dev": config["dev_limit"]}
    chosen, selected, counts = _metadata_selection(
        root, fields, partition, groups, mapping, sample_counts=caps
    )
    resolved = {}
    for identifier, source_row in _selected_source_rows(
        safe_path(root, fields["archive"]["path"]),
        selected,
        audit,
    ):
        candidate = selected[identifier]
        payload = input_payload(source_row)
        source = {
            "source_split": candidate["source"]["source_split"],
            "source_row": candidate["source"]["source_row"],
            "isbn13": isbn13(source_row["isbn"]),
            "provider_book_id": provider_id(source_row["url"]),
            "provider_url": source_row["url"],
            "attribution": source_row["copyright"],
            "language": source_row["language"],
        }
        if (
            identifier in resolved
            or not all(value.strip() for value in payload.values())
            or input_hash(payload) != candidate["input_sha256"]
            or source != candidate["source"]
            or [[depth, label] for depth, label in source_row["labels"]]
            != candidate["source_labels"]
            or map_source_labels(source_row["labels"], mapping) != candidate["fields"]
        ):
            raise ValueError(
                f"Selected source record failed verification: {identifier}; no resampling"
            )
        resolved[identifier] = {
            "record_id": identifier,
            "group_id": candidate["group"]["group_id"],
            "source_split": source["source_split"],
            "effective_split": candidate["group"]["proposed_split"],
            "input": payload,
            "input_sha256": candidate["input_sha256"],
            "source": source,
            "source_labels": candidate["source_labels"],
            "fields": candidate["fields"],
        }
    if resolved.keys() != selected.keys():
        raise ValueError("Selected source records were not all resolved")
    input_hashes = {
        role: {resolved[identifier]["input_sha256"] for identifier in identifiers}
        for role, identifiers in chosen.items()
    }
    if input_hashes["train"] & input_hashes["dev"]:
        raise ValueError("Selected training and development inputs overlap")
    counts["input_hash_overlap"] = 0
    return {
        **{role: [resolved[identifier] for identifier in ids] for role, ids in chosen.items()},
        "mapping": mapping,
        "provenance": {
            "policy": POLICY,
            "bindings": bindings,
            "counts": counts,
            "vocabulary": {facet: mapping["facets"][facet]["labels"] for facet in FACETS},
            "selected_record_ids": chosen,
            "selection": "SHA256(policy:role:record_id), then record_id; no label stratification or resampling",
            "exclusion_order": [
                "non_singleton_group",
                "cross_source_component",
                "review_required",
                "source_reserved",
                "effective_reserved",
                "source_effective_disagreement",
            ],
            "source_members_opened": [MEMBERS[role] for role in SAMPLE_COUNTS],
            "decoded_records": len(resolved),
            "test_records_decoded": 0,
            "source_omissions": "unknown",
            "negative_labels_created": 0,
            "human_labels_created": 0,
            "human_gold": False,
            "training_eligible": False,
            "claim": "Observed authored source-label retrieval on a bounded development subset only",
            "source_declared_license": audit["source_declared_license"],
        },
    }
