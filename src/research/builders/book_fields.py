"""Prepare weak BGC field references, without copying blurbs or starting training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path

from src.catalog.storage import write_json_atomic
from src.research.book_fields import (
    FACETS,
    input_hash,
    input_payload,
    map_source_labels,
    validate_mapping,
)
from src.research.builders.bgc_source import MEMBERS, SOURCE_SHA256, isbn13, provider_id, records
from src.research.candidate_io import create_or_verify, helper_hashes
from src.research.io import check_file, file_hash, parse_json, read_json, safe_path

from . import ROOT

MAX_REFERENCE_LINE = 32_000


def reference(root: Path, path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def checked(root: Path, item: dict) -> Path:
    errors = check_file(root, item)
    if errors:
        raise ValueError("; ".join(errors))
    return safe_path(root, item["path"])


def jsonl_rows(path: Path):
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as stream:
        while raw := stream.readline(MAX_REFERENCE_LINE + 1):
            if len(raw) > MAX_REFERENCE_LINE:
                raise ValueError("Candidate reference line exceeds bound")
            row = parse_json(raw)
            if not isinstance(row, dict):
                raise ValueError("Candidate reference must be a JSON object")
            yield row


def compressed(chunks):
    """Deterministic gzip stream: zlib's wrapper has zero mtime and no filename."""
    compressor = zlib.compressobj(level=6, wbits=31)
    for chunk in chunks:
        if block := compressor.compress(chunk):
            yield block
    yield compressor.flush()


def load_assignments(db, path: Path):
    db.execute(
        "CREATE TABLE assignments(record_id TEXT PRIMARY KEY, source_split TEXT, source_row INTEGER, group_id TEXT, proposed_split TEXT, review_required INTEGER, consumed INTEGER DEFAULT 0)"
    )
    for row in jsonl_rows(path):
        split, number = row.get("source_split"), row.get("source_row")
        if (
            split not in MEMBERS
            or type(number) is not int
            or number < 1
            or row.get("record_id") != f"bgc:{split}:{number}"
            or not isinstance(row.get("group_id"), str)
            or not re.fullmatch(r"bgc-group:[a-f0-9]{64}", row["group_id"])
            or row.get("proposed_split") not in MEMBERS
            or type(row.get("review_required")) is not bool
        ):
            raise ValueError("Invalid BGC leakage-group assignment")
        try:
            db.execute(
                "INSERT INTO assignments VALUES (?,?,?,?,?,?,0)",
                (
                    row["record_id"],
                    split,
                    number,
                    row["group_id"],
                    row["proposed_split"],
                    row["review_required"],
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("Repeated BGC group-assignment record") from error
    if db.execute(
        "SELECT group_id FROM assignments GROUP BY group_id HAVING COUNT(DISTINCT proposed_split)>1 LIMIT 1"
    ).fetchone():
        raise ValueError("A leakage group crosses proposed splits")


def prepare_fields(root: Path, mapping_path: Path, grouping_path: Path, directory: Path) -> dict:
    root, directory = root.resolve(), directory.resolve()
    if not directory.is_relative_to(root / "data/research_candidates/bgc"):
        raise ValueError("Field candidates must remain in ignored data/research_candidates/bgc")
    audit_path = root / "research/preparation/bgc_candidate_manifest.json"
    for path in (mapping_path, grouping_path):
        if not path.resolve().is_relative_to(root / "research/preparation"):
            raise ValueError(
                "Field mapping and group manifest must remain under research/preparation"
            )
    bindings = {
        "candidate_manifest": reference(root, audit_path),
        "grouping_manifest": reference(root, grouping_path),
        "mapping": reference(root, mapping_path),
    }
    audit, mapping, grouping = (
        read_json(audit_path),
        read_json(mapping_path),
        read_json(grouping_path),
    )
    if (
        audit.get("status") != "source_audited_not_admitted"
        or audit.get("training_authorized") is not False
    ):
        raise ValueError("Expected the unadmitted BGC source audit")
    archive_path = checked(root, audit["archive"])
    if audit["archive"]["sha256"] != SOURCE_SHA256:
        raise ValueError("BGC archive is not the reviewed source")
    if audit["preparation_script_sha256"] != file_hash(
        root / "src/research/builders/bgc_source.py"
    ) or audit["preparation_helper_sha256"] != helper_hashes(root):
        raise ValueError("BGC source audit dependencies changed")
    if (
        grouping.get("archive") != audit["archive"]
        or grouping.get("training_authorized") is not False
        or grouping.get("status") != "candidate_leakage_groups_not_admitted"
        or grouping.get("candidate_manifest") != bindings["candidate_manifest"]
    ):
        raise ValueError("Grouping must use the same archive and remain unadmitted")
    if grouping["preparation_script_sha256"] != file_hash(
        root / "src/research/builders/bgc_groups.py"
    ):
        raise ValueError("BGC grouping builder changed")
    group_helpers = {
        **helper_hashes(root),
        "src/research/book_groups.py": file_hash(root / "src/research/book_groups.py"),
        "src/research/builders/bgc_source.py": file_hash(
            root / "src/research/builders/bgc_source.py"
        ),
    }
    if grouping["preparation_helper_sha256"] != group_helpers:
        raise ValueError("BGC grouping dependencies changed or are incomplete")
    assignment_ref = grouping["assignments"]
    assignments_path = checked(root, assignment_ref)
    validate_mapping(
        mapping,
        source_labels=set(audit["observed"]["label_record_counts"]),
        archive_sha256=audit["archive"]["sha256"],
        hierarchy_sha256=audit["observed"]["members"]["hierarchy.txt"]["sha256"],
    )
    count: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    positive_counts: dict[str, Counter[str]] = {field: Counter() for field in FACETS}
    coverage: Counter[str] = Counter()
    proposed_labels: dict[str, dict[str, set[str]]] = {
        split: {field: set() for field in FACETS} for split in MEMBERS
    }
    output = directory / "field_references.jsonl.gz"
    uncompressed_bytes = 0
    with (
        zipfile.ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(prefix="leximind-fields-") as temporary,
        sqlite3.connect(Path(temporary) / "groups.sqlite") as db,
    ):
        load_assignments(db, assignments_path)
        hierarchy_raw = archive.read("hierarchy.txt")
        if hashlib.sha256(hierarchy_raw).hexdigest() != mapping["source_hierarchy_sha256"]:
            raise ValueError("BGC source hierarchy changed")
        parents: dict[str, set[str]] = {}
        for line in hierarchy_raw.decode("utf-8").splitlines():
            pair = line.split("\t")
            if len(pair) == 2:
                parents.setdefault(pair[1], set()).add(pair[0])

        def chunks():
            nonlocal uncompressed_bytes
            observed_labels: Counter = Counter()
            for split, member in MEMBERS.items():
                with archive.open(member) as stream:
                    for number, row in enumerate(records(stream), 1):
                        record_id = f"bgc:{split}:{number}"
                        assignment = db.execute(
                            "SELECT group_id,proposed_split,review_required FROM assignments WHERE record_id=? AND consumed=0",
                            (record_id,),
                        ).fetchone()
                        if assignment is None:
                            raise ValueError(f"Missing BGC group assignment: {record_id}")
                        db.execute(
                            "UPDATE assignments SET consumed=1 WHERE record_id=?", (record_id,)
                        )
                        present = {label for _, label in row["labels"]}
                        if any(not parents.get(label, set()) <= present for label in present):
                            raise ValueError("BGC record lacks a declared source parent")
                        observed_labels.update(present)
                        states = map_source_labels(row["labels"], mapping)
                        count[split] += 1
                        if not any(value["positive"] for value in states.values()):
                            coverage["records_without_any_mapped_positive"] += 1
                        for label in present:
                            decisions[mapping["mappings"][label]["decision"]] += 1
                        for field in FACETS:
                            positive_counts[field].update(states[field]["positive"])
                            coverage[field] += bool(states[field]["positive"])
                            proposed_labels[assignment[1]][field].update(states[field]["positive"])
                        candidate = {
                            "record_id": record_id,
                            "source": {
                                "source_split": split,
                                "source_row": number,
                                "isbn13": isbn13(row["isbn"]),
                                "provider_book_id": provider_id(row["url"]),
                                "provider_url": row["url"],
                                "attribution": row["copyright"],
                                "language": row["language"],
                            },
                            "input_sha256": input_hash(input_payload(row)),
                            "source_labels": [[depth, label] for depth, label in row["labels"]],
                            "fields": states,
                            "group": {
                                "group_id": assignment[0],
                                "proposed_split": assignment[1],
                                "review_required": bool(assignment[2]),
                            },
                        }
                        encoded = (
                            json.dumps(
                                candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        ).encode("utf-8")
                        if len(encoded) > MAX_REFERENCE_LINE:
                            raise ValueError("Candidate reference line exceeds bound")
                        uncompressed_bytes += len(encoded)
                        yield encoded
            expected = {split: item["rows"] for split, item in audit["observed"]["splits"].items()}
            if (
                dict(count) != expected
                or dict(observed_labels) != audit["observed"]["label_record_counts"]
            ):
                raise ValueError("BGC row/label counts differ from the source audit")
            if db.execute("SELECT COUNT(*) FROM assignments WHERE consumed=0").fetchone()[0]:
                raise ValueError("Unmatched extra BGC group assignments")
            for item in [audit["archive"], assignment_ref, *bindings.values()]:
                checked(root, item)

        output_ref = {
            "path": str(output.relative_to(root)),
            **create_or_verify(output, compressed(chunks())),
        }
    helpers = helper_hashes(root)
    for name in (
        "src/research/builders/bgc_source.py",
        "src/research/book_fields.py",
        "src/catalog/storage.py",
    ):
        helpers[name] = file_hash(root / name)
    return {
        "schema_version": 1,
        "status": "weak_field_candidates_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "archive": audit["archive"],
        **bindings,
        "assignments": assignment_ref,
        "field_references": output_ref,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": helpers,
        "observed": {
            "records": sum(count.values()),
            "source_splits": dict(count),
            "mapped_source_labels": dict(
                Counter(value["decision"] for value in mapping["mappings"].values())
            ),
            "source_label_record_decisions": dict(decisions),
            "positive_record_counts": {
                field: dict(sorted(values.items())) for field, values in positive_counts.items()
            },
            "negative_labels": 0,
            "records_without_any_mapped_positive": coverage["records_without_any_mapped_positive"],
            "records_with_positive_by_field": {field: coverage[field] for field in FACETS},
            "proposed_split_label_coverage": {
                split: {
                    field: {
                        "present_labels": len(labels),
                        "missing_labels": sorted(set(mapping["facets"][field]["labels"]) - labels),
                    }
                    for field, labels in fields.items()
                }
                for split, fields in proposed_labels.items()
            },
            "uncompressed_reference_bytes": uncompressed_bytes,
        },
        "contract": {
            "input": ["title", "description"],
            "text_storage": "Original ZIP only; references resolve source rows and verify canonical JSON input SHA-256.",
            "default_state": "unknown",
            "label_provenance": "Authored mapping of provider metadata; not human gold or complete negatives.",
            "group_semantics": "Conservative leakage constraints and proposed splits, not adjudicated works or admitted training splits.",
            "license": audit["source_declared_license"],
        },
        "remaining_gates": [
            "Review ambiguous/unsupported mappings and selected positive labels.",
            "Review grouping candidates and freeze a work-aware split.",
            "Collect explicit negatives or complete-label gold; positive-only masking does not establish a trainable objective.",
            "Partial-label runtime integration remains unimplemented; training remains paused.",
        ],
    }


def iter_resolved_candidates(root: Path, manifest: dict):
    """Stream candidate + exact title/blurb; all targets and provenance stay outside input."""
    if (
        manifest.get("status") != "weak_field_candidates_not_admitted"
        or manifest.get("training_authorized") is not False
        or manifest.get("human_gold") is not False
    ):
        raise ValueError("Resolver accepts only unadmitted weak field candidates")
    for name in ("candidate_manifest", "grouping_manifest", "mapping", "assignments"):
        checked(root, manifest[name])
    archive_path = checked(root, manifest["archive"])
    rows_path = checked(root, manifest["field_references"])
    with zipfile.ZipFile(archive_path) as archive:
        candidates = iter(jsonl_rows(rows_path))
        for split, member in MEMBERS.items():
            with archive.open(member) as stream:
                for number, source_row in enumerate(records(stream), 1):
                    candidate = next(candidates, None)
                    if candidate is None or candidate["record_id"] != f"bgc:{split}:{number}":
                        raise ValueError("Candidate/source row alignment changed")
                    payload = input_payload(source_row)
                    if candidate["input_sha256"] != input_hash(payload):
                        raise ValueError("Candidate input hash changed")
                    yield {"input": payload, "candidate": candidate}
        if next(candidates, None) is not None:
            raise ValueError("Extra candidate rows outside BGC source")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mapping", type=Path, default=ROOT / "research/preparation/book_field_mapping.json"
    )
    parser.add_argument(
        "--groups", type=Path, default=ROOT / "research/preparation/bgc_group_manifest.json"
    )
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "research/preparation/book_field_manifest.json",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.report.resolve().is_relative_to(
        ROOT / "research/preparation"
    ) or args.report.resolve() in {
        args.mapping.resolve(),
        args.groups.resolve(),
        ROOT / "research/preparation/bgc_candidate_manifest.json",
    }:
        parser.error("Report must be separate from source manifests under research/preparation")
    try:
        group = read_json(args.groups)
        directory = (
            args.candidate_dir
            or ROOT
            / "data/research_candidates/bgc"
            / SOURCE_SHA256
            / f"fields-{file_hash(args.mapping)[:12]}-{group['assignments']['sha256'][:12]}"
        )
        report = prepare_fields(ROOT, args.mapping, args.groups, directory)
        write_json_atomic(args.report, report)
    except (KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "records": report["observed"]["records"],
                "negative_labels": 0,
            }
        )
    )
    return 0
