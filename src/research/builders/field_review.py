"""Resolve a bounded BGC review packet locally; tracked reports contain no title/blurb text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.catalog.storage import write_json_atomic
from src.research.book_fields import FACETS
from src.research.builders.book_fields import (
    checked,
    iter_resolved_candidates,
    jsonl_rows,
    reference,
)
from src.research.candidate_io import create_or_verify, json_bytes
from src.research.field_reviews import human_reviewed_states, validate_review_record
from src.research.io import file_hash, read_json

from . import ROOT

MAX_RECORDS = 48
COVERAGE_LABELS = (
    "Beauty",
    "Classics",
    "Crafts, Home & Garden",
    "Food Memoir & Travel",
    "Health & Reference",
    "Home & Garden",
    "Religion & Philosophy",
    "Step Into Reading",
    "Weddings",
    "Women’s Fiction",
    "Fantasy",
    "Science Fiction",
    "Romance",
    "Mystery & Suspense",
    "Historical Fiction",
    "History",
    "Cooking",
    "Science",
    "Psychology",
    "Politics",
    "Poetry",
    "Graphic Novels & Manga",
    "Teen & Young Adult",
    "Children’s Middle Grade Books",
)
NEGATION_RECORDS = {
    "bgc:train:6018": "book-level romance denial",
    "bgc:train:25394": "book-level child-audience denial",
    "bgc:train:6753": "under-three safety warning is not a child-audience denial",
    "bgc:train:10611": "character dialogue is not a genre denial",
    "bgc:train:28677": "qualified novel denial is not a fiction denial",
    "bgc:train:34106": "memoir denial does not exclude every life-writing form",
    "bgc:dev:8007": "cookbook denial does not negate cooking aboutness",
    "bgc:test:5264": "autobiography denial does not exclude autobiographical elements",
}
CONTRACT = {
    "default_state": "unknown",
    "source_omissions": "unknown",
    "text_scope": "Literal source title and description only; metadata stays outside classifier input.",
    "evidence": "Unicode character offsets plus SHA-256; resolve passages from the pinned source ZIP.",
    "agent_reviews": "Assistant-authored candidates, not independent human annotation or gold.",
    "human_reviews": "Null until a person reviews source evidence; never inherit agent decisions.",
    "admission": "No review status authorizes training; source-positive conflicts require separate adjudication.",
}


def select_records(candidates) -> list[dict]:
    """Deterministic purposive coverage, not a representative sample or evaluation set."""
    pending = set(COVERAGE_LABELS)
    special = dict(NEGATION_RECORDS)
    selected = []
    for candidate in candidates:
        labels = {label for _, label in candidate["source_labels"]}
        matches = [label for label in COVERAGE_LABELS if label in labels and label in pending]
        reasons = []
        if matches:
            # One distinct source row per coverage target; overlap is reported, not extra sampling.
            label = matches[0]
            pending.remove(label)
            reasons.append("source_label:" + label)
        if candidate["record_id"] in special:
            reasons.append("negation_context:" + special.pop(candidate["record_id"]))
        if reasons:
            selected.append(
                {
                    "record_id": candidate["record_id"],
                    "input_sha256": candidate["input_sha256"],
                    "source_url": candidate["source"]["provider_url"],
                    "selection": reasons,
                    "agent_review": None,
                    "human_review": None,
                }
            )
        if not pending and not special:
            break
    if pending or special:
        raise ValueError("Source did not supply every declared review selection")
    if len(selected) > MAX_RECORDS:
        raise ValueError("Review selection exceeds the bounded packet")
    return selected


def _bindings(root: Path, manifest_path: Path, manifest: dict) -> dict:
    return {
        "field_manifest": reference(root, manifest_path),
        **{name: manifest[name] for name in ("archive", "mapping", "field_references")},
    }


def initialize_packet(root: Path, manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    rows_path = checked(root, manifest["field_references"])
    return {
        "schema_version": 1,
        "status": "candidate_field_reviews_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "bindings": _bindings(root, manifest_path, manifest),
        "contract": dict(CONTRACT),
        "selection": "First source row per declared category, plus eight hand-selected negation contexts; purposive and unsuitable for unbiased evaluation.",
        "records": select_records(jsonl_rows(rows_path)),
    }


def prepare_review(root: Path, packet_path: Path, manifest_path: Path, directory: Path) -> dict:
    root, directory = root.resolve(), directory.resolve()
    if not directory.is_relative_to(root / "data/research_candidates/bgc"):
        raise ValueError("Resolved review text must remain in ignored data/research_candidates/bgc")
    packet_ref = reference(root, packet_path)
    packet, manifest = read_json(packet_path), read_json(manifest_path)
    if (
        set(packet)
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
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
        or packet["status"] != "candidate_field_reviews_not_admitted"
        or packet["training_authorized"] is not False
        or packet["human_gold"] is not False
        or packet["contract"] != CONTRACT
        or not isinstance(packet["selection"], str)
        or not packet["selection"].strip()
    ):
        raise ValueError("Review packet must retain unknown states and remain unadmitted")
    if packet["bindings"] != _bindings(root, manifest_path, manifest):
        raise ValueError("Review source bindings changed; re-review rather than silently repinning")
    for item in packet["bindings"].values():
        checked(root, item)
    records = packet["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError("Review packet must contain 1 to 48 records")
    if any(
        not isinstance(row, dict) or not isinstance(row.get("record_id"), str) for row in records
    ):
        raise ValueError("Review records require source record IDs")
    requested = {row["record_id"]: row for row in records}
    if len(requested) != len(records):
        raise ValueError("Repeated review source record")
    mapping = read_json(checked(root, manifest["mapping"]))
    worksheet, found = [], set()
    states: dict[str, Counter[str]] = {kind: Counter() for kind in ("agent", "human")}
    reviewed: Counter[str] = Counter()
    source_labels: set[str] = set()
    human_label_count = 0
    for item in iter_resolved_candidates(root, manifest):
        candidate, payload = item["candidate"], item["input"]
        review = requested.get(candidate["record_id"])
        if review is None:
            continue
        validate_review_record(review, candidate, payload, mapping)
        found.add(candidate["record_id"])
        source_labels.update(label for _, label in candidate["source_labels"])
        for kind in states:
            value = review[kind + "_review"]
            if value is not None:
                reviewed[kind] += 1
                states[kind].update(decision["state"] for decision in value["decisions"])
        human_labels = human_reviewed_states(review, candidate, payload, mapping)
        human_label_count += sum(
            len(labels) for facet in human_labels.values() for labels in facet.values()
        )
        worksheet.append(
            {
                "review": review,
                "input": payload,
                "source": candidate["source"],
                "source_labels": candidate["source_labels"],
                "source_fields": candidate["fields"],
                "group": candidate["group"],
            }
        )
        if found == requested.keys():
            break
    if found != requested.keys():
        raise ValueError("Review records could not be resolved in the pinned BGC source")
    for item in packet["bindings"].values():
        checked(root, item)
    checked(root, packet_ref)
    worksheet_path = directory / "review_worksheet.json"
    resolved = {
        "path": str(worksheet_path.relative_to(root)),
        **create_or_verify(worksheet_path, [json_bytes(worksheet)]),
    }
    declined = {
        label for label, entry in mapping["mappings"].items() if entry["decision"] != "mapped"
    }
    return {
        "schema_version": 1,
        "status": "candidate_field_reviews_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "review_packet": packet_ref,
        "bindings": packet["bindings"],
        "local_worksheet": resolved,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": {
            name: file_hash(root / name)
            for name in (
                "src/research/builders/book_fields.py",
                "src/research/field_reviews.py",
                "src/research/book_fields.py",
                "src/research/candidate_io.py",
                "src/research/io.py",
                "src/catalog/storage.py",
            )
        },
        "observed": {
            "records": len(records),
            "source_labels_covered": len(source_labels),
            "declined_source_labels_covered": sorted(source_labels & declined),
            "declined_source_labels_missing": sorted(declined - source_labels),
            "reviewed_records": {kind: reviewed[kind] for kind in states},
            "decisions": {
                kind: {state: values[state] for state in ("positive", "negative", "unknown")}
                for kind, values in states.items()
            },
            "explicit_human_labels": human_label_count,
            "vocabulary_labels": sum(len(mapping["facets"][field]["labels"]) for field in FACETS),
        },
        "remaining_gates": [
            "Independent human review and source-positive conflict adjudication.",
            "Representative evaluation sampling; this purposive packet cannot estimate accuracy.",
            "Dataset admission, split freeze and training authorization remain separate.",
        ],
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fields", type=Path, default=ROOT / "research/preparation/book_field_manifest.json"
    )
    parser.add_argument(
        "--packet", type=Path, default=ROOT / "research/preparation/book_field_review.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/book_field_review_manifest.json"
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Create empty review slots once; never overwrite authored reviews",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    paths = [args.fields.resolve(), args.packet.resolve(), args.report.resolve()]
    if len(set(paths)) != len(paths) or any(
        not path.is_relative_to(ROOT / "research/preparation") for path in paths
    ):
        parser.error("Fields, packet and report must be distinct files under research/preparation")
    try:
        if args.initialize:
            create_or_verify(args.packet, [json_bytes(initialize_packet(ROOT, args.fields))])
        packet_hash = file_hash(args.packet)
        directory = ROOT / "data/research_candidates/bgc" / f"field-review-{packet_hash[:16]}"
        report = prepare_review(ROOT, args.packet, args.fields, directory)
        write_json_atomic(args.report, report)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report["observed"], ensure_ascii=False))
    return 0
