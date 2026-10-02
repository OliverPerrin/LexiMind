"""Prepare immutable BGC/catalogue/licensed split overlays from pinned identity evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.catalog.storage import write_json_atomic
from src.research.book_partitions import (
    POLICY,
    SPLITS,
    STRONG_KEYS,
    build_components,
    group_overrides,
)
from src.research.builders.book_groups_review import (
    external_records,
    file_ref,
    identity_keys,
    lines,
)
from src.research.candidate_io import create_or_verify
from src.research.io import check_file, file_hash, parse_json, read_json, safe_path

from . import ROOT


def field_support(root, inputs, bgc_rows, bgc_groups, overrides):
    """Count weak positive/negative/unknown columns without decoding source blurbs."""
    mapping = read_json(safe_path(root, inputs["field_mapping"]["path"]))
    vocabulary = {
        f"{facet}:{label}" for facet, spec in mapping["facets"].items() for label in spec["labels"]
    }
    rows: Counter[str] = Counter()
    seen: set[str] = set()
    support: dict[str, dict[str, Counter[str]]] = {
        split: {state: Counter() for state in ("positive", "negative")}
        for split in (*sorted(SPLITS), "quarantine")
    }
    original: dict[str, Counter[str]] = {split: Counter() for split in sorted(SPLITS)}
    with gzip.open(safe_path(root, inputs["field_references"]["path"]), "rb") as stream:
        for number, raw in enumerate(iter(lambda: stream.readline(256_001), b""), 1):
            if number > 100_000 or len(raw) > 256_000:
                raise ValueError("Field references exceed row or line bounds")
            row = parse_json(raw)
            rid, gid = row["record_id"], row["group"]["group_id"]
            if (
                rid in seen
                or bgc_rows.get(rid) != gid
                or row["group"]["proposed_split"] != bgc_groups[gid]
            ):
                raise ValueError("Field reference differs from its pinned base assignment")
            seen.add(rid)
            before = bgc_groups[gid]
            after = (
                (overrides[gid]["proposed_split"] or "quarantine") if gid in overrides else before
            )
            rows[after] += 1
            positives: set[str] = set()
            negatives: set[str] = set()
            for facet, values in row["fields"].items():
                positives.update(f"{facet}:{label}" for label in values["positive"])
                negatives.update(f"{facet}:{label}" for label in values["negative"])
            if (positives | negatives) - vocabulary or positives & negatives:
                raise ValueError("Field reference has unknown or conflicting column states")
            support[after]["positive"].update(positives)
            support[after]["negative"].update(negatives)
            original[before].update(positives)
    if seen != set(bgc_rows):
        raise ValueError("Field references do not cover every BGC assignment exactly once")
    return {
        "columns": len(vocabulary),
        "basis": "Pinned weak source-mapped labels only; missing columns remain unknown, never negative. No human review or negative evidence is inferred.",
        "by_proposed_split": {
            split: {
                "records": rows[split],
                "labels": {
                    label: {
                        "positive": states["positive"][label],
                        "negative": states["negative"][label],
                        "unknown": rows[split]
                        - states["positive"][label]
                        - states["negative"][label],
                        "positive_change_from_base": states["positive"][label]
                        - (original[split][label] if split in original else 0),
                    }
                    for label in sorted(vocabulary)
                },
                "missing_positive_columns": sorted(vocabulary - states["positive"].keys()),
            }
            for split, states in support.items()
        },
    }


def prepare(output: Path, *, root=ROOT):
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "data/research_candidates"):
        raise ValueError("Partition overlay must remain in ignored data/research_candidates")
    review_path = root / "research/preparation/book_group_review.json"
    review = read_json(review_path)
    if review.get("policy") != "book-group-evidence-review-v1":
        raise ValueError("Unknown cross-source review policy")
    inputs = {"review": file_ref(review_path, root), **review["inputs"], "packet": review["packet"]}
    fields_path = root / "research/preparation/book_field_manifest.json"
    fields = read_json(fields_path)
    if (
        fields["assignments"] != inputs["assignments"]
        or fields["grouping_manifest"] != inputs["groups"]
    ):
        raise ValueError("Field references use different base assignments or groups")
    inputs.update(
        {
            "fields": file_ref(fields_path, root),
            "field_mapping": fields["mapping"],
            "field_references": fields["field_references"],
        }
    )
    for reference in inputs.values():
        if errors := check_file(root, reference):
            raise ValueError("; ".join(errors))
        if safe_path(root, reference["path"]).is_relative_to(output):
            raise ValueError("Output cannot contain or replace pinned inputs")
    groups = read_json(safe_path(root, inputs["groups"]["path"]))
    if groups["assignments"] != inputs["assignments"]:
        raise ValueError("Review assignments differ from the pinned group manifest")
    licensed = read_json(safe_path(root, inputs["licensed_manifest"]["path"]))
    if licensed["source_inventory"] != inputs["licensed_sources"]:
        raise ValueError("Licensed work inventory changed since text preparation")
    external = external_records(
        read_json(safe_path(root, inputs["catalogue"]["path"])),
        read_json(safe_path(root, inputs["licensed_sources"]["path"])),
    )
    if {row["id"] for row in external if row["source"] == "licensed_text"} != {
        book["work_id"] for book in licensed["books"]
    }:
        raise ValueError("Licensed work identities differ from prepared texts")
    external_by_id = {f"{row['source']}:{row['id']}": row for row in external}
    evidence = [
        row
        for row in lines(safe_path(root, inputs["packet"]["path"]), 10_000)
        if row["kind"] == "cross_source_candidates"
    ]
    wanted = {row["record_id"] for entry in evidence for row in entry["bgc_candidates"]}
    referenced: dict[str, dict] = {}
    seen: dict[str, str] = {}
    bgc_groups: dict[str, str] = {}
    group_rows: Counter[str] = Counter()
    for row in lines(safe_path(root, inputs["assignments"]["path"]), 100_000):
        rid, gid, split = row["record_id"], row["group_id"], row["proposed_split"]
        if (
            rid in seen
            or split not in SPLITS
            or type(row["source_row"]) is not int
            or row["source_row"] < 1
            or row["source_split"] not in SPLITS
            or rid != f"bgc:{row['source_split']}:{row['source_row']}"
            or (gid in bgc_groups and bgc_groups[gid] != split)
        ):
            raise ValueError("Invalid, repeated or conflicting base assignment")
        seen[rid] = gid
        bgc_groups[gid] = split
        group_rows[gid] += 1
        if rid in wanted:
            referenced[rid] = row
    if len(seen) != groups["counts"]["records"] or len(bgc_groups) != groups["counts"]["groups"]:
        raise ValueError("Base record/group counts differ from the pinned manifest")
    strong: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    constraints, seen_external = [], set()
    for entry in evidence:
        other = entry["external"]
        node = f"{other['source']}:{other['id']}"
        if node in seen_external or external_by_id.get(node) != other:
            raise ValueError("Review external identity is repeated or differs from its source")
        seen_external.add(node)
        candidate_ids = set()
        for candidate in entry["bgc_candidates"]:
            rid = candidate["record_id"]
            if (
                rid in candidate_ids
                or rid not in referenced
                or any(
                    candidate[key] != referenced[rid][key]
                    for key in ("group_id", "source_split", "source_row", "proposed_split")
                )
            ):
                raise ValueError("Review candidate differs from its exact base assignment")
            candidate_ids.add(rid)
            kinds = set(candidate["matching_keys"])
            if (
                candidate["disposition"] == "cross_source_identity_candidate"
                and kinds
                and kinds <= STRONG_KEYS
            ):
                target = strong
            elif candidate["disposition"] == "title_only_review_do_not_merge" and kinds == {
                "normalized_title_only"
            }:
                target = ambiguous
            else:
                raise ValueError("Unsupported review constraint or matching evidence")
            pair = (node, candidate["group_id"])
            target.append(pair)
            constraints.append(
                {
                    "members": list(pair),
                    "matching_keys": sorted(kinds),
                    "evidence_record_id": rid,
                    "kind": "identity_candidate" if target is strong else "unresolved_title_only",
                }
            )
    if (
        len(strong) != review["counts"]["identity_candidate_pairs"]
        or len(ambiguous) != review["counts"]["title_only_pairs"]
    ):
        raise ValueError("Review constraint counts differ from its report")
    bgc_pair_counts = {
        "identity_candidate_pairs": len(strong),
        "unresolved_title_only_pairs": len(ambiguous),
    }
    # Include direct external-source matches even when no BGC row connects them.
    key_index: dict[tuple, set[str]] = defaultdict(set)
    matches: dict[tuple[str, str], set[str]] = defaultdict(set)
    for node, row in external_by_id.items():
        for key in identity_keys(row["title"], row["creators"], row["isbns"], row["urls"]):
            key_index[key].add(node)
    for (kind, _), nodes in key_index.items():
        for pair in combinations(sorted(nodes), 2):
            matches[pair].add(kind)
    external_counts: Counter[str] = Counter()
    for (left, right), matching in sorted(matches.items()):
        stronger = matching & STRONG_KEYS
        kind = "identity_candidate" if stronger else "unresolved_title_only"
        (strong if stronger else ambiguous).append((left, right))
        constraints.append(
            {
                "members": [left, right],
                "matching_keys": sorted(stronger or matching),
                "kind": kind,
            }
        )
        external_counts[kind] += 1
    components = build_components(bgc_groups, external_by_id, strong, ambiguous)
    overrides = group_overrides(components)
    by_node = {node: component for component in components for node in component["members"]}
    residual = sum(
        by_node[left]["component_id"] != by_node[right]["component_id"] for left, right in strong
    )
    if residual or any(
        by_node[node]["proposed_split"] is not None for pair in ambiguous for node in pair
    ):
        raise ValueError(
            "Identity constraints cross components or unresolved endpoints escaped quarantine"
        )
    coverage = field_support(root, inputs, seen, bgc_groups, overrides)
    counts: Counter[str] = Counter()
    for gid, original in bgc_groups.items():
        component = overrides.get(gid)
        proposed = component["proposed_split"] if component else original
        counts[proposed or "quarantine"] += group_rows[gid]
    # Verify every pinned input again before publishing an immutable overlay.
    for reference in inputs.values():
        if errors := check_file(root, reference):
            raise ValueError("Inputs changed during preparation: " + "; ".join(errors))
    references = {}
    for name, rows in (("components", components), ("constraints", constraints)):
        path = output / f"{name}.jsonl"
        reference = create_or_verify(
            path,
            (
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ),
        )
        references[name] = {"path": str(path.relative_to(root)), **reference}
    return {
        "schema_version": 1,
        "policy": POLICY,
        "status": "candidate_component_overlay_not_admitted",
        "training_authorized": False,
        "training_performed": False,
        "inputs": inputs,
        "implementation_sha256": {
            path: file_hash(root / path)
            for path in (
                "src/research/builders/book_partitions.py",
                "src/research/builders/book_groups_review.py",
                "src/research/builders/bgc_source.py",
                "src/research/book_partitions.py",
                "src/catalog/storage.py",
                "src/research/candidate_io.py",
                "src/research/io.py",
            )
        },
        **references,
        "assignment_resolution": "Resolve each BGC record through inputs.assignments. If its group_id appears in components.members, use that component_id/proposed_split/status; otherwise retain the base group_id/proposed_split as candidate_not_admitted. External namespaces must appear in components.members. A null proposed_split excludes the entire component from train/dev/test.",
        "split_policy": {
            "untouched_bgc_groups": "inherit_immutable_base_assignment",
            "overlay_components": {"train": 0.64, "dev": 0.16, "test": 0.2},
            "title_only_ambiguity": "quarantine_both_connected_endpoints_without_identity_merge",
        },
        "counts": {
            "bgc_records": len(seen),
            "bgc_groups": len(bgc_groups),
            "external_records": dict(Counter(row["source"] for row in external)),
            "overlay_components": len(components),
            "overridden_bgc_groups": len(overrides),
            "quarantined_components": sum(row["proposed_split"] is None for row in components),
            "quarantined_external_records": sum(
                node in external_by_id
                for row in components
                if row["proposed_split"] is None
                for node in row["members"]
            ),
            "effective_bgc_split_records": dict(sorted(counts.items())),
            "bgc_external_constraints": bgc_pair_counts,
            "external_external_constraints": dict(sorted(external_counts.items())),
            "cross_split_identity_constraints": residual,
        },
        "field_support": coverage,
        "interpretation": [
            "Components are conservative leakage constraints, not verified works, human identity adjudications or merged label gold. All candidates remain ineligible for training until separate admission.",
            "All source rows, labels and historical split files remain unchanged. This sparse versioned overlay contains only identities, references and proposed allocations, never copied blurbs or book text.",
            "Zero cross-split constraints covers the recorded exact ISBN/provider/title-creator links only. Title-only matches remain unresolved and quarantine both whole connected components; no match does not establish independence.",
            "Licensed work nodes cover every current and future chapter/page of that work. Source licenses and provenance still govern later use.",
            "Scope is the pinned BGC, catalogue and four licensed-work inventories. CR4 source-document identities and any other corpus are excluded until separate evidence provides namespaced work constraints; these are not globally finalized splits.",
            "Overlay hashing is independent of labels and model outputs and is not stratified. Untouched BGC assignments retain the historical proposal; merged components need separate support and admission review.",
        ],
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/research_candidates/book-partitions-v1"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/book_partition_manifest.json"
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.report.resolve() != ROOT / "research/preparation/book_partition_manifest.json":
        parser.error("Report must use research/preparation/book_partition_manifest.json")
    try:
        report = prepare(args.output_dir)
        write_json_atomic(args.report, report)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report["counts"]))
    return 0
