"""Inspect pinned leakage groups and cross-source identities without assigning gold or splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_bgc_candidate as source
from src.catalog.storage import write_json_atomic
from src.research.candidate_io import create_or_verify, sha
from src.research.io import check_file, file_hash, parse_json, read_json, safe_path

POLICY = "book-group-evidence-review-v1"


def file_ref(path, root=ROOT):
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def lines(path, limit):
    with path.open("rb") as stream:
        for number, line in enumerate(iter(lambda: stream.readline(256_001), b""), 1):
            if number > limit or len(line) > 256_000:
                raise ValueError("Review input exceeds row or line bound")
            yield parse_json(line)


def select_groups(reviews):
    """All direct conflicts and the three largest overall/title-author-only groups."""
    chosen = defaultdict(list)
    ordered = sorted(reviews, key=lambda item: (-item["records"], item["group_id"]))
    for row in ordered:
        if row["identity_label_conflicts"]:
            chosen[row["group_id"]].append("direct_identity_label_conflict")
    for row in ordered[:3]:
        chosen[row["group_id"]].append("largest_component")
    title_only = [
        row
        for row in ordered
        if set(row["shared_key_counts"]) == {"normalized_title_author_candidate"}
    ]
    for row in title_only[:3]:
        chosen[row["group_id"]].append("largest_title_author_only_component")
    return dict(sorted(chosen.items()))


def external_records(catalogue, licensed):
    """Retain creator roles as source metadata, never assume every creator is an author."""
    records = []
    for book in catalogue:
        urls = [book["source"]["url"]]
        if book.get("firstPublishedSource"):
            urls.append(book["firstPublishedSource"]["url"])
        records.append(
            {
                "source": "catalogue",
                "id": book["id"],
                "title": book["title"],
                "creators": book["authors"],
                "creator_basis": "catalogue_authors",
                "isbns": book["identifiers"]["isbns"],
                "urls": urls,
            }
        )
    for book in licensed["books"]:
        records.append(
            {
                "source": "licensed_text",
                "id": book["work_id"],
                "title": book["title"],
                "creators": book["creators"],
                "creator_basis": "source_creators_may_include_non_authors",
                "isbns": [book["provider_isbn"]] if book.get("provider_isbn") else [],
                "urls": [book["source_page"]],
            }
        )
    if len(records) > 10_000 or len({(r["source"], r["id"]) for r in records}) != len(records):
        raise ValueError("External inventory exceeds bound or repeats source identities")
    return records


def identity_keys(title, creators, isbns, urls):
    title = source.normalized(title)
    keys = {("normalized_title_only", title)} if title else set()
    names = {source.normalized(name) for name in creators if source.normalized(name)}
    # The ordered, comma-separated full credit is an additional exact candidate key.
    if len(creators) > 1:
        names.add(source.normalized(", ".join(creators)))
    if title:
        keys.update(("normalized_title_creator_candidate", (title, name)) for name in names)
    keys.update(("isbn13", value) for raw in isbns if (value := source.isbn13(raw)))
    keys.update(("provider_book_id", value) for raw in urls if (value := source.provider_id(raw)))
    return keys


def evidence_row(row, assignment):
    return {
        **assignment,
        "archive_member": source.MEMBERS[assignment["source_split"]],
        "title": row["title"],
        "author": row["author"],
        "isbn13": source.isbn13(row["isbn"]),
        "provider_book_id": source.provider_id(row["url"]),
        "provider_url": row["url"],
        "source_labels": sorted([list(pair) for pair in row["labels"]]),
        "body_sha256": sha(row["body"]),
        "normalized_body_sha256": sha(source.normalized(row["body"])),
        "body_utf8_bytes": len(row["body"].encode()),
    }


def group_summary(review, members, reasons):
    members = sorted(members, key=lambda row: row["record_id"])
    if {row["record_id"] for row in members} != {row["record_id"] for row in review["members"]}:
        raise ValueError("Selected group members differ from reviewed assignment source")
    expected = {row["record_id"]: row for row in review["members"]}
    if any(
        row["source_labels"] != sorted(expected[row["record_id"]]["source_labels"])
        or row["group_id"] != review["group_id"]
        or row["proposed_split"] != review["proposed_split"]
        for row in members
    ):
        raise ValueError("Selected member labels or assignment differs from pinned review")
    body_groups = defaultdict(list)
    for row in members:
        if row["body_utf8_bytes"]:
            body_groups[row["normalized_body_sha256"]].append(row)
    distinct_title_text = [
        {"sha256": digest, "records": len(rows), "titles": sorted({r["title"] for r in rows})}
        for digest, rows in sorted(body_groups.items())
        if len({source.normalized(row["title"]) for row in rows}) > 1
    ]
    all_labels = [set(tuple(pair) for pair in row["source_labels"]) for row in members]
    dispositions = ["keep_constraint", "review_identity"]
    if distinct_title_text:
        dispositions.append("distinct_titles_share_text")
    return {
        "group_id": review["group_id"],
        "selection": reasons,
        "dispositions": dispositions,
        "records": len(members),
        "distinct_titles": len({row["title"] for row in members}),
        "title_examples": sorted({row["title"] for row in members})[:5],
        "distinct_authors": len({row["author"] for row in members}),
        "distinct_isbn13": len({row["isbn13"] for row in members if row["isbn13"]}),
        "distinct_normalized_blurbs": len(body_groups),
        "distinct_source_label_sets": len({tuple(sorted(labels)) for labels in all_labels}),
        "labels_varying_between_members": [
            list(pair) for pair in sorted(set.union(*all_labels) - set.intersection(*all_labels))
        ],
        "identity_label_conflicts": review["identity_label_conflicts"],
        "distinct_title_shared_blurb_clusters": len(distinct_title_text),
        "shared_key_counts": review["shared_key_counts"],
        "proposed_split_unchanged": review["proposed_split"],
        "evidence_record_ids": [row["record_id"] for row in members[:3]],
        "packet": {"members": members, "distinct_title_shared_blurbs": distinct_title_text},
    }


def scan(rows, assignments, reviews, external):
    """One source pass, keeping metadata only for selected and cross-source candidate rows."""
    chosen = select_groups(reviews)
    review_by_id = {row["group_id"]: row for row in reviews}
    groups = defaultdict(list)
    index = defaultdict(set)
    for number, record in enumerate(external):
        for key in identity_keys(
            record["title"], record["creators"], record["isbns"], record["urls"]
        ):
            index[key].add(number)
    matched = defaultdict(list)
    seen = set()
    for record_id, row in rows:
        if record_id not in assignments or record_id in seen:
            raise ValueError("Source row is missing, repeated or absent from assignments")
        seen.add(record_id)
        assigned = assignments[record_id]
        candidates = defaultdict(set)
        for kind, value in identity_keys(
            row["title"], [row["author"]], [row["isbn"]], [row["url"]]
        ):
            for other in index.get((kind, value), ()):
                candidates[other].add(kind)
        if assigned["group_id"] not in chosen and not candidates:
            continue
        evidence = evidence_row(row, assigned)
        if assigned["group_id"] in chosen:
            groups[assigned["group_id"]].append(evidence)
        for other, kinds in sorted(candidates.items()):
            stronger = kinds - {"normalized_title_only"}
            matched[other].append(
                {
                    **evidence,
                    "matching_keys": sorted(stronger or kinds),
                    "disposition": "cross_source_identity_candidate"
                    if stronger
                    else "title_only_review_do_not_merge",
                }
            )
    if seen != set(assignments):
        raise ValueError("Source rows do not completely cover pinned assignments")
    results = [
        group_summary(review_by_id[gid], groups[gid], reasons) for gid, reasons in chosen.items()
    ]
    cross = [
        {
            "external": external[number],
            "bgc_candidates": sorted(candidates, key=lambda row: row["record_id"]),
        }
        for number, candidates in sorted(
            matched.items(), key=lambda pair: (external[pair[0]]["source"], external[pair[0]]["id"])
        )
    ]
    return results, cross


def prepare(output: Path, *, root=ROOT):
    output = output.resolve()
    if not output.is_relative_to(root / "data/research_candidates"):
        raise ValueError("Review packet must remain in ignored data/research_candidates")
    paths = {
        "groups": root / "research/preparation/bgc_group_manifest.json",
        "catalogue": root / "web/data/books.json",
        "licensed_sources": root / "research/preparation/licensed_books_sources.json",
        "licensed_manifest": root / "research/preparation/licensed_books_manifest.json",
    }
    inputs = {name: file_ref(path, root) for name, path in paths.items()}
    manifest = read_json(paths["groups"])
    if manifest["archive"]["sha256"] != source.SOURCE_SHA256:
        raise ValueError("Group review requires the pinned BGC source")
    inputs.update(
        {
            key: manifest[key]
            for key in ("archive", "candidate_manifest", "assignments", "review_groups")
        }
    )
    dependencies = {str(Path(__file__).relative_to(root)): file_hash(Path(__file__))}
    for path in (
        "scripts/prepare_bgc_candidate.py",
        "src/research/book_groups.py",
        "src/research/candidate_io.py",
        "src/research/io.py",
    ):
        dependencies[path] = file_hash(root / path)
        if manifest["preparation_helper_sha256"][path] != dependencies[path]:
            raise ValueError("Grouping source parser or helper changed; rebuild groups first")
    for ref in inputs.values():
        if errors := check_file(root, ref):
            raise ValueError("; ".join(errors))
    licensed_manifest = read_json(paths["licensed_manifest"])
    if licensed_manifest["source_inventory"] != inputs["licensed_sources"]:
        raise ValueError("Licensed source inventory differs from prepared texts")
    candidate = read_json(safe_path(root, inputs["candidate_manifest"]["path"]))
    assignments = {}
    for row in lines(safe_path(root, inputs["assignments"]["path"]), 100_000):
        if row["record_id"] in assignments:
            raise ValueError("Duplicate assignment identity")
        assignments[row["record_id"]] = row
    if len(assignments) != manifest["counts"]["records"]:
        raise ValueError("Assignment count differs from group manifest")
    reviews = list(lines(safe_path(root, inputs["review_groups"]["path"]), 100_000))
    if len({row["group_id"] for row in reviews}) != len(reviews):
        raise ValueError("Duplicate review group identity")
    external = external_records(read_json(paths["catalogue"]), read_json(paths["licensed_sources"]))

    def rows():
        with zipfile.ZipFile(safe_path(root, inputs["archive"]["path"])) as archive:
            for split, member in source.MEMBERS.items():
                digest, count = hashlib.sha256(), 0
                with archive.open(member) as stream:
                    for count, row in enumerate(source.records(stream, digest=digest), 1):
                        yield f"bgc:{split}:{count}", row
                expected = candidate["observed"]["members"][member]
                if count != expected["rows"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError("Source member differs from the pinned candidate audit")

    groups, cross = scan(rows(), assignments, reviews, external)
    for ref in inputs.values():
        if errors := check_file(root, ref):
            raise ValueError("Inputs changed during review: " + "; ".join(errors))
    packet = [{"kind": "group_review", **group} for group in groups] + [
        {"kind": "cross_source_candidates", **row} for row in cross
    ]
    packet_ref = create_or_verify(
        output / "review.jsonl",
        (
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            for row in packet
        ),
    )
    packet_ref["path"] = str((output / "review.jsonl").relative_to(root))
    cross_summary = []
    for row in cross:
        strong = [
            candidate
            for candidate in row["bgc_candidates"]
            if candidate["disposition"] == "cross_source_identity_candidate"
        ]
        cross_summary.append(
            {
                "source": row["external"]["source"],
                "id": row["external"]["id"],
                "title": row["external"]["title"],
                "identity_candidate_rows": len(strong),
                "title_only_rows": len(row["bgc_candidates"]) - len(strong),
                "identity_candidate_group_ids": sorted({item["group_id"] for item in strong}),
                "matching_keys": sorted(
                    {kind for item in strong for kind in item["matching_keys"]}
                ),
            }
        )
    return {
        "schema_version": 1,
        "policy": POLICY,
        "status": "bounded_evidence_review_not_human_gold",
        "training_authorized": False,
        "training_performed": False,
        "inputs": inputs,
        "implementation_sha256": dependencies,
        "counts": {
            "bgc_records_scanned": len(assignments),
            "available_multi_record_groups": len(reviews),
            "selected_groups": len(groups),
            "selected_records": sum(g["records"] for g in groups),
            "external_records": dict(Counter(row["source"] for row in external)),
            "external_records_with_identity_candidates": sum(
                bool(row["identity_candidate_rows"]) for row in cross_summary
            ),
            "external_records_with_title_only_candidates": sum(
                bool(row["title_only_rows"]) for row in cross_summary
            ),
            "identity_candidate_pairs": sum(
                row["identity_candidate_rows"] for row in cross_summary
            ),
            "title_only_pairs": sum(row["title_only_rows"] for row in cross_summary),
        },
        "selection_policy": "All direct identity conflicts; top three groups by size; top three connected only by exact normalized title-author; deterministic group-id tie break. Overlap selected once.",
        "reviewed_groups": [
            {key: value for key, value in row.items() if key != "packet"} for row in groups
        ],
        "cross_source_candidates": cross_summary,
        "packet": packet_ref,
        "interpretation": [
            "Dispositions are reproducible evidence rules inspected during preparation, not human identity or label gold. Source labels and proposed splits are unchanged.",
            "Keep every selected leakage constraint, including distinct titles sharing blurbs. Do not collapse those rows into a single literary work or union their labels.",
            "Same title/creator is only a candidate: editions, adaptations and translations can differ in audience and form. Shared ISBN/provider syntax is not independent identity confirmation.",
            "Cross-source keys use NFC and whitespace collapse, preserving case/punctuation. Title-only collisions never establish identity; author spelling and edition variants can remain undetected.",
            "Licensed creator lists include credited non-authors. Book Dash UUIDs and Open Library work IDs are not comparable with Penguin Random House book IDs.",
            "Coordinate all recorded cross-source identity candidate groups before admitting catalogue or licensed-text holdouts; this pass does not reserve or move any splits. No match does not establish independence.",
            "Review packets contain references, source metadata, labels and text hashes only; resolve archive member and 1-based source row for the original blurb.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/research_candidates/book-group-review-v1"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/book_group_review.json"
    )
    args = parser.parse_args()
    if args.report.resolve() != ROOT / "research/preparation/book_group_review.json":
        parser.error("Report must use research/preparation/book_group_review.json")
    try:
        report = prepare(args.output_dir)
        write_json_atomic(args.report, report)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
