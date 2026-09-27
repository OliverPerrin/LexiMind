"""Read-only, streaming inventory and split-integrity audit of prepared JSONL data.

This inspects source artifacts, not models or predictions. It never assigns splits,
repairs records, or grants training approval. --require-ready fails closed while
identity, leakage, or external protocol/source review remains unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.identity import author_identity, normalize_title
from src.catalog.storage import write_json_atomic

VERSION = "data-readiness/v1"
TASKS = {"books", "emotion", "summarization", "topic"}
SPLITS = {"train", "validation", "test"}
KINDS = (
    "input_raw",
    "input_normalized",
    "target_normalized",
    "declared_work",
    "declared_document",
    "title_author_candidate",
    "title_only_candidate",
)
PROVENANCE_FIELDS = (
    "identity_source",
    "description_source",
    "summary_source_url",
    "source_url",
    "url",
)


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def normalize_text(value: str) -> str:
    """NFC + whitespace collapse only; retain case, punctuation, and diacritics."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def nonblank(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def source_url(value: Any) -> bool:
    if not nonblank(value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"https", "http"} and bool(parsed.hostname)
    except ValueError:
        return False


def safe_category(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[\w./-]{1,96}", value):
        return value
    return "unknown" if value is None else "unrecognized:" + sha(json.dumps(value, sort_keys=True))


def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load_label_files(input_dir: Path) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    vocabularies = {}
    artifacts = []
    for task in ("emotion", "topic"):
        path = input_dir / task / "labels.json"
        if not path.exists():
            continue
        raw = path.read_bytes()
        entry: dict[str, Any] = {
            "path": str(path.relative_to(input_dir)),
            "bytes": len(raw),
            "sha256": sha(raw),
        }
        try:
            labels = json.loads(raw)
            if (
                not isinstance(labels, list)
                or not all(nonblank(label) for label in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError("expected unique nonblank label strings")
            vocabularies[task] = set(labels)
            entry.update({"status": "readable", "label_count": len(labels), "labels": labels})
        except (ValueError, TypeError, UnicodeError):
            entry["status"] = "invalid_label_vocabulary"
        artifacts.append(entry)
    return vocabularies, artifacts


def _reference(row: tuple, files: list[dict[str, Any]]) -> dict[str, Any]:
    file_id, line, record_hash = row
    return {"file": files[file_id]["path"], "line": line, "record_sha256": record_hash}


def _overlaps(
    db: sqlite3.Connection, files: list[dict[str, Any]], sample_limit: int
) -> dict[str, Any]:
    db.execute("CREATE INDEX occurrence_group ON occurrences(kind, digest, file_id, line)")
    db.execute(
        "CREATE TABLE groups AS SELECT kind,digest,file_id,COUNT(*) AS n FROM occurrences GROUP BY kind,digest,file_id"
    )
    db.execute("CREATE INDEX group_key ON groups(kind,digest,file_id)")
    summaries = {}
    for kind in KINDS:
        groups = db.execute(
            "SELECT digest,SUM(n),COUNT(*) FROM groups WHERE kind=? GROUP BY digest HAVING SUM(n)>1 ORDER BY digest",
            (kind,),
        )
        stats: dict[str, Any] = {
            "groups": 0,
            "records_in_groups": 0,
            "cross_file_groups": 0,
            "cross_split_groups": 0,
            "samples": [],
        }
        for digest, count, file_count in groups:
            stats["groups"] += 1
            stats["records_in_groups"] += count
            stats["cross_file_groups"] += file_count > 1
            members = [
                file_id
                for (file_id,) in db.execute(
                    "SELECT file_id FROM groups WHERE kind=? AND digest=? ORDER BY file_id",
                    (kind, digest),
                )
            ]
            cross_split = len({files[file_id]["split"] for file_id in members}) > 1
            stats["cross_split_groups"] += cross_split
            if len(stats["samples"]) < sample_limit:
                refs: list[tuple[int, int, str]] = []
                for member in members[:sample_limit]:
                    refs.extend(
                        db.execute(
                            "SELECT file_id,line,record_sha256 FROM occurrences WHERE kind=? AND digest=? AND file_id=? ORDER BY line LIMIT ?",
                            (kind, digest, member, sample_limit if len(members) == 1 else 1),
                        )
                    )
                stats["samples"].append(
                    {
                        "group_sha256": digest,
                        "records": count,
                        "cross_split": cross_split,
                        "references": [_reference(row, files) for row in refs],
                    }
                )
        summaries[kind] = stats
    pairs = []
    for kind, left, right, count, left_rows, right_rows in db.execute(
        "SELECT a.kind,a.file_id,b.file_id,COUNT(*),SUM(a.n),SUM(b.n) FROM groups a JOIN groups b ON a.kind=b.kind AND a.digest=b.digest AND a.file_id<b.file_id GROUP BY a.kind,a.file_id,b.file_id ORDER BY a.kind,a.file_id,b.file_id"
    ):
        pairs.append(
            {
                "kind": kind,
                "left": files[left]["path"],
                "right": files[right]["path"],
                "shared_groups": count,
                "left_records": left_rows,
                "right_records": right_rows,
                "cross_split": files[left]["split"] != files[right]["split"],
                "cross_task": files[left]["task"] != files[right]["task"],
            }
        )
    db.execute("CREATE INDEX occurrence_record ON occurrences(file_id,line,kind)")
    cross_title_targets: list[dict[str, Any]] = []
    cross_title_count = 0
    for digest, title_count in db.execute(
        "SELECT target.digest,COUNT(DISTINCT title.digest) FROM occurrences target JOIN occurrences title INDEXED BY occurrence_record ON target.file_id=title.file_id AND target.line=title.line WHERE target.kind='target_normalized' AND title.kind='title_only_candidate' GROUP BY target.digest HAVING COUNT(DISTINCT title.digest)>1 ORDER BY target.digest"
    ):
        cross_title_count += 1
        if len(cross_title_targets) < sample_limit:
            refs = list(
                db.execute(
                    "SELECT file_id,line,record_sha256 FROM occurrences WHERE kind='target_normalized' AND digest=? ORDER BY file_id,line LIMIT ?",
                    (digest, sample_limit),
                )
            )
            cross_title_targets.append(
                {
                    "target_sha256": digest,
                    "distinct_title_candidate_groups": title_count,
                    "references": [_reference(row, files) for row in refs],
                }
            )
    return {
        "duplicate_groups": summaries,
        "pairwise_intersections": pairs,
        "omitted_pair_counts_are_zero": True,
        "identical_targets_across_distinct_titles": {
            "groups": cross_title_count,
            "samples": cross_title_targets,
            "interpretation": "A source-pairing review signal; titles alone cannot establish distinct canonical works or prove a bad match.",
        },
    }


def _historical_counts(path: Path | None, files: list[dict[str, Any]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "historical_reference_unavailable",
            "note": "No model outputs were generated.",
        }
    raw = path.read_bytes()
    report = json.loads(raw)
    comparisons = []
    for task in ("summarization", "emotion", "topic"):
        stored = report.get(task, {})
        current = next(
            (entry for entry in files if entry["task"] == task and entry["split"] == "test"), None
        )
        count = stored.get("num_samples") if isinstance(stored, dict) else None
        if current is None or type(count) is not int:
            continue
        entry = {
            "task": task,
            "historical_report_rows": count,
            "current_test_rows": current["records"],
            "row_count_difference": current["records"] - count,
        }
        if task == "summarization":
            entry["historical_domains"] = {
                name: value["num_samples"]
                for name, value in stored.get("per_domain", {}).items()
                if isinstance(value, dict) and type(value.get("num_samples")) is int
            }
            entry["current_domains"] = current["domain_counts"]
        comparisons.append(entry)
    return {
        "status": "count_comparison_only",
        "report": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
        "report_sha256": sha(raw),
        "comparisons": comparisons,
        "note": "These are existing report sample counts, not new metrics. Equal counts do not establish identical examples or historical run provenance.",
    }


def audit_data(
    input_dir: Path, *, sample_limit: int = 4, historical_report: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= sample_limit <= 10:
        raise ValueError("sample_limit must be 1..10")
    input_dir = input_dir.resolve()
    paths = sorted(input_dir.rglob("*.jsonl"))
    initial_stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino) for path in paths
    }
    vocabularies, label_files = load_label_files(input_dir)
    files = []
    issues: Counter[str] = Counter()
    if not paths:
        issues["no_jsonl_inputs"] += 1
    issues["invalid_label_vocabularies"] = sum(
        entry["status"] != "readable" for entry in label_files
    )
    with tempfile.TemporaryDirectory(prefix="leximind-data-audit-") as temporary:
        db = sqlite3.connect(str(Path(temporary) / "hashes.sqlite"))
        try:
            db.execute(
                "CREATE TABLE occurrences(kind TEXT,digest TEXT,file_id INTEGER,line INTEGER,record_sha256 TEXT)"
            )
            pending: list[tuple[str, str, int, int, str]] = []
            for file_id, path in enumerate(paths):
                relative = str(path.relative_to(input_dir))
                task, split = path.parent.name, path.stem
                info: dict[str, Any] = {
                    "path": relative,
                    "task": task,
                    "split": split,
                    "bytes": 0,
                    "physical_lines": 0,
                    "blank_lines": 0,
                    "records": 0,
                    "malformed_records": 0,
                }
                domains: Counter[str] = Counter()
                labels: Counter[str] = Counter()
                fields: Counter[str] = Counter()
                coverage: Counter[str] = Counter()
                anomalies: list[dict[str, Any]] = []
                file_hash = hashlib.sha256()
                before = path.stat()
                if task not in TASKS or split not in SPLITS:
                    issues["unsupported_task_or_split"] += 1
                if task in {"emotion", "topic"} and task not in vocabularies:
                    issues["missing_label_vocabulary"] += 1
                with path.open("rb") as handle:
                    for number, line in enumerate(handle, 1):
                        info["physical_lines"] += 1
                        info["bytes"] += len(line)
                        file_hash.update(line)
                        if not line.strip():
                            info["blank_lines"] += 1
                            continue
                        record_hash = sha(line)
                        reference = {"file": relative, "line": number, "record_sha256": record_hash}
                        try:
                            row = json.loads(line, object_pairs_hook=object_without_duplicate_keys)
                            if not isinstance(row, dict):
                                raise ValueError("record is not an object")
                        except (ValueError, UnicodeError):
                            info["malformed_records"] += 1
                            issues["malformed_records"] += 1
                            if len(anomalies) < sample_limit:
                                anomalies.append({**reference, "reason": "malformed_json_object"})
                            continue
                        info["records"] += 1
                        fields.update(safe_category(field) for field in row)
                        domain = (
                            row.get("type")
                            if task in {"books", "summarization"}
                            else row.get("source")
                            if task == "topic"
                            else "emotion_comments_unverified_origin"
                        )
                        domains[safe_category(domain)] += 1
                        text = row.get("source") if task == "summarization" else row.get("text")
                        if not nonblank(text):
                            coverage["missing_or_invalid_input_text"] += 1
                            issues["missing_or_invalid_input_text"] += 1
                        occurrences: list[tuple[str, str]] = []
                        if nonblank(text):
                            occurrences.extend(
                                (
                                    ("input_raw", sha(text)),
                                    ("input_normalized", sha(normalize_text(text))),
                                )
                            )
                        if task == "summarization":
                            target = row.get("summary")
                            if nonblank(target):
                                occurrences.append(
                                    ("target_normalized", sha(normalize_text(target)))
                                )
                            else:
                                coverage["missing_or_invalid_summary"] += 1
                                issues["missing_or_invalid_summary"] += 1
                        if task in {"emotion", "topic"}:
                            values = (
                                row.get("emotions") if task == "emotion" else [row.get("topic")]
                            )
                            if (
                                not isinstance(values, list)
                                or not values
                                or not all(nonblank(value) for value in values)
                            ):
                                coverage["invalid_labels"] += 1
                                issues["invalid_labels"] += 1
                            else:
                                if len(set(values)) != len(values):
                                    coverage["duplicate_labels_within_record"] += 1
                                    issues["duplicate_labels_within_record"] += 1
                                for label in values:
                                    if label in vocabularies.get(task, set()):
                                        labels[label] += 1
                                    else:
                                        labels["unknown:" + sha(label)] += 1
                                        coverage["unknown_label_occurrences"] += 1
                                        issues["unknown_label_occurrences"] += 1
                        work, document = row.get("work_id"), row.get("document_id")
                        work_present, document_present = nonblank(work), nonblank(document)
                        coverage["declared_work_id"] += work_present
                        coverage["declared_document_id"] += document_present
                        coverage["missing_parent_identity"] += not (
                            work_present or document_present
                        )
                        issues["missing_parent_identity"] += not (work_present or document_present)
                        if nonblank(work):
                            occurrences.append(("declared_work", sha(work)))
                        if nonblank(document):
                            occurrences.append(("declared_document", sha(document)))
                        explicit_document = (
                            document_present
                            and row.get("identity_scope") == "provider_document"
                            and row.get("work_identity_status") == "unresolved"
                        )
                        coverage["explicit_provider_document_only"] += explicit_document
                        coverage["provider_dataset"] += nonblank(row.get("provider_dataset"))
                        provenanced = any(source_url(row.get(field)) for field in PROVENANCE_FIELDS)
                        coverage["source_url_provenance"] += provenanced
                        coverage["missing_source_url_provenance"] += not provenanced
                        issues["missing_source_url_provenance"] += not provenanced
                        title = row.get("title")
                        authors = author_identity(row)
                        coverage["title_present"] += nonblank(title)
                        coverage["author_identity_present"] += bool(authors)
                        title_author = nonblank(title) and bool(authors)
                        coverage["title_and_author_candidate"] += title_author
                        if nonblank(title):
                            occurrences.append(
                                ("title_only_candidate", sha(normalize_title(title)))
                            )
                            if title_author:
                                occurrences.append(
                                    (
                                        "title_author_candidate",
                                        sha(
                                            json.dumps(
                                                [normalize_title(title), authors],
                                                ensure_ascii=False,
                                            )
                                        ),
                                    )
                                )
                        literary = (
                            task == "books"
                            or (task == "summarization" and row.get("type") == "literary")
                            or (task == "topic" and row.get("source") == "gutenberg")
                        )
                        coverage["literary_rows"] += literary
                        # Row declarations cannot approve canonical reconciliation.
                        unresolved_literary = literary
                        coverage["unresolved_literary_work_identity"] += unresolved_literary
                        issues["unresolved_literary_work_identity"] += unresolved_literary
                        legacy_pair_risk = (
                            task == "summarization"
                            and row.get("type") == "literary"
                            and not (
                                row.get("identity_status") == "title_and_author_matched"
                                and work_present
                                and title_author
                                and source_url(row.get("description_source"))
                            )
                        )
                        coverage["legacy_literary_pair_identity_risk"] += legacy_pair_risk
                        issues["legacy_literary_pair_identity_risk"] += legacy_pair_risk
                        if (legacy_pair_risk or not (work_present or document_present)) and len(
                            anomalies
                        ) < sample_limit:
                            anomalies.append(
                                {
                                    **reference,
                                    "reason": "legacy_literary_pair_identity_unverified"
                                    if legacy_pair_risk
                                    else "parent_identity_missing",
                                }
                            )
                        pending.extend(
                            (kind, group_hash, file_id, number, record_hash)
                            for kind, group_hash in occurrences
                        )
                        if len(pending) >= 4096:
                            db.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?)", pending)
                            pending.clear()
                if info["records"] == 0:
                    issues["empty_input_files"] += 1
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ino,
                ):
                    raise RuntimeError(
                        f"Input changed during audit: {relative}; rerun against a stable snapshot"
                    )
                info.update(
                    {
                        "sha256": file_hash.hexdigest(),
                        "field_counts": dict(sorted(fields.items())),
                        "domain_counts": dict(sorted(domains.items())),
                        "label_counts": dict(sorted(labels.items())),
                        "identity_and_provenance": dict(sorted(coverage.items())),
                        "bounded_anomalies": anomalies,
                    }
                )
                files.append(info)
                print(f"Audited {relative}: {info['records']} records", file=sys.stderr)
            if sorted(input_dir.rglob("*.jsonl")) != paths or any(
                (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
                != initial_stats[path]
                for path in paths
            ):
                raise RuntimeError("Input corpus changed during audit; use a stable snapshot")
            if pending:
                db.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?)", pending)
            db.commit()
            overlaps = _overlaps(db, files, sample_limit)
        finally:
            db.close()
    for kind in (
        "input_normalized",
        "target_normalized",
        "declared_work",
        "declared_document",
        "title_author_candidate",
    ):
        count = overlaps["duplicate_groups"][kind]["cross_split_groups"]
        if count:
            issues["cross_split_" + kind + "_groups"] = count
    all_tasks = sorted({entry["task"] for entry in files})
    for task, split in itertools.product(all_tasks, sorted(SPLITS)):
        if not any(entry["task"] == task and entry["split"] == split for entry in files):
            issues["missing_task_split_files"] += 1
    inventory = {
        "schema_version": 1,
        "audit_version": VERSION,
        "input_root": str(input_dir.relative_to(ROOT))
        if input_dir.is_relative_to(ROOT)
        else input_dir.name,
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "records": sum(entry["records"] for entry in files),
        "bytes": sum(entry["bytes"] for entry in files),
        "files": files,
        "label_files": label_files,
        "normalization": "NFC Unicode normalization and whitespace collapse; case, punctuation and diacritics preserved. Titles use a separate noncanonical candidate key.",
        "report_privacy": "No input text, summaries, titles, authors or model outputs are copied; references contain file/line/hash only.",
    }
    active_issues = {name: count for name, count in sorted(issues.items()) if count}
    task_status = []
    for task in all_tasks:
        entries = [entry for entry in files if entry["task"] == task]
        quarantined = any(
            entry["malformed_records"]
            or any(
                entry["identity_and_provenance"].get(name, 0)
                for name in (
                    "missing_parent_identity",
                    "missing_source_url_provenance",
                    "unresolved_literary_work_identity",
                    "invalid_labels",
                    "unknown_label_occurrences",
                )
            )
            for entry in entries
        ) or any(
            pair["cross_split"]
            and pair["kind"] != "title_only_candidate"
            and any(path.startswith(task + "/") for path in (pair["left"], pair["right"]))
            for pair in overlaps["pairwise_intersections"]
        )
        task_status.append(
            {
                "task": task,
                "records": sum(entry["records"] for entry in entries),
                "status": "quarantined_pending_source_and_partition_review"
                if quarantined
                else "requires_protocol_review",
                "structurally_readable": all(not entry["malformed_records"] for entry in entries),
                "training_eligible": False,
            }
        )
    audit = {
        "schema_version": 1,
        "audit_version": VERSION,
        "inventory_sha256": sha(json.dumps(inventory, sort_keys=True, ensure_ascii=False)),
        "status": "blocked" if active_issues else "requires_protocol_review",
        "mechanical_checks_passed": not active_issues,
        "training_authorized": False,
        "blocking_findings": active_issues,
        "external_gates": [
            "Dataset/source eligibility and rights decision",
            "Canonical work/edition reconciliation where work generalization is claimed",
            "Frozen dataset selection, calibration/model-selection split policy and contamination review",
            "Explicit user authorization before any training or research evaluation",
        ],
        "task_disposition": task_status,
        **overlaps,
        "historical_count_comparison": _historical_counts(historical_report, files),
        "limitations": [
            "Exact hashes do not detect near-duplicates, overlapping excerpts, translations or related editions.",
            "Declared IDs and title/author candidate groups are evidence to review, not independently verified canonical identities.",
            "Title-only intersections indicate identity risk and are not proof that two records represent the same work.",
            "Document-only grouping can prevent paragraph leakage while work-level generalization remains unresolved.",
            "No model execution, output scoring, data repair or split reassignment was performed.",
        ],
    }
    return inventory, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "research/preparation/data_inventory.json"
    )
    parser.add_argument("--audit", type=Path, default=ROOT / "research/preparation/data_audit.json")
    parser.add_argument(
        "--historical-report",
        type=Path,
        default=ROOT / "research/results/historical/leximind_test.json",
    )
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    destinations = (args.inventory.resolve(), args.audit.resolve())
    if destinations[0] == destinations[1] or any(
        path.is_relative_to(args.input_dir.resolve()) for path in destinations
    ):
        parser.error(
            "Report destinations must differ and remain outside the read-only input directory"
        )
    if args.historical_report.resolve() in destinations or Path(__file__).resolve() in destinations:
        parser.error("Reports cannot overwrite the auditor or its historical reference")
    inventory, audit = audit_data(
        args.input_dir, sample_limit=args.sample_limit, historical_report=args.historical_report
    )
    write_json_atomic(args.inventory, inventory)
    write_json_atomic(args.audit, audit)
    print(
        json.dumps(
            {
                "records": inventory["records"],
                "status": audit["status"],
                "blocking_findings": audit["blocking_findings"],
            },
            sort_keys=True,
        )
    )
    return 2 if args.require_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
