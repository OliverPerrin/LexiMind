"""Deterministic source-record assignments; no copied text, training or scoring."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_io import create_or_verify, file_hash
from .io import check_file, parse_json, safe_path

POLICY = "source-partitions-v1"
SALT = "leximind-source-partitions-v1"
ROLES = ("train", "model_selection", "calibration", "test")
POLICIES = {
    "google-research-datasets/go_emotions": {"train": 10000, "validation": 5000, "test": 10000},
    "fancyzhx/ag_news": {"train": 9000, "test": 10000},
}


def text_group(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise ValueError("A source text must be nonempty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def role_for(repo: str, split: str, group: str) -> str:
    """Hash group allocation has no dependence on labels, input order or model output."""
    if repo not in POLICIES or split not in POLICIES[repo]:
        raise ValueError("Unknown source or source split")
    if split == "test":
        return "test"
    if repo.endswith("go_emotions") and split == "train":
        return "train"
    bucket = int(hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest(), 16) % 10000
    if split == "validation":
        return "model_selection" if bucket < 5000 else "calibration"
    return "train" if bucket < 9000 else "model_selection" if bucket < 9500 else "calibration"


def prepare_partitions(root: Path, candidate_manifest: Path, output_dir: Path) -> dict[str, Any]:
    """Write immutable ID assignments and return a compact, unadmitted preparation report."""
    root = root.resolve()
    candidate_manifest = candidate_manifest.resolve()
    if not candidate_manifest.is_relative_to(root) or not output_dir.resolve().is_relative_to(root):
        raise ValueError("Preparation paths must stay inside the repository")
    manifest_bytes = candidate_manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = parse_json(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Expected a versioned candidate manifest object")
    if not isinstance(manifest.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["revision"]
    ):
        raise ValueError("Candidate requires a full pinned provider revision")
    repo = manifest.get("repo")
    if repo not in POLICIES:
        raise ValueError("Only reviewed GoEmotions and AG News source layouts are supported")
    candidate = safe_path(root, manifest["local_candidate_directory"])
    sources = manifest["prepared_files"]
    label_names = manifest.get("label_names")
    if (
        not isinstance(label_names, list)
        or not label_names
        or any(not isinstance(label, str) or not label for label in label_names)
        or len(set(label_names)) != len(label_names)
    ):
        raise ValueError("Candidate must pin a unique, nonempty label vocabulary")
    if not isinstance(sources, dict) or set(sources) != set(POLICIES[repo]):
        raise ValueError("Candidate must contain exactly the original source splits")
    for reference in sources.values():
        if (
            not isinstance(reference, dict)
            or type(reference.get("rows")) is not int
            or reference["rows"] < 1
        ):
            raise ValueError("Prepared source files require positive integer row counts")
    output = output_dir.resolve()
    # Keep all writes separate from source artifacts, including reports supplied by callers.
    if output == candidate or any(
        safe_path(candidate, ref["path"]).is_relative_to(output) for ref in sources.values()
    ):
        raise ValueError("Assignments must not contain or replace source files")
    for reference in sources.values():
        errors = check_file(candidate, reference)
        if errors:
            raise ValueError("; ".join(errors))
    counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="leximind-partitions-") as temporary:
        with sqlite3.connect(Path(temporary) / "index.sqlite") as db:
            db.execute(
                "CREATE TABLE rows (id TEXT PRIMARY KEY, text_group TEXT, role TEXT, source_split TEXT)"
            )

            def assignments():
                for split in POLICIES[repo]:
                    path = safe_path(candidate, sources[split]["path"])
                    row_count = 0
                    with path.open("rb") as stream:
                        for number, raw in enumerate(stream, 1):
                            row = parse_json(raw)
                            identifier = (
                                row.get("document_id")
                                if repo.endswith("go_emotions")
                                else row.get("record_id")
                            )
                            if (
                                not isinstance(identifier, str)
                                or not identifier.strip()
                                or row.get("provider_split") != split
                            ):
                                raise ValueError(
                                    f"Missing record identity or mismatched source split at {split}:{number}"
                                )
                            if (
                                row.get("source_row") != number
                                or type(row.get("source_row")) is not int
                            ):
                                raise ValueError(
                                    "Source row numbers must identify exact candidate/source order"
                                )
                            group = text_group(row["text"])
                            role = role_for(repo, split, group)
                            try:
                                db.execute(
                                    "INSERT INTO rows VALUES (?,?,?,?)",
                                    (identifier, group, role, split),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(
                                    "Source identities repeat within/across original splits"
                                ) from exc
                            counts[role] += 1
                            labels = (
                                row.get("emotions")
                                if repo.endswith("go_emotions")
                                else [row.get("topic")]
                            )
                            if (
                                not isinstance(labels, list)
                                or not labels
                                or any(not isinstance(label, str) or not label for label in labels)
                            ):
                                raise ValueError("Candidate labels must be nonempty strings")
                            if not set(labels) <= set(label_names):
                                raise ValueError(
                                    "Candidate row contains labels outside its pinned vocabulary"
                                )
                            label_counts[role].update(labels)
                            record = {
                                "record_id": identifier,
                                "source_split": split,
                                "source_row": number,
                                "partition": role,
                                "text_sha256": group,
                            }
                            yield (
                                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                            ).encode()
                            row_count += 1
                    if (
                        row_count != sources[split]["rows"]
                        or file_hash(path) != sources[split]["sha256"]
                    ):
                        raise ValueError("Candidate source changed during partition preparation")
                if file_hash(candidate_manifest) != manifest_hash:
                    raise ValueError("Candidate manifest changed during preparation")
                if any(counts[role] == 0 for role in ROLES):
                    raise ValueError(
                        "The fixed policy produced an empty role; do not silently reseed it"
                    )

            reference = create_or_verify(output / "assignments.jsonl", assignments())
            db.execute("CREATE INDEX by_group ON rows(text_group,role)")
            cross = db.execute(
                "SELECT COUNT(*) FROM (SELECT text_group FROM rows GROUP BY text_group HAVING COUNT(DISTINCT role)>1)"
            ).fetchone()[0]
            within = db.execute(
                "SELECT COUNT(*) FROM (SELECT source_split,text_group FROM rows GROUP BY source_split,text_group HAVING COUNT(DISTINCT role)>1)"
            ).fetchone()[0]
            pair_counts = [
                {
                    "partitions": [first, second],
                    "text_groups": db.execute(
                        "SELECT COUNT(*) FROM (SELECT text_group FROM rows WHERE role IN (?,?) GROUP BY text_group HAVING COUNT(DISTINCT role)=2)",
                        (first, second),
                    ).fetchone()[0],
                }
                for index, first in enumerate(ROLES)
                for second in ROLES[index + 1 :]
            ]
    return {
        "schema_version": 1,
        "status": "candidate_assignments_not_admitted",
        "training_authorized": False,
        "policy": POLICY,
        "salt": SALT,
        "repo": repo,
        "revision": manifest["revision"],
        "candidate_manifest": {
            "path": str(candidate_manifest.resolve().relative_to(root.resolve())),
            "sha256": manifest_hash,
            "bytes": len(manifest_bytes),
        },
        "implementation_sha256": file_hash(Path(__file__)),
        "assignments": {
            "path": str((output / "assignments.jsonl").relative_to(root.resolve())),
            **reference,
        },
        "source_files": sources,
        "counts": {role: counts[role] for role in ROLES},
        "label_names": label_names,
        "label_counts": {
            role: {label: label_counts[role][label] for label in label_names} for role in ROLES
        },
        "cross_partition_normalized_text_groups": cross,
        "within_original_split_cross_partition_text_groups": within,
        "overlap_by_pair": pair_counts,
        "interpretation": [
            "Original official test membership is unchanged; original candidates are never overwritten.",
            "GoEmotions original validation groups split 50/50 into selection/calibration; original train stays train.",
            "AG News original train groups split 90/5/5; fractions are hash thresholds, not exact row quotas.",
            "Normalization is NFC plus whitespace collapse; case/punctuation remain significant.",
            "Groups repeated across original official partitions remain visible, not deleted or reassigned.",
            "No unseen-text or unseen-article claim; AG identities are pinned source rows, not article IDs.",
            "No labels or test scores influence assignment. Rare-label coverage is reported, not optimized.",
            "These are unadmitted source assignments, not evidence of source-use clearance or a frozen protocol.",
        ],
    }
