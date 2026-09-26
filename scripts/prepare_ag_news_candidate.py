"""Reconstruct a pinned, unadmitted AG News candidate without model execution.

Default: verify/reuse local sources. --fetch acquires only two public Parquet
files plus the pinned source card. Source rows have reproducible local identities;
the release does not supply upstream article IDs or a validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.locking import advisory_lock
from src.catalog.storage import write_json_atomic
from src.research.candidate_io import (
    create_or_verify,
    file_hash,
    helper_hashes,
    json_bytes,
    parquet_rows,
    sha,
)
from src.research.io import read_json

REPO = "fancyzhx/ag_news"
REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
FILES: dict[str, dict[str, Any]] = {
    "train": {
        "path": "data/train-00000-of-00001.parquet",
        "bytes": 18585438,
        "rows": 120000,
        "sha256": "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee",
    },
    "test": {
        "path": "data/test-00000-of-00001.parquet",
        "bytes": 1234829,
        "rows": 7600,
        "sha256": "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
    },
}
MAX_SOURCE_BYTES = 50_000_000
CARD_GIT_BLOB = "fc01bffaf6b7ff9e9df5818a6fa2bd5cb4a27438"
CARD_BYTES = 8070
NORMALIZATION = (
    "Unicode NFC and whitespace collapse only; case, punctuation and diacritics preserved"
)


def url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"


def verify_sources(directory: Path, receipt: dict[str, Any]) -> None:
    if (
        not isinstance(receipt, dict)
        or any(
            receipt.get(key) != value
            for key, value in {
                "repo": REPO,
                "revision": REVISION,
                "config": "default",
                "label_names": LABELS,
            }.items()
        )
        or receipt.get("public") is not True
        or receipt.get("gated") is not False
    ):
        raise ValueError("Acquisition receipt differs from the pinned AG News release")
    if receipt.get("provider_license_declaration") not in ("unknown", ["unknown"]):
        raise ValueError(
            "License declaration differs from the reviewed source; inspect before reuse"
        )
    if receipt.get("files") != {
        split: {**record, "url": url(record["path"])} for split, record in FILES.items()
    }:
        raise ValueError("Source file receipt differs from immutable file metadata")
    for record in FILES.values():
        path = directory / "raw" / record["path"]
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError("Cached AG News source hash/size mismatch")
    raw_card = (directory / "raw/README.md").read_bytes()
    blob = hashlib.sha1(f"blob {len(raw_card)}\0".encode() + raw_card).hexdigest()
    if (
        len(raw_card) != CARD_BYTES
        or blob != CARD_GIT_BLOB
        or receipt.get("source_card")
        != {
            "path": "README.md",
            "url": url("README.md"),
            "bytes": len(raw_card),
            "sha256": sha(raw_card),
            "git_blob_sha1": blob,
        }
    ):
        raise ValueError("Pinned source-card provenance differs")


def acquire_sources(directory: Path, *, fetch: bool) -> dict[str, Any]:
    receipt_path = directory / "raw/acquisition.json"
    if receipt_path.exists():
        receipt: dict[str, Any] = read_json(receipt_path)
        verify_sources(directory, receipt)
        return receipt
    if not fetch:
        raise FileNotFoundError(
            "Pinned AG News source cache is absent; use --fetch for bounded acquisition"
        )
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise ValueError("Source is not the expected public, ungated revision")
    card = info.card_data.to_dict()
    schema = card["dataset_info"]
    names = next(
        feature["dtype"]["class_label"]["names"]
        for feature in schema["features"]
        if feature["name"] == "label"
    )
    if [names[str(index)] for index in range(len(names))] != LABELS or card.get("license") not in (
        "unknown",
        ["unknown"],
    ):
        raise ValueError("Source labels or license declaration changed")
    siblings = {entry.rfilename: entry for entry in info.siblings}
    expected_paths = {record["path"] for record in FILES.values()}
    if {path for path in siblings if path.startswith("data/")} != expected_paths or sum(
        record["bytes"] for record in FILES.values()
    ) >= MAX_SOURCE_BYTES:
        raise ValueError("Unexpected source layout or download bound")
    if siblings["README.md"].size != CARD_BYTES or siblings["README.md"].blob_id != CARD_GIT_BLOB:
        raise ValueError("Source-card metadata changed")
    for record in FILES.values():
        sibling = siblings[record["path"]]
        if (
            sibling.size != record["bytes"]
            or not sibling.lfs
            or sibling.lfs.sha256 != record["sha256"]
        ):
            raise ValueError("Provider-published source size/hash changed")
        hf_hub_download(
            REPO,
            record["path"],
            repo_type="dataset",
            revision=REVISION,
            local_dir=directory / "raw",
            token=False,
        )
    hf_hub_download(
        REPO,
        "README.md",
        repo_type="dataset",
        revision=REVISION,
        local_dir=directory / "raw",
        token=False,
    )
    raw_card = (directory / "raw/README.md").read_bytes()
    receipt = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "config": "default",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "public": True,
        "gated": False,
        "provider_license_declaration": card.get("license"),
        "label_names": LABELS,
        "provider_schema": schema,
        "files": {split: {**record, "url": url(record["path"])} for split, record in FILES.items()},
        "source_card": {
            "path": "README.md",
            "url": url("README.md"),
            "bytes": len(raw_card),
            "sha256": sha(raw_card),
            "git_blob_sha1": CARD_GIT_BLOB,
        },
    }
    verify_sources(directory, receipt)
    create_or_verify(receipt_path, [json_bytes(receipt)])
    receipt = read_json(receipt_path)
    return receipt


def convert_record(
    row: dict[str, Any], split: str, number: int, source_hash: str
) -> dict[str, Any]:
    if (
        split not in FILES
        or type(number) is not int
        or number < 1
        or not isinstance(source_hash, str)
        or not re.fullmatch(r"[a-f0-9]{64}", source_hash)
    ):
        raise ValueError("Invalid source split or row number")
    if (
        not isinstance(row, dict)
        or set(row) != {"text", "label"}
        or not isinstance(row["text"], str)
        or not row["text"]
        or type(row["label"]) is not int
        or row["label"] not in range(len(LABELS))
    ):
        raise ValueError(f"Invalid AG News source row: {split}:{number}")
    return {
        "record_id": f"hf:{REPO}:sha256:{source_hash}:row:{number}",
        "source_row": number,
        "provider_split": split,
        "text": row["text"],
        "topic": LABELS[row["label"]],
        "label_id": row["label"],
    }


def duplicate_report(db: sqlite3.Connection, column: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "groups": 0,
        "records_in_groups": 0,
        "cross_split_groups": 0,
        "different_label_groups": 0,
        "bounded_samples": [],
    }
    for digest, count, splits, labels in db.execute(
        f"SELECT {column},COUNT(*),COUNT(DISTINCT split),COUNT(DISTINCT label) FROM records GROUP BY {column} HAVING COUNT(*)>1 ORDER BY {column}"
    ):
        report["groups"] += 1
        report["records_in_groups"] += count
        report["cross_split_groups"] += splits > 1
        report["different_label_groups"] += labels > 1
        if len(report["bounded_samples"]) < 4:
            references = [
                {"split": split, "source_row": row}
                for split, row in db.execute(
                    f"SELECT split,MIN(source_row) FROM records WHERE {column}=? GROUP BY split ORDER BY split",
                    (digest,),
                )
            ]
            report["bounded_samples"].append(
                {
                    "text_sha256": digest,
                    "records": count,
                    "cross_split": splits > 1,
                    "references": references,
                }
            )
    return report


def compare_legacy(db: sqlite3.Connection, legacy_dir: Path) -> dict[str, Any]:
    source_hashes = {digest for (digest,) in db.execute("SELECT DISTINCT exact FROM records")}
    result = {}
    for path in sorted(legacy_dir.glob("*.jsonl")):
        rows = overlap = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError("Legacy topic schema differs from its text-field contract")
                rows += 1
                overlap += sha(row["text"]) in source_hashes
        result[path.name] = {
            "rows": rows,
            "sha256": file_hash(path),
            "exact_text_matches_to_candidate": overlap,
        }
    return {
        "scope": "Exact decoded text only; no taxonomy conversion or article-identity inference",
        "files": result,
    }


def prepare_candidate(directory: Path, receipt: dict[str, Any], legacy_dir: Path) -> dict[str, Any]:
    if set(receipt.get("files", {})) != set(FILES):
        raise ValueError("Candidate preparation requires every original source split")
    prepared, label_counts = {}, {}
    with tempfile.TemporaryDirectory(prefix="leximind-ag-news-index-") as temporary:
        db = sqlite3.connect(str(Path(temporary) / "index.sqlite"))
        try:
            db.execute(
                "CREATE TABLE records(split TEXT,source_row INTEGER,exact TEXT,normalized TEXT,label INTEGER)"
            )
            for split in FILES:
                source = receipt["files"][split]
                counts: Counter[str] = Counter()
                rows = 0

                def encoded(
                    split: str = split, source: dict = source, counts: Counter[str] = counts
                ) -> Iterable[bytes]:
                    nonlocal rows
                    for number, raw in enumerate(
                        parquet_rows(directory / "raw" / source["path"], ("text", "label")), 1
                    ):
                        row = convert_record(raw, split, number, source["sha256"])
                        rows += 1
                        counts[row["topic"]] += 1
                        db.execute(
                            "INSERT INTO records VALUES (?,?,?,?,?)",
                            (
                                split,
                                number,
                                sha(row["text"]),
                                sha(" ".join(unicodedata.normalize("NFC", row["text"]).split())),
                                row["label_id"],
                            ),
                        )
                        yield (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        ).encode()
                    if rows != source["rows"]:
                        raise ValueError("Source row count differs from pinned metadata")

                artifact = create_or_verify(directory / "prepared" / f"{split}.jsonl", encoded())
                prepared[split] = {
                    "path": f"prepared/{split}.jsonl",
                    "rows": rows,
                    "source_split": split,
                    **artifact,
                }
                label_counts[split] = dict(sorted(counts.items()))
                print(f"Prepared unadmitted {split}: {rows} source rows", file=sys.stderr)
            for field in ("exact", "normalized"):
                db.execute(f"CREATE INDEX records_{field} ON records({field},split)")
            audit = {
                "exact_text": duplicate_report(db, "exact"),
                "normalized_text": duplicate_report(db, "normalized"),
                "normalization": NORMALIZATION,
            }
            legacy = compare_legacy(db, legacy_dir)
        finally:
            db.close()
    labels = create_or_verify(directory / "prepared/labels.json", [json_bytes(LABELS)])
    return {
        "schema_version": 1,
        "prepared_record_version": 1,
        "status": "candidate_prepared_not_admitted",
        "training_authorized": False,
        "repo": REPO,
        "revision": REVISION,
        "config": "default",
        "local_candidate_directory": str(directory.resolve().relative_to(ROOT))
        if directory.resolve().is_relative_to(ROOT)
        else directory.name,
        "identity_scope": "provider_source_row",
        "row_identity_field": "record_id",
        "upstream_article_ids_available": False,
        "row_provenance": "record_id encodes the immutable source-file SHA256 and one-based source_row. provider_split resolves acquisition.files; these are row references, not invented article/document IDs.",
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": helper_hashes(ROOT),
        "acquisition_receipt_sha256": file_hash(directory / "raw/acquisition.json"),
        "acquisition": receipt,
        "prepared_files": prepared,
        "prepared_label_map": {"path": "prepared/labels.json", **labels},
        "label_names": LABELS,
        "label_counts": label_counts,
        "duplicate_review": audit,
        "legacy_comparison": legacy,
        "provider_validation_split_available": False,
        "tooling": {
            "python_version": platform.python_version(),
            "pyarrow_version": version("pyarrow"),
            "model_execution": False,
        },
        "rights_review": {
            "status": "unresolved",
            "provider_license_declaration": "unknown",
            "scope_evidence": "Pinned Hub card describes research and non-commercial purposes; this is not general reuse clearance",
            "original_provider_url": "https://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            "original_provider_direct_verification": "timed_out_during_source_review",
        },
        "not_admitted_reasons": [
            "Original-source rights and intended-use review remains unresolved",
            "Source-row identity does not establish article or syndicated-document identity",
            "Duplicate policy and training/selection/calibration roles are not assigned by this reconstruction",
            "No model or research execution authorization is implied",
        ],
        "privacy_boundary": "Only code, hashes, counts and source-row references are committed; source news text remains in ignored local candidate storage.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--candidate-dir", type=Path, default=ROOT / "data/research_candidates/ag_news" / REVISION
    )
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "data/processed/topic")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "research/preparation/ag_news_candidate_manifest.json",
    )
    args = parser.parse_args()
    directory, legacy, manifest = (
        args.candidate_dir.resolve(),
        args.legacy_dir.resolve(),
        args.manifest.resolve(),
    )
    if (
        directory == legacy
        or directory.is_relative_to(legacy)
        or legacy.is_relative_to(directory)
        or manifest.is_relative_to(legacy)
    ):
        parser.error("Candidate/report locations must not overlap read-only legacy data")
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(
        ROOT / "data/research_candidates"
    ):
        parser.error("In-repository candidate text must stay in ignored data/research_candidates")
    if any(manifest.is_relative_to(directory / part) for part in ("raw", "prepared")):
        parser.error("Manifest cannot replace a source/candidate artifact")
    with advisory_lock(directory / ".prepare.lock"):
        receipt = acquire_sources(directory, fetch=args.fetch)
        report = prepare_candidate(directory, receipt, legacy)
        write_json_atomic(manifest, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": {split: row["rows"] for split, row in report["prepared_files"].items()},
                "training_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
