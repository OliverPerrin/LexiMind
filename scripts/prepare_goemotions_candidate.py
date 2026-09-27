"""Prepare a new, unadmitted GoEmotions candidate from one pinned public release.

Only --fetch permits acquiring the three simplified Parquet files and small
provenance documents. Legacy data are read-only. No models, datasets library,
training, inference, scoring, deduplication, or split reassignment are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.locking import advisory_lock
from src.catalog.storage import write_json_atomic

REPO = "google-research-datasets/go_emotions"
REVISION = "add492243ff905527e67aeb8b80c082af02207c3"
SPLITS = ("train", "validation", "test")
MAX_DOWNLOAD_BYTES = 100_000_000
LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
PINNED_FILES = {
    "train": {
        "bytes": 2767678,
        "sha256": "b7d74279616ae7c9b8374ab62ea9f9d6504d36a577bb17f745d720dc2b0d4e76",
        "rows": 43410,
    },
    "validation": {
        "bytes": 350063,
        "sha256": "d46ad5633c4fa41829d22d549743a7bf858d94129536a02af7280c281db5e63a",
        "rows": 5426,
    },
    "test": {
        "bytes": 346630,
        "sha256": "fd0953e535ba2569edc6a1daaa1133f8e4b9071691d540c9bab812fda132bc26",
        "rows": 5427,
    },
}
DOCUMENTS = {
    "google_readme.md": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md",
    "google_license.txt": "https://raw.githubusercontent.com/google-research/google-research/master/LICENSE",
}


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_path(split: str) -> str:
    return f"simplified/{split}-00000-of-00001.parquet"


def source_url(split: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{source_path(split)}"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _create_or_verify(path: Path, chunks: Iterable[bytes]) -> dict[str, Any]:
    """Never replace a differing candidate or source artifact at this location."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        expected = digest.hexdigest()
        if path.exists():
            if file_hash(path) != expected:
                raise ValueError(
                    f"Existing candidate artifact differs; use a separately reviewed location: {path.name}"
                )
        else:
            os.replace(temporary, path)
        return {"bytes": size, "sha256": expected}
    finally:
        temporary.unlink(missing_ok=True)


def _check_cached_sources(candidate_dir: Path, receipt: dict[str, Any]) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("Acquisition receipt must be an object")
    if (
        receipt.get("repo") != REPO
        or receipt.get("revision") != REVISION
        or receipt.get("config") != "simplified"
        or receipt.get("label_names") != LABELS
    ):
        raise ValueError("Candidate acquisition receipt does not match the pinned release")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("public") is not True
        or receipt.get("gated") is not False
        or receipt.get("provider_license_declaration") not in ("apache-2.0", ["apache-2.0"])
    ):
        raise ValueError("Source access/license declaration differs from the reviewed receipt")
    for split in SPLITS:
        expected = PINNED_FILES[split]
        path = candidate_dir / "raw" / source_path(split)
        observed = receipt["files"][split]
        if (
            observed.get("sha256") != expected["sha256"]
            or observed.get("bytes") != expected["bytes"]
            or observed.get("rows") != expected["rows"]
            or observed.get("url") != source_url(split)
        ):
            raise ValueError(f"Source receipt disagrees with pinned metadata: {split}")
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or file_hash(path) != expected["sha256"]
        ):
            raise ValueError(f"Missing or corrupt pinned source: {split}")
    documents = receipt.get("documents")
    expected_documents = {
        **DOCUMENTS,
        "README.md": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md",
    }
    if (
        not isinstance(documents, list)
        or not all(isinstance(document, dict) for document in documents)
        or len(documents) != len(expected_documents)
        or {document.get("name") for document in documents} != set(expected_documents)
    ):
        raise ValueError("Acquisition receipt lacks its reviewed provenance documents")
    for document in documents:
        if (
            document.get("url") != expected_documents[document["name"]]
            or not isinstance(document.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", document["sha256"])
        ):
            raise ValueError("Invalid provenance document receipt")
        path = candidate_dir / "raw" / "documents" / document["name"]
        if not path.is_file() or file_hash(path) != document["sha256"]:
            raise ValueError("Missing or corrupt source documentation")


def acquire_sources(candidate_dir: Path, *, fetch: bool) -> dict[str, Any]:
    receipt_path = candidate_dir / "raw/acquisition.json"
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text())
        _check_cached_sources(candidate_dir, receipt)
        return receipt
    if not fetch:
        raise FileNotFoundError(
            "No pinned source receipt; use --fetch to acquire the bounded simplified release"
        )
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise ValueError("Pinned source is not the expected public, ungated revision")
    card = info.card_data.to_dict()
    if card.get("license") not in ("apache-2.0", ["apache-2.0"]):
        raise ValueError("Provider license declaration changed; source review is required")
    dataset_info = next(
        (item for item in card.get("dataset_info", []) if item.get("config_name") == "simplified"),
        None,
    )
    if not dataset_info:
        raise ValueError("Pinned card lacks the simplified schema")
    names = next(
        feature["sequence"]["class_label"]["names"]
        for feature in dataset_info["features"]
        if feature["name"] == "labels"
    )
    if [names[str(i)] for i in range(len(names))] != LABELS:
        raise ValueError("Provider label map differs from the reviewed map")
    siblings = {entry.rfilename: entry for entry in info.siblings}
    if not siblings.get("README.md") or siblings["README.md"].size > 100_000:
        raise ValueError("Pinned source card exceeds the small documentation bound")
    simplified_files = {name for name in siblings if name.startswith("simplified/")}
    if simplified_files != {source_path(split) for split in SPLITS}:
        raise ValueError("Unexpected simplified file layout")
    if sum(siblings[name].size for name in simplified_files) >= MAX_DOWNLOAD_BYTES:
        raise ValueError("Bounded download limit exceeded")
    files = {}
    for split in SPLITS:
        expected, sibling = PINNED_FILES[split], siblings[source_path(split)]
        if (
            sibling.size != expected["bytes"]
            or sibling.lfs is None
            or sibling.lfs.sha256 != expected["sha256"]
        ):
            raise ValueError(f"Provider metadata changed for {split}")
        downloaded = Path(
            hf_hub_download(
                REPO,
                source_path(split),
                repo_type="dataset",
                revision=REVISION,
                local_dir=candidate_dir / "raw",
                token=False,
            )
        )
        if (
            downloaded.stat().st_size != expected["bytes"]
            or file_hash(downloaded) != expected["sha256"]
        ):
            raise ValueError(f"Downloaded payload does not match the pinned SHA-256: {split}")
        files[split] = {**expected, "path": source_path(split), "url": source_url(split)}
    documents = []
    card_path = Path(
        hf_hub_download(
            REPO,
            "README.md",
            repo_type="dataset",
            revision=REVISION,
            local_dir=candidate_dir / "raw/documents",
            token=False,
        )
    )
    documents.append(
        {
            "name": "README.md",
            "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md",
            "sha256": file_hash(card_path),
            "bytes": card_path.stat().st_size,
        }
    )
    for name, url in DOCUMENTS.items():
        request = Request(url, headers={"User-Agent": "LexiMind-source-preparation/1.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read(100_001)
        if len(content) > 100_000:
            raise ValueError("Source documentation exceeds the small-file bound")
        details = _create_or_verify(candidate_dir / "raw/documents" / name, [content])
        documents.append({"name": name, "url": url, **details})
    receipt = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "config": "simplified",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "public": True,
        "gated": False,
        "provider_license_declaration": card.get("license"),
        "label_names": LABELS,
        "provider_schema": dataset_info,
        "files": files,
        "documents": documents,
        "scope": "Three simplified source files and small provenance documents only; no raw/full annotation release or model weights.",
    }
    _create_or_verify(receipt_path, [_json_bytes(receipt)])
    _check_cached_sources(candidate_dir, receipt)
    return receipt


def iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        parquet = import_module("pyarrow.parquet")
    except ImportError as error:
        raise RuntimeError(
            "Parquet decoding needs PyArrow only; no datasets or model stack is required"
        ) from error
    source = parquet.ParquetFile(path)
    if set(source.schema_arrow.names) != {"text", "labels", "id"}:
        raise ValueError("Unexpected simplified Parquet columns")
    for batch in source.iter_batches(batch_size=1024, columns=["text", "labels", "id"]):
        yield from batch.to_pylist()


def convert_record(
    row: dict[str, Any], split: str, number: int, source_sha256: str
) -> dict[str, Any]:
    if split not in SPLITS or not isinstance(row, dict):
        raise ValueError("Unsupported split or source record")
    text, comment_id, ids = row.get("text"), row.get("id"), row.get("labels")
    if (
        not isinstance(text, str)
        or not text
        or not isinstance(comment_id, str)
        or not comment_id
        or not isinstance(ids, list)
        or not ids
        or not all(type(index) is int and 0 <= index < len(LABELS) for index in ids)
    ):
        raise ValueError(f"Invalid simplified source record at {split}:{number}")
    return {
        "text": text,
        "emotions": [LABELS[index] for index in ids],
        "label_ids": list(ids),
        "document_id": f"hf:{REPO}:comment:{comment_id}",
        "provider_comment_id": comment_id,
        "provider_dataset": REPO,
        "provider_revision": REVISION,
        "provider_config": "simplified",
        "provider_split": split,
        "split": split,
        "identity_scope": "provider_document",
        "document_identity_basis": "provider_comment_id",
        "identity_source": source_url(split),
        "source_provenance": {"file_sha256": source_sha256, "row": number},
        "candidate_status": "not_admitted",
    }


def exact_match_key(text: str, emotions: list[str]) -> str:
    """Compare exact text and original label order; no normalization or sorting."""
    return sha(json.dumps([text, emotions], ensure_ascii=False, separators=(",", ":")))


def _duplicate_report(db: sqlite3.Connection, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "groups": 0,
        "records_in_groups": 0,
        "cross_split_groups": 0,
        "bounded_samples": [],
    }
    for key, count, splits in db.execute(
        f"SELECT {column},COUNT(*),COUNT(DISTINCT split) FROM candidates GROUP BY {column} HAVING COUNT(*)>1 ORDER BY {column}"
    ):
        result["groups"] += 1
        result["records_in_groups"] += count
        result["cross_split_groups"] += splits > 1
        if len(result["bounded_samples"]) < 4:
            references = [
                {"split": split, "source_row": number, "comment_id_sha256": identifier}
                for split, number, identifier in db.execute(
                    f"SELECT split,row_number,comment_id_sha256 FROM candidates WHERE {column}=? ORDER BY split,row_number LIMIT 4",
                    (key,),
                )
            ]
            result["bounded_samples"].append(
                {
                    "group_sha256": key,
                    "records": count,
                    "cross_split": splits > 1,
                    "references": references,
                }
            )
    return result


def compare_legacy(db: sqlite3.Connection, legacy_dir: Path) -> dict[str, Any]:
    global_counts = dict(db.execute("SELECT match_key,COUNT(*) FROM candidates GROUP BY match_key"))
    split_counts = {
        (split, key): count
        for split, key, count in db.execute(
            "SELECT split,match_key,COUNT(*) FROM candidates GROUP BY split,match_key"
        )
    }
    result: dict[str, Any] = {}
    for split in SPLITS:
        path = legacy_dir / f"{split}.jsonl"
        if not path.exists():
            result[split] = {"status": "missing_legacy_split"}
            continue
        counts: Counter[str] = Counter(
            {
                **{
                    f"{scope}_{status}": 0
                    for scope in ("original_split", "all_splits")
                    for status in ("unmatched", "unique_exact_candidate", "ambiguous_candidates")
                },
                "rows": 0,
                "matched_only_in_other_splits": 0,
            }
        )
        fingerprint = hashlib.sha256()
        samples: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                fingerprint.update(line)
                if not line.strip():
                    continue
                counts["rows"] += 1
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("text"), str)
                    or not isinstance(row.get("emotions"), list)
                    or not all(isinstance(label, str) for label in row["emotions"])
                ):
                    raise ValueError(f"Malformed legacy emotion record at {split}:{number}")
                key = exact_match_key(row["text"], row["emotions"])
                within, total = split_counts.get((split, key), 0), global_counts.get(key, 0)
                for scope, count in (("original_split", within), ("all_splits", total)):
                    status = (
                        "unmatched"
                        if count == 0
                        else "unique_exact_candidate"
                        if count == 1
                        else "ambiguous_candidates"
                    )
                    counts[f"{scope}_{status}"] += 1
                counts["matched_only_in_other_splits"] += within == 0 and total > 0
                if total != 1 and len(samples) < 4:
                    samples.append(
                        {
                            "legacy_line": number,
                            "legacy_record_sha256": sha(line),
                            "match_key_sha256": key,
                            "candidate_rows_all_splits": total,
                        }
                    )
        result[split] = {
            "status": "compared_without_modification",
            "legacy_sha256": fingerprint.hexdigest(),
            "counts": dict(sorted(counts.items())),
            "bounded_ambiguous_or_unmatched_samples": samples,
        }
    return {
        "comparison_rule": "Exact text plus labels in their recorded order; no text normalization, label sorting, fuzzy match, or split reassignment.",
        "splits": result,
        "identity_limit": "Only unique exact source-record candidates are identified by this join. Ambiguous matches remain ambiguous; legacy rows are never rewritten with guessed comment IDs.",
    }


def prepare_candidate(
    candidate_dir: Path, receipt: dict[str, Any], legacy_dir: Path
) -> dict[str, Any]:
    prepared = {}
    label_counts = {}
    duplicate_label_rows = 0
    with tempfile.TemporaryDirectory(prefix="leximind-goemotions-index-") as temporary:
        db = sqlite3.connect(str(Path(temporary) / "hashes.sqlite"))
        try:
            db.execute(
                "CREATE TABLE candidates(split TEXT,row_number INTEGER,comment_id_sha256 TEXT,text_sha256 TEXT,match_key TEXT,label_set_sha256 TEXT)"
            )
            for split in SPLITS:
                counts: Counter[str] = Counter()
                rows = 0

                def encoded_rows(
                    split: str = split, counts: Counter[str] = counts
                ) -> Iterable[bytes]:
                    nonlocal rows, duplicate_label_rows
                    for number, raw in enumerate(
                        iter_parquet_rows(candidate_dir / "raw" / source_path(split)), 1
                    ):
                        row = convert_record(raw, split, number, receipt["files"][split]["sha256"])
                        rows += 1
                        counts.update(row["emotions"])
                        duplicate_label_rows += len(set(row["label_ids"])) != len(row["label_ids"])
                        db.execute(
                            "INSERT INTO candidates VALUES (?,?,?,?,?,?)",
                            (
                                split,
                                number,
                                sha(row["provider_comment_id"]),
                                sha(row["text"]),
                                exact_match_key(row["text"], row["emotions"]),
                                sha(json.dumps(sorted(set(row["label_ids"])))),
                            ),
                        )
                        yield (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        ).encode()
                    if rows != receipt["files"][split]["rows"]:
                        raise ValueError(
                            f"Source row count differs from its pinned metadata: {split}"
                        )

                details = _create_or_verify(
                    candidate_dir / "prepared" / f"{split}.jsonl", encoded_rows()
                )
                prepared[split] = {"path": f"prepared/{split}.jsonl", "rows": rows, **details}
                label_counts[split] = dict(sorted(counts.items()))
                print(f"Prepared unadmitted {split}: {rows} source records", file=sys.stderr)
            db.execute("CREATE INDEX candidate_match ON candidates(match_key,split)")
            db.execute("CREATE INDEX candidate_text ON candidates(text_sha256)")
            db.execute("CREATE INDEX candidate_comment ON candidates(comment_id_sha256)")
            db.commit()
            duplicate_reports = {
                "provider_comment_id": _duplicate_report(db, "comment_id_sha256"),
                "exact_text": _duplicate_report(db, "text_sha256"),
                "exact_text_and_ordered_labels": _duplicate_report(db, "match_key"),
            }
            conflicting_text = db.execute(
                "SELECT COUNT(*) FROM (SELECT text_sha256 FROM candidates GROUP BY text_sha256 HAVING COUNT(DISTINCT label_set_sha256)>1)"
            ).fetchone()[0]
            conflicting_ids = db.execute(
                "SELECT COUNT(*) FROM (SELECT comment_id_sha256 FROM candidates GROUP BY comment_id_sha256 HAVING COUNT(DISTINCT match_key)>1)"
            ).fetchone()[0]
            comparison = compare_legacy(db, legacy_dir)
        finally:
            db.close()
    prepared_labels = _create_or_verify(
        candidate_dir / "prepared/labels.json", [_json_bytes(LABELS)]
    )
    return {
        "schema_version": 1,
        "status": "candidate_prepared_not_admitted",
        "training_authorized": False,
        "repo": REPO,
        "revision": REVISION,
        "config": "simplified",
        "local_candidate_directory": str(candidate_dir.resolve().relative_to(ROOT))
        if candidate_dir.resolve().is_relative_to(ROOT)
        else candidate_dir.name,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "acquisition_receipt_sha256": file_hash(candidate_dir / "raw/acquisition.json"),
        "tooling": {
            "python_version": platform.python_version(),
            "pyarrow_version": version("pyarrow"),
            "model_or_datasets_imports": False,
        },
        "acquisition": receipt,
        "prepared_files": prepared,
        "prepared_label_map": {"path": "prepared/labels.json", **prepared_labels},
        "label_names": LABELS,
        "label_counts": label_counts,
        "duplicate_review": {
            **duplicate_reports,
            "exact_text_groups_with_different_label_sets": conflicting_text,
            "comment_ids_with_conflicting_records": conflicting_ids,
            "records_with_duplicate_label_ids": duplicate_label_rows,
            "policy": "All source records, duplicates, annotation differences and original partitions are retained unchanged.",
        },
        "legacy_comparison": comparison,
        "not_admitted_reasons": [
            "Duplicate-text and conflicting-annotation policy remains unapproved",
            "Model-selection versus calibration grouping is not assigned or frozen",
            "Provider rights/bias/source-policy review remains a separate admission gate",
            "No user authorization for training or research evaluation is implied by preparation",
        ],
        "privacy_boundary": "Only code, provenance metadata, hashes, counts and bounded hashed references are committed. Raw and prepared Reddit text remain in ignored local candidate storage. No user profiles or Reddit endpoints are queried.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "data/research_candidates/go_emotions" / REVISION,
    )
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "data/processed/emotion")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "research/preparation/goemotions_candidate_manifest.json",
    )
    args = parser.parse_args()
    legacy = args.legacy_dir.resolve()
    candidate = args.candidate_dir.resolve()
    manifest_path = args.manifest.resolve()
    if (
        candidate == legacy
        or candidate.is_relative_to(legacy)
        or legacy.is_relative_to(candidate)
        or manifest_path.is_relative_to(legacy)
    ):
        parser.error("Candidate artifacts must remain separate from read-only legacy data")
    if candidate.is_relative_to(ROOT) and not candidate.is_relative_to(
        ROOT / "data/research_candidates"
    ):
        parser.error("In-repository candidate text must stay under data/research_candidates")
    if any(
        manifest_path.is_relative_to(candidate / subdirectory)
        for subdirectory in ("raw", "prepared")
    ):
        parser.error("The committed manifest cannot replace raw or prepared candidate artifacts")
    with advisory_lock(args.candidate_dir / ".prepare.lock"):
        receipt = acquire_sources(args.candidate_dir, fetch=args.fetch)
        manifest = prepare_candidate(args.candidate_dir, receipt, args.legacy_dir)
        write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": {split: manifest["prepared_files"][split]["rows"] for split in SPLITS},
                "training_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
