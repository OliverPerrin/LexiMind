"""Prepare deterministic BGC leakage groups from the pinned local source; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_bgc_candidate as source
from src.catalog.storage import write_json_atomic
from src.research.book_groups import POLICY, BookGroupRecord, prepare_book_groups, text_keys
from src.research.candidate_io import helper_hashes
from src.research.io import check_file, file_hash, read_json, safe_path


def grouping_record(row, split, number):
    keys = text_keys(row["title"], row["author"], row["body"])
    if isbn := source.isbn13(row["isbn"]):
        keys["isbn13"] = isbn
    if provider := source.provider_id(row["url"]):
        keys["provider_book_id"] = provider
    return BookGroupRecord(
        record_id=f"bgc:{split}:{number}",
        source_split=split,
        source_row=number,
        keys=keys,
        labels=tuple(row["labels"]),
    )


def prepare(candidate_manifest: Path, output: Path) -> dict:
    candidate_manifest, output = candidate_manifest.resolve(), output.resolve()
    if not candidate_manifest.is_relative_to(ROOT / "research/preparation"):
        raise ValueError("Candidate manifest must remain under research/preparation")
    if not output.is_relative_to(ROOT / "data/research_candidates"):
        raise ValueError("Assignment files must remain in ignored data/research_candidates")
    manifest = read_json(candidate_manifest)
    manifest_ref = {
        "path": str(candidate_manifest.relative_to(ROOT)),
        "bytes": candidate_manifest.stat().st_size,
        "sha256": file_hash(candidate_manifest),
    }
    archive_ref = manifest["archive"]
    if archive_ref["sha256"] != source.SOURCE_SHA256 or archive_ref["bytes"] != source.SOURCE_BYTES:
        raise ValueError("BGC grouping only accepts the reviewed original archive")
    dependencies = {
        **helper_hashes(ROOT),
        "src/research/book_groups.py": file_hash(ROOT / "src/research/book_groups.py"),
        "scripts/prepare_bgc_candidate.py": file_hash(Path(source.__file__)),
    }
    if manifest["preparation_script_sha256"] != dependencies[
        "scripts/prepare_bgc_candidate.py"
    ] or manifest["preparation_helper_sha256"] != helper_hashes(ROOT):
        raise ValueError("BGC candidate audit dependencies changed")
    errors = check_file(ROOT, archive_ref)
    if errors:
        raise ValueError("; ".join(errors))
    archive_path = safe_path(ROOT, archive_ref["path"])

    def rows():
        with zipfile.ZipFile(archive_path) as archive:
            for split, member in source.MEMBERS.items():
                digest, count = hashlib.sha256(), 0
                with archive.open(member) as stream:
                    for count, row in enumerate(source.records(stream, digest=digest), 1):
                        yield grouping_record(row, split, count)
                expected = manifest["observed"]["members"][member]
                if count != expected["rows"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError("Source member differs from the candidate audit")
        errors = check_file(ROOT, archive_ref) + check_file(ROOT, manifest_ref)
        if errors:
            raise ValueError("Source changed during grouping: " + "; ".join(errors))

    observed = prepare_book_groups(rows(), output, source.SOURCE_SHA256)
    for key in ("assignments", "review_groups"):
        observed[key]["path"] = str((output / f"{key}.jsonl").relative_to(ROOT))
    return {
        "schema_version": 1,
        "status": "candidate_leakage_groups_not_admitted",
        "training_authorized": False,
        "policy": POLICY,
        "candidate_manifest": manifest_ref,
        "archive": archive_ref,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": dependencies,
        "split_policy": {"train": 0.64, "dev": 0.16, "test": 0.20},
        **observed,
        "interpretation": [
            "Connected groups are conservative leakage constraints, not verified literary works; title/author and shared blurbs can conflate unrelated editions or books.",
            "ISBN prefix/checksum and provider URL parsing validate syntax only, not identity registration or metadata correctness.",
            "Text matches use NFC and whitespace collapse only, preserving case/punctuation; empty blurbs and incomplete title-author pairs produce no shared key.",
            "All original row IDs, splits and source labels remain unchanged; the proposed split is a separate deterministic hash allocation of complete groups.",
            "Every multi-record group remains unadjudicated; conflicting label sets are reported per member and never combined into gold labels.",
            "Zero residuals apply only to these exact matching keys; paraphrases, translated titles, author variants and unobserved work identities can still leak.",
            "The proposed allocation is not a reproduction of the published benchmark splits, is not label-stratified and remains unadmitted.",
            "Source rights and per-field mappings still govern any later use; no model, training, scoring or network requests occurred.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "research/preparation/bgc_candidate_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/research_candidates/bgc" / source.SOURCE_SHA256 / "groups-v1",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/bgc_group_manifest.json"
    )
    args = parser.parse_args()
    if (
        not args.report.resolve().is_relative_to(ROOT / "research/preparation")
        or args.report.resolve() == args.candidate_manifest.resolve()
    ):
        parser.error("Report must be separate from the candidate under research/preparation")
    try:
        report = prepare(args.candidate_manifest, args.output_dir)
        write_json_atomic(args.report, report)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], **report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
