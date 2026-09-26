"""Preserve the author arXiv archive once and stream ID/split metadata, not models.

Only --fetch allows network acquisition. Source text stays inside the original
ZIP; generated indexes contain locators, identifiers, hashes and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterator
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog.locking import advisory_lock
from src.research.candidate_io import create_or_verify, file_hash, helper_hashes, json_bytes, sha
from src.research.io import parse_json, read_json

REVISION = "6ef082e22b8f49e7195f10c1cdeb5abcf428ff5e"
README_URL = f"https://raw.githubusercontent.com/armancohan/long-summarization/{REVISION}/README.md"
METADATA_URL = "https://archive.org/metadata/armancohan-long-summarization-paper-code"
ARCHIVE_URL = (
    "https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip"
)
DRIVE_URL = "https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view?usp=sharing"
LICENSE_URL = "https://info.arxiv.org/help/license/index.html"
EXPECTED_BYTES = 3_624_420_843
PROVIDER_CHECKSUMS = {
    "sha1": "26f95b9e0f37e9d2bcef31dd1dcd3d25d9367b4d",
    "md5": "6242aaf5cfcc7814473eee8b779c1b9f",
}
MAX_DOWNLOAD_BYTES = 4_000_000_000
MAX_LINE_BYTES = 8_000_000
MAX_UNCOMPRESSED_BYTES = 30_000_000_000
MAX_ROWS = 1_000_000
SPLITS = ("train", "val", "test")
DEFAULT_CANDIDATE = ROOT / "data/research_candidates/arxiv/author-release-26f95b9e0f37"
DOCUMENTS = {
    "author_readme.md": README_URL,
    "archive_metadata.json": METADATA_URL,
    "arxiv_license_information.html": LICENSE_URL,
}
_ARXIV_ID = re.compile(
    r"(?P<base>(?:[0-9]{4}\.[0-9]{4,5}|[A-Za-z][A-Za-z.-]*/[0-9]{7}))(?P<version>v[1-9][0-9]*)?\Z"
)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_storage(candidate: Path) -> None:
    """Reject local aliases before requests or writes inside the candidate tree."""
    base = candidate.resolve()
    directories = [candidate, candidate / "raw", candidate / "raw/documents", candidate / "index"]
    for directory in directories:
        if directory.is_symlink() or not directory.resolve().is_relative_to(base):
            raise ValueError("Candidate directory alias escapes the reviewed storage boundary")
        if directory.exists() and not directory.is_dir():
            raise ValueError("Candidate storage path is not a directory")
    files = [candidate / ".prepare.lock"]
    files += [
        candidate / "raw" / name
        for name in (
            "arxiv-dataset.zip",
            "arxiv-dataset.zip.partial",
            "arxiv-dataset.zip.ranged.partial",
            "acquisition.json",
        )
    ]
    files += [
        candidate / "raw/documents" / suffix
        for name in DOCUMENTS
        for suffix in (name, name + ".receipt.json")
    ]
    files += [candidate / "index" / f"{split}.jsonl" for split in SPLITS]
    for path in files:
        if path.is_symlink():
            raise ValueError("Candidate files must not be symlinks")
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Candidate files must be regular and singly linked")


@contextmanager
def _partial_writer(candidate: Path, offset: int) -> Iterator[BinaryIO]:
    """Open relative to a checked directory FD, without following a partial alias."""
    _validate_storage(candidate)
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("This platform lacks safe directory-relative no-follow source writes")
    raw = candidate / "raw"
    partial = raw / "arxiv-dataset.zip.partial"
    directory_info = raw.stat(follow_symlinks=False)
    previous = partial.lstat() if partial.exists() else None
    directory_fd = os.open(raw, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened_directory = os.fstat(directory_fd)
        if (opened_directory.st_dev, opened_directory.st_ino) != (
            directory_info.st_dev,
            directory_info.st_ino,
        ):
            raise ValueError("Raw directory changed during acquisition")
        flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
        if previous is None:
            flags |= os.O_CREAT | os.O_EXCL
        descriptor = os.open(partial.name, flags, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size != offset:
            raise ValueError("Partial source is aliased, nonregular, or changed")
        if previous is not None and (opened.st_dev, opened.st_ino) != (
            previous.st_dev,
            previous.st_ino,
        ):
            raise ValueError("Partial source changed before opening")
        os.lseek(descriptor, 0, os.SEEK_END)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "ab") as handle:
        yield handle


def verify_archive(path: Path) -> dict[str, str]:
    if path.stat().st_size != EXPECTED_BYTES or EXPECTED_BYTES > MAX_DOWNLOAD_BYTES:
        raise ValueError("Archive length differs from the reviewed acquisition boundary")
    digests = {name: hashlib.new(name) for name in ("sha256", "sha1", "md5")}
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    observed = {name: digest.hexdigest() for name, digest in digests.items()}
    if any(observed[name] != value for name, value in PROVIDER_CHECKSUMS.items()):
        raise ValueError("Archive differs from the author-linked mirror's declared checksums")
    return observed


def acquire_archive(candidate: Path, *, fetch: bool, progress: Callable[[dict], None]) -> dict:
    _validate_storage(candidate)
    raw = candidate / "raw"
    archive, partial = raw / "arxiv-dataset.zip", raw / "arxiv-dataset.zip.partial"
    receipt_path = raw / "acquisition.json"
    if archive.exists():
        observed = verify_archive(archive)
        if not receipt_path.exists():
            raise ValueError("Local archive has no acquisition receipt; preserve it for review")
        receipt = read_json(receipt_path)
        if not isinstance(receipt, dict):
            raise ValueError("Acquisition receipt must be an object")
        if (
            type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
            or receipt.get("status") != "acquired_checksum_verified_not_admitted"
            or receipt.get("training_authorized") is not False
            or receipt.get("requested_url") != ARCHIVE_URL
            or type(receipt.get("bytes")) is not int
            or receipt.get("bytes") != EXPECTED_BYTES
            or receipt.get("provider_declared_checksums") != PROVIDER_CHECKSUMS
            or receipt.get("observed_checksums") != observed
        ):
            raise ValueError("Acquisition receipt does not bind this archive")
        return receipt
    if (raw / "arxiv-dataset.zip.ranged.partial").exists():
        raise ValueError(
            "A ranged source acquisition is incomplete; never use or duplicate its staged bytes"
        )
    if not fetch:
        raise FileNotFoundError("Archive not acquired; --fetch explicitly permits acquisition")
    raw.mkdir(parents=True, exist_ok=True)
    retained = partial.stat().st_size if partial.exists() else 0
    if retained > EXPECTED_BYTES or EXPECTED_BYTES > MAX_DOWNLOAD_BYTES:
        raise ValueError("Source or partial file exceeds reviewed size")
    if shutil.disk_usage(raw).free < EXPECTED_BYTES - retained + 1_000_000_000:
        raise ValueError("Insufficient free space for source and bounded metadata preparation")
    response_headers: dict[str, str | None] = {}
    resolved_url = ARCHIVE_URL
    for attempt in range(1, 4):
        _validate_storage(candidate)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == EXPECTED_BYTES:
            break
        headers = {"User-Agent": "LexiMind-source-preparation/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(ARCHIVE_URL, headers=headers), timeout=45) as response:
                if offset and (
                    response.status != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {offset}-{EXPECTED_BYTES - 1}/{EXPECTED_BYTES}"
                ):
                    raise ValueError("Server did not honor resumed range; partial source retained")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != EXPECTED_BYTES - offset:
                    raise ValueError("Response length differs from reviewed archive size")
                resolved_url = response.url
                response_headers = {
                    key: response.headers.get(key)
                    for key in (
                        "Content-Length",
                        "Content-Type",
                        "Last-Modified",
                        "ETag",
                        "Content-Range",
                    )
                }
                last_report = offset
                with _partial_writer(candidate, offset) as handle:
                    while block := response.read(1024 * 1024):
                        if offset + len(block) > EXPECTED_BYTES:
                            raise ValueError("Response exceeds reviewed byte limit")
                        handle.write(block)
                        offset += len(block)
                        if offset - last_report >= 128 * 1024 * 1024:
                            progress(
                                {
                                    "stage": "download",
                                    "bytes": offset,
                                    "total_bytes": EXPECTED_BYTES,
                                }
                            )
                            last_report = offset
                    handle.flush()
                    os.fsync(handle.fileno())
                if offset != EXPECTED_BYTES:
                    raise OSError("Source ended early; partial bytes retained")
            break
        except OSError as error:
            progress({"stage": "download_retry", "attempt": attempt, "error": str(error)})
            if attempt == 3:
                raise
    observed = verify_archive(partial)
    os.link(partial, archive)
    partial.unlink()
    receipt = {
        "schema_version": 1,
        "status": "acquired_checksum_verified_not_admitted",
        "requested_url": ARCHIVE_URL,
        "resolved_url": resolved_url,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "bytes": EXPECTED_BYTES,
        "provider_declared_checksums": PROVIDER_CHECKSUMS,
        "observed_checksums": observed,
        "response_headers": response_headers,
        "source_use_status": "unreviewed_not_admitted",
        "training_authorized": False,
    }
    create_or_verify(receipt_path, [json_bytes(receipt)])
    return receipt


def provenance_documents(candidate: Path, *, fetch: bool) -> list[dict]:
    _validate_storage(candidate)
    receipts = []
    for name, url in DOCUMENTS.items():
        path = candidate / "raw/documents" / name
        receipt_path = path.with_name(path.name + ".receipt.json")
        if path.exists() and receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                receipt.get("url") != url
                or receipt.get("sha256") != file_hash(path)
                or receipt.get("bytes") != path.stat().st_size
            ):
                raise ValueError(f"Provenance document changed: {name}")
        else:
            if path.exists() or receipt_path.exists():
                raise ValueError(f"Incomplete provenance pair needs review: {name}")
            if not fetch:
                raise FileNotFoundError(f"Missing provenance document: {name}; use --fetch")
            with urlopen(
                Request(url, headers={"User-Agent": "LexiMind-source-preparation/1.0"}), timeout=30
            ) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Provenance document exceeded byte bound")
            receipt = {
                "url": url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                **create_or_verify(path, [data]),
            }
            create_or_verify(receipt_path, [json_bytes(receipt)])
        receipts.append({"path": _relative(path), **receipt})
    metadata = read_json(candidate / "raw/documents/archive_metadata.json")
    entries = [
        entry for entry in metadata.get("files", []) if entry.get("name") == "arxiv-dataset.zip"
    ]
    if (
        len(entries) != 1
        or int(entries[0]["size"]) != EXPECTED_BYTES
        or any(entries[0].get(key) != value for key, value in PROVIDER_CHECKSUMS.items())
    ):
        raise ValueError("Archive metadata disagrees with reviewed source checksum/size")
    readme = (candidate / "raw/documents/author_readme.md").read_text(encoding="utf-8")
    if ARCHIVE_URL not in readme or "article_id" not in readme:
        raise ValueError("Pinned author README does not document the expected source")
    return receipts


def _sentences(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(sentence, str) for sentence in value):
        raise ValueError(f"{field} must be an array of strings")
    return value


def record_metadata(record: Any, split: str, line: int, offset: int, raw: bytes) -> dict:
    required = {"article_id", "abstract_text", "article_text", "section_names", "sections"}
    if not isinstance(record, dict) or not required <= set(record):
        raise ValueError("Original record lacks the documented article identity/content fields")
    identifier = record["article_id"]
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 512:
        raise ValueError("Original article_id must be present; never invent a missing ID")
    article = _sentences(record["article_text"], "article_text")
    abstract = _sentences(record["abstract_text"], "abstract_text")
    names = _sentences(record["section_names"], "section_names")
    sections = record["sections"]
    if not isinstance(sections, list):
        raise ValueError("sections must be an array of sentence arrays")
    for section in sections:
        _sentences(section, "section")
    recognized = _ARXIV_ID.fullmatch(identifier)
    article_joined, abstract_joined = "\n".join(article), "\n".join(abstract)
    normalized = unicodedata.normalize("NFC", " ".join(article_joined.split()))
    flags = []
    if not article_joined.strip():
        flags.append("empty_article")
    if not abstract_joined.strip():
        flags.append("empty_abstract")
    if len(names) != len(sections):
        flags.append("section_name_count_mismatch")
    if sum(map(len, sections)) != len(article):
        flags.append("section_sentence_count_mismatch")
    if recognized is None:
        flags.append("unresolved_arxiv_identifier_format")
    return {
        "source_split": split,
        "source_line": line,
        "uncompressed_byte_offset": offset,
        "source_line_bytes": len(raw),
        "source_line_sha256": sha(raw),
        "article_id": identifier,
        "arxiv_base_id": recognized.group("base") if recognized else None,
        "article_version": recognized.group("version") if recognized else None,
        "source_url": f"https://arxiv.org/abs/{identifier}" if recognized else None,
        "article_sentence_count": len(article),
        "abstract_sentence_count": len(abstract),
        "article_char_count": len(article_joined),
        "abstract_char_count": len(abstract_joined),
        "section_count": len(sections),
        "section_name_count": len(names),
        "article_joined_sha256": sha(article_joined),
        "abstract_joined_sha256": sha(abstract_joined),
        "article_normalized_sha256": sha(normalized),
        "flags": flags,
        "additional_fields": sorted(set(record) - required),
    }


def _members(archive: zipfile.ZipFile) -> tuple[dict[str, zipfile.ZipInfo], list[dict]]:
    members = archive.infolist()
    if len(members) > 1000 or sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Archive member count or expanded size exceeds reviewed bound")
    selected: dict[str, zipfile.ZipInfo] = {}
    inventory = []
    seen = set()
    for item in members:
        name = PurePosixPath(item.filename)
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in item.filename
            or item.filename in seen
        ):
            raise ValueError("Unsafe or duplicate archive member name")
        seen.add(item.filename)
        if stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
            raise ValueError("Symlink/encrypted members are outside the source contract")
        split = name.stem if name.name in {f"{key}.txt" for key in SPLITS} else None
        if split:
            if split in selected or item.is_dir():
                raise ValueError("Multiple or invalid files for a provider split")
            selected[split] = item
        inventory.append(
            {
                "name": item.filename,
                "bytes": item.file_size,
                "compressed_bytes": item.compress_size,
                "crc32": f"{item.CRC:08x}",
                "source_split": split,
            }
        )
    if set(selected) != set(SPLITS):
        raise ValueError("Archive must contain exactly one train.txt, val.txt and test.txt")
    return selected, inventory


def _groups(database: sqlite3.Connection, key: str) -> dict:
    if key not in {
        "article_id",
        "arxiv_base_id",
        "article_hash",
        "normalized_hash",
        "abstract_hash",
    }:
        raise ValueError("Unknown audit key")
    groups, rows, cross = database.execute(
        f"SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(s>1),0) FROM "
        f"(SELECT COUNT(*) n, COUNT(DISTINCT split) s FROM records WHERE {key} IS NOT NULL "
        f"GROUP BY {key} HAVING COUNT(*)>1)"
    ).fetchone()
    return {"groups": groups, "rows": rows, "cross_split_groups": cross}


def index_archive(
    archive_path: Path, candidate: Path, *, progress: Callable[[dict], None] | None = None
) -> dict:
    """Stream a local ZIP into metadata-only indexes; do not extract a text corpus."""
    progress = progress or (lambda event: None)
    _validate_storage(candidate)
    candidate.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(prefix=".arxiv-audit-", dir=candidate) as temporary,
    ):
        selected, members = _members(archive)
        database = sqlite3.connect(Path(temporary) / "index.sqlite")
        try:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE records (article_id TEXT, arxiv_base_id TEXT, split TEXT, article_hash TEXT, normalized_hash TEXT, abstract_hash TEXT)"
            )
            outputs: dict[str, dict] = {}
            total_rows = 0
            for split in SPLITS:
                rows, lines, offset = 0, 0, 0
                flags: Counter[str] = Counter()
                lengths = {
                    key: {"sum": 0, "max": 0}
                    for key in ("article_char_count", "abstract_char_count")
                }
                member_hash = hashlib.sha256()

                def chunks(
                    source_split: str = split,
                    source_hash: Any = member_hash,
                    anomaly_counts: Counter[str] = flags,
                    length_stats: dict = lengths,
                ) -> Iterator[bytes]:
                    nonlocal rows, lines, offset, total_rows
                    with archive.open(selected[source_split]) as stream:
                        while raw := stream.readline(MAX_LINE_BYTES + 1):
                            if len(raw) > MAX_LINE_BYTES:
                                raise ValueError(
                                    f"{source_split}: source record exceeds bounded line size"
                                )
                            lines += 1
                            start = offset
                            offset += len(raw)
                            source_hash.update(raw)
                            if not raw.strip():
                                anomaly_counts["blank_lines"] += 1
                                continue
                            record = record_metadata(
                                parse_json(raw), source_split, lines, start, raw
                            )
                            rows += 1
                            total_rows += 1
                            if total_rows > MAX_ROWS:
                                raise ValueError("Source row count exceeds preparation bound")
                            anomaly_counts.update(record["flags"])
                            anomaly_counts["version_not_stated"] += (
                                record["article_version"] is None
                            )
                            for key in length_stats:
                                length_stats[key]["sum"] += record[key]
                                length_stats[key]["max"] = max(
                                    length_stats[key]["max"], record[key]
                                )
                            database.execute(
                                "INSERT INTO records VALUES (?,?,?,?,?,?)",
                                (
                                    record["article_id"],
                                    record["arxiv_base_id"],
                                    source_split,
                                    record["article_joined_sha256"],
                                    record["article_normalized_sha256"],
                                    record["abstract_joined_sha256"],
                                ),
                            )
                            if rows % 10_000 == 0:
                                progress({"stage": "index", "split": source_split, "rows": rows})
                            yield (
                                json.dumps(
                                    record,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            ).encode()
                    # The generator must finish validation before create_or_verify
                    # publishes its staged file, including when it yielded no rows.
                    if offset != selected[source_split].file_size or rows == 0:
                        raise ValueError(f"{source_split}: source member was incomplete or empty")

                index_path = candidate / "index" / f"{split}.jsonl"
                receipt = create_or_verify(index_path, chunks())
                database.commit()
                outputs[split] = {
                    "rows": rows,
                    "source_lines": lines,
                    "archive_member": selected[split].filename,
                    "uncompressed_bytes": offset,
                    "uncompressed_sha256": member_hash.hexdigest(),
                    "index": {"path": _relative(index_path), **receipt},
                    "flags": dict(sorted(flags.items())),
                    "lengths": lengths,
                }
            groups = {
                key: _groups(database, key)
                for key in (
                    "article_id",
                    "arxiv_base_id",
                    "article_hash",
                    "normalized_hash",
                    "abstract_hash",
                )
            }
            unique_ids, recognized_ids = database.execute(
                "SELECT COUNT(DISTINCT article_id), COUNT(DISTINCT arxiv_base_id) FROM records"
            ).fetchone()
            return {
                "rows": total_rows,
                "unique_original_article_ids": unique_ids,
                "unique_recognized_arxiv_base_ids": recognized_ids,
                "splits": outputs,
                "archive_members": members,
                "duplicate_groups": groups,
            }
        finally:
            database.close()


def prepare(candidate: Path, *, fetch: bool, progress: Callable[[dict], None]) -> dict:
    documents = provenance_documents(candidate, fetch=fetch)
    acquisition = acquire_archive(candidate, fetch=fetch, progress=progress)
    archive = candidate / "raw/arxiv-dataset.zip"
    audit = index_archive(archive, candidate, progress=progress)
    return {
        "schema_version": 1,
        "status": "candidate_prepared_not_admitted",
        "training_authorized": False,
        "source": {
            "author_repository": "https://github.com/armancohan/long-summarization",
            "author_revision": REVISION,
            "readme_url": README_URL,
            "archive_url": ARCHIVE_URL,
            "alternative_author_link": DRIVE_URL,
            "metadata_url": METADATA_URL,
            "provider_declared_checksums": PROVIDER_CHECKSUMS,
            "observed_archive": {
                "path": _relative(archive),
                "bytes": archive.stat().st_size,
                "sha256": acquisition["observed_checksums"]["sha256"],
            },
            "acquisition_receipt": {
                "path": _relative(candidate / "raw/acquisition.json"),
                "sha256": file_hash(candidate / "raw/acquisition.json"),
            },
            "documents": documents,
        },
        "conversion": {
            "script": "scripts/prepare_arxiv_candidate.py",
            "script_sha256": file_hash(Path(__file__)),
            "helper_sha256": helper_hashes(ROOT),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "source_splits": list(SPLITS),
            "split_aliases": {"validation": "val"},
            "split_reassignment": False,
            "text_corpus_copied": False,
            "row_locator": "archive member + 1-based source line + uncompressed byte offset",
            "normalization": "article sentences joined by newline, whitespace collapsed, Unicode NFC; case and punctuation retained",
            "max_line_bytes": MAX_LINE_BYTES,
            "max_rows": MAX_ROWS,
            "max_expanded_bytes": MAX_UNCOMPRESSED_BYTES,
        },
        "audit": audit,
        "source_use": {
            "status": "unreviewed_not_admitted",
            "legal_clearance_asserted": False,
            "provider_readme_dataset_license": "not specified in the inspected README",
            "arxiv_terms_url": LICENSE_URL,
            "article_level_licenses_recovered": False,
            "note": "Public access and repository code licensing do not grant one uniform article-text license.",
        },
        "remaining_gates": [
            "article-level source-use and version review",
            "duplicate/identifier generalization policy",
            "selection/calibration partition",
            "reviewed preprocessing and length policy",
            "explicit experiment resumption",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "research/preparation/arxiv_source_manifest.json"
    )
    args = parser.parse_args()
    try:
        if not args.candidate_dir.resolve().is_relative_to(
            (ROOT / "data/research_candidates").resolve()
        ):
            raise ValueError("Candidate output must remain under ignored data/research_candidates")
        _validate_storage(args.candidate_dir)
        args.candidate_dir.mkdir(parents=True, exist_ok=True)
        with advisory_lock(args.candidate_dir / ".prepare.lock"):
            report = prepare(
                args.candidate_dir,
                fetch=args.fetch,
                progress=lambda row: print(json.dumps(row), flush=True),
            )
            create_or_verify(args.manifest, [json_bytes(report)])
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "rows": report["audit"]["rows"],
                    "manifest": _relative(args.manifest),
                }
            )
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, sqlite3.Error) as error:
        print(f"arXiv preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
