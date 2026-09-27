"""Audit the pinned BGC source archive without exporting text or training labels.

The archive uses XML-like framing with unescaped ampersands. Parse its literal
record frame instead of repairing text or invoking an XML entity processor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.storage import write_json_atomic
from src.research.candidate_io import create_or_verify, helper_hashes
from src.research.io import check_file, file_hash

SOURCE_URL = "https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip"
SOURCE_SHA256 = "41e6d70c2db2b4ec8dd644be7d08371f8f879c6d698732a829c9a24a42f38d7c"
SOURCE_BYTES = 48_690_771
MAX_ARCHIVE_BYTES = 100_000_000
MAX_UNCOMPRESSED_BYTES = 200_000_000
MAX_RECORD_BYTES = 256_000
MEMBERS = {split: f"BlurbGenreCollection_EN_{split}.txt" for split in ("train", "dev", "test")}
FIELDS = ("title", "body", "copyright", "author", "published", "page_num", "isbn", "url")
FRAME = re.compile(
    r'<book date="(?P<record_date>[^"]+)" xml:lang="(?P<language>[^"]+)">\s*'
    + "".join(rf"<{key}>(?P<{key}>.*?)</{key}>\s*" for key in FIELDS[:3])
    + r"<metadata>\s*<topics>(?P<topics>.*?)</topics>\s*"
    + "".join(rf"<{key}>(?P<{key}>.*?)</{key}>\s*" for key in FIELDS[3:])
    + r"</metadata>\s*</book>\s*",
    re.DOTALL,
)
TOPIC = re.compile(r"<d([0-9]+)>(.*?)</d\1>", re.DOTALL)


def records(stream, *, digest=None):
    """Yield literal fields from one bounded record at a time; fail on bad framing."""
    parts, size, number = [], 0, 0
    while raw := stream.readline(MAX_RECORD_BYTES + 1):
        if digest is not None:
            digest.update(raw)
        if len(raw) > MAX_RECORD_BYTES:
            raise ValueError("BGC line exceeds record bound")
        if not parts and not raw.strip():
            continue
        parts.append(raw)
        size += len(raw)
        if size > MAX_RECORD_BYTES:
            raise ValueError("BGC record exceeds memory bound")
        if raw.strip() == b"</book>":
            number += 1
            match = FRAME.fullmatch(b"".join(parts).decode("utf-8"))
            if match is None:
                raise ValueError(f"Invalid BGC record frame at source row {number}")
            row = match.groupdict()
            pairs = TOPIC.findall(row.pop("topics"))
            raw_topics = match.group("topics")
            if (
                not pairs
                or TOPIC.sub("", raw_topics).strip()
                or any(not label.strip() for _, label in pairs)
            ):
                raise ValueError(f"Invalid BGC topic frame at source row {number}")
            row["labels"] = [(int(depth), label) for depth, label in pairs]
            yield row
            parts, size = [], 0
    if parts:
        raise ValueError("Truncated BGC record")


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def isbn13(value):
    value = re.sub(r"[\s-]", "", value)
    if (
        re.fullmatch(r"[0-9]{13}", value)
        and value.startswith(("978", "979"))
        and not value.startswith("9790")  # ISMN sheet-music identifiers, not ISBNs.
        and sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(value)) % 10 == 0
    ):
        return value
    return None


def provider_id(value):
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.username is not None
        or parsed.password is not None
        or port not in (None, {"http": 80, "https": 443}.get(parsed.scheme))
    ):
        return None
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "penguinrandomhouse.com",
        "www.penguinrandomhouse.com",
    }:
        return None
    match = re.match(r"^/books/([0-9]+)(?:/|$)", parsed.path)
    return match[1] if match else None


def edition_date(value):
    match = re.fullmatch(r"([A-Z][a-z]{2}) ([0-9]{2}), ([0-9]{4})", value.strip())
    if not match:
        return None
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    try:
        return date(int(match[3]), months.index(match[1]) + 1, int(match[2])).isoformat()
    except ValueError:
        return None


def duplicate_summary(db, column):
    if column not in {"isbn", "provider", "exact", "normalized", "title_author"}:
        raise ValueError("Unknown duplicate-audit column")
    duplicates = f"SELECT {column}, COUNT(*) n FROM records WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(*) > 1"
    groups, rows = db.execute(f"SELECT COUNT(*), COALESCE(SUM(n),0) FROM ({duplicates})").fetchone()
    cross = db.execute(
        f"SELECT COUNT(*) FROM (SELECT {column} FROM records WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(DISTINCT split)>1)"
    ).fetchone()[0]
    disagree = db.execute(
        f"SELECT COUNT(*) FROM (SELECT {column} FROM records WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(DISTINCT labels)>1)"
    ).fetchone()[0]
    examples = []
    for value, count in db.execute(f"{duplicates} ORDER BY n DESC,{column} LIMIT 3"):
        refs = [
            dict(split=s, source_row=r)
            for s, r in db.execute(
                f"SELECT split,source_row FROM records WHERE {column}=? ORDER BY split,source_row LIMIT 4",
                (value,),
            )
        ]
        examples.append({"value": value, "records": count, "references": refs})
    return {
        "unique_values": db.execute(f"SELECT COUNT(DISTINCT {column}) FROM records").fetchone()[0],
        "duplicate_groups": groups,
        "records_in_duplicate_groups": rows,
        "cross_split_groups": cross,
        "different_label_set_groups": disagree,
        "bounded_examples": examples,
    }


def audit_archive(path):
    """Read local archive only; temporary SQLite stores hashes/IDs, never blurb text."""
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds 100 MB source bound")
    splits, labels, depth_labels = {}, Counter(), {}
    member_refs, observed_dates, languages = {}, Counter(), Counter()
    identity = Counter()
    with (
        zipfile.ZipFile(path) as archive,
        tempfile.TemporaryDirectory(prefix="leximind-bgc-index-") as temporary,
    ):
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        allowed = set(MEMBERS.values()) | {"README.txt", "hierarchy.txt", "description.pdf"}
        if len(set(names)) != len(names) or set(names) - allowed - {
            n for n in names if n.startswith("__MACOSX/")
        }:
            raise ValueError("Unexpected or duplicate archive member")
        if not set(MEMBERS.values()) | {"README.txt", "hierarchy.txt"} <= set(names):
            raise ValueError("Missing BGC source member")
        if sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Uncompressed archive exceeds audit bound")
        for name in ("README.txt", "hierarchy.txt"):
            if archive.getinfo(name).file_size > MAX_RECORD_BYTES:
                raise ValueError("Source metadata exceeds bound")
            raw = archive.read(name)
            member_refs[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        readme = archive.read("README.txt").decode("utf-8")
        if "CC BY-NC 4.0" not in readme:
            raise ValueError("Source README does not declare the reviewed license")
        edges, declared_roots = [], set()
        for line in archive.read("hierarchy.txt").decode("utf-8").splitlines():
            pair = line.split("\t")
            if len(pair) == 1 and pair[0].strip():
                declared_roots.add(pair[0])
            elif len(pair) == 2 and all(part.strip() for part in pair):
                edges.append(tuple(pair))
            else:
                raise ValueError("Invalid hierarchy edge or standalone root")
        parents = {}
        for parent, child in edges:
            parents.setdefault(child, set()).add(parent)
        with sqlite3.connect(Path(temporary) / "audit.sqlite") as db:
            db.execute(
                "CREATE TABLE records(split TEXT,source_row INTEGER,isbn TEXT,provider TEXT,exact TEXT,normalized TEXT,title_author TEXT,labels TEXT)"
            )
            for split, member in MEMBERS.items():
                count, split_labels, dates, years = 0, set(), Counter(), Counter()
                date_count, date_min, date_max = 0, None, None
                missing_fields = Counter()
                hierarchy_missing = repeated_labels = 0
                digest = hashlib.sha256()
                with archive.open(member) as stream:
                    for count, row in enumerate(records(stream, digest=digest), start=1):
                        row_labels = {label for _, label in row["labels"]}
                        repeated_labels += len(row["labels"]) != len(set(row["labels"]))
                        hierarchy_missing += any(
                            not parents.get(label, set()) <= row_labels for label in row_labels
                        )
                        labels.update(row_labels)
                        split_labels.update(row_labels)
                        for depth, label in row["labels"]:
                            depth_labels.setdefault(depth, set()).add(label)
                        observed_dates[row["record_date"]] += 1
                        languages[row["language"]] += 1
                        for field in FIELDS:
                            if not row[field].strip():
                                missing_fields[field] += 1
                        parsed = edition_date(row["published"])
                        if parsed:
                            date_count += 1
                            date_min = min(date_min, parsed) if date_min else parsed
                            date_max = max(date_max, parsed) if date_max else parsed
                            years[parsed[:3] + "0s"] += 1
                        else:
                            dates["missing" if not row["published"].strip() else "unparsed"] += 1
                        isbn, provider = isbn13(row["isbn"]), provider_id(row["url"])
                        identity["valid_isbn13" if isbn else "invalid_or_missing_isbn13"] += 1
                        identity[
                            "provider_book_id_present"
                            if provider
                            else "unparsed_or_missing_provider_url"
                        ] += 1
                        title_author = json.dumps(
                            [normalized(row["title"]), normalized(row["author"])],
                            ensure_ascii=False,
                        )
                        db.execute(
                            "INSERT INTO records VALUES(?,?,?,?,?,?,?,?)",
                            (
                                split,
                                count,
                                isbn,
                                provider,
                                hashlib.sha256(row["body"].encode()).hexdigest(),
                                hashlib.sha256(normalized(row["body"]).encode()).hexdigest(),
                                hashlib.sha256(title_author.encode()).hexdigest(),
                                json.dumps(sorted(row_labels)),
                            ),
                        )
                member_refs[member] = {
                    "bytes": archive.getinfo(member).file_size,
                    "sha256": digest.hexdigest(),
                    "rows": count,
                }
                splits[split] = {
                    "rows": count,
                    "unique_labels": len(split_labels),
                    "empty_fields": dict(missing_fields),
                    "records_with_repeated_depth_label_pairs": repeated_labels,
                    "records_missing_declared_parent_labels": hierarchy_missing,
                    "edition_dates": {
                        "parsed": date_count,
                        "missing": dates["missing"],
                        "unparsed": dates["unparsed"],
                        "minimum": date_min,
                        "maximum": date_max,
                        "decades": dict(sorted(years.items())),
                    },
                }
            duplicates = {
                key: duplicate_summary(db, column)
                for key, column in (
                    ("isbn13", "isbn"),
                    ("provider_book_id", "provider"),
                    ("exact_blurb", "exact"),
                    ("normalized_blurb", "normalized"),
                    ("normalized_title_author_candidate", "title_author"),
                )
            }
    nodes = {node for pair in edges for node in pair} | declared_roots
    return {
        "splits": splits,
        "total_records": sum(split["rows"] for split in splits.values()),
        "unique_label_names": len(labels),
        "label_record_counts": dict(sorted(labels.items())),
        "labels_by_depth": {
            str(depth): len(values) for depth, values in sorted(depth_labels.items())
        },
        "hierarchy": {
            "edges": len(edges),
            "unique_edges": len(set(edges)),
            "nodes": len(nodes),
            "standalone_roots": sorted(declared_roots),
            "observed_labels_missing_from_hierarchy": sorted(set(labels) - nodes),
            "hierarchy_labels_absent_from_records": sorted(nodes - set(labels)),
        },
        "members": member_refs,
        "identity": dict(identity),
        "duplicates": duplicates,
        "provider_record_dates": dict(observed_dates),
        "languages": dict(languages),
    }


def preserve_source(source: Path, destination: Path) -> dict:
    """Publish only a complete byte-identical pinned archive; never overwrite one."""

    def chunks():
        digest, size = hashlib.sha256(), 0
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                size += len(block)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds 100 MB source bound")
                digest.update(block)
                yield block
        if size != SOURCE_BYTES or digest.hexdigest() != SOURCE_SHA256:
            raise ValueError("Archive differs from the reviewed official BGC source")

    return create_or_verify(destination, chunks())


def prepare_candidate(source: Path, directory: Path) -> dict:
    root, directory = ROOT.resolve(), directory.resolve()
    if not directory.is_relative_to(root / "data/research_candidates"):
        raise ValueError("BGC source cache must remain in ignored data/research_candidates")
    destination = directory / "blurbgenrecollectionen.zip"
    reference = {
        "path": str(destination.relative_to(root)),
        "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256,
    }
    if destination.exists():
        errors = check_file(root, reference)
        if errors:
            raise ValueError("; ".join(errors))
    else:
        preserve_source(source, destination)
    statistics = audit_archive(destination)
    errors = check_file(root, reference)
    if errors:
        raise ValueError("Source changed during audit: " + "; ".join(errors))
    return {
        "schema_version": 1,
        "status": "source_audited_not_admitted",
        "training_authorized": False,
        "source_url": SOURCE_URL,
        "provider_page": "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html",
        "source_declared_license": "CC BY-NC 4.0; README retains Penguin Random House/content-provider attribution",
        "archive": reference,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": helper_hashes(root),
        "provider_advertised_records": 91892,
        "observed": statistics,
        "interpretation": [
            "Original archive, split membership, field text and source label names are preserved; no training export is created.",
            "Counts describe source records, ISBN editions and publisher URL identifiers, not reconciled literary works.",
            "ISBN checks cover prefix/format/checksum, not registration or metadata correctness; publisher IDs are parsed only from the supplied URLs.",
            "The published field is an edition publication date, possibly prospective at collection time, not an original-work date or proof of modern writing.",
            "Exact-blurb hashes use literal UTF-8 field content; normalized hashes use NFC and whitespace collapse only, preserving case/punctuation.",
            "Title-author hashes are review candidates, not work identities; blank authors and matching titles can conflate works.",
            "Source genre categories and hierarchy are silver metadata; missing labels are unknown, not negative or adjudicated gold.",
            "No duplicate is removed, relabeled or reassigned; cross-split identities/text require an explicit work-group policy before admission.",
            "No model, scoring, website redistribution or publisher-page scraping occurred; archive payload stays ignored locally.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Already downloaded official ZIP; no network fallback"
    )
    parser.add_argument(
        "--candidate-dir", type=Path, default=ROOT / "data/research_candidates/bgc" / SOURCE_SHA256
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/bgc_candidate_manifest.json"
    )
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(ROOT / "research/preparation"):
        parser.error("Report must remain separate from source data, under research/preparation")
    try:
        report = prepare_candidate(
            args.archive or args.candidate_dir / "blurbgenrecollectionen.zip", args.candidate_dir
        )
        write_json_atomic(args.report, report)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1
    observed = report["observed"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "records": observed["total_records"],
                "unique_labels": observed["unique_label_names"],
                "splits": {key: value["rows"] for key, value in observed["splits"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
