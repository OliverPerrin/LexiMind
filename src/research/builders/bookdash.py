"""Pinned, independent Book Dash cohort; whole-work splits before model use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

from src.catalog.storage import write_json_atomic
from src.research.candidate_io import create_or_verify, json_bytes, sha
from src.research.io import check_file, file_hash, read_json, safe_path

from . import ROOT, bgc_source, licensed_books

POLICY = "bookdash-independent-cohort-v1"
REGISTRY = "research/preparation/bookdash_sources.json"
MANIFEST = "research/preparation/bookdash_manifest.json"
CACHE = "data/research_candidates/bookdash"
COMMIT = licensed_books.BOOKDASH_COMMIT
RAW = licensed_books.BOOKDASH_BASE
TREE_URL = f"https://api.github.com/repos/bookdash/bookdash-books/git/trees/{COMMIT}?recursive=1"


def reference(path, root):
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def acquire(root, descriptor, url, *, fetch=False):
    """Read exact pinned bytes, retaining the acquisition receipt separately."""
    if (
        descriptor.get("url") != url
        or type(descriptor["bytes"]) is not int
        or not 0 < descriptor["bytes"] <= 500_000
        or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
    ):
        raise ValueError("Invalid bounded Book Dash source")
    path = safe_path(root, descriptor["path"])
    if not path.is_relative_to(root / CACHE / "sources"):
        raise ValueError("Book Dash sources must remain in candidate storage")
    receipt = path.with_name(path.name + ".receipt.json")
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"Missing {path.name}; use --fetch for the pinned source")
        request = urllib.request.Request(
            url, headers={"Accept-Encoding": "identity", "User-Agent": "LexiMind research"}
        )
        with urllib.request.build_opener(licensed_books.PinnedRedirects()).open(
            request, timeout=30
        ) as response:
            raw = licensed_books.verified_bytes(response.read(descriptor["bytes"] + 1), descriptor)
            event = {
                "url": url,
                "final_url": response.url,
                "bytes": len(raw),
                "sha256": sha(raw),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "acquisition": "direct_source_bytes",
            }
        create_or_verify(path, [raw])
        create_or_verify(receipt, [json_bytes(event)])
    if errors := check_file(root, descriptor):
        raise ValueError("; ".join(errors))
    if receipt.stat().st_size > 32_000:
        raise ValueError("Oversized Book Dash receipt")
    event = read_json(receipt)
    if (
        any(event.get(key) != descriptor[key] for key in ("url", "bytes", "sha256"))
        or datetime.fromisoformat(event["retrieved_at"]).tzinfo is None
    ):
        raise ValueError("Book Dash acquisition receipt mismatch")
    final = event.get("final_url")
    if final != url and not (
        event.get("acquisition") == "github_git_blob_base64"
        and final
        == f"https://api.github.com/repos/bookdash/bookdash-books/git/blobs/{descriptor.get('git_blob')}"
    ):
        raise ValueError("Book Dash acquisition location changed")
    return path.read_bytes(), reference(receipt, root)


def prepare_book(book, raw, metadata):
    """Bind publisher metadata and the separately declared source heading exactly."""
    slug = book["slug"]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise ValueError("Invalid Book Dash slug")
    match = re.search(
        r"^  " + re.escape(slug) + r":\n(.*?)(?=^  [a-z]|\Z)", metadata.decode(), re.M | re.S
    )
    fields = dict(re.findall(r'^    ([a-z]+): "([^"\n]*)"', match[1], re.M)) if match else {}
    if (
        any(
            fields.get(key) != book[value]
            for key, value in (
                ("title", "title"),
                ("date", "publication_date"),
                ("identifier", "provider_id"),
                ("source", "provider_isbn"),
            )
        )
        or fields.get("language") != "en"
        or fields.get("publisher") != "Book Dash"
    ):
        raise ValueError("Book Dash publisher metadata mismatch")
    headings = re.findall(r"^# (.+)$", raw.decode(), re.M)
    if headings != [book["source_heading"]]:
        raise ValueError("Book Dash exact source heading mismatch")
    record = {
        "work_id": "bookdash-" + slug,
        "title": book["title"],
        "source_heading": book["source_heading"],
        "creators": book["creators"],
        "publisher": "Book Dash",
        "language": "en",
        "publication_date": book["publication_date"],
        "date_basis": "Pinned publisher repository date, not independently verified original publication",
        "provider_id": book["provider_id"],
        "provider_isbn": book["provider_isbn"],
        "source": slug + ".md",
        "source_page": f"https://bookdash.github.io/bookdash-books/{slug}/en/",
        "extractor": "bookdash_markdown",
        "license": {
            "id": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "evidence_source": "bookdash-meta.yml",
            "evidence_anchor": f"titles.{slug}.rights",
            "obligations": [
                "Credit original creators and Book Dash",
                "Link source and license; mark changes; do not imply endorsement",
            ],
        },
        "source_sha256": sha(raw),
        "changes": "Exclude frontmatter, title heading, images and image descriptions; normalize trailing whitespace and excess blank lines; preserve page boundaries including empty pages",
    }
    # Existing helper binds all creator credits, IDs and CC-BY evidence. The
    # heading projection is explicit; the original metadata title stays above.
    licensed_books.validate_license(
        {**record, "title": book["source_heading"]},
        {record["source"]: raw, "bookdash-meta.yml": metadata},
    )
    sections = licensed_books.bookdash(raw)
    words = sum(len(section["text"].split()) for section in sections)
    if words != book["expected_words"] or words < 100:
        raise ValueError("Book Dash source text differs from reviewed cohort")
    return {**record, "sections": sections, "whitespace_words": words}


def normalized(text):
    return bgc_source.normalized(text).casefold().replace("’", "'")


def identity_keys(title, isbns, aliases=()):
    return {
        *(("title", normalized(value)) for value in (title, *aliases)),
        *(("isbn", value) for raw in isbns if (value := bgc_source.isbn13(raw))),
    } - {("title", "")}


def book_keys(book):
    keys = identity_keys(
        book["title"], book["isbns"], [book["source_heading"]] if book.get("source_heading") else []
    )
    texts = [section["text"] for section in book.get("sections", [])]
    for kind, text in [("body", "\n".join(texts)), *(("page", text) for text in texts)]:
        if value := normalized(text):
            keys.add((kind, sha(value)))
    return keys


def screen(books, external):
    """Quarantine even title-only matches; duplicate pages never cross splits."""
    keys, matches = defaultdict(set), defaultdict(list)
    for book in books:
        for key in book_keys({**book, "isbns": [book["provider_isbn"]]}):
            keys[key].add(book["work_id"])
    for other in external:
        found = defaultdict(set)
        for key in book_keys(other):
            for work_id in keys.get(key, ()):
                found[work_id].add(key[0])
        for work_id, reasons in found.items():
            matches[work_id].append(
                {"source": other["source"], "id": other["id"], "matching_keys": sorted(reasons)}
            )
    seen = defaultdict(set)
    for book in books:
        work_id = book["work_id"]
        for key in book_keys({**book, "isbns": [book["provider_isbn"]]}):
            seen[key].add(work_id)
    for (kind, _), work_ids in sorted(seen.items()):
        if len(work_ids) > 1:
            for work_id in sorted(work_ids):
                matches[work_id].append(
                    {
                        "source": "cohort",
                        "id": sorted(work_ids - {work_id}),
                        "matching_keys": [kind],
                    }
                )
    return dict(matches)


def assign_splits(books, matches):
    """Fixed approximate 70/15/15 ranked hash allocation, independent of model scores."""
    eligible = sorted(
        (book["work_id"] for book in books if book["work_id"] not in matches),
        key=lambda work_id: sha(f"{POLICY}:{work_id}"),
    )
    if len(eligible) < 3:
        raise ValueError("Need at least three independent eligible works")
    held = max(1, round(len(eligible) * 0.15))
    train_end = len(eligible) - 2 * held
    return {
        work_id: "train" if i < train_end else "dev" if i < train_end + held else "test"
        for i, work_id in enumerate(eligible)
    }


def prepare(*, root=ROOT, fetch=False):
    root = root.resolve()
    registry = read_json(root / REGISTRY)
    if (
        registry["schema_version"] != 1
        or registry["policy"] != POLICY
        or registry["commit"] != COMMIT
        or not 3 <= len(registry["books"]) <= 40
    ):
        raise ValueError("Unsupported bounded Book Dash cohort")
    slugs = [book["slug"] for book in registry["books"]]
    if len(set(slugs)) != len(slugs) or any(
        not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in slugs
    ):
        raise ValueError("Invalid or repeated Book Dash work")
    inputs = {
        name: reference(root / path, root)
        for name, path in {
            "registry": REGISTRY,
            "legacy_sources": "research/preparation/licensed_books_sources.json",
            "legacy_manifest": "research/preparation/licensed_books_manifest.json",
            "bgc_manifest": "research/preparation/bgc_candidate_manifest.json",
            "catalogue": "web/data/books.json",
        }.items()
    }
    metadata, metadata_receipt = acquire(
        root, registry["metadata"], RAW + "_data/meta.yml", fetch=fetch
    )
    tree_raw, tree_receipt = acquire(root, registry["tree"], TREE_URL, fetch=fetch)
    tree = json.loads(tree_raw)
    if tree.get("sha") != COMMIT or tree.get("truncated") is not False:
        raise ValueError("Book Dash source tree is incomplete or changed")
    blobs = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    inputs.update(
        metadata=registry["metadata"],
        metadata_receipt=metadata_receipt,
        tree=registry["tree"],
        tree_receipt=tree_receipt,
    )
    legacy = read_json(root / inputs["legacy_sources"]["path"])
    legacy_manifest = read_json(root / inputs["legacy_manifest"]["path"])
    if legacy_manifest["source_inventory"] != inputs["legacy_sources"]:
        raise ValueError("Prior licensed texts do not bind the source inventory")
    prior_artifacts = {}
    for book in legacy_manifest["books"]:
        artifact = {
            **book["artifact"],
            "path": str(
                safe_path(
                    root / legacy_manifest["cache_root"], book["artifact"]["path"]
                ).relative_to(root)
            ),
        }
        if errors := check_file(root, artifact):
            raise ValueError("; ".join(errors))
        inputs["prior_work:" + book["work_id"]] = artifact
        prior_artifacts[book["work_id"]] = read_json(safe_path(root, artifact["path"]))
    old_ids = {book["work_id"] for book in legacy["books"]}
    if set(prior_artifacts) != old_ids:
        raise ValueError("Prior licensed prepared identities are incomplete")
    books = []
    for book in registry["books"]:
        slug, source = book["slug"], book["source"]
        if "bookdash-" + slug in old_ids or blobs.get(f"{slug}/en/index.md") != source["git_blob"]:
            raise ValueError("Source reuses a prior work or differs from the pinned tree")
        url = RAW + slug + "/en/index.md"
        descriptor = {**source, "url": url, "path": f"{CACHE}/sources/{source['sha256']}/{slug}.md"}
        raw, receipt = acquire(root, descriptor, url, fetch=fetch)
        if (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            != source["git_blob"]
        ):
            raise ValueError("Book Dash git object hash mismatch")
        inputs["source:" + slug], inputs["receipt:" + slug] = descriptor, receipt
        books.append(prepare_book(book, raw, metadata))
    bgc = read_json(root / inputs["bgc_manifest"]["path"])
    archive = bgc["archive"]
    if archive["bytes"] != bgc_source.SOURCE_BYTES or archive["sha256"] != bgc_source.SOURCE_SHA256:
        raise ValueError("Expected the pinned BGC screening archive")
    inputs["bgc_archive"] = archive
    if errors := check_file(root, archive):
        raise ValueError("; ".join(errors))
    scanned: Counter[str] = Counter()

    def external():
        for book in legacy["books"]:
            scanned["prior_licensed"] += 1
            yield {
                "source": "prior_licensed",
                "id": book["work_id"],
                "title": book["title"],
                "isbns": [book["provider_isbn"]] if book.get("provider_isbn") else [],
                "sections": prior_artifacts[book["work_id"]]["sections"],
            }
        for book in read_json(root / inputs["catalogue"]["path"]):
            scanned["catalogue"] += 1
            yield {
                "source": "catalogue",
                "id": book["id"],
                "title": book["title"],
                "isbns": book["identifiers"]["isbns"],
            }
        with zipfile.ZipFile(safe_path(root, archive["path"])) as opened:
            for split, member in bgc_source.MEMBERS.items():
                digest, count = hashlib.sha256(), 0
                with opened.open(member) as stream:
                    for count, row in enumerate(bgc_source.records(stream, digest=digest), 1):
                        scanned["bgc"] += 1
                        yield {
                            "source": "bgc",
                            "id": f"bgc:{split}:{count}",
                            "title": row["title"],
                            "isbns": [row["isbn"]],
                        }
                expected = bgc["observed"]["members"][member]
                if count != expected["rows"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError("BGC screening source differs from its pinned audit")

    matches = screen(books, external())
    splits = assign_splits(books, matches)
    implementation = {
        name: file_hash(root / name)
        for name in (
            "src/research/builders/bookdash.py",
            "src/research/builders/licensed_books.py",
            "src/research/builders/bgc_source.py",
            "src/research/candidate_io.py",
            "src/research/io.py",
            "src/catalog/storage.py",
        )
    }
    build_key = sha(
        json_bytes({"inputs": inputs, "implementation": implementation, "policy": POLICY})
    )
    records = []
    for book in books:
        work_id = book["work_id"]
        group_id = "bookdash-work:" + sha(work_id)
        split = splits.get(work_id, "quarantine")
        path = root / CACHE / "prepared" / build_key / (work_id + ".json")
        payload = {
            "schema_version": 1,
            **book,
            "group_id": group_id,
            "proposed_split": split,
            "candidate_status": "unadmitted",
            "training_authorized": False,
        }
        artifact = create_or_verify(path, [json_bytes(payload)])
        records.append(
            {
                "work_id": work_id,
                "title": book["title"],
                "group_id": group_id,
                "proposed_split": split,
                "license": book["license"]["id"],
                "whitespace_words": book["whitespace_words"],
                "sections": len(book["sections"]),
                "matches": matches.get(work_id, []),
                "artifact": {"path": str(path.relative_to(root)), **artifact},
            }
        )
    for item in inputs.values():
        if errors := check_file(root, item):
            raise ValueError("Book Dash inputs changed during preparation: " + "; ".join(errors))
    return {
        "schema_version": 1,
        "policy": POLICY,
        "candidate_status": "unadmitted",
        "training_authorized": False,
        "training_performed": False,
        "inputs": inputs,
        "implementation_sha256": implementation,
        "build_key": build_key,
        "books": records,
        "counts": {
            "works": len(records),
            "works_by_split": dict(Counter(book["proposed_split"] for book in records)),
            "whitespace_words": sum(book["whitespace_words"] for book in records),
            "screened_records": dict(scanned),
        },
        "contract": {
            "split": "Rank stable SHA-256(policy:work_id) among unambiguous works; reserve max(1, round(15%)) each for dev and test, all remaining for train. Assigned before model use. Every page and future window inherits the whole-work split. Never train on dev/test/quarantine.",
            "screening": "Quarantine any title or explicit source-heading alias match (NFC/whitespace/casefold/curly-apostrophe normalization) or valid ISBN match against BGC, the catalogue, prior licensed works, or this cohort. Also quarantine fullbody/page duplicates within this cohort or against prior licensed texts using the same normalization. No fuzzy match, translated edition or pretraining contamination guarantee.",
            "scope": "Previously unused English picture books from one publisher, 100+ narrative words each. Images omitted; narrow genre and audience coverage. No human field labels or blanket training authorization.",
            "license": "Per-title CC-BY-4.0 evidence and every creator in the pinned metadata retained. Current publisher guidance (https://bookdash.org/re-using-the-book-dash-content/) additionally requests author/illustrator/designer/editor credits, source and logo. The older metadata does not assign roles or establish whether an editor credit is missing; credit completeness needs review before redistribution. This preparation is local text only. Repository MIT terms do not replace source conditions.",
        },
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fetch", action="store_true", help="Acquire missing exact pinned sources")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    manifest = prepare(fetch=args.fetch)
    write_json_atomic(ROOT / MANIFEST, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    return 0
