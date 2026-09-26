"""Small, cached Open Library catalogue imports using the public metadata API."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeGuard, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .identity import matched_work_id
from .storage import json_text, write_json_atomic

API_BASE = "https://openlibrary.org"
USER_AGENT = "LexiMind/0.2 (oliver.t.perrin@gmail.com; https://github.com/OliverPerrin/LexiMind)"

# Display labels are deterministic mappings of actual source subjects, never
# predictions derived from the title, cover, description, or sampling query.
GENRE_RULES = {
    "Science fiction": ("science fiction", "science-fiction", "fiction, science fiction, general"),
    "Fantasy": ("fantasy", "fantasy fiction"),
    "Mystery": ("mystery", "mystery and detective stories", "detective and mystery stories"),
    "Romance": (
        "romance",
        "romance fiction",
        "love stories",
        "love stories, fiction",
        "fiction, romance, contemporary",
    ),
    "Historical fiction": ("historical fiction",),
    "Horror": ("horror", "horror fiction", "horror tales"),
    "Literary fiction": ("literary fiction", "fiction, literary"),
    "Adventure": ("adventure", "adventure stories", "adventure fiction"),
    "Biography": ("biography", "autobiography", "memoir", "memoirs"),
    "History": ("history",),
    "Science": ("science", "popular science", "natural history"),
    "Philosophy": ("philosophy",),
    "Poetry": ("poetry", "poems"),
    "Essays": ("essays", "american essays", "literary collections"),
    "Children's books": ("children's fiction", "juvenile fiction", "children's stories"),
    "Fiction": ("fiction",),
}


class SourceRecordError(ValueError):
    """An individual source record lacks the metadata needed for admission."""


class CatalogueIntegrityError(ValueError):
    """Corrupt provenance/review/publication; abort rather than shrink the output."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.tzinfo is not None and stamp.utcoffset() is not None
    except ValueError:
        return False


def _strings(value: Any, *, nonempty: bool = False) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def validate_response(envelope: Any, expected_url: str) -> None:
    if (
        not isinstance(envelope, dict)
        or not isinstance(envelope.get("data"), dict)
        or envelope.get("url") != expected_url
        or not _valid_timestamp(envelope.get("retrievedAt"))
    ):
        raise CatalogueIntegrityError(f"Invalid source response metadata: {expected_url}")
    if envelope.get("sha256") != digest(envelope["data"]):
        raise CatalogueIntegrityError(f"Source response checksum mismatch: {expected_url}")


class OpenLibraryClient:
    def __init__(self, cache_dir: Path, *, offline: bool = False, interval: float = 1.1):
        self.cache_dir = cache_dir
        self.offline = offline
        self.interval = max(interval, 1.0)
        self.last_request = 0.0
        self.requests = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(path, str) or not (
            path == "/search.json" or re.fullmatch(r"/works/OL\d+W\.json", path)
        ):
            raise SourceRecordError("Only Open Library search and work JSON paths are allowed")
        if params is not None and path != "/search.json":
            raise SourceRecordError("Work lookups cannot override request parameters")
        url = API_BASE + path + ("?" + urlencode(params) if params else "")
        cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if cache.exists():
            try:
                envelope = json.loads(cache.read_text(encoding="utf-8"))
                validate_response(envelope, url)
            except (ValueError, UnicodeError) as error:
                raise CatalogueIntegrityError(f"Corrupt API cache: {cache}: {error}") from error
            return cast(dict[str, Any], envelope)
        if self.offline:
            raise FileNotFoundError(f"No cached Open Library response: {url}")
        for attempt in range(3):
            time.sleep(max(0, self.last_request + self.interval - time.monotonic()))
            self.last_request = time.monotonic()
            try:
                self.requests += 1
                request = Request(
                    url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
                )
                with urlopen(request, timeout=40) as response:
                    if response.geturl() != url:
                        raise CatalogueIntegrityError(
                            f"Unexpected source redirect: {url} -> {response.geturl()}"
                        )
                    data = json.load(response)
                    if not isinstance(data, dict):
                        raise CatalogueIntegrityError(f"Source response must be an object: {url}")
                break
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        envelope = {
            "url": url,
            "retrievedAt": datetime.now(timezone.utc).isoformat(),
            "sha256": digest(data),
            "data": data,
        }
        validate_response(envelope, url)
        write_json_atomic(cache, envelope)
        return envelope


def genres_from_subjects(subjects: list[str]) -> list[str]:
    values = {s.casefold().strip() for s in subjects}
    genres = [genre for genre, aliases in GENRE_RULES.items() if values.intersection(aliases)]
    if len(genres) > 1 and "Fiction" in genres:
        genres.remove("Fiction")
    return genres


def make_book(search: dict[str, Any], work_response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(search, dict) or not isinstance(search.get("key"), str):
        raise SourceRecordError("Search result must identify an Open Library work")
    key = search["key"]
    if not re.fullmatch(r"/works/OL\d+W", key):
        raise SourceRecordError("Search/work identity mismatch")
    validate_response(work_response, API_BASE + key + ".json")
    work = work_response["data"]
    if work.get("key") != key:
        raise SourceRecordError("Search/work identity mismatch")
    authors = search.get("author_name")
    author_keys = search.get("author_key")
    if not _strings(authors, nonempty=True):
        raise SourceRecordError(f"Missing author names: {key}")
    if (
        not _strings(author_keys, nonempty=True)
        or len(author_keys) != len(authors)
        or any(not re.fullmatch(r"OL\d+A", author) for author in author_keys)
    ):
        raise SourceRecordError(f"Malformed search author identity: {key}")
    work_author_records = work.get("authors")
    if not isinstance(work_author_records, list):
        raise SourceRecordError(f"Missing work authors: {key}")
    work_authors = set()
    for record in work_author_records:
        author_key = (
            record.get("author", {}).get("key")
            if isinstance(record, dict) and isinstance(record.get("author"), dict)
            else None
        )
        if not isinstance(author_key, str) or not re.fullmatch(r"/authors/OL\d+A", author_key):
            raise SourceRecordError(f"Malformed work author identity: {key}")
        work_authors.add(author_key.rsplit("/", 1)[-1])
    if set(author_keys) != work_authors:
        raise SourceRecordError(f"Search/work author identity mismatch: {key}")
    description = work.get("description", "")
    if isinstance(description, dict):
        description = description.get("value", "")
    if not isinstance(description, str):
        description = ""
    source_subjects = work.get("subjects") or search.get("subject", [])
    if not _strings(source_subjects):
        raise SourceRecordError(f"Subjects must be a list of nonempty strings: {key}")
    subjects = sorted({subject.strip() for subject in source_subjects})
    source_covers = work.get("covers", [])
    if not isinstance(source_covers, list):
        raise SourceRecordError(f"Covers must be a list: {key}")
    covers = [cover for cover in source_covers if type(cover) is int and cover > 0]
    cover_id = covers[0] if covers else search.get("cover_i")
    year = search.get("first_publish_year")
    book = {
        "id": key.split("/")[-1],
        "title": work.get("title", ""),
        "authors": authors,
        "description": description.strip(),
        "descriptionSource": API_BASE + key if description.strip() else None,
        "coverUrl": f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        if type(cover_id) is int and cover_id > 0
        else None,
        "genres": genres_from_subjects(subjects),
        "subjects": subjects,
        "moods": [],
        "firstPublished": year if type(year) is int else None,
        "source": {
            "name": "Open Library",
            "url": API_BASE + key,
            "retrievedAt": work_response["retrievedAt"],
        },
        "identifiers": {"openLibraryWork": key, "isbns": []},
        "sourceRevision": work.get("revision"),
        "sourceContentHash": work_response["sha256"],
    }
    validate_book(book)
    return book


def apply_review(book: dict[str, Any], review: dict[str, Any]) -> dict[str, Any] | None:
    """Apply explicit, source-pinned exclusions without altering raw evidence."""
    for section in ("excludedWorks", "withheldDescriptions"):
        decision = review.get(section, {}).get(book["id"])
        if not decision:
            continue
        if decision["sourceContentHash"] != book["sourceContentHash"]:
            raise CatalogueIntegrityError(
                f"Source changed; catalogue review required for {book['id']}"
            )
        if section == "excludedWorks":
            return None
        book = {
            **book,
            "description": "",
            "descriptionSource": None,
            "descriptionStatus": "withheld_after_source_review",
        }
    return book


def validate_review(review: Any) -> None:
    if not isinstance(review, dict) or review.get("schemaVersion") != 1:
        raise CatalogueIntegrityError("Invalid catalogue review schema")
    for section in ("excludedWorks", "withheldDescriptions"):
        decisions = review.get(section)
        if not isinstance(decisions, dict):
            raise CatalogueIntegrityError(f"Missing review section: {section}")
        for work_id, decision in decisions.items():
            if (
                not re.fullmatch(r"OL\d+W", work_id)
                or not isinstance(decision, dict)
                or not isinstance(decision.get("sourceContentHash"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", decision["sourceContentHash"])
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()
            ):
                raise CatalogueIntegrityError(f"Invalid review decision for {work_id}")


def validate_book(book: Any) -> None:
    if (
        not isinstance(book, dict)
        or not isinstance(book.get("id"), str)
        or not re.fullmatch(r"OL\d+W", book["id"])
    ):
        raise SourceRecordError("Invalid Open Library work ID")
    key = "/works/" + book["id"]
    identifiers, source = book.get("identifiers"), book.get("source")
    if (
        not isinstance(identifiers, dict)
        or identifiers.get("openLibraryWork") != key
        or not isinstance(source, dict)
        or source.get("name") != "Open Library"
        or source.get("url") != API_BASE + key
    ):
        raise SourceRecordError("Book source and work identifier disagree")
    if not _valid_timestamp(source.get("retrievedAt")):
        raise SourceRecordError("Book source requires a timezone-aware retrieval timestamp")
    if (
        not isinstance(book.get("title"), str)
        or not book["title"].strip()
        or not _strings(book.get("authors"), nonempty=True)
    ):
        raise SourceRecordError("Books require a title and authors")
    for field in ("genres", "subjects", "moods"):
        if not _strings(book.get(field)):
            raise SourceRecordError(f"Book {field} must be a list of nonempty strings")
    if not _strings(identifiers.get("isbns")):
        raise SourceRecordError("Book ISBNs must be a list of strings")
    if not isinstance(book.get("description"), str):
        raise SourceRecordError("Description must be a string")
    expected_description_source = source["url"] if book["description"] else None
    if book.get("descriptionSource") != expected_description_source:
        raise SourceRecordError("Descriptions require matching source attribution")
    if book.get("firstPublished") is not None and type(book["firstPublished"]) is not int:
        raise SourceRecordError("First publication year must be an integer or null")
    cover = book.get("coverUrl")
    if cover is not None and (
        not isinstance(cover, str)
        or not re.fullmatch(r"https://covers\.openlibrary\.org/b/id/[1-9]\d*-[SML]\.jpg", cover)
    ):
        raise SourceRecordError("Invalid Open Library cover URL")
    content_hash = book.get("sourceContentHash")
    if not isinstance(content_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", content_hash):
        raise SourceRecordError("Book requires a source content hash")
    if book["moods"]:
        raise SourceRecordError("This importer has no validated source for book mood labels")


def validate_catalogue(catalogue: Any, manifest: Any | None = None) -> None:
    if not isinstance(catalogue, list) or not catalogue:
        raise CatalogueIntegrityError("Catalogue must be a nonempty book array")
    seen_ids: set[str] = set()
    seen_identity: dict[str, str] = {}
    for book in catalogue:
        validate_book(book)
        if book["id"] in seen_ids:
            raise CatalogueIntegrityError(f"Duplicate work ID: {book['id']}")
        seen_ids.add(book["id"])
        identity = matched_work_id(book)
        if identity in seen_identity:
            raise CatalogueIntegrityError(
                f"Duplicate title/author identity requires review: {seen_identity[identity]}, {book['id']}"
            )
        seen_identity[identity] = book["id"]
    if manifest is not None and (
        not isinstance(manifest, dict)
        or manifest.get("catalogueSha256") != digest(catalogue)
        or manifest.get("count") != len(catalogue)
        or (
            "catalogueFileSha256" in manifest
            and manifest["catalogueFileSha256"]
            != hashlib.sha256(json_text(catalogue).encode()).hexdigest()
        )
    ):
        raise CatalogueIntegrityError("Catalogue and manifest disagree; rebuild both artifacts")


def load_catalogue(catalogue_path: Path, receipt_path: Path | None = None) -> list[dict[str, Any]]:
    """Read one byte snapshot and verify its deployed publication receipt."""
    raw = catalogue_path.read_bytes()
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("schemaVersion")) is not int
            or receipt["schemaVersion"] != 1
            or receipt.get("catalogueFileSha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise CatalogueIntegrityError("Catalogue and publication receipt disagree")
    catalogue = json.loads(raw)
    validate_catalogue(catalogue)
    if receipt_path is not None and (
        type(receipt.get("count")) is not int or receipt["count"] != len(catalogue)
    ):
        raise CatalogueIntegrityError("Catalogue and publication receipt counts disagree")
    return cast(list[dict[str, Any]], catalogue)
