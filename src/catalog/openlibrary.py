"""Small, cached Open Library catalogue imports using the public metadata API."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    "Children's books": ("children's fiction", "juvenile fiction", "children's stories"),
    "Fiction": ("fiction",),
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class OpenLibraryClient:
    def __init__(self, cache_dir: Path, *, offline: bool = False, interval: float = 1.1):
        self.cache_dir = cache_dir
        self.offline = offline
        self.interval = max(interval, 1.0)
        self.last_request = 0.0
        self.requests = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = API_BASE + path + ("?" + urlencode(params) if params else "")
        cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if cache.exists():
            envelope = json.loads(cache.read_text())
            if envelope["url"] != url or envelope["sha256"] != digest(envelope["data"]):
                raise ValueError(f"Corrupt API cache: {cache}")
            return envelope
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
                    data = json.load(response)
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
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n")
        return envelope


def genres_from_subjects(subjects: list[str]) -> list[str]:
    values = {s.casefold().strip() for s in subjects}
    genres = [genre for genre, aliases in GENRE_RULES.items() if values.intersection(aliases)]
    if len(genres) > 1 and "Fiction" in genres:
        genres.remove("Fiction")
    return genres


def make_book(search: dict[str, Any], work_response: dict[str, Any]) -> dict[str, Any]:
    work = work_response["data"]
    key = work.get("key", "")
    if not re.fullmatch(r"/works/OL\d+W", key) or search.get("key") != key:
        raise ValueError("Search/work identity mismatch")
    authors = search.get("author_name", [])
    if not authors or not all(isinstance(a, str) and a.strip() for a in authors):
        raise ValueError(f"Missing author names: {key}")
    # Search and work are fetched under the same work identifier. Confirm that
    # search author IDs are actually present on the authoritative work record.
    work_authors = {
        a.get("author", {}).get("key", "").split("/")[-1] for a in work.get("authors", [])
    }
    search_authors = set(search.get("author_key", []))
    if not search_authors or not search_authors.issubset(work_authors):
        raise ValueError(f"Search/work author identity mismatch: {key}")
    description = work.get("description", "")
    if isinstance(description, dict):
        description = description.get("value", "")
    if not isinstance(description, str):
        description = ""
    subjects = sorted(
        {s.strip() for s in work.get("subjects", []) if isinstance(s, str) and s.strip()}
    )
    if not subjects:
        subjects = sorted(
            {s.strip() for s in search.get("subject", []) if isinstance(s, str) and s.strip()}
        )
    covers = [cover for cover in work.get("covers", []) if isinstance(cover, int) and cover > 0]
    cover_id = covers[0] if covers else search.get("cover_i")
    year = search.get("first_publish_year")
    book = {
        "id": key.split("/")[-1],
        "title": work.get("title", ""),
        "authors": authors,
        "description": description.strip(),
        "descriptionSource": API_BASE + key if description.strip() else None,
        "coverUrl": f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        if isinstance(cover_id, int) and cover_id > 0
        else None,
        "genres": genres_from_subjects(subjects),
        "subjects": subjects,
        "moods": [],
        "firstPublished": year if isinstance(year, int) else None,
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
            raise ValueError(f"Source changed; catalogue review required for {book['id']}")
        if section == "excludedWorks":
            return None
        book = {
            **book,
            "description": "",
            "descriptionSource": None,
            "descriptionStatus": "withheld_after_source_review",
        }
    return book


def validate_book(book: dict[str, Any]) -> None:
    if not re.fullmatch(r"OL\d+W", book["id"]):
        raise ValueError("Invalid Open Library work ID")
    key = "/works/" + book["id"]
    if book["identifiers"]["openLibraryWork"] != key or book["source"]["url"] != API_BASE + key:
        raise ValueError("Book source and work identifier disagree")
    if not book["title"] or not book["authors"]:
        raise ValueError("Books require a title and authors")
    if book["description"] and book.get("descriptionSource") != book["source"]["url"]:
        raise ValueError("Descriptions require matching source attribution")
    if book["moods"]:
        raise ValueError("This importer has no validated source for book mood labels")
