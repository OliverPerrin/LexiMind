"""Conservative identity matching for legacy Gutenberg/description joins.

Titles alone are not identifiers. Preserve subtitles and reject records without
author evidence; ambiguous descriptions must be reviewed rather than selected.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from typing import Any


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def normalize_author(author: str) -> str:
    # Gutenberg records often use "Austen, Jane, 1775-1817".
    author = re.sub(r",?\s*\b\d{4}\s*[-–]\s*(?:\d{4})?\b", "", author)
    parts = [part.strip() for part in author.split(",") if part.strip()]
    if len(parts) == 2:
        author = f"{parts[1]} {parts[0]}"
    return normalize_title(author)


def author_names(record: dict[str, Any]) -> list[str]:
    value = record.get("authors") or record.get("author") or record.get("Author")
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    names = []
    for author in value:
        if isinstance(author, dict):
            author = author.get("name", "")
        if not isinstance(author, str) or not author.strip():
            return []  # Never silently discard a missing/malformed coauthor.
        if normalize_author(author) in {"unknown", "anonymous", "various", "n a"}:
            return []  # These labels do not establish a person's identity.
        names.append(author.strip())
    return names


def author_identity(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({normalize_author(a) for a in author_names(record)} - {""}))


def match_description(
    book: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    title = normalize_title(book["title"]) if isinstance(book.get("title"), str) else ""
    authors = author_identity(book)
    if not title or not authors:
        return None
    matches = [
        row
        for row in candidates
        if isinstance(row, dict)
        and isinstance(row.get("title"), str)
        and normalize_title(row["title"]) == title
        and author_identity(row) == authors
    ]
    # Repeated identical source records are harmless; conflicting blurbs are not.
    unique = {json.dumps(row, sort_keys=True): row for row in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def matched_work_id(book: dict[str, Any]) -> str:
    """Stable conservative grouping key, not a claim of canonical authority."""
    title = normalize_title(book["title"]) if isinstance(book.get("title"), str) else ""
    authors = author_identity(book)
    if not title or not authors:
        raise ValueError("Work identity requires a complete title and author evidence")
    value = json.dumps([title, authors], ensure_ascii=False, separators=(",", ":"))
    return "title-author:" + hashlib.sha256(value.encode()).hexdigest()[:24]
