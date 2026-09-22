"""Safe, dependency-free adapter for the historical Gradio discovery demo."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from .openlibrary import validate_book


def has_validated_tone(item: dict[str, Any]) -> bool:
    return (
        item.get("emotion_status") in {"validated_editorial", "validated_in_domain"}
        and bool(item.get("emotion_source"))
        and item.get("emotion") not in {None, "", "neutral", "Unknown"}
    )


def load_demo_items(
    catalogue_path: Path, legacy_paths: list[Path]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonical books plus historical papers; never reuse old literary joins."""
    items: list[dict[str, Any]] = []
    notices = []
    if catalogue_path.exists():
        books = json.loads(catalogue_path.read_text())
        if not isinstance(books, list):
            raise ValueError("Book catalogue must be an array")
        for book in books:
            validate_book(book)
            items.append(
                {
                    "id": book["id"],
                    "title": book["title"],
                    "authors": book["authors"],
                    "source_type": "literary",
                    "dataset": "openlibrary",
                    "topic": book["genres"][0] if book["genres"] else "",
                    "topics": book["genres"],
                    "genres": book["genres"],
                    "text": " ".join([book["description"], *book["subjects"], *book["authors"]]),
                    "reference_summary": book["description"],
                    "generated_summary": "",
                    "source_url": book["source"]["url"],
                    "description_source": book["descriptionSource"],
                    "emotion": "Unknown",
                    "emotion_status": "unvalidated_domain_abstention",
                }
            )
    else:
        notices.append(
            "The verified book catalogue is unavailable. Historical book descriptions are withheld because their identity was not verified."
        )
    for path in legacy_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("source_type") != "academic":
                continue
            if not has_validated_tone(item):
                item = {
                    **item,
                    "emotion": "Unknown",
                    "emotion_confidence": 0.0,
                    "emotion_status": "unvalidated_domain_abstention",
                }
            items.append(item)
        break
    return items, notices


def _plain_markdown(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+!|>])", r"\\\1", html.escape(value, quote=False))


def format_book_card(item: dict[str, Any]) -> str:
    title = _plain_markdown(item.get("title", "Untitled"))
    authors = ", ".join(item.get("authors", []))
    parts = ["Book"]
    if authors:
        parts.append(_plain_markdown(authors))
    if item.get("genres"):
        parts.append("Genres: " + _plain_markdown(", ".join(item["genres"])))
    if has_validated_tone(item):
        parts.append("Tone: " + _plain_markdown(item["emotion"].title()))
    card = f"### {title}\n\n*{' | '.join(parts)}*\n\n"
    description = item.get("reference_summary", "").strip()
    card += (
        _plain_markdown(description)
        if description
        else "No description is available in the reviewed source record."
    ) + "\n\n"
    # Source is validated against a fixed Open Library work URL by the loader.
    card += f"[Book record and description source: Open Library]({item['source_url']})\n\n---\n\n"
    return card
