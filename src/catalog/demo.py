"""Safe, dependency-free adapter for the historical Gradio discovery demo."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .openlibrary import load_catalogue


def has_validated_tone(item: dict[str, Any]) -> bool:
    source = item.get("emotion_source")
    emotion = item.get("emotion")
    status = item.get("emotion_status")
    if not isinstance(source, str) or not isinstance(emotion, str) or not isinstance(status, str):
        return False
    try:
        url = urlsplit(source)
        valid_source = url.scheme == "https" and bool(url.hostname)
    except ValueError:
        return False
    return (
        status in {"validated_editorial", "validated_in_domain"}
        and valid_source
        and emotion.strip().casefold() not in {"", "neutral", "unknown"}
    )


def load_demo_items(
    catalogue_path: Path, legacy_paths: list[Path], receipt_path: Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonical books plus historical papers; never reuse old literary joins."""
    items: list[dict[str, Any]] = []
    notices = []
    if catalogue_path.exists():
        books = load_catalogue(catalogue_path, receipt_path)
        for book in books:
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
        with path.open(encoding="utf-8") as stream:
            papers = _read_papers(stream)
        items.extend(papers)
        break
    return items, notices


def _read_papers(lines: Any) -> list[dict[str, Any]]:
    papers = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("source_type") != "academic":
            continue
        if not has_validated_tone(item):
            item = {
                **item,
                "emotion": "Unknown",
                "emotion_confidence": 0.0,
                "emotion_status": "unvalidated_domain_abstention",
            }
        papers.append(item)
    return papers


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
