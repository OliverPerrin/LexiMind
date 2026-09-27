"""Provider-scoped parent document identities for future literary preparation.

These identifiers prevent paragraphs/chapters of a provider document crossing
splits. They do not reconcile editions, translations, or works across sources.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlsplit


def _numeric_id(value: Any) -> str | None:
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value):
        return value
    return None


def _identity(dataset: str, key: str, basis: str, provider_id: str) -> dict[str, Any]:
    return {
        "document_id": f"hf:{dataset}:{key}",
        "identity_scope": "provider_document",
        "document_identity_basis": basis,
        "provider_document_id": provider_id,
        "provider_dataset": dataset,
        "identity_source": f"https://huggingface.co/datasets/{dataset}",
        "work_identity_status": "unresolved",
    }


def gutenberg_document_identity(
    item: dict[str, Any], metadata: dict[str, Any], dataset: str, full_text: str
) -> dict[str, Any]:
    if dataset == "sedthh/gutenberg_english":
        text_id = _numeric_id(metadata.get("text_id"))
        if text_id is not None:
            return _identity(dataset, f"text_id:{text_id}", "metadata.text_id", text_id)
    elif dataset == "deepmind/pg19":
        url = item.get("url")
        if isinstance(url, str):
            parsed = urlsplit(url)
            if (
                parsed.scheme in {"http", "https"}
                and parsed.hostname in {"gutenberg.org", "www.gutenberg.org"}
                and re.fullmatch(r"/ebooks/[1-9]\d*/?", parsed.path)
            ):
                ebook_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
                return _identity(dataset, f"ebook_id:{ebook_id}", "url.ebook_id", url)
    else:
        raise ValueError(f"Unsupported Gutenberg provider: {dataset}")
    if not isinstance(full_text, str) or not full_text:
        raise ValueError("Document-only grouping requires the complete source text")
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    return _identity(
        dataset, f"document-sha256:{content_hash}", "full_source_text_sha256", content_hash
    )


def booksum_document_identity(item: dict[str, Any]) -> dict[str, Any]:
    bid = _numeric_id(item.get("bid"))
    chapter_path = item.get("chapter_path")
    match = (
        re.fullmatch(r"all_chapterized_books/([1-9]\d*)-chapters/[^/]+", chapter_path)
        if isinstance(chapter_path, str)
        else None
    )
    parent_id = match.group(1) if match else None
    if bid and parent_id and bid != parent_id:
        raise ValueError("BookSum bid and chapter_path parent disagree; review source identity")
    parent_id = bid or parent_id
    if not parent_id:
        raise ValueError(
            "BookSum row lacks parent identity: retain bid or a recognized chapter_path before grouping chapters"
        )
    result = _identity(
        "kmfoda/booksum", f"bid:{parent_id}", "bid" if bid else "chapter_path_parent", parent_id
    )
    # Keep the provider's chapter-bearing book_id intact; never use title splitting
    # to infer a parent identity or match a work from another dataset.
    result["provider_ids"] = {
        key: item[key]
        for key in ("bid", "book_id", "chapter_path", "summary_id")
        if isinstance(item.get(key), (str, int)) and not isinstance(item.get(key), bool)
    }
    return result


def booksum_display_title(book_id: str) -> str:
    return re.sub(r"\.(?:chapters?|parts?|sections?)\b.*$", "", book_id, flags=re.IGNORECASE)
