"""Deterministic work/document partitions for future source preparation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SPLITS = ("train", "validation", "test")


def split_source_records(
    records: list[dict[str, Any]], *, seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    """Keep a work in one split, preserve source splits, and reject conflicts.

    Hash assignment is stable when input order or other works change. The 90/5/5
    fractions are expected across many groups, not exact quotas for small samples.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("split") is not None and record["split"] not in SPLITS:
            raise ValueError(f"Unknown source split: {record['split']!r}")
        if record.get("type") in {"literary", "gutenberg"}:
            key = record.get("work_id")
            if isinstance(key, str) and key.strip():
                key = "literary:" + key
            elif (
                isinstance(record.get("document_id"), str)
                and record["document_id"].strip()
                and record.get("identity_scope") == "provider_document"
                and record.get("work_identity_status") == "unresolved"
            ):
                key = "document:" + record["document_id"]
            else:
                raise ValueError(
                    "Literary splitting requires a verified work_id or explicit provider-document identity"
                )
        else:
            source = record.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Summarization records require source text")
            key = "text:" + hashlib.sha256(source.encode()).hexdigest()
        groups.setdefault(key, []).append(record)
    result: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for key in sorted(groups):
        group = groups[key]
        pinned = {record["split"] for record in group if record.get("split") is not None}
        if len(pinned) > 1:
            raise ValueError(f"Conflicting source splits for work {key}")
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
        partition = (
            next(iter(pinned))
            if pinned
            else "train"
            if bucket < 90
            else "validation"
            if bucket < 95
            else "test"
        )
        for record in sorted(
            group, key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
        ):
            result[partition].append(
                {name: value for name, value in record.items() if name != "split"}
            )
    return result


def split_summarization_records(
    records: list[dict[str, Any]], *, seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    """Compatibility entry point; document grouping does not prove work isolation."""
    return split_source_records(records, seed=seed)
