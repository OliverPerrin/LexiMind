"""Streaming CR4 source observations; no label ontology, cleaning, or training export."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

COLUMNS = (
    "file_id",
    "classification_id",
    "user_id",
    "workflow_name",
    "created_at",
    "subject_ids",
    "passage",
    "highlighted_char",
    "Category",
    "Genre",
    "Code",
    "PUBL_DATE",
    "t0",
    "t1",
)
NO_CHARACTER = "No (Highlight is not a character or there is some error with the passage.)"
MAX_RECORD_BYTES = 1024 * 1024  # The release contains a 222,345-character raw response.


def normalized(value: str) -> str:
    """Comparison only: NFC, whitespace collapse and casefold; never a target rewrite."""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def fingerprint(*values: str) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def source_rows(path: Path):
    """Read exact CSV values with bounded records, including quoted multiline responses."""
    with path.open("rb") as stream:
        record_bytes = 0

        def lines():
            nonlocal record_bytes
            while raw := stream.readline(MAX_RECORD_BYTES + 1):
                record_bytes += len(raw)
                if record_bytes > MAX_RECORD_BYTES:
                    raise ValueError("CR4 CSV record exceeds the 1 MiB bound")
                yield raw.decode("utf-8")

        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, MAX_RECORD_BYTES))
        try:
            reader = csv.reader(lines(), strict=True)
            if tuple(next(reader, ())) != COLUMNS:
                raise ValueError("CR4 CSV columns differ from the pinned human-response schema")
            record_bytes = 0
            for number, values in enumerate(reader, 1):
                if len(values) != len(COLUMNS):
                    raise ValueError(f"CR4 row {number} has an invalid field count")
                yield number, dict(zip(COLUMNS, values, strict=True))
                record_bytes = 0
        except csv.Error as error:
            raise ValueError(f"Invalid CR4 CSV: {error}") from error
        finally:
            csv.field_size_limit(previous_limit)


def validity(row: dict[str, str]) -> dict:
    """Conservative review flags, never categorical emotion or neutral labels."""
    passage, character, response = row["passage"], row["highlighted_char"], row["t1"]
    flags = []
    for key in (
        "passage",
        "highlighted_char",
        "file_id",
        "workflow_name",
        "subject_ids",
        "classification_id",
        "user_id",
    ):
        if not row[key].strip() or row[key] == "NA":
            flags.append(f"missing_{key}")
    occurrences = passage.count(character) if character else 0
    if occurrences != 1:
        flags.append("highlight_not_found" if occurrences == 0 else "highlight_span_ambiguous")
    state = {"Yes": "yes", NO_CHARACTER: "no"}.get(row["t0"], "unknown")
    response_state = (
        "blank" if not response.strip() else "na_marker" if response == "NA" else "text"
    )
    if state != "yes":
        flags.append("character_rejected_or_passage_error" if state == "no" else "unknown_t0")
    if response_state != "text":
        flags.append(f"response_{response_state}")
    if state == "no" and response_state == "text":
        flags.append("response_with_rejected_character")
    if len(response) > 1024:
        flags.append("long_response_requires_review")
    if response_state == "text" and normalized(response) == normalized(passage):
        flags.append("response_copies_passage")
    return {
        "character_decision": state,
        "response_state": response_state,
        "literal_highlight_occurrences": occurrences,
        "unambiguous_input_condition": bool(passage.strip() and character.strip())
        and passage != "NA"
        and character != "NA"
        and occurrences == 1,
        "flags": flags,
    }


def annotation(number: int, row: dict[str, str]) -> dict:
    """Keep condition, raw human response and source provenance in separate objects."""
    return {
        "condition": {"passage": row["passage"], "highlighted_character": row["highlighted_char"]},
        "human_response": {"t0": row["t0"], "t1": row["t1"]},
        "source": {
            "dataset": "doi:10.5683/SP3/XN4ZYZ",
            "version": "1.0",
            "source_row": number,
            "document_namespace": "cr4:document:"
            + fingerprint(row["workflow_name"], row["file_id"]),
            **{
                key: row[key]
                for key in COLUMNS
                if key not in {"passage", "highlighted_char", "t0", "t1"}
            },
        },
        "validity": validity(row),
    }


def iter_annotations(path: Path):
    """Local-only observations, without dropping disagreement or unknown responses."""
    for number, row in source_rows(path):
        yield annotation(number, row)
