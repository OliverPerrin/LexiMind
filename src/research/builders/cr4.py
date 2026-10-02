"""Verify and index CR4 human annotations locally; --download fetches missing pinned files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from src.catalog.storage import write_json_atomic
from src.research.candidate_io import create_or_verify, helper_hashes
from src.research.cr4 import COLUMNS, fingerprint, normalized, source_rows, validity
from src.research.io import check_file, file_hash

from . import ROOT

API = "https://borealisdata.ca/api"
EXPECTED_ROWS = 207721
FILES = {
    "CR4NarrEmote_All.csv": {
        "url": f"{API}/access/datafile/980979?format=original",
        "bytes": 53_511_582,
        "sha256": "1c358adf610babec70663a4f674f997ba2b8ce50fb693c5eaa4ce8d74d0ed41d",
        "provider_md5": "c18956b7a98251f9a10837b025a5b34d",
    },
    "CR4NarrEmote_ReadMe.txt": {
        "url": f"{API}/access/datafile/980977",
        "bytes": 2064,
        "sha256": "a1f04cfa096a1d6bac9cef9a9d0c991981260145c38eb69cae607d76752cf541",
        "provider_md5": "1ff86daef3bf561df1e682e6ce00bbac",
    },
    "dataset_metadata.json": {
        "url": f"{API}/datasets/:persistentId/?persistentId=doi:10.5683/SP3/XN4ZYZ",
        "bytes": 7155,
        "sha256": "91973824af906cfc49010e908badf9dab5867b61b241ebc204bd8e62d46c0796",
    },
    "schema.xml": {
        "url": f"{API}/access/datafile/980979/metadata",
        "bytes": 5807,
        "sha256": "44ecadd2319d22e669123aaa921c0b75ab9481ffa85b087ac835540afe383d9a",
    },
}


def download_file(path: Path, spec: dict) -> dict:
    """One bounded request; publish only complete pinned bytes, never partial downloads."""

    def chunks():
        sha256, md5, size = hashlib.sha256(), hashlib.md5(), 0
        with urlopen(spec["url"], timeout=45) as response:
            if urlsplit(response.geturl()).hostname != "borealisdata.ca":
                raise ValueError("CR4 download redirected outside the reviewed provider")
            while chunk := response.read(min(1024 * 1024, spec["bytes"] - size + 1)):
                size += len(chunk)
                if size > spec["bytes"]:
                    raise ValueError("CR4 download exceeds the pinned byte count")
                sha256.update(chunk)
                md5.update(chunk)
                yield chunk
        if size != spec["bytes"] or sha256.hexdigest() != spec["sha256"]:
            raise ValueError("CR4 download differs from the pinned source")
        if "provider_md5" in spec and md5.hexdigest() != spec["provider_md5"]:
            raise ValueError("CR4 download differs from the provider checksum")

    result: dict = create_or_verify(path, chunks())
    return result


def _index_chunks(db):
    """One small row-reference record per subject/condition; no copied passage or target text."""
    compressor = zlib.compressobj(wbits=31)
    query = """SELECT workflow,file_id,subject,condition,source_row
        FROM annotations ORDER BY workflow,file_id,subject,condition,source_row"""
    current, references = None, []

    def encode(key, rows):
        workflow, file_id, subject, condition = key
        return (
            json.dumps(
                {
                    "document_namespace": "cr4:document:" + fingerprint(workflow, file_id),
                    "workflow_name": workflow,
                    "file_id": file_id,
                    "subject_id": subject,
                    "condition_sha256": condition,
                    "source_rows": rows,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    for *key, source_row in db.execute(query):
        if current is not None and key != current:
            yield compressor.compress(encode(current, references))
            references = []
        current = key
        references.append(source_row)
    if current is not None:
        yield compressor.compress(encode(current, references))
    yield compressor.flush()


def audit_source(path: Path, index: Path) -> dict:
    """One CSV pass and a temporary hash/ID database; all original annotations remain intact."""
    missing: dict[str, Counter[str]] = {key: Counter() for key in COLUMNS}
    t0: Counter[str] = Counter()
    workflows: Counter[str] = Counter()
    raw_responses: Counter[str] = Counter()
    norm_responses: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    response_states: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    count = 0
    with tempfile.TemporaryDirectory(prefix="leximind-cr4-") as directory:
        with sqlite3.connect(Path(directory) / "audit.sqlite") as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("""CREATE TABLE annotations(source_row INTEGER,classification TEXT,
                user TEXT,workflow TEXT,file_id TEXT,subject TEXT,condition TEXT,passage TEXT,
                t0 TEXT,response TEXT,normalized_response TEXT)""")
            for count, row in source_rows(path):
                state = validity(row)
                for key, value in row.items():
                    if not value.strip():
                        missing[key]["blank"] += 1
                    elif value == "NA":
                        missing[key]["literal_NA"] += 1
                t0[row["t0"]] += 1
                workflows[row["workflow_name"]] += 1
                flags.update(state["flags"])
                response_states[f"{state['character_decision']}:{state['response_state']}"] += 1
                lengths["t0_missing_rows"] += not row["t0"].strip() or row["t0"] == "NA"
                lengths["t0_unknown_rows"] += state["character_decision"] == "unknown"
                lengths["t0_no_with_nonblank_t1_rows"] += state[
                    "character_decision"
                ] == "no" and bool(row["t1"].strip())
                lengths["t0_no_with_nonblank_non_NA_t1_rows"] += (
                    state["character_decision"] == "no" and state["response_state"] == "text"
                )
                lengths["maximum_response_characters"] = max(
                    lengths["maximum_response_characters"], len(row["t1"])
                )
                lengths["maximum_passage_characters"] = max(
                    lengths["maximum_passage_characters"], len(row["passage"])
                )
                lengths["unambiguous_input_rows"] += state["unambiguous_input_condition"]
                response_hash = norm_hash = None
                if state["response_state"] == "text":
                    raw_responses[row["t1"]] += 1
                    norm_responses[normalized(row["t1"])] += 1
                    response_hash = fingerprint(row["t1"])
                    norm_hash = fingerprint(normalized(row["t1"]))
                    lengths["responses_containing_commas"] += "," in row["t1"]
                    lengths["responses_containing_semicolons"] += ";" in row["t1"]
                for key in ("Category", "Genre", "Code", "PUBL_DATE"):
                    categories.setdefault(key, Counter())[row[key]] += 1
                db.execute(
                    "INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        count,
                        row["classification_id"],
                        row["user_id"],
                        row["workflow_name"],
                        row["file_id"],
                        row["subject_ids"],
                        fingerprint(row["passage"], row["highlighted_char"]),
                        fingerprint(row["passage"]),
                        state["character_decision"],
                        response_hash,
                        norm_hash,
                    ),
                )
            if not count:
                raise ValueError("CR4 source contains no annotations")

            def scalar(query):
                return db.execute(query).fetchone()[0]

            def grouped(group, having):
                return scalar(
                    f"SELECT COUNT(*) FROM (SELECT 1 FROM annotations GROUP BY {group} HAVING {having})"
                )

            identity = {
                "annotators": scalar("SELECT COUNT(DISTINCT user) FROM annotations"),
                "source_document_ids": scalar("SELECT COUNT(DISTINCT file_id) FROM annotations"),
                "namespaced_source_documents": grouped("workflow,file_id", "1"),
                "subject_ids": scalar("SELECT COUNT(DISTINCT subject) FROM annotations"),
                "unique_passage_texts": scalar("SELECT COUNT(DISTINCT passage) FROM annotations"),
                "unique_passage_character_conditions": scalar(
                    "SELECT COUNT(DISTINCT condition) FROM annotations"
                ),
                "index_records": grouped("workflow,file_id,subject,condition", "1"),
                "duplicate_classification_ids": grouped("classification", "COUNT(*)>1"),
                "file_ids_reused_across_workflows": grouped(
                    "file_id", "COUNT(DISTINCT workflow)>1"
                ),
                "subjects_with_multiple_conditions": grouped(
                    "subject", "COUNT(DISTINCT condition)>1"
                ),
                "subjects_with_multiple_documents": grouped("subject", "COUNT(DISTINCT file_id)>1"),
                "passage_texts_across_documents": grouped("passage", "COUNT(DISTINCT file_id)>1"),
                "conditions_across_documents": grouped("condition", "COUNT(DISTINCT file_id)>1"),
                "annotators_across_workflows": grouped("user", "COUNT(DISTINCT workflow)>1"),
                "annotators_across_documents": grouped("user", "COUNT(DISTINCT file_id)>1"),
                "repeated_annotator_subject_pairs": grouped("user,subject", "COUNT(*)>1"),
            }
            disagreement = {
                "conditions_with_multiple_annotators": grouped(
                    "condition", "COUNT(DISTINCT user)>1"
                ),
                "conditions_with_mixed_character_decisions": grouped(
                    "condition", "COUNT(DISTINCT t0)>1"
                ),
                "conditions_with_multiple_raw_responses": grouped(
                    "condition", "COUNT(DISTINCT response)>1"
                ),
                "conditions_with_multiple_normalized_responses": grouped(
                    "condition", "COUNT(DISTINCT normalized_response)>1"
                ),
                "conditions_with_only_missing_responses": grouped("condition", "COUNT(response)=0"),
            }
            indexed = create_or_verify(index, _index_chunks(db))

    return {
        "annotation_rows": count,
        "t0_exact_counts": dict(t0),
        "response_state_counts": dict(response_states),
        "missingness": {key: dict(value) for key, value in missing.items() if value},
        "workflow_annotation_counts": dict(workflows),
        "source_metadata_counts": categories,
        "identity": identity,
        "disagreement": disagreement,
        "validity_flags": dict(flags),
        "responses": {
            "nonblank_non_NA_rows": sum(raw_responses.values()),
            "unique_raw_responses": len(raw_responses),
            "unique_comparison_normalized_responses": len(norm_responses),
            "top_20_short_raw_responses": dict(
                [
                    (value, n)
                    for value, n in raw_responses.most_common()
                    if len(value) <= 64 and "\n" not in value
                ][:20]
            ),
            **dict(lengths),
        },
        "index": indexed,
    }


def prepare_candidate(directory: Path, *, download: bool = False) -> dict:
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "data/research_candidates/cr4"):
        raise ValueError("CR4 cache must remain under ignored data/research_candidates/cr4")
    references = {}
    for name, spec in FILES.items():
        path = directory / "raw" / name
        reference = {"path": str(path.relative_to(ROOT)), **spec}
        if not path.exists() and download:
            download_file(path, spec)
        errors = check_file(ROOT, reference)
        if errors:
            raise ValueError("; ".join(errors))
        references[name] = reference
    index = directory / "index-v2" / "subject_references.jsonl.gz"
    observed = audit_source(directory / "raw/CR4NarrEmote_All.csv", index)
    if observed["annotation_rows"] != EXPECTED_ROWS:
        raise ValueError("CR4 row count differs from the pinned source schema")
    for reference in references.values():
        errors = check_file(ROOT, reference)
        if errors:
            raise ValueError("Source changed during audit: " + "; ".join(errors))
    observed["index"]["path"] = str(index.relative_to(ROOT))
    return {
        "schema_version": 1,
        "status": "source_audited_not_admitted",
        "training_authorized": False,
        "dataset": "doi:10.5683/SP3/XN4ZYZ",
        "dataset_version": "1.0",
        "paper_url": "https://aclanthology.org/2025.emnlp-main.493/",
        "provider_declared_license": "CC0-1.0; individual excerpt rights are not independently verified",
        "source_files": references,
        "observed": observed,
        "preparation_script_sha256": file_hash(Path(__file__)),
        "preparation_helper_sha256": {
            **helper_hashes(ROOT),
            "src/research/cr4.py": file_hash(ROOT / "src/research/cr4.py"),
        },
        "policy": {
            "input_condition": ["passage", "highlighted_character"],
            "human_response_columns": ["t0", "t1"],
            "derived_columns_not_acquired": [
                "t1_corrected",
                "t1_unified",
                "label_context",
                "NRC_*",
                "NRCBERT_*",
                "EMO_*",
            ],
            "emotion_ontology": None,
            "assigned_splits": None,
            "training_export": None,
            "missing_t1": "Blank means no supplied inference; literal NA is retained as an ambiguous source missing-value marker. Neither becomes neutral or a negative emotion label.",
            "t0_no": "Rejects the highlighted character OR reports a passage error; not proof that the sentence has no emotion.",
            "highlight_scope": "Exact passage and highlighted string are mandatory. Multiple literal occurrences need span review; no offsets, character identity, or surrounding context are invented.",
            "response_normalization": "NFC, whitespace collapse and casefold are comparison diagnostics only. No stemming, spelling correction, synonym mapping, comma splitting or voting changes targets.",
            "derived_provenance": "Paper Appendix A.1 describes GPT-4o filtering/translation and GPT-o1 morphological aggregation. Cleaned responses are source-derived, partly LLM-assisted; NRC/NRCBERT/EMO are lexical/model mappings, not raw human judgments.",
            "identity_and_splits": "Namespace file_id by workflow as a source-document key, not a reconciled literary work. Before any split, group documents/duplicate conditions and reconcile editions and other corpora; audit overlapping annotators separately.",
            "disagreement": "Keep every source annotation including rejections and missing responses. Distinct response strings measure lexical variation, not semantic disagreement or gold consensus.",
            "index": "Gzip JSONL contains only IDs, hashes and 1-based CSV data-row references, without source passages, response text, user IDs or inferred labels.",
            "download_format": "Provider MD5 matches format=original CSV; the archival TAB export has different bytes, and the default TAB download also inserts a header.",
        },
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-dir", type=Path, default=ROOT / "data/research_candidates/cr4/v1.0"
    )
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing reviewed source files only"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "research/preparation/cr4_candidate_manifest.json"
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.report.resolve().is_relative_to(ROOT / "research/preparation"):
        parser.error("Report must remain under research/preparation")
    try:
        result = prepare_candidate(args.candidate_dir, download=args.download)
        write_json_atomic(args.report, result)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["observed"]["annotation_rows"],
                "index_bytes": result["observed"]["index"]["bytes"],
            }
        )
    )
    return 0
