"""Build a blind, train-only worksheet from a pinned field diagnostic cohort."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.research.book_fields import input_hash
from src.research.book_partitions import group_overrides
from src.research.builders.book_fields import checked, jsonl_rows, reference
from src.research.builders.field_review import CONTRACT
from src.research.candidate_io import create_or_verify, json_bytes, sha
from src.research.io import read_json
from src.research.review_app import build

REVIEW_SALT = "bgc-field-review-v1"
REVIEW_COUNT = 32


def select_review_rows(train_rows: list[dict], dev_rows: list[dict]) -> list[dict]:
    """Select by ID alone; never choose based on targets, ranker outputs or misses."""
    if len(train_rows) < REVIEW_COUNT:
        raise ValueError("The fixed review cohort needs at least 32 training records")
    if len({r["record_id"] for r in train_rows}) != len(train_rows):
        raise ValueError("Repeated training review identity")
    train_groups = {r["group_id"] for r in train_rows}
    if len(train_groups) != len(train_rows) or train_groups & {r["group_id"] for r in dev_rows}:
        raise ValueError("Review groups must be unique and disjoint from development")
    if any(r["source_split"] != "train" or r["effective_split"] != "train" for r in train_rows):
        raise ValueError("Blind review accepts original and effective training roles only")
    return sorted(
        train_rows, key=lambda row: (sha(f"{REVIEW_SALT}:{row['record_id']}"), row["record_id"])
    )[:REVIEW_COUNT]


def build_review(root: Path, config_path: Path, examples_path: Path, output_dir: Path) -> dict:
    """Bind ordinary review-app candidates to the diagnostic's additional split evidence.

    No model predictions, agent assertions or human judgments are created here.
    Source metadata stays hidden initially in the existing local review interface.
    """
    root, output_dir = root.resolve(), output_dir.resolve()
    if not output_dir.is_relative_to(root / "data/research_candidates/bgc"):
        raise ValueError("Blind review files must remain in ignored BGC candidate storage")
    if output_dir.exists():
        raise FileExistsError("Existing review evidence is preserved; use a fresh directory")
    examples_path = examples_path.resolve()
    if not examples_path.is_relative_to(root / "outputs"):
        raise ValueError("Review inputs must be a saved local diagnostic cohort")
    config_ref = reference(root, config_path)
    examples_ref = reference(root, examples_path)
    config, examples = read_json(config_path), read_json(examples_path)
    if examples.get("config_reference") != config_ref:
        raise ValueError("Review examples belong to a different diagnostic protocol")
    review_config = config.get("review", {})
    if (
        review_config.get("records") != REVIEW_COUNT
        or review_config.get("role") != "train"
        or review_config.get("selection_salt") != REVIEW_SALT
        or review_config.get("mode") != "blind_to_ranker_outputs"
        or review_config.get("human_review") is not None
        or review_config.get("agent_review") is not None
    ):
        raise ValueError("Unsupported fixed blind-review protocol")
    fields = read_json(checked(root, config["field_manifest"]))
    partitions = read_json(checked(root, config["partition_manifest"]))
    if partitions["inputs"]["fields"] != config["field_manifest"]:
        raise ValueError("Review field and partition sources disagree")
    bindings = {
        "protocol": config_ref,
        "examples": examples_ref,
        "field_manifest": config["field_manifest"],
        "partition_manifest": config["partition_manifest"],
        "components": partitions["components"],
        **{
            name: fields[name] for name in ("archive", "mapping", "assignments", "field_references")
        },
    }
    for item in bindings.values():
        checked(root, item)
    for name in ("archive", "assignments", "field_references"):
        if fields[name] != partitions["inputs"][name]:
            raise ValueError("Review source dependencies differ from partition overlay")
    if fields["mapping"] != partitions["inputs"]["field_mapping"]:
        raise ValueError("Review field mapping differs from partition overlay")
    mapping = read_json(checked(root, bindings["mapping"]))
    selected = select_review_rows(examples["train"], examples["dev"])
    by_id = {row["record_id"]: row for row in selected}
    group_counts: Counter = Counter()
    assignments = {}
    for assignment in jsonl_rows(checked(root, bindings["assignments"])):
        group_counts[assignment["group_id"]] += 1
        if assignment["record_id"] in by_id:
            assignments[assignment["record_id"]] = assignment
    touched = group_overrides(jsonl_rows(checked(root, bindings["components"])))
    candidates = {
        candidate["record_id"]: candidate
        for candidate in jsonl_rows(checked(root, bindings["field_references"]))
        if candidate["record_id"] in by_id
    }
    if set(candidates) != set(by_id) or set(assignments) != set(by_id):
        raise ValueError("Review identities are missing from the pinned source references")
    records, worksheet = [], []
    for row in selected:
        rid = row["record_id"]
        candidate, assignment = candidates[rid], assignments[rid]
        group = row["group_id"]
        if (
            candidate["group"]["group_id"] != group
            or assignment["group_id"] != group
            or group in touched
            or group_counts[group] != 1
            or candidate["group"]["review_required"]
            or assignment["review_required"]
            or candidate["group"]["proposed_split"] != "train"
            or assignment["proposed_split"] != "train"
            or assignment["source_split"] != "train"
            or candidate["source"]["source_split"] != "train"
            or row["input_sha256"] != candidate["input_sha256"]
            or input_hash(row["input"]) != candidate["input_sha256"]
            or row["source"] != candidate["source"]
            or row["fields"] != candidate["fields"]
            or row["source_labels"] != candidate["source_labels"]
        ):
            raise ValueError("Review candidate violates source, input or training-group evidence")
        review = {
            "record_id": rid,
            "input_sha256": row["input_sha256"],
            "source_url": row["source"]["provider_url"],
            "selection": ["fixed_hash_training_cohort_blind_to_ranker_outputs"],
            "agent_review": None,
            "human_review": None,
        }
        records.append(review)
        worksheet.append(
            {
                "review": review,
                "input": row["input"],
                "source": row["source"],
                "source_labels": row["source_labels"],
                "source_fields": row["fields"],
                "group": candidate["group"],
            }
        )
    packet = {
        "schema_version": 1,
        "status": "candidate_field_reviews_not_admitted",
        "training_authorized": False,
        "human_gold": False,
        "bindings": bindings,
        "contract": dict(CONTRACT),
        "selection": "32 fixed-hash singleton training groups; source and effective roles agree; no predictions shown or used for selection.",
        "records": records,
    }

    def save(name, value):
        path = output_dir / name
        create_or_verify(path, [json_bytes(value)])
        return reference(root, path)

    packet_ref = save("review_packet.json", packet)
    worksheet_ref = save("review_worksheet.json", worksheet)
    manifest = {
        **{
            key: packet[key]
            for key in ("schema_version", "status", "training_authorized", "human_gold")
        },
        "review_packet": packet_ref,
        "local_worksheet": worksheet_ref,
        "bindings": bindings,
        "selection": packet["selection"],
        "records": len(records),
        "observed_weak_label_counts": {
            facet: sum(bool(row["fields"][facet]["positive"]) for row in selected)
            for facet in mapping["facets"]
        },
        "new_human_labels": 0,
        "source_negative_count": 0,
        "record_ids": [row["record_id"] for row in selected],
        "group_ids": [row["group_id"] for row in selected],
        "effective_role": "train",
        "formal_study_admitted": False,
        "import_boundary": "Use research.py review import with this manifest. Imports remain candidates. Source-conflicting decisions need separate adjudication and are rejected by the existing importer; preserve the exported draft.",
    }
    manifest_ref = save("review_manifest.json", manifest)
    page_path = output_dir / "index.html"
    build(root, root / manifest_ref["path"], page_path)
    for item in bindings.values():
        checked(root, item)
    return {
        "manifest": manifest_ref,
        "packet": packet_ref,
        "worksheet": worksheet_ref,
        "page": reference(root, page_path),
        "records": len(records),
        "effective_role": "train",
        "new_human_labels": 0,
        "blind_to_ranker_outputs": True,
    }
