"""Synthetic source-bound, blind training worksheets; no real book text is opened."""

import json
from copy import deepcopy

import pytest

from src.research.book_fields import FACETS, input_hash
from src.research.builders.book_fields import reference
from src.research.candidate_io import json_bytes, sha
from src.research.field_baseline_review import build_review, select_review_rows
from src.research.review_app import _load


def row(number, role="train"):
    payload = {"title": f"Synthetic title {number}", "description": f"Synthetic text {number}."}
    return {
        "record_id": f"bgc:{role}:{number}",
        "group_id": "bgc-group:" + sha(f"{role}:{number}"),
        "source_split": role,
        "effective_split": role,
        "input": payload,
        "input_sha256": input_hash(payload),
        "source": {
            "source_split": role,
            "source_row": number,
            "provider_url": f"https://example.test/{number}",
        },
        "source_labels": [[0, "Alpha"]],
        "fields": {facet: {"positive": ["alpha"], "negative": []} for facet in FACETS},
    }


def test_review_selection_is_blind_and_rejects_split_or_group_leakage():
    train = [row(i) for i in range(40)]
    dev = [row(100, "dev")]
    selected = select_review_rows(train, dev)
    assert len(selected) == 32
    assert selected == select_review_rows(list(reversed(train)), dev)
    changed = deepcopy(train)
    for item in changed:
        item["fields"] = {facet: {"positive": [], "negative": []} for facet in FACETS}
        item["model_score"] = 999
    assert [r["record_id"] for r in select_review_rows(changed, dev)] == [
        r["record_id"] for r in selected
    ]
    for key in ("source_split", "effective_split"):
        bad = deepcopy(train)
        bad[0][key] = "test"
        with pytest.raises(ValueError, match="training roles"):
            select_review_rows(bad, dev)
    with pytest.raises(ValueError, match="disjoint"):
        select_review_rows(train, [{**dev[0], "group_id": train[0]["group_id"]}])
    with pytest.raises(ValueError, match="Repeated"):
        select_review_rows(train + [train[0]], dev)


@pytest.fixture
def review_source(tmp_path):
    def write(path, value, lines=False):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            b"".join((json.dumps(r) + "\n").encode() for r in value) if lines else json_bytes(value)
        )
        return reference(tmp_path, target)

    train, dev = [row(i) for i in range(1, 41)], [row(101, "dev")]
    mapping = {
        "facets": {
            facet: {"labels": ["alpha", "beta"], "definition": "Synthetic facet"}
            for facet in FACETS
        }
    }
    candidates, assignments = [], []
    for item in train + dev:
        group = {
            "group_id": item["group_id"],
            "proposed_split": item["effective_split"],
            "review_required": False,
        }
        candidates.append(
            {
                **{
                    key: item[key]
                    for key in ("record_id", "input_sha256", "source", "source_labels", "fields")
                },
                "group": group,
            }
        )
        assignments.append(
            {
                "record_id": item["record_id"],
                "source_split": item["source_split"],
                "source_row": item["source"]["source_row"],
                **group,
            }
        )
    fields = {
        "archive": write("data/source-receipt.json", {"synthetic": True}),
        "mapping": write("research/mapping.json", mapping),
        "assignments": write("data/assignments.jsonl", assignments, True),
        "field_references": write("data/fields.jsonl", candidates, True),
    }
    fields_ref = write("research/fields.json", fields)
    partitions = {
        "components": write("data/components.jsonl", [], True),
        "inputs": {
            "fields": fields_ref,
            "field_mapping": fields["mapping"],
            **{key: fields[key] for key in ("archive", "assignments", "field_references")},
        },
    }
    config = {
        "field_manifest": fields_ref,
        "partition_manifest": write("research/partitions.json", partitions),
        "review": {
            "records": 32,
            "role": "train",
            "selection_salt": "bgc-field-review-v1",
            "mode": "blind_to_ranker_outputs",
            "human_review": None,
            "agent_review": None,
        },
    }
    config_ref = write("configs/diagnostic.json", config)
    examples_path = "outputs/run/examples.json"
    examples = {"config_reference": config_ref, "train": train, "dev": dev}
    write(examples_path, examples)
    return tmp_path / config_ref["path"], tmp_path / examples_path, examples, fields


def test_review_roundtrips_through_existing_app_with_blank_human_slots(tmp_path, review_source):
    config, examples, _, _ = review_source
    output = tmp_path / "data/research_candidates/bgc/review"
    result = build_review(tmp_path, config, examples, output)
    assert result["records"] == 32 and result["new_human_labels"] == 0
    loaded = _load(tmp_path, tmp_path / result["manifest"]["path"])
    assert len(loaded["rows"]) == 32
    assert all(
        r["human_review"] is None and r["agent_review"] is None for r in loaded["packet"]["records"]
    )
    assert "partition_manifest" in loaded["bindings"]["sources"]
    assert "components" in loaded["bindings"]["sources"]
    assert (output / "index.html").is_file()
    with pytest.raises(FileExistsError):
        build_review(tmp_path, config, examples, output)


def test_self_rehashed_fabricated_text_cannot_enter_review(tmp_path, review_source):
    config, path, examples, _ = review_source
    selected = select_review_rows(examples["train"], examples["dev"])[0]
    selected["input"]["description"] = "Forged source description"
    selected["input_sha256"] = input_hash(selected["input"])
    path.write_bytes(json_bytes(examples))
    with pytest.raises(ValueError, match="source, input"):
        build_review(tmp_path, config, path, tmp_path / "data/research_candidates/bgc/review")


def test_changed_source_receipt_and_mismatched_protocol_fail(tmp_path, review_source):
    config, path, examples, fields = review_source
    examples["config_reference"]["sha256"] = "0" * 64
    path.write_bytes(json_bytes(examples))
    with pytest.raises(ValueError, match="different diagnostic protocol"):
        build_review(tmp_path, config, path, tmp_path / "data/research_candidates/bgc/review")
    examples["config_reference"] = reference(tmp_path, config)
    path.write_bytes(json_bytes(examples))
    (tmp_path / fields["field_references"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        build_review(tmp_path, config, path, tmp_path / "data/research_candidates/bgc/review")
