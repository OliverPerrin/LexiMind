"""Synthetic source receipts for the local continuation pilot; no model execution."""

from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.research.builders.licensed_books import _rpt_window
from src.research.candidate_io import json_bytes, sha
from src.research.io import read_json
from src.training.pilot import PILOT_WORKS, prepare_pilot_data


@pytest.mark.parametrize(
    "mutation",
    [
        {"sft_steps": 257},
        {"rl_steps": -1},
        {"mps_memory_fraction": 0.9},
        {"device": "cuda"},
        {"minimum_rewarded_content_tokens": 1},
    ],
)
def test_pilot_rejects_expansion_outside_bounded_contract(mutation):
    from src.training.pilot import validate_pilot_config

    root = Path(__file__).resolve().parents[2]
    config = read_json(root / "configs/research/macbook_pilot.json")
    validate_pilot_config(config)
    with pytest.raises(ValueError):
        validate_pilot_config({**config, **mutation})


def test_prefix_reward_rejects_undefined_and_empty_token_stuffing():
    from types import SimpleNamespace

    import torch

    from src.training.pilot import continuation_rewards

    root = Path(__file__).resolve().parents[2]
    tokenizer = Tokenizer.from_file(str(root / "artifacts/hf_tokenizer/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids = tokenizer.encode("The old green forest", add_special_tokens=False).ids
    text = tokenizer.decode(ids)
    prefixes = [tokenizer.decode(ids[:i], skip_special_tokens=False) for i in range(len(ids) + 1)]
    row = {
        "continuation_target": {
            "byte_encoding": "utf-8",
            "observed_bytes": text,
            "token_bytes": [b[len(a) :] for a, b in zip(prefixes[:-1], prefixes[1:], strict=True)],
            "tokenizer_sha256": "a" * 64,
            "source_evidence_sha256": "b" * 64,
            "work_group": "synthetic",
            "split": "train",
        }
    }

    def reward(tokens):
        generated = SimpleNamespace(
            response_ids=torch.tensor([tokens]),
            response_mask=torch.ones((1, len(tokens)), dtype=torch.bool),
        )
        return continuation_rewards([row], generated, tokenizer, minimum_tokens=4)[0]

    assert reward(ids + [1])["reward"] == 1
    assert tokenizer.decode(ids + [32127]) == text
    assert reward(ids + [32127, 1])["reward"] == 0
    assert reward(ids[:1] + [3, 3, 3, 1])["reward"] == 0
    assert reward(ids + [2, 1])["reward"] == 0


def test_pilot_records_failure_and_preserves_existing_runs(tmp_path, monkeypatch):
    import src.training.pilot as pilot

    output = tmp_path / "outputs" / "new"

    def failure(*args, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text('{"status":"running"}')
        raise FloatingPointError("synthetic nonfinite loss")

    monkeypatch.setattr(pilot, "_run_pilot", failure)
    with pytest.raises(FloatingPointError):
        pilot.run_pilot(tmp_path, tmp_path / "config.json", output)
    report = read_json(output / "report.json")
    assert report["status"] == "stopped" and "FloatingPointError" in report["error"]
    saved = (output / "report.json").read_bytes()

    def reject(*args, **kwargs):
        raise FileExistsError("Existing run")

    monkeypatch.setattr(pilot, "_run_pilot", reject)
    with pytest.raises(FileExistsError):
        pilot.run_pilot(tmp_path, tmp_path / "config.json", output)
    assert (output / "report.json").read_bytes() == saved


def write(root, relative, value, *, raw=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else json_bytes(value)
    path.write_bytes(data)
    return {"path": relative, "bytes": len(data), "sha256": sha(data)}


@pytest.fixture
def pilot_root(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    tokenizer_bytes = (repo / "artifacts/hf_tokenizer/tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_ref = write(tmp_path, "tokenizer.json", tokenizer_bytes, raw=True)
    text = (
        "The children walked slowly through the old green forest and listened to the birds in the tall trees. "
        * 6
    )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    window = _rpt_window(tokenizer, ids, 8, {0, 1, 2})
    assert window is not None
    inventory = write(tmp_path, "sources.json", {"fixture": "synthetic"})
    works, components = {}, []
    for work_id in PILOT_WORKS:
        work = {
            "work_id": work_id,
            "source_sha256": sha(text),
            "title": "Synthetic source",
            "creators": ["Synthetic creator"],
            "source_page": "https://example.com/synthetic-fixture",
            "license": {"id": "synthetic-only"},
            "sections": [
                {"section_id": str(i), "text": text, "source_lines": [1, 10]} for i in range(8)
            ],
        }
        works[work_id] = write(tmp_path, f"works/{work_id}.json", work)
        components.append(
            {
                "component_id": "group:" + work_id,
                "proposed_split": "train",
                "members": ["licensed_text:" + work_id],
            }
        )
    component_ref = write(
        tmp_path,
        "components.jsonl",
        b"".join(json_bytes(row).replace(b"\n", b"") + b"\n" for row in components),
        raw=True,
    )
    licensed = write(tmp_path, "licensed.json", {"source_inventory": inventory})
    partition = write(
        tmp_path,
        "partitions.json",
        {"inputs": {"licensed_manifest": licensed}, "components": component_ref},
    )
    records = []
    for work_id, artifact in works.items():
        for i in range(8):
            evidence = {
                "artifact": artifact,
                "source_sha256": sha(text),
                "section_id": str(i),
                "source_lines": [1, 10],
                "source_section_sha256": sha(text),
                "partition_manifest_sha256": partition["sha256"],
                "prompt_token_span": window["prompt_token_span"],
                "target_token_span": window["target_token_span"],
            }
            records.append(
                {
                    "record_id": work_id + str(i),
                    "work_id": work_id,
                    "source_evidence": evidence,
                    "attribution": {
                        key: work[key] for key in ("title", "creators", "source_page", "license")
                    },
                    "prompt_ids": window["prompt_ids"],
                    "target_ids": window["target_ids"],
                    "prompt_normalized_text": window["prompt_normalized_text"],
                    "continuation_target": {
                        "split": "train",
                        "work_group": "group:" + work_id,
                        "tokenizer_sha256": tokenizer_ref["sha256"],
                        "source_evidence_sha256": sha(json_bytes(evidence)),
                        "observed_bytes": window["observed_bytes"],
                        "token_bytes": window["token_bytes"],
                    },
                }
            )
    # Held-out test content is deliberately incompatible with the training schema:
    # the loader must ignore it before consulting any content or source fields.
    records.append({"continuation_target": {"split": "test"}, "must_not_be_returned": True})
    builder = write(tmp_path, "builder.py", b"# synthetic builder\n", raw=True)
    manifest = {
        "configuration": {"policy": "licensed-book-normalized-continuation-v1"},
        "inputs": {
            "licensed_manifest": licensed,
            "partition_manifest": partition,
            "tokenizer": tokenizer_ref,
            "components": component_ref,
            "source_inventory": inventory,
            **{"work:" + key: val for key, val in works.items()},
        },
        "implementation_sha256": {builder["path"]: builder["sha256"]},
    }
    repin(tmp_path, manifest, records)
    return tmp_path, manifest, records


def repin(root, manifest, records):
    manifest["continuations"] = write(
        root,
        "rows.jsonl",
        b"".join(json_bytes(row).replace(b"\n", b"") + b"\n" for row in records),
        raw=True,
    )
    write(root, "research/preparation/rpt_candidate_manifest.json", manifest)


def test_pilot_uses_only_train_works_and_preserves_global_roles(pilot_root):
    root, _, _ = pilot_root
    result = prepare_pilot_data(root)
    assert len(result["train"]) == 16 and len(result["diagnostic"]) == 8
    assert {row["work_id"] for row in result["diagnostic"]} == {"bookdash-sizwes-smile"}
    for role in ("train", "diagnostic"):
        for row in result[role]:
            assert row["source_split"] == row["continuation_target"]["split"] == "train"
            assert row["pilot_role"] == role
            assert row["labels"] == row["target_ids"] + [1]
            assert row["input_ids"][-1] == 1 and len(row["input_ids"]) <= 160
    assert not result["provenance"]["global_test_used"]
    assert not result["provenance"]["study_admission"]


def test_pilot_rejects_stale_source_receipts(pilot_root):
    root, _, _ = pilot_root
    (root / "sources.json").write_text("{}")
    with pytest.raises(ValueError, match="Pilot source changed"):
        prepare_pilot_data(root)


def test_pilot_rejects_forged_target_after_outer_rehash(pilot_root):
    root, manifest, records = pilot_root
    records[0]["target_ids"] = list(reversed(records[0]["target_ids"]))
    repin(root, manifest, records)
    with pytest.raises(ValueError, match="observed source tokens"):
        prepare_pilot_data(root)


def test_pilot_rejects_overlapping_diagnostic_windows(pilot_root):
    root, manifest, records = pilot_root
    diagnostic = [row for row in records if row.get("work_id") == "bookdash-sizwes-smile"]
    diagnostic[1]["source_evidence"] = deepcopy(diagnostic[0]["source_evidence"])
    diagnostic[1]["continuation_target"]["source_evidence_sha256"] = sha(
        json_bytes(diagnostic[1]["source_evidence"])
    )
    repin(root, manifest, records)
    with pytest.raises(ValueError, match="diagnostic windows overlap"):
        prepare_pilot_data(root)


def test_pilot_rejects_missing_whole_work(pilot_root):
    root, manifest, records = pilot_root
    records[:] = [row for row in records if row.get("work_id") != "bookdash-sizwes-smile"]
    repin(root, manifest, records)
    with pytest.raises(ValueError, match="exactly eight"):
        prepare_pilot_data(root)


def test_pilot_rejects_changed_builder(pilot_root):
    root, _, _ = pilot_root
    (root / "builder.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="implementation changed"):
        prepare_pilot_data(root)


def test_pilot_rejects_reassigned_source_work(pilot_root):
    root, manifest, records = pilot_root
    components = [read_json(root / "works" / f"{work}.json") for work in PILOT_WORKS]
    rows = [
        {
            "component_id": "group:" + row["work_id"],
            "proposed_split": "test",
            "members": ["licensed_text:" + row["work_id"]],
        }
        for row in components
    ]
    ref = write(
        root,
        "components.jsonl",
        b"".join(json_bytes(row).replace(b"\n", b"") + b"\n" for row in rows),
        raw=True,
    )
    manifest["inputs"]["components"] = ref
    partition = read_json(root / "partitions.json")
    partition["components"] = ref
    manifest["inputs"]["partition_manifest"] = write(root, "partitions.json", partition)
    repin(root, manifest, records)
    with pytest.raises(ValueError, match="effective train partition"):
        prepare_pilot_data(root)
