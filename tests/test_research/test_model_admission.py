"""Synthetic metadata only: consistency is not human review or a research run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.research.admission import validate_model_admission


def digest(value):
    return hashlib.sha256(value).hexdigest()


def payload_hash(value):
    return digest(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    )


class Bundle:
    """All asserted observations below are labeled unit fixtures in temporary files."""

    def __init__(self, root):
        self.root = root
        self.counter = 0
        runtime = {
            "transformers_version": "fixture-1",
            "torch_version": "fixture-1",
            "peft_version": "fixture-1",
            "environment_sha256": "e" * 64,
            "implementation_commit": "c" * 40,
        }
        self.plan = {
            "study_id": "unit-fixture-only",
            "model_study": {
                "backbone": {
                    "selected_repo": "fixture/model",
                    "selected_revision": "a" * 40,
                    "selected_runtime": runtime,
                },
                "adaptation_family": "lora",
                "shared_scope": "encoder_attention_effective_deltas",
                "merge_space": "effective_weight_deltas",
                "materialization": "into_same_base_weights_without_rank_recompression",
                "pretrained_base_updates": False,
                "embedding_and_layernorm_updates": False,
                "head_initialization": "identical task-specific initialization within each matched training seed",
                "merge_private_policy": "retain_corresponding_specialist_private_modules_without_updates",
                "post_merge_training": False,
                "private_scope": {
                    "emotion": "pooler_and_classifier",
                    "topic": "pooler_and_classifier",
                },
                "tasks": [{"task_id": "emotion"}, {"task_id": "topic"}],
                "training_seeds": {"values": [17, 42]},
                "arms": [
                    {"arm_id": "joint", "role": "joint"},
                    {"arm_id": "specialists", "role": "specialists"},
                ],
                "budget": {
                    "status": "frozen",
                    "primary_unit": "synchronized_training_wall_seconds",
                    "B": 100.0,
                    "hardware_id": "fixture-cpu",
                    "timing_boundary_id": "fixture-window",
                    "equal_selection_access": True,
                    "count_all_expert_training": True,
                    "report_total_development_cost": True,
                    "tokens_are_auxiliary_not_compute_equivalence": True,
                    "specialist_allocation": {"emotion": 0.5, "topic": 0.5},
                    "overshoot_rule": {
                        "name": "finish_current_optimizer_window",
                        "max_fraction": 0.05,
                    },
                    "selection_allowance": {
                        "unit": "synchronized_selection_wall_seconds",
                        "per_recipe_seconds": 2.0,
                        "max_trials_per_recipe": 3,
                        "data_access": "shared_development_partitions",
                    },
                },
            },
        }
        self.model = self.plan["model_study"]
        self.binding = {
            "study_id": self.plan["study_id"],
            "backbone_repo": "fixture/model",
            "backbone_revision": "a" * 40,
            "runtime": runtime,
        }
        self.support = self.write_bytes(
            b"UNIT FIXTURE ONLY. No actual observation, rights review or run."
        )
        self.task_entries = []
        for task in ("emotion", "topic"):
            label = self.artifact(
                "task_label_contract",
                task_id=task,
                output_type="multi_label" if task == "emotion" else "single_label",
                label_order=["fixture_a", "fixture_b"],
                tokenizer={"repo": "fixture/tokenizer", "revision": "d" * 40},
            )
            splits = {
                name: self.write_bytes(
                    json.dumps({"id": task + name, "text": "repeated fixture text"}).encode(),
                    ".jsonl",
                )
                for name in ("train", "model_selection", "calibration", "test")
            }
            split_hashes = {name: ref["sha256"] for name, ref in splits.items()}
            overlap = self.artifact(
                "dataset_overlap_report",
                task_id=task,
                split_sha256=split_hashes,
                cross_split_normalized_text_groups=1,
            )
            groups = self.artifact(
                "dataset_group_assignment",
                task_id=task,
                split_sha256=split_hashes,
                grouping_policy={
                    "unit": "provider_document_id",
                    "generalization_claim": "new_source_ids",
                    "text_overlap_policy": "preserve_official_splits_report_overlap",
                },
                overlap_report=self.write(overlap),
                group_ids_by_split={split: [task + "-" + split] for split in splits},
            )
            self.task_entries.append(
                {
                    "task_id": task,
                    "label_contract": self.write(label),
                    "splits": splits,
                    "group_assignment": self.write(groups),
                }
            )
        self.data_payload = {"tasks": self.task_entries}
        self.seal_dataset()
        shared_module = "encoder.layer.0.self_attn.q"
        shared = [
            shared_module + ".lora_A.default.weight",
            shared_module + ".lora_B.default.weight",
        ]
        private = {
            task: ["heads." + task + ".bias", "heads." + task + ".weight"]
            for task in ("emotion", "topic")
        }
        frozen = [
            "decoder.embed_tokens.weight",
            "encoder.embed_tokens.weight",
            shared_module + ".weight",
        ]
        inventory = {}
        for name in shared + frozen + [name for values in private.values() for name in values]:
            origin = "base" if name in frozen else "lora" if name in shared else "private_head"
            identity = "embedding" if "embed_tokens" in name else name
            inventory[name] = {
                "shape": [2] if name.endswith("bias") else [2, 2],
                "origin": origin,
                "initial_sha256": digest((identity + ":17").encode()),
            }
        initials = {}
        for seed in (17, 42):
            states = {}
            for task, parameters in private.items():
                record = self.artifact(
                    "private_initialization_manifest",
                    seed=seed,
                    task_id=task,
                    parameter_initial_sha256={
                        name: digest((name + ":" + str(seed)).encode()) for name in parameters
                    },
                )
                states[task] = self.write(record)
            initials[str(seed)] = {arm: copy.deepcopy(states) for arm in ("joint", "specialists")}
        self.module_payload = {
            "dataset_admission_sha256": self.model["selected_dataset_manifest"]["sha256"],
            "label_contract_sha256": {
                task["task_id"]: task["label_contract"]["sha256"] for task in self.task_entries
            },
            "private_scope": self.model["private_scope"],
            "inspection_seed": 17,
            "lora_config": {
                "r": 2,
                "alpha": 2.0,
                "dropout": 0.0,
                "bias": "none",
                "use_rslora": False,
                "use_dora": False,
            },
            "module_allowlist": {
                "shared": [shared_module],
                "private": {task: ["heads." + task] for task in private},
            },
            "shared_parameters": sorted(shared),
            "private_parameters": private,
            "frozen_parameters": sorted(frozen),
            "parameter_inventory": inventory,
            "tied_parameter_groups": [
                ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]
            ],
            "private_initialization_files": initials,
        }
        self.seal_modules()
        self.seal_feasibility()

    def write_bytes(self, content, suffix=".md"):
        self.counter += 1
        path = self.root / f"fixture-{self.counter}{suffix}"
        path.write_bytes(content)
        return {"path": path.name, "sha256": digest(content), "bytes": len(content)}

    def write(self, record):
        return self.write_bytes(json.dumps(record, sort_keys=True).encode(), ".json")

    def read(self, ref):
        return json.loads((self.root / ref["path"]).read_text())

    def artifact(self, kind, **fields):
        return {
            "schema_version": 1,
            "kind": kind,
            "status": "reviewed",
            "binding": copy.deepcopy(self.binding),
            **fields,
        }

    def receipt(self, kind, payload, checks):
        return self.write(
            self.artifact(
                kind,
                payload_sha256=payload_hash(payload),
                outcome="passed",
                recorded_by="unit-fixture-only",
                checks={name: True for name in checks},
                evidence_files=[self.support],
            )
        )

    def seal_dataset(self):
        receipt = self.receipt(
            "dataset_admission_receipt",
            self.data_payload,
            [
                "source_use_reviewed",
                "group_disjointness_verified",
                "labels_verified",
                "split_files_verified",
                "no_quarantined_inputs",
                "grouping_policy_reviewed",
            ],
        )
        self.model["selected_dataset_manifest"] = self.write(
            self.artifact("dataset_admission", payload=self.data_payload, receipt=receipt)
        )

    def seal_modules(self):
        receipt = self.receipt(
            "adapter_structure_receipt",
            self.module_payload,
            [
                "complete_parameter_inventory",
                "encoder_only_shared_scope",
                "private_task_scope",
                "frozen_base_and_aliases",
                "matched_head_initialization",
                "save_reload_verified",
            ],
        )
        self.model["adapter_module_manifest"] = self.write(
            self.artifact("adapter_module_manifest", payload=self.module_payload, receipt=receipt)
        )

    def seal_feasibility(self):
        budget = self.model["budget"]
        self.feasibility_payload = {
            "dataset_admission_sha256": self.model["selected_dataset_manifest"]["sha256"],
            "adapter_module_manifest_sha256": self.model["adapter_module_manifest"]["sha256"],
            "label_contract_sha256": self.module_payload["label_contract_sha256"],
            "budget_contract": {
                key: budget[key]
                for key in (
                    "B",
                    "hardware_id",
                    "timing_boundary_id",
                    "primary_unit",
                    "specialist_allocation",
                    "overshoot_rule",
                    "selection_allowance",
                )
            },
            "measurements": [
                dict(
                    task_id=task,
                    hardware_id="fixture-cpu",
                    timing_boundary_id="fixture-window",
                    training_window_seconds=1.5,
                    optimizer_windows=1,
                    source_length=8,
                    target_length=0,
                    batch_size=2,
                    raw_receipt=self.support,
                )
                for task in ("emotion", "topic")
            ],
        }
        self.store_feasibility()

    def store_feasibility(self):
        receipt = self.receipt(
            "feasibility_review_receipt",
            self.feasibility_payload,
            [
                "runtime_and_tokenizer_verified",
                "timing_boundary_verified",
                "hardware_feasible",
                "allocation_and_overshoot_reviewed",
                "selection_allowance_reviewed",
            ],
        )
        self.model["budget"]["feasibility_evidence"] = self.write(
            self.artifact(
                "feasibility_timing_evidence", payload=self.feasibility_payload, receipt=receipt
            )
        )

    def blockers(self):
        return validate_model_admission(self.root, self.plan)


@pytest.fixture
def bundle(tmp_path):
    return Bundle(tmp_path)


def test_complete_synthetic_contract_is_consistent_and_read_only(bundle):
    before = {p: p.read_bytes() for p in bundle.root.iterdir()}
    assert bundle.blockers() == []
    assert {p: p.read_bytes() for p in bundle.root.iterdir()} == before
    assert "training_authorized" not in bundle.plan  # The helper cannot grant permission.


def test_actual_draft_has_clear_blockers_without_needing_data_or_model_files():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "configs/research/study_design.json").read_text())
    errors = validate_model_admission(root, plan)
    assert errors
    assert any("Missing pinned dataset admission" in e for e in errors)
    assert any("Missing pinned feasibility" in e for e in errors)


@pytest.mark.parametrize("value", [{}, "", False, "old-report.json"])
def test_nonnull_placeholders_do_not_admit_evidence(bundle, value):
    bundle.model["selected_dataset_manifest"] = value
    assert bundle.blockers()


def test_historical_metric_json_cannot_be_repurposed_as_admission(bundle):
    bundle.model["selected_dataset_manifest"] = bundle.write({"summarization": {"rougeL": 0.5}})
    assert any("schema_version" in e or "Expected reviewed" in e for e in bundle.blockers())


def test_validly_rehashed_evidence_for_another_runtime_is_rejected(bundle):
    record = bundle.read(bundle.model["selected_dataset_manifest"])
    record["binding"]["runtime"]["environment_sha256"] = "f" * 64
    bundle.model["selected_dataset_manifest"] = bundle.write(record)
    assert any("does not bind this study" in e for e in bundle.blockers())


@pytest.mark.parametrize(
    "allocation",
    [
        {},
        {"emotion": 1.0},
        {"emotion": 0.8, "topic": 0.8},
        {"emotion": True, "topic": 0.0},
        {"emotion": float("nan"), "topic": 0.5},
    ],
)
def test_specialist_allocations_require_exact_tasks_finite_fractions_and_total_one(
    bundle, allocation
):
    bundle.model["budget"]["specialist_allocation"] = allocation
    assert bundle.blockers()


@pytest.mark.parametrize(
    "key,value",
    [
        ("B", 0),
        ("B", float("inf")),
        ("overshoot_rule", ""),
        ("overshoot_rule", {"name": "whatever", "max_fraction": 0.1}),
        ("selection_allowance", {}),
        ("hardware_id", "unknown"),
    ],
)
def test_budget_placeholders_and_invalid_values_are_rejected(bundle, key, value):
    bundle.model["budget"][key] = value
    assert bundle.blockers()


def test_raw_supporting_receipt_tampering_is_detected(bundle):
    (bundle.root / bundle.support["path"]).write_bytes(b"rewritten")
    assert any("changed" in e for e in bundle.blockers())


def test_rehashed_adapter_for_different_data_is_rejected(bundle):
    bundle.module_payload["dataset_admission_sha256"] = "f" * 64
    bundle.seal_modules()
    assert any("admitted dataset and label" in e for e in bundle.blockers())


def test_base_alias_cannot_be_declared_trainable(bundle):
    name = "encoder.embed_tokens.weight"
    bundle.module_payload["frozen_parameters"].remove(name)
    bundle.module_payload["shared_parameters"] = sorted(
        bundle.module_payload["shared_parameters"] + [name]
    )
    bundle.seal_modules()
    assert any("base parameters must stay frozen" in e for e in bundle.blockers())


def test_shared_private_parameter_intersection_is_rejected(bundle):
    bundle.module_payload["private_parameters"]["topic"].append(
        bundle.module_payload["shared_parameters"][0]
    )
    bundle.module_payload["private_parameters"]["topic"].sort()
    bundle.seal_modules()
    assert any("parameter partitions must be disjoint" in e for e in bundle.blockers())


def test_ordered_allowlist_is_required(bundle):
    bundle.module_payload["shared_parameters"].reverse()
    bundle.seal_modules()
    assert any("lexicographic" in e for e in bundle.blockers())


def test_head_initialization_differs_between_matched_arms(bundle):
    reference = bundle.module_payload["private_initialization_files"]["42"]["specialists"][
        "emotion"
    ]
    record = bundle.read(reference)
    name = next(iter(record["parameter_initial_sha256"]))
    record["parameter_initial_sha256"][name] = "f" * 64
    bundle.module_payload["private_initialization_files"]["42"]["specialists"]["emotion"] = (
        bundle.write(record)
    )
    bundle.seal_modules()
    assert any("differ across matched" in e for e in bundle.blockers())


def test_preserved_text_overlap_does_not_imply_group_leakage(bundle):
    assert bundle.blockers() == []  # Fixture text repeats across distinct provider IDs.
    entry = bundle.task_entries[0]
    group = bundle.read(entry["group_assignment"])
    group["group_ids_by_split"]["test"] = group["group_ids_by_split"]["train"]
    entry["group_assignment"] = bundle.write(group)
    bundle.seal_dataset()
    assert any("groups overlap" in e for e in bundle.blockers())


def test_unseen_text_claim_cannot_keep_reported_text_overlap(bundle):
    entry = bundle.task_entries[0]
    group = bundle.read(entry["group_assignment"])
    group["grouping_policy"]["generalization_claim"] = "unseen_normalized_text"
    entry["group_assignment"] = bundle.write(group)
    bundle.seal_dataset()
    assert any("contradicts" in e for e in bundle.blockers())


def test_feasibility_must_cover_each_task_and_exact_budget(bundle):
    bundle.feasibility_payload["measurements"] = bundle.feasibility_payload["measurements"][:1]
    bundle.store_feasibility()
    assert any("cover every admitted task" in e for e in bundle.blockers())
    bundle.seal_feasibility()
    bundle.feasibility_payload["budget_contract"]["B"] = 101
    bundle.store_feasibility()
    assert any("exact data, adapters, labels and budget" in e for e in bundle.blockers())


def test_freeze_and_initialization_declarations_are_not_optional(bundle):
    bundle.model["embedding_and_layernorm_updates"] = True
    bundle.model["head_initialization"] = None
    errors = bundle.blockers()
    assert any("explicitly be false" in e for e in errors)
    assert any("initialization" in e for e in errors)


def test_outside_repository_reference_is_rejected(bundle):
    bundle.model["selected_dataset_manifest"]["path"] = "../outside.json"
    assert any("inside the repository" in e for e in bundle.blockers())


def test_feasibility_budget_boolean_cannot_impersonate_numeric_allowance(bundle):
    bundle.model["budget"]["B"] = 1
    bundle.seal_feasibility()
    bundle.feasibility_payload["budget_contract"]["B"] = True
    bundle.store_feasibility()
    assert any("exact data, adapters, labels and budget" in e for e in bundle.blockers())


def test_malformed_seed_declaration_returns_blockers(bundle):
    bundle.model["training_seeds"] = []
    assert any("training_seeds must be an object" in e for e in bundle.blockers())
