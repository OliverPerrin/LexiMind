"""Inspect research preparation without loading models, scoring systems or granting execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.research.admission import validate_model_admission
from src.research.io import check_file, read_json, safe_path
from src.research.ledger import validate_ledger
from src.research.manifest import ARTIFACTS


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def validate_study_design(plan: dict[str, Any]) -> None:
    """Validate declared comparison semantics, not scientific truth or readiness."""
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("Unsupported study schema")
    if plan.get("status") not in {"draft", "frozen"}:
        raise ValueError("Study status must be draft or frozen")
    if type(plan.get("training_authorized")) is not bool:
        raise ValueError("Study must explicitly declare its authorization status")
    model = _object(plan.get("model_study"), "model_study")
    if model.get("adaptation_family") != "lora":
        raise ValueError("The first comparison declares a common LoRA adaptation family")
    if (
        model.get("pretrained_base_updates") is not False
        or model.get("embedding_and_layernorm_updates") is not False
    ):
        raise ValueError(
            "The first comparison freezes pretrained base, embedding and layer-norm parameters during training"
        )
    if (
        model.get("head_initialization")
        != "identical task-specific initialization within each matched training seed"
    ):
        raise ValueError("Task-head initialization must be matched within each training seed")
    if model.get("access_regime") != "all_task_training_data_available":
        raise ValueError(
            "This comparison requires the common-data regime; existing-expert reuse is separate"
        )
    if model.get("merge_space") != "effective_weight_deltas":
        raise ValueError("Separate LoRA factor averaging is not effective-delta merging")
    if model.get("materialization") != "into_same_base_weights_without_rank_recompression":
        raise ValueError("A recompressed or routed model is a separate deployment comparison")
    if model.get("shared_scope") != "encoder_attention_effective_deltas":
        raise ValueError("The draft must declare its shared encoder interface")
    if (
        model.get("merge_private_policy")
        != "retain_corresponding_specialist_private_modules_without_updates"
    ):
        raise ValueError("Specify private-module retention; unrelated heads must not be averaged")
    if model.get("post_merge_training") is not False:
        raise ValueError("Post-merge training needs a separate budgeted arm")
    tasks = _rows(model.get("tasks"), "tasks")
    task_ids = [row.get("task_id") for row in tasks]
    if not task_ids or any(not isinstance(value, str) or not value for value in task_ids):
        raise ValueError("Tasks require nonempty string IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs must be unique")
    private = _object(model.get("private_scope"), "private_scope")
    if set(private) != set(task_ids):
        raise ValueError("Every candidate task must have an explicit private-module policy")
    if any(not isinstance(value, str) or not value for value in private.values()):
        raise ValueError("Private-module policies must be named explicitly")
    arms = _rows(model.get("arms"), "arms")
    if any(not isinstance(row.get("arm_id"), str) or not row["arm_id"] for row in arms):
        raise ValueError("Arms require nonempty string IDs")
    by_id = {arm["arm_id"]: arm for arm in arms}
    if len(by_id) != len(arms):
        raise ValueError("Arm IDs must be unique")
    if not any(arm.get("role") == "joint" for arm in arms) or not any(
        arm.get("role") == "specialists" for arm in arms
    ):
        raise ValueError("The comparison requires joint and specialist controls")
    for arm in arms:
        role = arm.get("role")
        if role in {"joint", "specialists"}:
            if arm.get("adaptation_family") != model.get("adaptation_family"):
                raise ValueError(
                    "Primary joint and specialist arms must share an adaptation family"
                )
            expected = "B_total" if role == "joint" else "B_total_across_tasks"
            if arm.get("training_allocation") != expected:
                raise ValueError("Budget is total across specialists, not B per specialist")
            if arm.get("selection_policy") != "same_predeclared_validation_allowance":
                raise ValueError("Training arms must declare comparable selection access")
        elif role == "merge":
            source_id = arm.get("reuses_arm")
            source = by_id.get(source_id) if isinstance(source_id, str) else None
            if source is None or source.get("role") != "specialists":
                raise ValueError("A merge must reuse the identified specialist control")
            if arm.get("include_expert_training_cost") is not True:
                raise ValueError("Merge-equivalent cost must include expert creation")
            if type(arm.get("deployed_encoders")) is not int or arm["deployed_encoders"] != 1:
                raise ValueError("The primary merge must deploy one encoder, not a routed ensemble")
            if arm.get("operator") not in {
                "weighted_effective_delta_sum",
                "ties_on_effective_encoder_deltas",
            }:
                raise ValueError("Merge operator is outside the first comparison")
        else:
            raise ValueError(f"Unknown arm role: {role}")
    budget = _object(model.get("budget"), "budget")
    if budget.get("primary_unit") != "synchronized_training_wall_seconds":
        raise ValueError("This proposed budget uses a timed training boundary, not token equality")
    for key in (
        "tokens_are_auxiliary_not_compute_equivalence",
        "equal_selection_access",
        "count_all_expert_training",
        "report_total_development_cost",
    ):
        if budget.get(key) is not True:
            raise ValueError(f"Budget safeguard missing: {key}")
    value = budget.get("B")
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
    ):
        raise ValueError("B must be unknown or a finite positive training allowance")
    applied = _object(plan.get("applied_study"), "applied_study")
    if applied.get("independent_from_model_study") is not True:
        raise ValueError("Book utility and benchmark-task retention are separate questions")
    gains = applied.get("gain_mapping")
    if (
        applied.get("ranking") != "all_eligible_catalogue_works"
        or gains != [0, 1, 3, 7]
        or any(type(value) is not int for value in gains)
    ):
        raise ValueError("The draft must state full-catalogue ranking and its graded gain mapping")
    if applied.get("mood_queries_enabled") is not False:
        raise ValueError("The current study has no admitted whole-work mood evidence")


def inspect_preparation(root: Path, manifest_path: Path, target: str) -> dict[str, Any]:
    """Check only common and requested-target evidence. Never authorize a run."""
    if target not in {"model_study", "book_study"}:
        raise ValueError("Target must be model_study or book_study")
    manifest = _object(read_json(safe_path(root, str(manifest_path))), "manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported preparation manifest schema")
    artifacts = _rows(manifest.get("artifacts"), "artifacts")
    errors: list[str] = []
    ids: set[str] = set()
    files: dict[str, Path] = {}
    required = {key for key, (scope, _) in ARTIFACTS.items() if scope in {"common", target}}
    for artifact in artifacts:
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in ids:
            errors.append("Preparation artifact IDs must be unique nonempty strings")
            continue
        ids.add(artifact_id)
        if artifact_id not in ARTIFACTS:
            errors.append(f"Unregistered preparation artifact: {artifact_id}")
            continue
        scope, expected_path = ARTIFACTS[artifact_id]
        if artifact.get("scope") != scope or artifact.get("path") != expected_path:
            errors.append(f"Preparation scope/path mismatch: {artifact_id}")
            continue
        if artifact_id not in required:
            continue
        errors.extend(check_file(root, artifact))
        files[artifact_id] = safe_path(root, expected_path)
    errors.extend(f"Missing preparation artifact: {key}" for key in sorted(required - files.keys()))
    blockers: list[str] = []
    observations: list[str] = []
    if not errors:
        try:
            plan = _object(read_json(files["study_design"]), "study design")
            validate_study_design(plan)
            status = _object(read_json(files["preparation_status"]), "preparation status")
            if type(status.get("schema_version")) is not int or status["schema_version"] != 1:
                raise ValueError("Unsupported preparation status schema")
            if status.get("paused_by_user") is not False:
                blockers.append("Training and research evaluation remain paused by the user")
            if plan.get("status") != "frozen":
                blockers.append("The prospective protocol is a draft, not frozen")
            study = plan["model_study"] if target == "model_study" else plan["applied_study"]
            if study.get("unresolved") != []:
                blockers.append("The target study still declares unresolved design decisions")
            if target == "model_study":
                if study.get("task_suite_status") != "frozen":
                    blockers.append("The model task suite is still a candidate proposal")
                if study["budget"].get("status") != "frozen":
                    blockers.append("The model budget policy is not frozen")
                inventory = _object(read_json(files["data_inventory"]), "data inventory")
                audit = _object(read_json(files["data_audit"]), "data audit")
                if (
                    inventory.get("audit_script_sha256")
                    != hashlib.sha256(files["data_auditor"].read_bytes()).hexdigest()
                ):
                    errors.append("Data inventory was produced by a different auditor revision")
                # Matches audit_research_data.py's documented inventory serialization.
                canonical = json.dumps(
                    inventory, sort_keys=True, ensure_ascii=False, allow_nan=False
                ).encode()
                if audit.get("inventory_sha256") != hashlib.sha256(canonical).hexdigest():
                    errors.append("Data audit does not bind the supplied inventory")
                backbone = _object(read_json(files["backbone_candidates"]), "backbone candidates")
                if (
                    backbone.get("metadata_source_sha256")
                    != hashlib.sha256(files["repository_metadata"].read_bytes()).hexdigest()
                ):
                    errors.append("Backbone review does not bind the supplied repository metadata")
                for prefix in ("goemotions", "ag_news"):
                    candidate = _object(
                        read_json(files[prefix + "_candidate"]), prefix + " candidate"
                    )
                    if (
                        candidate.get("preparation_script_sha256")
                        != hashlib.sha256(files[prefix + "_builder"].read_bytes()).hexdigest()
                    ):
                        errors.append(
                            f"{prefix} candidate was produced by a different preparation script"
                        )
                    helpers = _object(
                        candidate.get("preparation_helper_sha256"), "source helper hashes"
                    )
                    expected_helpers = {
                        ARTIFACTS[key][1]: files[key]
                        for key in ("candidate_io", "file_integrity_contract")
                    }
                    if set(helpers) != set(expected_helpers) or any(
                        helpers.get(path) != hashlib.sha256(local.read_bytes()).hexdigest()
                        for path, local in expected_helpers.items()
                    ):
                        errors.append(
                            f"{prefix} candidate does not bind current preparation helpers"
                        )
                    partition = _object(
                        read_json(files[prefix + "_partitions"]), prefix + " partitions"
                    )
                    expected_candidate_ref = {
                        "path": ARTIFACTS[prefix + "_candidate"][1],
                        "bytes": files[prefix + "_candidate"].stat().st_size,
                        "sha256": hashlib.sha256(
                            files[prefix + "_candidate"].read_bytes()
                        ).hexdigest(),
                    }
                    if partition.get("candidate_manifest") != expected_candidate_ref or any(
                        partition.get(key) != candidate.get(key) for key in ("repo", "revision")
                    ):
                        errors.append(f"{prefix} assignments refer to a different source candidate")
                    errors.extend(check_file(root, partition["candidate_manifest"]))
                    if partition.get("source_files") != candidate.get("prepared_files"):
                        errors.append(
                            f"{prefix} assignments do not bind current prepared source files"
                        )
                    if (
                        partition.get("implementation_sha256")
                        != hashlib.sha256(files["partition_contract"].read_bytes()).hexdigest()
                    ):
                        errors.append(
                            f"{prefix} assignments used a different partition implementation"
                        )
                arxiv = _object(read_json(files["arxiv_source"]), "arXiv source report")
                conversion = _object(arxiv.get("conversion"), "arXiv conversion")
                if (
                    conversion.get("script") != ARTIFACTS["arxiv_builder"][1]
                    or conversion.get("script_sha256")
                    != hashlib.sha256(files["arxiv_builder"].read_bytes()).hexdigest()
                ):
                    errors.append("arXiv source report used a different reconstruction script")
                if conversion.get("helper_sha256") != {
                    path: hashlib.sha256(local.read_bytes()).hexdigest()
                    for path, local in expected_helpers.items()
                }:
                    errors.append("arXiv source report does not bind current preparation helpers")
                validate_ledger(read_json(files["compute_ledger_template"]))
                observations.append(
                    "Legacy corpus audit is historical preparation context; fresh dataset admission is checked separately"
                )
                blockers.extend(validate_model_admission(root, plan))
            else:
                from src.research.annotations import validate_packet
                from src.research.book_admission import validate_book_admission

                packet = _object(read_json(files["annotation_packet"]), "annotation packet")
                validate_packet(
                    packet,
                    safe_path(root, packet["catalog"]["path"]),
                    {"mood": files["mood_rubric"], "recommendation": files["relevance_rubric"]},
                    root=root,
                )
                blockers.extend(validate_book_admission(root, plan, packet))
            gates = _object(status.get("gates"), "preparation gates")
            relevant_paths = {ARTIFACTS[key][1] for key in required}
            for name, gate in gates.items():
                for reference in _rows(_object(gate, name).get("evidence"), name + " evidence"):
                    if reference.get("path") in relevant_paths:
                        errors.extend(check_file(root, reference))
            literature = _object(gates.get("literature_gate"), "literature gate")
            if literature.get("status") != "resolved":
                blockers.append("Literature framing decision is not recorded")
            evidence = _rows(literature.get("evidence"), "literature evidence")
            review_id = "model_methods_review" if target == "model_study" else "book_methods_review"
            review_path = ARTIFACTS[review_id][1]
            reviews = [row for row in evidence if row.get("path") == review_path]
            if len(reviews) != 1:
                errors.append(f"Literature framing requires one pinned {target} methods review")
            else:
                errors.extend(check_file(root, reviews[0]))
        except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
            errors.append(str(exc))
    return {
        "schema_version": 1,
        "target": target,
        "artifacts_valid": not errors,
        "errors": errors,
        "blockers": blockers,
        "observations": observations,
        "ready_for_requested_stage": not errors and not blockers,
        "training_authorized": False,
        "execution_performed": False,
        "scope": "Preparation evidence inspection only; no models, scoring, training or network calls",
        "authority": "Readiness flags do not grant permission or establish scientific/human truth",
    }
