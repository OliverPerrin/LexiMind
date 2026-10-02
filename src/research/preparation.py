"""Inspect research preparation without loading models, scoring systems or granting execution."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
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


def _inspect_source_archive(root: Path, files: dict[str, Path]) -> list[str]:
    """Optional integrity checks for retained controls; not a book-data prerequisite."""
    errors: list[str] = []
    inventory = _object(read_json(files["data_inventory"]), "data inventory")
    audit = _object(read_json(files["data_audit"]), "data audit")
    if (
        inventory.get("audit_script_sha256")
        != hashlib.sha256(files["data_auditor"].read_bytes()).hexdigest()
    ):
        errors.append("Data inventory was produced by a different auditor revision")
    # Matches audit_research_data.py's documented inventory serialization.
    canonical = json.dumps(inventory, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if audit.get("inventory_sha256") != hashlib.sha256(canonical).hexdigest():
        errors.append("Data audit does not bind the supplied inventory")
    for prefix in ("goemotions", "ag_news"):
        candidate = _object(read_json(files[prefix + "_candidate"]), prefix + " candidate")
        if (
            candidate.get("preparation_script_sha256")
            != hashlib.sha256(files[prefix + "_builder"].read_bytes()).hexdigest()
        ):
            errors.append(f"{prefix} candidate was produced by a different preparation script")
        helpers = _object(candidate.get("preparation_helper_sha256"), "source helper hashes")
        expected_helpers = {
            ARTIFACTS[key][1]: files[key] for key in ("candidate_io", "file_integrity_contract")
        }
        if set(helpers) != set(expected_helpers) or any(
            helpers.get(path) != hashlib.sha256(local.read_bytes()).hexdigest()
            for path, local in expected_helpers.items()
        ):
            errors.append(f"{prefix} candidate does not bind current preparation helpers")
        partition = _object(read_json(files[prefix + "_partitions"]), prefix + " partitions")
        expected_candidate_ref = {
            "path": ARTIFACTS[prefix + "_candidate"][1],
            "bytes": files[prefix + "_candidate"].stat().st_size,
            "sha256": hashlib.sha256(files[prefix + "_candidate"].read_bytes()).hexdigest(),
        }
        if partition.get("candidate_manifest") != expected_candidate_ref or any(
            partition.get(key) != candidate.get(key) for key in ("repo", "revision")
        ):
            errors.append(f"{prefix} assignments refer to a different source candidate")
        errors.extend(check_file(root, partition["candidate_manifest"]))
        if partition.get("source_files") != candidate.get("prepared_files"):
            errors.append(f"{prefix} assignments do not bind current prepared source files")
        if (
            partition.get("implementation_sha256")
            != hashlib.sha256(files["partition_contract"].read_bytes()).hexdigest()
        ):
            errors.append(f"{prefix} assignments used a different partition implementation")
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
    return errors


def _inspect_book_candidates(files: dict[str, Path], bgc: dict[str, Any]) -> list[str]:
    """Check candidate provenance without requiring ignored local corpora in CI."""
    errors: list[str] = []
    reports: dict[str, dict[str, Any]] = {}
    shared_helpers = ("candidate_io", "file_integrity_contract")
    contracts = (
        (
            "bgc_groups",
            "bgc_group_builder",
            (*shared_helpers, "book_group_contract", "bgc_builder"),
            {"candidate_manifest": "bgc_candidate"},
        ),
        (
            "book_fields",
            "book_field_builder",
            (*shared_helpers, "book_field_contract", "bgc_builder", "catalogue_storage"),
            {
                "candidate_manifest": "bgc_candidate",
                "grouping_manifest": "bgc_groups",
                "mapping": "book_field_mapping",
            },
        ),
        (
            "licensed_books",
            "licensed_book_builder",
            shared_helpers,
            {"source_inventory": "licensed_book_sources"},
        ),
    )
    for report_id, builder_id, helper_ids, references in contracts:
        report = _object(read_json(files[report_id]), report_id)
        reports[report_id] = report
        if report.get("training_authorized") is not False:
            errors.append(f"{report_id} must remain a preparation-only candidate")
        if (
            report.get("preparation_script_sha256")
            != hashlib.sha256(files[builder_id].read_bytes()).hexdigest()
        ):
            errors.append(f"{report_id} used a different preparation script")
        expected_helpers = {
            ARTIFACTS[key][1]: hashlib.sha256(files[key].read_bytes()).hexdigest()
            for key in helper_ids
        }
        if report.get("preparation_helper_sha256") != expected_helpers:
            errors.append(f"{report_id} does not bind current preparation helpers")
        for reference_key, artifact_id in references.items():
            expected_reference = {
                "path": ARTIFACTS[artifact_id][1],
                "bytes": files[artifact_id].stat().st_size,
                "sha256": hashlib.sha256(files[artifact_id].read_bytes()).hexdigest(),
            }
            if report.get(reference_key) != expected_reference:
                errors.append(f"{report_id} does not bind current {reference_key}")
        if report_id in {"bgc_groups", "book_fields"} and report.get("archive") != bgc.get(
            "archive"
        ):
            errors.append(f"{report_id} refers to a different BGC archive")
    if reports["book_fields"].get("assignments") != reports["bgc_groups"].get("assignments"):
        errors.append("book_fields refers to different group assignments")
    return errors


def _inspect_book_reviews(files: dict[str, Path]) -> list[str]:
    """Bind bounded review evidence; a historical catalogue screen is not admission."""
    errors: list[str] = []

    def reference(key):
        return {
            "path": ARTIFACTS[key][1],
            "bytes": files[key].stat().st_size,
            "sha256": hashlib.sha256(files[key].read_bytes()).hexdigest(),
        }

    def hashes(keys):
        return {ARTIFACTS[key][1]: reference(key)["sha256"] for key in keys}

    fields = _object(read_json(files["book_fields"]), "book fields")
    groups = _object(read_json(files["bgc_groups"]), "book groups")
    review = _object(read_json(files["book_field_review_report"]), "field review report")
    packet = _object(read_json(files["book_field_review"]), "field review packet")
    expected_bindings = {
        "archive": fields["archive"],
        "field_manifest": reference("book_fields"),
        "field_references": fields["field_references"],
        "mapping": reference("book_field_mapping"),
    }
    if review.get("bindings") != expected_bindings or packet.get("bindings") != expected_bindings:
        errors.append("Field review does not bind current source, fields and mapping")
    if review.get("review_packet") != reference("book_field_review"):
        errors.append("Field review report refers to a different review packet")
    if review.get("preparation_script_sha256") != reference("book_field_reviewer")["sha256"]:
        errors.append("Field review used a different reviewer script")
    if review.get("preparation_helper_sha256") != hashes(
        (
            "book_field_builder",
            "book_field_review_contract",
            "book_field_contract",
            "candidate_io",
            "file_integrity_contract",
            "catalogue_storage",
        )
    ):
        errors.append("Field review does not bind current reviewer helpers")
    group_review = _object(read_json(files["book_group_review"]), "group review")
    if group_review.get("implementation_sha256") != hashes(
        (
            "book_group_reviewer",
            "bgc_builder",
            "book_group_contract",
            "candidate_io",
            "file_integrity_contract",
        )
    ):
        errors.append("Group review does not bind current reviewer implementation")
    inputs = _object(group_review.get("inputs"), "group review inputs")
    # Catalogue matching describes the pinned snapshot recorded by the reviewer.
    # A later catalogue edit must not invalidate the independent model-study packet;
    # new candidate admission must refresh its own cross-source population screen.
    for key, expected in {
        "groups": reference("bgc_groups"),
        "licensed_sources": reference("licensed_book_sources"),
        "licensed_manifest": reference("licensed_books"),
        "candidate_manifest": reference("bgc_candidate"),
        **{key: groups[key] for key in ("archive", "assignments", "review_groups")},
    }.items():
        if inputs.get(key) != expected:
            errors.append(f"Group review does not bind current {key}")
    if any(
        value.get("training_authorized") is not False for value in (review, packet, group_review)
    ):
        errors.append("Review preparation cannot authorize training")
    return errors


def _inspect_extensions(files: dict[str, Path], plan: dict[str, Any]) -> list[str]:
    """Validate new source/objective receipts without requiring local corpora in CI."""
    errors: list[str] = []

    def ref(key):
        raw = files[key].read_bytes()
        return {
            "path": ARTIFACTS[key][1],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def hashes(keys):
        return {ARTIFACTS[key][1]: ref(key)["sha256"] for key in keys}

    cr4 = _object(read_json(files["cr4_candidate"]), "CR4 source audit")
    if cr4.get("preparation_script_sha256") != ref("cr4_builder")["sha256"] or cr4.get(
        "preparation_helper_sha256"
    ) != hashes(("candidate_io", "file_integrity_contract", "cr4_contract")):
        errors.append("CR4 audit does not bind its current parser and helpers")
    parts = _object(read_json(files["book_partitions"]), "book partitions")
    review = _object(read_json(files["book_group_review"]), "book group review")
    fields = _object(read_json(files["book_fields"]), "book fields")
    expected = {
        "review": ref("book_group_review"),
        **review["inputs"],
        "packet": review["packet"],
        "fields": ref("book_fields"),
        "field_mapping": ref("book_field_mapping"),
        "field_references": fields["field_references"],
    }
    if parts.get("inputs") != expected:
        errors.append("Book partitions do not bind the current review and field references")
    if parts.get("implementation_sha256") != hashes(
        (
            "book_partition_builder",
            "book_group_reviewer",
            "bgc_builder",
            "book_partition_contract",
            "catalogue_storage",
            "candidate_io",
            "file_integrity_contract",
        )
    ):
        errors.append("Book partitions used a different implementation")
    rpt = _object(read_json(files["rpt_candidate"]), "RPT candidate")
    licensed = _object(read_json(files["licensed_books"]), "licensed source")
    rpt_inputs = _object(rpt.get("inputs"), "RPT input bindings")
    for key, expected in {
        "licensed_manifest": ref("licensed_books"),
        "partition_manifest": ref("book_partitions"),
        "tokenizer": ref("rpt_tokenizer"),
        "source_inventory": ref("licensed_book_sources"),
        "components": parts["components"],
        **{
            f"work:{row['work_id']}": {
                **row["artifact"],
                "path": f"{licensed['cache_root']}/{row['artifact']['path']}",
            }
            for row in licensed["books"]
        },
    }.items():
        if rpt_inputs.get(key) != expected:
            errors.append(f"RPT candidate does not bind current {key}")
    if rpt.get("implementation_sha256") != hashes(
        ("licensed_book_builder", "candidate_io", "file_integrity_contract")
    ):
        errors.append("RPT candidate used a different tokenizer-preparation implementation")
    methods = _object(read_json(files["rl_methods"]), "RL source register")
    pilot = _object(read_json(files["pilot_observations"]), "Local pilot observations")
    if (
        pilot.get("kind") != "local_macbook_training_observations"
        or pilot["protocol"].get("baseline_config") != ref("pilot_config")
        or pilot["data"].get("rpt_manifest") != ref("rpt_candidate")
        or pilot["data"].get("global_test_used") is not False
    ):
        errors.append("Local pilot does not bind its scoped configuration and training-only data")
    cutoff = date.fromisoformat(methods["as_of"])
    for source in _rows(methods.get("sources"), "RL sources"):
        if (
            not date.fromisoformat(source["first_submitted"])
            <= date.fromisoformat(source["version_date"])
            <= cutoff
        ):
            errors.append("RL source version lies outside the declared literature cutoff")
    if any(report.get("training_authorized") is not False for report in (cr4, parts, rpt, methods)):
        errors.append("Source and RL preparation reports cannot authorize training")
    extension = _object(plan.get("rl_extension"), "RL extension")
    if extension.get("enabled") is not False or extension.get("training_authorized") is not False:
        errors.append("RL preparation remains disabled until a separately admitted experiment")
    return errors


def inspect_preparation(
    root: Path, manifest_path: Path, target: str, *, check_archive: bool = False
) -> dict[str, Any]:
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
    scopes = {"common", target} | ({"archive"} if check_archive else set())
    required = {key for key, (scope, _) in ARTIFACTS.items() if scope in scopes}
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
                backbone = _object(read_json(files["backbone_candidates"]), "backbone candidates")
                if (
                    backbone.get("metadata_source_sha256")
                    != hashlib.sha256(files["repository_metadata"].read_bytes()).hexdigest()
                ):
                    errors.append("Backbone review does not bind the supplied repository metadata")
                bgc = _object(read_json(files["bgc_candidate"]), "BGC source audit")
                if (
                    bgc.get("preparation_script_sha256")
                    != hashlib.sha256(files["bgc_builder"].read_bytes()).hexdigest()
                ):
                    errors.append("BGC audit used a different source parser")
                expected_helpers = {
                    ARTIFACTS[key][1]: hashlib.sha256(files[key].read_bytes()).hexdigest()
                    for key in ("candidate_io", "file_integrity_contract")
                }
                if bgc.get("preparation_helper_sha256") != expected_helpers:
                    errors.append("BGC audit does not bind current preparation helpers")
                errors.extend(_inspect_book_candidates(files, bgc))
                errors.extend(_inspect_book_reviews(files))
                errors.extend(_inspect_extensions(files, plan))
                validate_ledger(read_json(files["compute_ledger_template"]))
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
            if check_archive:
                errors.extend(_inspect_source_archive(root, files))
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
        "source_archive_checked": check_archive,
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
