"""Check cross-bound model-study admission records using files and JSON only.

Passing these consistency checks does not verify human reviews, measurements,
source rights or execution permission. Draft references must remain null until
actual evidence exists. No model, training or scoring code is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .io import file_hash, read_json

HEAD_INITIALIZATION = "identical task-specific initialization within each matched training seed"
SPLITS = ("train", "model_selection", "calibration", "test")
RUNTIME_FIELDS = {
    "transformers_version",
    "torch_version",
    "peft_version",
    "environment_sha256",
    "implementation_commit",
}


class AdmissionError(ValueError):
    """An evidence reference or declared contract is incomplete/inconsistent."""


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise AdmissionError(f"{name} must be an object, not a placeholder")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AdmissionError(f"{name} must be a nonblank string")
    if value.casefold() in {"unknown", "unresolved", "tbd", "none", "null", "main", "latest"}:
        raise AdmissionError(f"{name} is unresolved")
    return value


def _digest(value: Any, name: str, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise AdmissionError(f"{name} must be a complete lowercase {length}-digit digest")
    return value


def _number(value: Any, name: str, *, maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise AdmissionError(f"{name} must be finite and positive")
    try:
        valid = math.isfinite(value) and value > 0 and (maximum is None or value <= maximum)
    except OverflowError:
        valid = False
    if not valid:
        raise AdmissionError(f"{name} must be finite and positive")
    return float(value)


def _names(value: Any, name: str, *, nonempty: bool = True, ordered: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise AdmissionError(f"{name} must be {'nonempty ' if nonempty else ''}an array")
    result = [_text(item, name) for item in value]
    if len(result) != len(set(result)):
        raise AdmissionError(f"{name} contains duplicate names")
    if ordered and result != sorted(result):
        raise AdmissionError(f"{name} must use canonical lexicographic order")
    return result


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict:
    return _object(read_json(path), path.name)


def _reference(root: Path, reference: Any, name: str) -> tuple[Path, dict]:
    ref = _object(reference, name)
    if set(ref) != {"path", "bytes", "sha256"}:
        raise AdmissionError(f"{name} requires exactly path, bytes and sha256")
    relative = Path(_text(ref["path"], name + ".path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise AdmissionError(f"{name} must stay inside the repository")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise AdmissionError(f"{name} is missing or escapes the repository")
    if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
        raise AdmissionError(f"{name}.bytes must be a positive integer")
    expected = _digest(ref["sha256"], name + ".sha256")
    if path.stat().st_size != ref["bytes"]:
        raise AdmissionError(f"{name} byte count changed")
    if file_hash(path) != expected:
        raise AdmissionError(f"{name} SHA-256 changed")
    return path, ref


def _artifact(root: Path, ref: Any, kind: str, binding: dict) -> dict:
    descriptor = _object(ref, kind)
    if not isinstance(descriptor.get("path"), str) or Path(descriptor["path"]).suffix != ".json":
        raise AdmissionError(f"{kind} must reference a JSON evidence record")
    if type(descriptor.get("bytes")) is not int or not 0 < descriptor["bytes"] <= 64 * 1024 * 1024:
        raise AdmissionError(f"{kind} must be a bounded JSON evidence record")
    path, _ = _reference(root, ref, kind)
    if path.suffix != ".json" or path.stat().st_size > 64 * 1024 * 1024:
        raise AdmissionError(f"{kind} must be a bounded JSON evidence record")
    record = _read_json(path)
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise AdmissionError(f"{kind} has an unsupported schema_version")
    if record.get("kind") != kind or record.get("status") != "reviewed":
        raise AdmissionError(
            f"Expected reviewed {kind}; unrelated reports are not admission evidence"
        )
    if record.get("binding") != binding:
        raise AdmissionError(f"{kind} does not bind this study, backbone and runtime")
    return record


def _receipt(
    root: Path, ref: Any, kind: str, binding: dict, payload: dict, checks: set[str]
) -> None:
    record = _artifact(root, ref, kind, binding)
    if record.get("payload_sha256") != _canonical(payload):
        raise AdmissionError(f"{kind} does not bind the supplied payload")
    if record.get("outcome") != "passed":
        raise AdmissionError(f"{kind} has no recorded passing outcome")
    _text(record.get("recorded_by"), kind + ".recorded_by")
    assertions = _object(record.get("checks"), kind + ".checks")
    if set(assertions) != checks or any(assertions[key] is not True for key in checks):
        raise AdmissionError(f"{kind} is missing required review/measurement assertions")
    evidence = record.get("evidence_files")
    if not isinstance(evidence, list) or not evidence:
        raise AdmissionError(f"{kind} requires pinned supporting evidence files")
    for item in evidence:
        _reference(root, item, kind + ".supporting_evidence")


def _binding(plan: dict, model: dict) -> dict:
    backbone = _object(model.get("backbone"), "backbone")
    repo = _text(backbone.get("selected_repo"), "selected backbone repo")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise AdmissionError("selected backbone repo must be an owner/repository identifier")
    revision = _digest(backbone.get("selected_revision"), "selected backbone revision", 40)
    runtime = _object(backbone.get("selected_runtime"), "selected runtime")
    if set(runtime) != RUNTIME_FIELDS:
        raise AdmissionError(
            "selected runtime requires exact library versions, environment hash and implementation commit"
        )
    for field in RUNTIME_FIELDS:
        _text(runtime[field], "runtime." + field)
    _digest(runtime["environment_sha256"], "runtime.environment_sha256")
    _digest(runtime["implementation_commit"], "runtime.implementation_commit", 40)
    return {
        "study_id": _text(plan.get("study_id"), "study_id"),
        "backbone_repo": repo,
        "backbone_revision": revision,
        "runtime": runtime,
    }


def _budget(model: dict, tasks: list[str]) -> dict:
    budget = _object(model.get("budget"), "budget")
    _number(budget.get("B"), "B")
    _text(budget.get("hardware_id"), "hardware_id")
    _text(budget.get("timing_boundary_id"), "timing_boundary_id")
    if budget.get("primary_unit") != "synchronized_training_wall_seconds":
        raise AdmissionError("Budget must use the declared synchronized training-window unit")
    if budget.get("status") != "frozen":
        raise AdmissionError("Budget and timing policy are not frozen")
    for flag in (
        "equal_selection_access",
        "count_all_expert_training",
        "report_total_development_cost",
        "tokens_are_auxiliary_not_compute_equivalence",
    ):
        if budget.get(flag) is not True:
            raise AdmissionError(f"Budget safeguard must be true: {flag}")
    allocation = _object(budget.get("specialist_allocation"), "specialist_allocation")
    if set(allocation) != set(tasks):
        raise AdmissionError("Specialist fractions must cover exactly the admitted tasks")
    fractions = [_number(allocation[task], "allocation." + task, maximum=1) for task in tasks]
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise AdmissionError("Specialist fractions must sum to one total B")
    overshoot = _object(budget.get("overshoot_rule"), "overshoot_rule")
    if (
        set(overshoot) != {"name", "max_fraction"}
        or overshoot["name"] != "finish_current_optimizer_window"
    ):
        raise AdmissionError("A supported named optimizer-window stopping rule is required")
    _number(overshoot["max_fraction"], "max overshoot fraction", maximum=1)
    selection = _object(budget.get("selection_allowance"), "selection_allowance")
    if set(selection) != {"unit", "per_recipe_seconds", "max_trials_per_recipe", "data_access"}:
        raise AdmissionError(
            "Selection allowance requires its unit, seconds, trial cap and common data access"
        )
    if (
        selection["unit"] != "synchronized_selection_wall_seconds"
        or selection["data_access"] != "shared_development_partitions"
    ):
        raise AdmissionError(
            "Selection must use comparable timed allowances and common development partitions"
        )
    _number(selection["per_recipe_seconds"], "selection seconds")
    if (
        type(selection["max_trials_per_recipe"]) is not int
        or selection["max_trials_per_recipe"] < 1
    ):
        raise AdmissionError("Selection trial cap must be a positive integer")
    return {
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
    }


def _dataset(root: Path, ref: Any, binding: dict, tasks: list[str]) -> tuple[dict, dict]:
    record = _artifact(root, ref, "dataset_admission", binding)
    payload = _object(record.get("payload"), "dataset payload")
    entries = payload.get("tasks")
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise AdmissionError("Dataset tasks must be an ordered array of objects")
    if [row.get("task_id") for row in entries] != tasks:
        raise AdmissionError("Dataset task order must match the admitted study tasks")
    labels = {}
    for task in entries:
        task_id = task["task_id"]
        label = _artifact(root, task.get("label_contract"), "task_label_contract", binding)
        if label.get("task_id") != task_id:
            raise AdmissionError("Label contract is for a different task")
        output_type = label.get("output_type")
        if output_type not in {"single_label", "multi_label", "sequence"}:
            raise AdmissionError("Label contract must declare its output type")
        _names(label.get("label_order"), "label_order", nonempty=output_type != "sequence")
        if output_type == "sequence" and label["label_order"]:
            raise AdmissionError(
                "Sequence targets must use a tokenizer contract, not classification labels"
            )
        tokenizer = _object(label.get("tokenizer"), "tokenizer contract")
        _text(tokenizer.get("repo"), "tokenizer repo")
        _digest(tokenizer.get("revision"), "tokenizer revision", 40)
        labels[task_id] = task["label_contract"]["sha256"]
        splits = _object(task.get("splits"), task_id + ".splits")
        if set(splits) != set(SPLITS):
            raise AdmissionError(
                "Each task needs train/model_selection/calibration/test split references"
            )
        split_hashes = {}
        for split in SPLITS:
            _, split_ref = _reference(root, splits[split], task_id + "." + split)
            split_hashes[split] = split_ref["sha256"]
        if len(set(split_hashes.values())) != len(SPLITS):
            raise AdmissionError("Different partitions cannot be the same file content")
        groups = _artifact(root, task.get("group_assignment"), "dataset_group_assignment", binding)
        if groups.get("task_id") != task_id or groups.get("split_sha256") != split_hashes:
            raise AdmissionError("Group assignment does not bind the task's exact split files")
        policy = _object(groups.get("grouping_policy"), "grouping_policy")
        if policy.get("unit") not in {
            "provider_document_id",
            "canonical_work_id",
            "reviewed_group_id",
        }:
            raise AdmissionError(
                "Grouping must declare provider-document, canonical-work or reviewed group identity"
            )
        if policy.get("generalization_claim") not in {
            "new_source_ids",
            "unseen_normalized_text",
            "unseen_canonical_works",
        }:
            raise AdmissionError("Grouping must declare its held-out generalization claim")
        if policy.get("text_overlap_policy") not in {
            "preserve_official_splits_report_overlap",
            "deduplicated_variant",
        }:
            raise AdmissionError(
                "An explicit preserve/report or deduplicated text-overlap policy is required"
            )
        overlap = _artifact(root, groups.get("overlap_report"), "dataset_overlap_report", binding)
        if overlap.get("task_id") != task_id or overlap.get("split_sha256") != split_hashes:
            raise AdmissionError("Overlap report does not bind the admitted task and splits")
        count = overlap.get("cross_split_normalized_text_groups")
        if type(count) is not int or count < 0:
            raise AdmissionError("Normalized text-overlap count must be a nonnegative integer")
        if (
            policy["generalization_claim"] == "unseen_normalized_text"
            or policy["text_overlap_policy"] == "deduplicated_variant"
        ) and count:
            raise AdmissionError(
                "Observed text overlap contradicts the declared unseen-text/deduplicated policy"
            )
        if (
            policy["generalization_claim"] == "unseen_canonical_works"
            and policy["unit"] != "canonical_work_id"
        ):
            raise AdmissionError("Provider document IDs do not establish unseen canonical works")
        by_split = _object(groups.get("group_ids_by_split"), "group_ids_by_split")
        if set(by_split) != set(SPLITS):
            raise AdmissionError("Group assignment must cover every partition")
        seen: set[str] = set()
        for split in SPLITS:
            names = set(_names(by_split[split], "group IDs"))
            if seen & names:
                raise AdmissionError("Parent/document groups overlap across study partitions")
            seen |= names
    _receipt(
        root,
        record.get("receipt"),
        "dataset_admission_receipt",
        binding,
        payload,
        {
            "source_use_reviewed",
            "group_disjointness_verified",
            "labels_verified",
            "split_files_verified",
            "no_quarantined_inputs",
            "grouping_policy_reviewed",
        },
    )
    return payload, labels


def _modules(
    root: Path, ref: Any, binding: dict, tasks: list[str], data_ref: dict, labels: dict, model: dict
) -> dict:
    record = _artifact(root, ref, "adapter_module_manifest", binding)
    payload = _object(record.get("payload"), "adapter payload")
    if (
        payload.get("dataset_admission_sha256") != data_ref["sha256"]
        or payload.get("label_contract_sha256") != labels
    ):
        raise AdmissionError(
            "Adapter manifest does not bind the admitted dataset and label contracts"
        )
    if payload.get("private_scope") != model.get("private_scope") or set(
        payload["private_scope"]
    ) != set(tasks):
        raise AdmissionError("Adapter private-component policy differs from the study plan")
    lora = _object(payload.get("lora_config"), "lora_config")
    if set(lora) != {"r", "alpha", "dropout", "bias", "use_rslora", "use_dora"}:
        raise AdmissionError("LoRA rank, scaling, dropout and variant must be pinned")
    if type(lora["r"]) is not int or lora["r"] < 1:
        raise AdmissionError("LoRA rank must be a positive integer")
    _number(lora["alpha"], "LoRA alpha")
    if type(lora["dropout"]) not in (int, float) or not 0 <= lora["dropout"] < 1:
        raise AdmissionError("LoRA dropout must be a finite number in [0,1)")
    if lora["bias"] != "none" or lora["use_rslora"] is not False or lora["use_dora"] is not False:
        raise AdmissionError("This contract specifies standard LoRA without trainable base bias")
    modules = _object(payload.get("module_allowlist"), "module_allowlist")
    shared_modules = _names(modules.get("shared"), "shared module allowlist", ordered=True)
    if any(
        not re.search(r"(?:^|\.)encoder(?:\.|$)", name)
        or not re.search(r"(?:SelfAttention|self_attn)\.", name)
        for name in shared_modules
    ):
        raise AdmissionError("Shared allowlist must be explicitly encoder attention modules")
    private_modules = _object(modules.get("private"), "private module allowlists")
    private = _object(payload.get("private_parameters"), "private_parameters")
    if set(private) != set(tasks) or set(private_modules) != set(tasks):
        raise AdmissionError("Every task requires its private allowlist and parameter partition")
    groups = {
        "shared": _names(payload.get("shared_parameters"), "shared_parameters", ordered=True),
        "frozen": _names(payload.get("frozen_parameters"), "frozen_parameters", ordered=True),
    }
    prefixes = {"shared": shared_modules}
    for task in tasks:
        groups[task] = _names(private[task], "private parameters for " + task, ordered=True)
        prefixes[task] = _names(private_modules[task], "private modules for " + task, ordered=True)
        if any(re.search(r"(?:^|\.)encoder(?:\.|$)", name) for name in prefixes[task]):
            raise AdmissionError("Task-private allowlists cannot contain the shared encoder")
    module_names = [name for names in prefixes.values() for name in names]
    if len(module_names) != len(set(module_names)):
        raise AdmissionError("Shared and private module allowlists must be disjoint")
    if any(
        first != second and first.startswith(second + ".")
        for first in module_names
        for second in module_names
    ):
        raise AdmissionError("Overlapping parent/child module allowlists are ambiguous")
    parameter_names = [name for names in groups.values() for name in names]
    if len(parameter_names) != len(set(parameter_names)):
        raise AdmissionError("Shared, private and frozen parameter partitions must be disjoint")
    inventory = _object(payload.get("parameter_inventory"), "parameter_inventory")
    if set(inventory) != set(parameter_names):
        raise AdmissionError("Parameter partitions must cover the complete declared inventory")
    for partition, allowed in prefixes.items():
        if any(
            not any(name.startswith(module + ".") for name in groups[partition])
            for module in allowed
        ):
            raise AdmissionError("An allowed trainable module has no declared parameters")
    for partition, names in groups.items():
        for name in names:
            parameter = _object(inventory[name], "parameter " + name)
            shape = parameter.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or any(type(n) is not int or n <= 0 for n in shape)
            ):
                raise AdmissionError("Every parameter requires its concrete positive shape")
            _digest(parameter.get("initial_sha256"), name + ".initial_sha256")
            origin = parameter.get("origin")
            if origin not in {"base", "lora", "private_head"}:
                raise AdmissionError("Every parameter must declare base/lora/private_head origin")
            if origin == "base" and partition != "frozen":
                raise AdmissionError("All pretrained base parameters must stay frozen")
            if partition == "shared" and origin != "lora":
                raise AdmissionError("Only encoder LoRA parameters belong in the shared update")
            if partition in tasks:
                policy = payload["private_scope"][partition]
                expected_origin = {
                    "pooler_and_classifier": "private_head",
                    "decoder_attention_adapters": "lora",
                }.get(policy)
                if expected_origin is None or origin != expected_origin:
                    raise AdmissionError(
                        "Private parameter origin contradicts its task-interface policy"
                    )
            if partition != "frozen":
                if not any(name.startswith(module + ".") for module in prefixes[partition]):
                    raise AdmissionError(
                        "Trainable parameter lies outside its ordered module allowlist"
                    )
    aliases = payload.get("tied_parameter_groups")
    if not isinstance(aliases, list):
        raise AdmissionError("Tied-parameter groups must be explicitly recorded (possibly empty)")
    alias_names: set[str] = set()
    for tied in aliases:
        names = _names(tied, "tied group", ordered=True)
        if len(names) < 2 or not set(names) <= set(groups["frozen"]) or alias_names & set(names):
            raise AdmissionError("Tied groups must be disjoint groups of frozen inventory aliases")
        if len({inventory[name]["initial_sha256"] for name in names}) != 1:
            raise AdmissionError("Tied aliases disagree on initial tensor identity")
        if len({tuple(inventory[name]["shape"]) for name in names}) != 1:
            raise AdmissionError("Tied aliases disagree on shape")
        alias_names.update(names)
    initializations = _object(
        payload.get("private_initialization_files"), "private_initialization_files"
    )
    seeds = _object(model.get("training_seeds"), "training_seeds").get("values")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise AdmissionError("Study requires unique integer training seeds")
    inspection_seed = payload.get("inspection_seed")
    if type(inspection_seed) is not int or inspection_seed not in seeds:
        raise AdmissionError("Adapter inventory must identify its inspected initialization seed")
    if set(initializations) != {str(seed) for seed in seeds}:
        raise AdmissionError("Private initialization records must cover every matched seed")
    arm_rows = model.get("arms")
    if not isinstance(arm_rows, list) or any(not isinstance(arm, dict) for arm in arm_rows):
        raise AdmissionError("Training-arm declarations must be objects")
    selected_arms = [arm for arm in arm_rows if arm.get("role") in {"joint", "specialists"}]
    training_arms = [_text(arm.get("arm_id"), "training arm ID") for arm in selected_arms]
    if (
        len(training_arms) != 2
        or len(set(training_arms)) != 2
        or {arm["role"] for arm in selected_arms} != {"joint", "specialists"}
    ):
        raise AdmissionError(
            "Private initialization comparison requires distinct joint and specialist arms"
        )
    for seed in seeds:
        by_arm = _object(initializations[str(seed)], "initializations by arm")
        if set(by_arm) != set(training_arms):
            raise AdmissionError("Private initialization files must cover the same training arms")
        hashes: dict[str, set[str]] = {task: set() for task in tasks}
        for arm, task_refs in by_arm.items():
            task_refs = _object(task_refs, arm + ".initializations")
            if set(task_refs) != set(tasks):
                raise AdmissionError("Each arm requires all corresponding private initializations")
            for task, state_ref in task_refs.items():
                initial = _artifact(root, state_ref, "private_initialization_manifest", binding)
                if (
                    initial.get("seed") != seed
                    or type(initial.get("seed")) is not int
                    or initial.get("task_id") != task
                ):
                    raise AdmissionError("Private initialization record has the wrong seed/task")
                states = _object(
                    initial.get("parameter_initial_sha256"), "initial parameter hashes"
                )
                if set(states) != set(groups[task]):
                    raise AdmissionError(
                        "Private initializations do not cover the exact private parameter set"
                    )
                for digest in states.values():
                    _digest(digest, "private initial tensor")
                if seed == inspection_seed and any(
                    states[name] != inventory[name]["initial_sha256"] for name in states
                ):
                    raise AdmissionError(
                        "Private initialization disagrees with the inspected parameter inventory"
                    )
                hashes[task].add(_canonical(states))
        if any(len(values) != 1 for values in hashes.values()):
            raise AdmissionError("Task-private initial tensors differ across matched training arms")
    _receipt(
        root,
        record.get("receipt"),
        "adapter_structure_receipt",
        binding,
        payload,
        {
            "complete_parameter_inventory",
            "encoder_only_shared_scope",
            "private_task_scope",
            "frozen_base_and_aliases",
            "matched_head_initialization",
            "save_reload_verified",
        },
    )
    return payload


def _feasibility(
    root: Path,
    ref: Any,
    binding: dict,
    data_ref: dict,
    modules_ref: dict,
    labels: dict,
    budget: dict,
    tasks: list[str],
) -> None:
    record = _artifact(root, ref, "feasibility_timing_evidence", binding)
    payload = _object(record.get("payload"), "feasibility payload")
    if (
        payload.get("dataset_admission_sha256") != data_ref["sha256"]
        or payload.get("adapter_module_manifest_sha256") != modules_ref["sha256"]
        or payload.get("label_contract_sha256") != labels
        or _canonical(payload.get("budget_contract")) != _canonical(budget)
    ):
        raise AdmissionError(
            "Feasibility evidence does not bind the exact data, adapters, labels and budget"
        )
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise AdmissionError("Feasibility evidence requires recorded timing observations")
    covered = set()
    for measurement in measurements:
        item = _object(measurement, "timing measurement")
        if item.get("task_id") not in tasks:
            raise AdmissionError("Timing observation must identify an admitted task")
        covered.add(item["task_id"])
        if (
            item.get("hardware_id") != budget["hardware_id"]
            or item.get("timing_boundary_id") != budget["timing_boundary_id"]
        ):
            raise AdmissionError("Timing observations use different hardware or boundaries")
        _number(item.get("training_window_seconds"), "observed training-window seconds")
        if type(item.get("optimizer_windows")) is not int or item["optimizer_windows"] < 1:
            raise AdmissionError("Timing observation needs a positive optimizer-window count")
        for key in ("source_length", "batch_size"):
            if type(item.get(key)) is not int or item[key] < 1:
                raise AdmissionError(f"Timing observation needs a positive {key}")
        if type(item.get("target_length")) is not int or item["target_length"] < 0:
            raise AdmissionError("Timing observation requires nonnegative target_length")
        _reference(root, item.get("raw_receipt"), "raw timing receipt")
    if covered != set(tasks):
        raise AdmissionError("Feasibility timing does not cover every admitted task")
    _receipt(
        root,
        record.get("receipt"),
        "feasibility_review_receipt",
        binding,
        payload,
        {
            "runtime_and_tokenizer_verified",
            "timing_boundary_verified",
            "hardware_feasible",
            "allocation_and_overshoot_reviewed",
            "selection_allowance_reviewed",
        },
    )


def validate_model_admission(root: Path, plan: dict) -> list[str]:
    """Return blockers only. Empty means consistent receipts, never permission/truth."""
    blockers = []
    try:
        model = _object(plan.get("model_study"), "model_study")
        task_rows = model.get("tasks")
        if not isinstance(task_rows, list) or any(not isinstance(row, dict) for row in task_rows):
            raise AdmissionError("Study tasks must be objects")
        tasks = _names([row.get("task_id") for row in task_rows], "study task IDs")
    except (ValueError, TypeError, AttributeError) as error:
        return [str(error)]
    for field in ("pretrained_base_updates", "embedding_and_layernorm_updates"):
        if model.get(field) is not False:
            blockers.append(f"{field} must explicitly be false")
    if model.get("head_initialization") != HEAD_INITIALIZATION:
        blockers.append(
            "Identical task-private initialization within each matched seed must be declared"
        )
    if (
        model.get("merge_private_policy")
        != "retain_corresponding_specialist_private_modules_without_updates"
    ):
        blockers.append("Private modules must be retained unchanged from corresponding specialists")
    for key, expected in {
        "adaptation_family": "lora",
        "shared_scope": "encoder_attention_effective_deltas",
        "merge_space": "effective_weight_deltas",
        "materialization": "into_same_base_weights_without_rank_recompression",
    }.items():
        if model.get(key) != expected:
            blockers.append(f"{key} must match the declared encoder-only LoRA comparison")
    if model.get("post_merge_training") is not False:
        blockers.append("Post-merge training is outside this admission contract")
    binding = budget = None
    try:
        binding = _binding(plan, model)
    except (ValueError, TypeError, OSError) as error:
        blockers.append(str(error))
    try:
        budget = _budget(model, tasks)
    except (ValueError, TypeError, OSError) as error:
        blockers.append(str(error))
    data_ref = model.get("selected_dataset_manifest")
    module_ref = model.get("adapter_module_manifest")
    feasibility_ref = (
        model.get("budget", {}).get("feasibility_evidence")
        if isinstance(model.get("budget"), dict)
        else None
    )
    for name, value in (
        ("dataset admission", data_ref),
        ("adapter module manifest", module_ref),
        ("feasibility/timing evidence", feasibility_ref),
    ):
        if value is None:
            blockers.append(f"Missing pinned {name}")
        elif not isinstance(value, dict):
            blockers.append(f"{name} must be a typed hash reference, not a placeholder")
    if (
        binding is None
        or budget is None
        or not isinstance(data_ref, dict)
        or not isinstance(module_ref, dict)
        or not isinstance(feasibility_ref, dict)
    ):
        return blockers
    try:
        _, labels = _dataset(root, data_ref, binding, tasks)
        _modules(root, module_ref, binding, tasks, data_ref, labels, model)
        _feasibility(root, feasibility_ref, binding, data_ref, module_ref, labels, budget, tasks)
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        blockers.append(str(error))
    return blockers
