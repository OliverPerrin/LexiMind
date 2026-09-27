"""Validate and summarize recorded compute without executing research workloads.

A schema-valid observed ledger is a claim backed by its external measurement
receipts, not proof of those receipts and never authorization to run training.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

SCHEMA_VERSION = 1
PHASES = (
    "training",
    "validation",
    "calibration",
    "merge_search",
    "merge",
    "final_evaluation",
    "teacher",
    "preprocessing",
)
OUTCOMES = ("completed", "failed", "interrupted")
COUNT_METRICS = (
    "source_tokens",
    "target_tokens",
    "padded_source_tokens",
    "padded_target_tokens",
    "examples",
    "optimizer_steps",
    "teacher_input_tokens",
    "teacher_output_tokens",
)
QUANTITY_METRICS = ("wall_seconds", "device_window_seconds", "estimated_flops", "external_cost_usd")
METRICS = COUNT_METRICS + QUANTITY_METRICS
RUN_FIELDS = (
    "run_id",
    "hardware_id",
    "environment_sha256",
    "backbone_revision",
    "code_commit",
    "timing_boundary_id",
)


class LedgerValidationError(ValueError):
    """The ledger cannot be safely interpreted under this schema."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LedgerValidationError(f"{context} must be an object")
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise LedgerValidationError(
            f"{context} must be a nonblank identifier without surrounding whitespace"
        )
    if value.casefold() in {"unknown", "unresolved", "tbd", "none", "null"}:
        raise LedgerValidationError(f"{context} is unresolved")
    return value


def _unique_records(values: Any, id_field: str, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise LedgerValidationError(f"{context} must be an array")
    indexed = {}
    for value in values:
        row = _object(value, context)
        identifier = _identifier(row.get(id_field), f"{context}.{id_field}")
        if identifier in indexed:
            raise LedgerValidationError(f"Duplicate {id_field}: {identifier}")
        indexed[identifier] = row
    return indexed


def validate_ledger(ledger: Any) -> None:
    """Validate structure and references, without verifying measurements."""
    ledger = _object(ledger, "ledger")
    if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != SCHEMA_VERSION:
        raise LedgerValidationError("Unsupported compute-ledger schema_version")
    if ledger.get("status") not in ("unobserved_template", "observed"):
        raise LedgerValidationError("status must be unobserved_template or observed")
    runs = _unique_records(ledger.get("runs"), "run_id", "runs")
    events = _unique_records(ledger.get("events"), "event_id", "events")
    recipes = _unique_records(ledger.get("recipes"), "recipe_id", "recipes")
    if ledger["status"] == "unobserved_template":
        if runs or events or recipes:
            raise LedgerValidationError(
                "An unobserved_template must have empty runs, events, and recipes"
            )
        return
    if not runs or not events:
        raise LedgerValidationError("An observed ledger requires runs and recorded events")
    for run_id, run in runs.items():
        for field in RUN_FIELDS:
            _identifier(run.get(field), f"run {run_id}.{field}")
        if not re.fullmatch(r"[a-f0-9]{64}", run["environment_sha256"]):
            raise LedgerValidationError(f"run {run_id}.environment_sha256 must be a full SHA-256")
        if not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", run["code_commit"]):
            raise LedgerValidationError(f"run {run_id}.code_commit must be a full Git commit hash")
        if not re.fullmatch(r"[a-f0-9]{40}", run["backbone_revision"]):
            raise LedgerValidationError(
                f"run {run_id}.backbone_revision must be a full lowercase 40-character HF commit SHA for an immutable revision"
            )
    run_events: Counter[str] = Counter()
    for event_id, event in events.items():
        run_id = _identifier(event.get("run_id"), f"event {event_id}.run_id")
        if run_id not in runs:
            raise LedgerValidationError(f"event {event_id} references nonexistent run {run_id}")
        run_events[run_id] += 1
        if event.get("phase") not in PHASES:
            raise LedgerValidationError(f"event {event_id} has an unsupported phase")
        if event.get("outcome") not in OUTCOMES:
            raise LedgerValidationError(f"event {event_id} has an unsupported outcome")
        metrics = _object(event.get("metrics"), f"event {event_id}.metrics")
        if unknown := set(metrics) - set(METRICS):
            raise LedgerValidationError(
                f"event {event_id} has unsupported metrics: {sorted(unknown)}"
            )
        for metric, value in metrics.items():
            if value is None:
                continue
            if metric in COUNT_METRICS:
                if type(value) is not int or value < 0:
                    raise LedgerValidationError(
                        f"event {event_id}.{metric} must be a nonnegative exact integer or null"
                    )
            else:
                try:
                    finite = type(value) in {int, float} and math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite or value < 0:
                    raise LedgerValidationError(
                        f"event {event_id}.{metric} must be finite, nonnegative, and numeric or null"
                    )
        for actual, padded in (
            ("source_tokens", "padded_source_tokens"),
            ("target_tokens", "padded_target_tokens"),
        ):
            if (
                metrics.get(actual) is not None
                and metrics.get(padded) is not None
                and metrics[padded] < metrics[actual]
            ):
                raise LedgerValidationError(f"event {event_id}.{padded} is less than {actual}")
    if empty_runs := sorted(set(runs) - set(run_events)):
        raise LedgerValidationError(f"Observed runs without events: {empty_runs}")
    for recipe_id, recipe in recipes.items():
        members = recipe.get("run_ids")
        if not isinstance(members, list) or not members:
            raise LedgerValidationError(
                f"recipe {recipe_id}.run_ids must contain leaf run references"
            )
        checked = [_identifier(member, f"recipe {recipe_id}.run_ids") for member in members]
        if len(set(checked)) != len(checked):
            raise LedgerValidationError(f"recipe {recipe_id} has duplicate run references")
        if absent := sorted(set(checked) - set(runs)):
            raise LedgerValidationError(
                f"recipe {recipe_id} references nonexistent leaf runs: {absent}"
            )


def _aggregate(events: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    members = sorted({event["run_id"] for event in events})
    totals = {}
    for metric in METRICS:
        values = [event["metrics"].get(metric) for event in events]
        known = [value for value in values if value is not None]
        unknown = sorted(
            event["event_id"] for event in events if event["metrics"].get(metric) is None
        )
        try:
            subtotal = sum(known) if metric in COUNT_METRICS else math.fsum(known)
        except OverflowError as error:
            raise LedgerValidationError(
                f"Aggregate overflow in {metric}; retain smaller disjoint accounting scopes"
            ) from error
        totals[metric] = {
            "value": subtotal if events and not unknown else None,
            "known_subtotal": subtotal if known else None,
            "known_event_count": len(known),
            "unknown_event_ids": unknown,
        }
    outcomes = Counter(event["outcome"] for event in events)
    return {
        "event_count": len(events),
        "run_ids": members,
        "hardware_ids": sorted({runs[run_id]["hardware_id"] for run_id in members}),
        "timing_boundary_ids": sorted({runs[run_id]["timing_boundary_id"] for run_id in members}),
        "outcome_counts": {outcome: outcomes[outcome] for outcome in OUTCOMES},
        "metrics": totals,
    }


def _phases(events: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        phase: _aggregate([event for event in events if event["phase"] == phase], runs)
        for phase in PHASES
    }


def summarize_ledger(ledger: Any) -> dict[str, Any]:
    """Count each recorded event once per project and once per referencing recipe."""
    validate_ledger(ledger)
    runs = {run["run_id"]: run for run in ledger["runs"]}
    events = sorted(ledger["events"], key=lambda event: event["event_id"])
    project = _aggregate(events, runs)
    recipes = []
    warnings: list[dict[str, Any]] = []
    for recipe in sorted(ledger["recipes"], key=lambda row: row["recipe_id"]):
        members = set(recipe["run_ids"])
        included = [event for event in events if event["run_id"] in members]
        recipes.append(
            {
                "recipe_id": recipe["recipe_id"],
                "run_ids": sorted(members),
                "recipe_equivalent": _aggregate(included, runs),
                "by_phase": _phases(included, runs),
            }
        )
    for scope, aggregate in [
        ("project_unique", project),
        *[("recipe:" + recipe["recipe_id"], recipe["recipe_equivalent"]) for recipe in recipes],
    ]:
        if len(aggregate["hardware_ids"]) > 1:
            warnings.append(
                {
                    "code": "mixed_hardware",
                    "scope": scope,
                    "hardware_ids": aggregate["hardware_ids"],
                    "message": "Summed device/window seconds are recorded usage, not hardware-normalized compute.",
                }
            )
        if len(aggregate["timing_boundary_ids"]) > 1:
            warnings.append(
                {
                    "code": "mixed_timing_boundaries",
                    "scope": scope,
                    "timing_boundary_ids": aggregate["timing_boundary_ids"],
                    "message": "Timing windows need a reviewed comparison policy before matched-budget claims.",
                }
            )
    if unknown := [metric for metric in METRICS if project["metrics"][metric]["unknown_event_ids"]]:
        warnings.append(
            {
                "code": "unknown_metric_values",
                "metrics": unknown,
                "message": "Missing/null quantities remain unknown; known subtotals are not complete costs.",
            }
        )
    if spent := [event["event_id"] for event in events if event["outcome"] != "completed"]:
        warnings.append({"code": "failed_or_interrupted_spend_included", "event_ids": spent})
    referenced = {run_id for recipe in ledger["recipes"] for run_id in recipe["run_ids"]}
    if unassigned := sorted(set(runs) - referenced):
        warnings.append(
            {
                "code": "runs_not_in_recipes",
                "run_ids": unassigned,
                "message": "These runs still count in project-unique recorded cost.",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_status": ledger["status"],
        "schema_valid": True,
        "measurements_independently_verified": False,
        "training_authorized": False,
        "accounting_scope": "Recorded events only; absent phases and unrecorded spending are not proven zero.",
        "project_unique": project,
        "project_by_phase": _phases(events, runs),
        "recipes": recipes,
        "phases_without_observations": [
            phase for phase in PHASES if not any(event["phase"] == phase for event in events)
        ],
        "warnings": warnings,
        "interpretation": [
            "A reused expert's events count in each referencing recipe's equivalent cost, but once in the project total.",
            "Recipe-equivalent totals must not be added together as project expenditure.",
            "Failed and interrupted event spending is included in every applicable aggregate.",
            "No token-to-compute, device-time-to-FLOPs, currency, or hardware equivalence conversion is performed.",
            "Budget selection, head/data exposure, measurement receipts, and training authorization remain external protocol gates.",
        ],
    }
