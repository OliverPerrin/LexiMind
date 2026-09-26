"""Synthetic accounting fixtures; no timings, profiles, or experiments are run."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.research.ledger import (
    COUNT_METRICS,
    QUANTITY_METRICS,
    LedgerValidationError,
    summarize_ledger,
    validate_ledger,
)

ROOT = Path(__file__).resolve().parents[2]


def observed_fixture():
    runs = [
        {
            "run_id": name,
            "hardware_id": "fixture-device-a",
            "environment_sha256": "a" * 64,
            "backbone_revision": "b" * 40,
            "code_commit": "c" * 40,
            "timing_boundary_id": "fixture-boundary-v1",
        }
        for name in ("expert-a", "expert-b", "merge")
    ]
    events = []
    for event_id, run_id, phase, outcome, tokens, cost in (
        ("train-a", "expert-a", "training", "completed", 10, 1.0),
        ("train-b", "expert-b", "training", "failed", 20, 2.0),
        ("merge-attempt", "merge", "merge", "interrupted", 0, 0.5),
    ):
        metrics = {**dict.fromkeys(COUNT_METRICS, 0), **dict.fromkeys(QUANTITY_METRICS, 0.0)}
        metrics.update(source_tokens=tokens, padded_source_tokens=tokens, external_cost_usd=cost)
        events.append(
            {
                "event_id": event_id,
                "run_id": run_id,
                "phase": phase,
                "outcome": outcome,
                "metrics": metrics,
            }
        )
    return {
        "schema_version": 1,
        "status": "observed",
        "runs": runs,
        "events": events,
        "recipes": [
            {"recipe_id": "specialist-a", "run_ids": ["expert-a"]},
            {"recipe_id": "specialist-b", "run_ids": ["expert-b"]},
            {"recipe_id": "merged", "run_ids": ["expert-a", "expert-b", "merge"]},
        ],
    }


def test_reused_experts_cost_each_recipe_but_only_once_project_wide():
    report = summarize_ledger(observed_fixture())
    recipes = {recipe["recipe_id"]: recipe for recipe in report["recipes"]}
    assert recipes["merged"]["recipe_equivalent"]["metrics"]["external_cost_usd"]["value"] == 3.5
    assert recipes["merged"]["by_phase"]["training"]["metrics"]["external_cost_usd"]["value"] == 3.0
    assert report["project_unique"]["metrics"]["external_cost_usd"]["value"] == 3.5
    assert (
        sum(
            recipe["recipe_equivalent"]["metrics"]["external_cost_usd"]["value"]
            for recipe in recipes.values()
        )
        == 6.5
    )
    assert report["project_unique"]["outcome_counts"] == {
        "completed": 1,
        "failed": 1,
        "interrupted": 1,
    }
    assert report["project_unique"]["metrics"]["source_tokens"]["value"] == 30


def test_missing_and_null_metrics_remain_unknown_with_a_known_subtotal():
    ledger = observed_fixture()
    del ledger["events"][0]["metrics"]["external_cost_usd"]
    ledger["events"][1]["metrics"]["external_cost_usd"] = None
    report = summarize_ledger(ledger)
    cost = report["project_unique"]["metrics"]["external_cost_usd"]
    assert cost == {
        "value": None,
        "known_subtotal": 0.5,
        "known_event_count": 1,
        "unknown_event_ids": ["train-a", "train-b"],
    }
    assert report["project_by_phase"]["teacher"]["metrics"]["teacher_input_tokens"]["value"] is None
    assert (
        report["project_by_phase"]["teacher"]["metrics"]["teacher_input_tokens"]["known_subtotal"]
        is None
    )


@pytest.mark.parametrize(
    "metric,value",
    [
        ("source_tokens", True),
        ("optimizer_steps", 1.0),
        ("examples", -1),
        ("external_cost_usd", True),
        ("external_cost_usd", -0.1),
        ("wall_seconds", float("nan")),
        ("device_window_seconds", float("inf")),
        ("estimated_flops", "100"),
    ],
)
def test_bad_metric_types_and_costs_are_rejected(metric, value):
    ledger = observed_fixture()
    ledger["events"][0]["metrics"][metric] = value
    with pytest.raises(LedgerValidationError):
        validate_ledger(ledger)


@pytest.mark.parametrize(
    "revision", ["refs/heads/main", "refs/tags/v1.0", "v1.0", "b" * 7, "b" * 64, "B" * 40]
)
def test_backbone_revision_requires_a_canonical_full_hf_commit(revision):
    ledger = observed_fixture()
    ledger["runs"][0]["backbone_revision"] = revision
    with pytest.raises(LedgerValidationError, match="40-character HF commit"):
        validate_ledger(ledger)


def test_padded_counts_cannot_be_smaller_than_known_actual_counts():
    ledger = observed_fixture()
    ledger["events"][0]["metrics"]["padded_source_tokens"] = 9
    with pytest.raises(LedgerValidationError, match="less than"):
        validate_ledger(ledger)
    ledger["events"][0]["metrics"]["source_tokens"] = None
    validate_ledger(ledger)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value["runs"].append(copy.deepcopy(value["runs"][0])), "Duplicate run_id"),
        (
            lambda value: value["events"].append(copy.deepcopy(value["events"][0])),
            "Duplicate event_id",
        ),
        (
            lambda value: value["recipes"][0]["run_ids"].append("expert-a"),
            "duplicate run references",
        ),
        (lambda value: value["recipes"][0]["run_ids"].append("absent"), "nonexistent leaf"),
        (lambda value: value["events"][0].update(run_id="absent"), "nonexistent run"),
        (lambda value: value["events"].pop(), "without events"),
        (lambda value: value["recipes"][0].update(run_ids=[]), "leaf run references"),
        (lambda value: value["events"][0]["metrics"].update(gpu_hours=10), "unsupported metrics"),
        (lambda value: value["runs"][0].update(environment_sha256="unknown"), "unresolved"),
        (lambda value: value["runs"][0].update(backbone_revision="main"), "immutable revision"),
        (lambda value: value["runs"][0].update(code_commit="abc"), "full Git commit"),
    ],
)
def test_references_and_observation_metadata_fail_closed(mutation, message):
    ledger = observed_fixture()
    mutation(ledger)
    with pytest.raises(LedgerValidationError, match=message):
        validate_ledger(ledger)


def test_mixed_hardware_and_timing_are_reported_without_normalization():
    ledger = observed_fixture()
    ledger["runs"][1].update(
        hardware_id="fixture-device-b", timing_boundary_id="fixture-boundary-v2"
    )
    report = summarize_ledger(ledger)
    assert any(
        warning["code"] == "mixed_hardware" and warning["scope"] == "recipe:merged"
        for warning in report["warnings"]
    )
    assert any(warning["code"] == "mixed_timing_boundaries" for warning in report["warnings"])
    assert report["project_unique"]["metrics"]["external_cost_usd"]["value"] == 3.5
    assert report["training_authorized"] is False
    assert report["measurements_independently_verified"] is False


def test_report_is_deterministic_and_does_not_mutate_ledger():
    ledger = observed_fixture()
    original = copy.deepcopy(ledger)
    first = summarize_ledger(ledger)
    assert ledger == original
    ledger["runs"].reverse()
    ledger["events"].reverse()
    ledger["recipes"].reverse()
    for recipe in ledger["recipes"]:
        recipe["run_ids"].reverse()
    assert summarize_ledger(ledger) == first
    assert summarize_ledger(original) == first


def test_unassigned_runs_are_in_project_cost_and_unknown_empty_phases_are_not_zero():
    ledger = observed_fixture()
    ledger["recipes"] = []
    report = summarize_ledger(ledger)
    assert report["project_unique"]["metrics"]["external_cost_usd"]["value"] == 3.5
    assert any(warning["code"] == "runs_not_in_recipes" for warning in report["warnings"])
    assert report["project_by_phase"]["validation"]["metrics"]["wall_seconds"]["value"] is None


def test_aggregate_overflow_is_not_serialized_as_infinity():
    ledger = observed_fixture()
    for event in ledger["events"]:
        event["metrics"]["estimated_flops"] = 1e308
    with pytest.raises(LedgerValidationError, match="Aggregate overflow"):
        summarize_ledger(ledger)


def test_committed_template_contains_no_fake_observations():
    template = json.loads((ROOT / "research/preparation/compute_ledger_template.json").read_text())
    validate_ledger(template)
    assert template == {
        "schema_version": 1,
        "status": "unobserved_template",
        "runs": [],
        "events": [],
        "recipes": [],
    }
    report = summarize_ledger(template)
    assert report["project_unique"]["event_count"] == 0
    assert all(metric["value"] is None for metric in report["project_unique"]["metrics"].values())


def test_template_cannot_hide_observations_under_unobserved_status():
    ledger = observed_fixture()
    ledger["status"] = "unobserved_template"
    with pytest.raises(LedgerValidationError, match="empty runs"):
        validate_ledger(ledger)


def test_cli_observed_gate_and_source_preservation(tmp_path):
    source = tmp_path / "ledger.json"
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_compute_ledger.py"),
        str(source),
        "--report",
        str(report_path),
        "--require-observed",
    ]
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "unobserved_template",
                "runs": [],
                "events": [],
                "recipes": [],
            }
        )
    )
    assert subprocess.run(command, capture_output=True).returncode == 2
    source.write_text(json.dumps(observed_fixture()))
    before = source.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert source.read_bytes() == before
    assert json.loads(report_path.read_text())["training_authorized"] is False
    assert subprocess.run(command[:-2] + [str(source)], capture_output=True).returncode == 2


def test_cli_rejects_duplicate_json_fields(tmp_path):
    source = tmp_path / "ledger.json"
    source.write_text(
        '{"schema_version":1,"status":"observed","status":"unobserved_template","runs":[],"events":[],"recipes":[]}'
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_compute_ledger.py"), str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Duplicate JSON field" in result.stderr
