"""Preparation consistency tests, with no research scoring or model imports."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.research.manifest import ARTIFACTS, build_manifest
from src.research.preparation import inspect_preparation, validate_study_design

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("research/preparation/manifest.json")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def packet_root(tmp_path):
    for _, relative in ARTIFACTS.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    catalog = tmp_path / "web/data/books.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "web/data/books.json", catalog)
    write_json(tmp_path / MANIFEST, build_manifest(tmp_path))
    return tmp_path


def test_committed_preparation_valid_but_both_studies_blocked():
    for target in ("model_study", "book_study"):
        report = inspect_preparation(ROOT, MANIFEST, target)
        assert report["artifacts_valid"], report["errors"]
        assert report["blockers"]
        assert not report["ready_for_requested_stage"]
        assert not report["execution_performed"]
        assert not report["training_authorized"]


def test_changed_book_catalogue_does_not_block_model_packet(packet_root):
    (packet_root / "web/data/books.json").write_text("{}")
    model = inspect_preparation(packet_root, MANIFEST, "model_study")
    books = inspect_preparation(packet_root, MANIFEST, "book_study")
    assert model["artifacts_valid"], model["errors"]
    assert not books["artifacts_valid"]


def test_missing_model_artifact_does_not_block_book_packet(packet_root):
    (packet_root / ARTIFACTS["data_inventory"][1]).unlink()
    assert inspect_preparation(packet_root, MANIFEST, "book_study")["artifacts_valid"]
    assert not inspect_preparation(packet_root, MANIFEST, "model_study")["artifacts_valid"]


def test_stale_artifact_rejected_without_auto_refresh(packet_root):
    before = (packet_root / MANIFEST).read_bytes()
    path = packet_root / ARTIFACTS["study_decisions"][1]
    path.write_text(path.read_text() + "\nChanged evidence.\n")
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("SHA-256 changed" in error for error in result["errors"])
    assert (packet_root / MANIFEST).read_bytes() == before


def test_rehashing_does_not_hide_stale_nested_data_binding(packet_root):
    path = packet_root / ARTIFACTS["data_inventory"][1]
    value = json.loads(path.read_text())
    value["audit_script_sha256"] = "a" * 64
    write_json(path, value)
    write_json(packet_root / MANIFEST, build_manifest(packet_root))
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("auditor revision" in error for error in result["errors"])
    assert any("supplied inventory" in error for error in result["errors"])


def test_rehashing_does_not_hide_stale_literature_evidence(packet_root):
    path = packet_root / ARTIFACTS["model_methods_review"][1]
    path.write_text(path.read_text() + "\nChanged review.\n")
    write_json(packet_root / MANIFEST, build_manifest(packet_root))
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("SHA-256 changed" in error for error in result["errors"])


def test_scope_cannot_hide_required_evidence(packet_root):
    path = packet_root / MANIFEST
    value = json.loads(path.read_text())
    next(row for row in value["artifacts"] if row["id"] == "data_inventory")["scope"] = "book_study"
    write_json(path, value)
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("scope/path mismatch" in error for error in result["errors"])


def test_placeholder_flags_cannot_admit_model_study(packet_root):
    path = packet_root / ARTIFACTS["study_design"][1]
    value = json.loads(path.read_text())
    value["status"] = "frozen"
    value["model_study"]["task_suite_status"] = "frozen"
    value["model_study"]["selected_dataset_manifest"] = {"approved": True}
    value["model_study"]["adapter_module_manifest"] = {"approved": True}
    write_json(path, value)
    status_path = packet_root / ARTIFACTS["preparation_status"][1]
    status = json.loads(status_path.read_text())
    status["paused_by_user"] = False  # Synthetic fixture, never written to the real registry.
    reference = next(
        row
        for row in status["gates"]["protocol"]["evidence"]
        if row["path"] == ARTIFACTS["study_design"][1]
    )
    raw = path.read_bytes()
    reference.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    write_json(status_path, status)
    write_json(packet_root / MANIFEST, build_manifest(packet_root))
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert result["artifacts_valid"], result["errors"]
    assert result["blockers"]
    assert not result["ready_for_requested_stage"]
    assert not result["training_authorized"]


@pytest.mark.parametrize(
    "mutation", ["factor_merge", "free_experts", "per_task_budget", "mood", "boolean_gains"]
)
def test_invalid_comparison_semantics_rejected(mutation):
    plan = json.loads((ROOT / ARTIFACTS["study_design"][1]).read_text())
    model = plan["model_study"]
    if mutation == "factor_merge":
        model["merge_space"] = "lora_factors"
    elif mutation == "free_experts":
        model["arms"][2]["include_expert_training_cost"] = False
    elif mutation == "per_task_budget":
        model["arms"][1]["training_allocation"] = "B_per_task"
    elif mutation == "mood":
        plan["applied_study"]["mood_queries_enabled"] = True
    else:
        plan["applied_study"]["gain_mapping"] = [False, True, 3, 7]
    with pytest.raises(ValueError):
        validate_study_design(plan)


def test_strict_json_rejects_duplicate_manifest_keys(packet_root):
    (packet_root / MANIFEST).write_text('{"schema_version":1,"schema_version":1,"artifacts":[]}')
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        inspect_preparation(packet_root, MANIFEST, "model_study")


def test_cli_distinguishes_consistent_from_ready():
    command = [sys.executable, str(ROOT / "scripts/audit_research_preparation.py")]
    default = subprocess.run(command, capture_output=True, text=True, check=False)
    required = subprocess.run(
        command + ["--require-ready"], capture_output=True, text=True, check=False
    )
    assert default.returncode == 0, default.stdout + default.stderr
    assert required.returncode == 2, required.stdout + required.stderr
    assert not json.loads(required.stdout)["execution_performed"]


@pytest.mark.parametrize("mutation", ["manifest_reference", "repo", "revision"])
def test_partition_report_must_bind_its_exact_candidate(packet_root, mutation):
    path = packet_root / ARTIFACTS["ag_news_partitions"][1]
    report = json.loads(path.read_text())
    if mutation == "manifest_reference":
        other = packet_root / ARTIFACTS["data_inventory"][1]
        raw = other.read_bytes()
        report["candidate_manifest"] = {
            "path": ARTIFACTS["data_inventory"][1],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    else:
        report[mutation] = "different/provider" if mutation == "repo" else "f" * 40
    write_json(path, report)
    write_json(packet_root / MANIFEST, build_manifest(packet_root))
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("different source candidate" in error for error in result["errors"])


def test_arxiv_report_cannot_outlive_its_converter_version(packet_root):
    path = packet_root / ARTIFACTS["arxiv_source"][1]
    report = json.loads(path.read_text())
    report["conversion"]["script_sha256"] = "f" * 64
    write_json(path, report)
    write_json(packet_root / MANIFEST, build_manifest(packet_root))
    result = inspect_preparation(packet_root, MANIFEST, "model_study")
    assert not result["artifacts_valid"]
    assert any("different reconstruction script" in error for error in result["errors"])
