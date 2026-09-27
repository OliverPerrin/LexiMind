"""Explicit, target-scoped inventory of preparation evidence; no data or model loads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts.audit_research_artifacts import safe_path

# A book catalogue change must not invalidate the independent model-study packet.
ARTIFACTS: dict[str, tuple[str, str]] = {
    "study_design": ("common", "configs/research/study_design.json"),
    "preparation_status": ("common", "configs/research/preparation.json"),
    "research_index": ("common", "docs/research/README.md"),
    "study_decisions": ("common", "docs/research/study_decisions.md"),
    "evaluation_protocol": ("common", "docs/eval_protocol.md"),
    "preflight_contract": ("common", "src/research/preparation.py"),
    "file_integrity_contract": ("common", "scripts/audit_research_artifacts.py"),
    "manifest_contract": ("common", "src/research/manifest.py"),
    "model_literature": ("model_study", "research/preparation/model_literature.json"),
    "model_methods_review": ("model_study", "docs/research/model_recipe_review.md"),
    "backbone_candidates": ("model_study", "research/preparation/backbone_candidates.json"),
    "backbone_review": ("model_study", "docs/research/backbone_interface_review.md"),
    "repository_metadata": ("model_study", "research/preparation/repository_metadata.json"),
    "data_inventory": ("model_study", "research/preparation/data_inventory.json"),
    "data_audit": ("model_study", "research/preparation/data_audit.json"),
    "data_auditor": ("model_study", "scripts/audit_research_data.py"),
    "data_readiness": ("model_study", "docs/research/data_readiness.md"),
    "dataset_decisions": ("model_study", "docs/research/dataset_decisions.md"),
    "goemotions_candidate": (
        "model_study",
        "research/preparation/goemotions_candidate_manifest.json",
    ),
    "goemotions_builder": ("model_study", "scripts/prepare_goemotions_candidate.py"),
    "goemotions_reconstruction": ("model_study", "docs/research/goemotions_reconstruction.md"),
    "compute_ledger_template": ("model_study", "research/preparation/compute_ledger_template.json"),
    "compute_contract": ("model_study", "src/research/ledger.py"),
    "compute_accounting": ("model_study", "docs/research/compute_accounting.md"),
    "model_admission_contract": ("model_study", "src/research/admission.py"),
    "model_admission_guide": ("model_study", "docs/research/admission_contracts.md"),
    "book_literature": ("book_study", "research/preparation/book_literature.json"),
    "book_methods_review": ("book_study", "docs/research/book_discovery_review.md"),
    "annotation_packet": ("book_study", "research/preparation/annotation_packet.json"),
    "annotation_contract": ("common", "src/research/annotations.py"),
    "annotation_guide": ("book_study", "docs/research/annotation_preparation.md"),
    "mood_rubric": ("book_study", "docs/mood_annotation_guide.md"),
    "relevance_rubric": ("book_study", "docs/recommendation_judgments.md"),
    "book_admission_contract": ("book_study", "src/research/book_admission.py"),
    "book_admission_guide": ("book_study", "docs/research/book_admission_contract.md"),
}


def build_manifest(root: Path) -> dict[str, Any]:
    """Snapshot reviewed files. Building a snapshot does not admit their claims."""
    rows = []
    for artifact_id, (scope, relative) in ARTIFACTS.items():
        data = safe_path(root, relative).read_bytes()
        rows.append(
            {
                "id": artifact_id,
                "scope": scope,
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "Research preparation evidence, not execution authority",
        "artifacts": rows,
    }
