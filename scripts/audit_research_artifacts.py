"""Validate archived report bytes without importing ML code or running experiments.

The default audit needs only the small files in Git. --check-local additionally
checks optional, untracked checkpoint/data observations. --require-ready reports
preparation blockers and deliberately exits nonzero while research is paused.
This is a preparation check, not an interlock on the legacy training scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("research/results/manifest.json")
DEFAULT_PREPARATION = Path("configs/research/preparation.json")


def load_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def safe_path(root: Path, relative: str) -> Path:
    """Never follow an archive manifest outside its repository, including symlinks."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected repository-relative path: {relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return target


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_file(root: Path, item: dict) -> list[str]:
    if not isinstance(item, dict):
        return ["Invalid artifact: expected an object"]
    try:
        path = safe_path(root, item["path"])
        if not path.is_file():
            return [f"Missing file: {item['path']}"]
        errors = []
        if path.stat().st_size != item["bytes"]:
            errors.append(f"Byte size changed: {item['path']}")
        if sha256(path) != item["sha256"]:
            errors.append(f"SHA-256 changed: {item['path']}")
        if path.suffix == ".json":
            load_json(path)
        return errors
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return [f"Invalid artifact {item.get('path', '<missing path>')}: {exc}"]


def audit_archive(root: Path, manifest: dict, check_local: bool = False) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported or missing manifest schema_version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return errors + ["Manifest must contain archived artifacts"]
    ids = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            errors.append("Every artifact must have a nonempty string id")
            continue
        if item["id"] in ids:
            errors.append(f"Duplicate artifact id: {item['id']}")
        ids.add(item["id"])
        errors.extend(check_file(root, item))
    if check_local:
        for item in manifest.get("checkpoint_observations", []):
            errors.extend(check_file(root, item))
        snapshots = [
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("id") == "local_data_snapshot"
        ]
        for item in snapshots:
            try:
                snapshot = load_json(safe_path(root, item["path"]))
                for data_file in snapshot["files"]:
                    errors.extend(check_file(root, data_file))
            except (KeyError, ValueError, OSError) as exc:
                errors.append(f"Cannot check local data snapshot: {exc}")
    return errors


def preparation_blockers(root: Path, preparation: dict) -> list[str]:
    """Fail closed on unresolved gates; never starts a run or treats time as approval."""
    blockers = []
    if preparation.get("schema_version") != 1:
        blockers.append("Preparation schema_version is missing or unsupported")
    if preparation.get("paused_by_user") is not False:
        blockers.append("Training and experiments remain paused by the user")
    if preparation.get("status") != "ready_for_review":
        blockers.append("Preparation is draft; no experiment protocol has been frozen")
    required = (
        "literature_gate",
        "protocol",
        "dataset_manifest",
        "group_split_audit",
        "backbone_revision",
        "budget_definition",
        "human_judged_recommendation_set",
    )
    gates = preparation.get("gates", {})
    if not isinstance(gates, dict):
        return blockers + ["Preparation gates must be an object"]
    for name in required:
        gate = gates.get(name, {})
        if not isinstance(gate, dict) or gate.get("status") != "resolved":
            blockers.append(f"Unresolved gate: {name}")
            continue
        evidence = gate.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            blockers.append(f"Missing pinned evidence: {name}")
            continue
        for item in evidence:
            blockers.extend(f"{name}: {error}" for error in check_file(root, item))
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--check-local", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    try:
        manifest = load_json(safe_path(args.root, str(args.manifest)))
        errors = audit_archive(args.root, manifest, args.check_local)
        blockers = preparation_blockers(
            args.root, load_json(safe_path(args.root, str(args.preparation)))
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"archive_valid": False, "errors": [str(exc)]}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "archive_valid": not errors,
                "errors": errors,
                "local_observations_checked": args.check_local,
                "preparation_ready": not blockers,
                "preparation_blockers": blockers,
                "scope": "File integrity only; no models loaded, metrics recomputed, or experiments run.",
            },
            indent=2,
        )
    )
    if errors:
        return 1
    return 2 if args.require_ready and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
