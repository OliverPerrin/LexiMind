"""Shared JSON and file-integrity checks for research source evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_json(raw: str | bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_finite_json)


def read_json(path: Path) -> Any:
    return parse_json(path.read_bytes())


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected repository-relative path: {relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return target


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(root: Path, item: dict) -> list[str]:
    """Check a pinned artifact without rewriting it or trusting declared readiness."""
    if not isinstance(item, dict):
        return ["Invalid artifact: expected an object"]
    try:
        path = safe_path(root, item["path"])
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("Artifact byte count must be a nonnegative integer")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Artifact requires a full lowercase SHA-256")
        if not path.is_file():
            return [f"Missing file: {item['path']}"]
        errors = []
        if path.stat().st_size != item["bytes"]:
            errors.append(f"Byte size changed: {item['path']}")
        if file_hash(path) != item["sha256"]:
            errors.append(f"SHA-256 changed: {item['path']}")
        if path.suffix == ".json":
            read_json(path)
        return errors
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return [f"Invalid artifact {item.get('path', '<missing path>')}: {exc}"]
