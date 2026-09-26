"""Atomic single-file writes and rollback for catalogue/manifest publication.

POSIX cannot atomically replace two independent paths. Stage both files before
publishing, restore the previous files on ordinary failures, and require readers
to verify the manifest hash to detect a crash between the two replacements.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .locking import advisory_lock


def _stage(path: Path, chunks: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        staged.chmod(mode)
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, chunks: Iterable[str]) -> None:
    """Never expose partial JSON/JSONL, even when serialization fails midway."""
    staged = _stage(path, chunks)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, [json_text(value)])


def publish_catalogue(
    catalogue_path: Path,
    catalogue: list[dict[str, Any]],
    manifest_path: Path,
    manifest: dict[str, Any],
    receipt_path: Path | None = None,
) -> None:
    """Stage all artifacts, publish data last, and roll back failed replacements."""
    from .openlibrary import validate_catalogue

    validate_catalogue(catalogue, manifest)
    artifacts: list[tuple[Path, Any]] = [(manifest_path, manifest)]
    if receipt_path is not None:
        artifacts.append(
            (
                receipt_path,
                {
                    "schemaVersion": 1,
                    "count": len(catalogue),
                    "catalogueFileSha256": hashlib.sha256(
                        json_text(catalogue).encode()
                    ).hexdigest(),
                    "sourceManifestSha256": hashlib.sha256(
                        json_text(manifest).encode()
                    ).hexdigest(),
                },
            )
        )
    artifacts.append((catalogue_path, catalogue))
    if len({path.resolve() for path, _ in artifacts}) != len(artifacts):
        raise ValueError("Catalogue, manifest, and receipt paths must differ")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    with advisory_lock(lock_path):
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path | None] = {}
        replaced: list[Path] = []
        recovery_files: set[Path] = set()
        try:
            for path, value in artifacts:
                staged[path] = _stage(path, [json_text(value)])
                backups[path] = (
                    _stage(path, [path.read_bytes().decode("utf-8")]) if path.exists() else None
                )
            for path, temporary in staged.items():
                os.replace(temporary, path)
                replaced.append(path)
        except BaseException as publication_error:
            rollback_errors = []
            for path in reversed(replaced):
                backup = backups[path]
                try:
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
                    if backup is not None:
                        recovery_files.add(backup)
            if rollback_errors:
                raise RuntimeError(
                    f"Catalogue publication and rollback failed; recovery files retained: {sorted(str(path) for path in recovery_files)}"
                ) from publication_error
            raise
        finally:
            for cleanup_path in [*staged.values(), *backups.values()]:
                if cleanup_path is not None and cleanup_path not in recovery_files:
                    cleanup_path.unlink(missing_ok=True)
