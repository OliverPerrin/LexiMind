"""Small shared helpers for immutable local source candidates; no model imports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

from .io import file_hash as file_hash


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def create_or_verify(path: Path, chunks: Iterable[bytes]) -> dict[str, Any]:
    """Publish a completed file once, or verify it without overwriting old bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        expected = digest.hexdigest()
        try:
            # A hard link publishes atomically without replacing a file that a
            # concurrent writer created after the existence check.
            os.link(temporary, path)
        except FileExistsError:
            if file_hash(path) != expected:
                raise ValueError(
                    f"Existing candidate artifact differs; use a separately reviewed location: {path.name}"
                ) from None
        return {"bytes": size, "sha256": expected}
    finally:
        temporary.unlink(missing_ok=True)


def parquet_rows(path: Path, columns: tuple[str, ...]) -> Iterable[dict[str, Any]]:
    try:
        parquet = import_module("pyarrow.parquet")
    except ImportError as error:
        raise RuntimeError(
            "Parquet decoding needs PyArrow only, without a datasets/model stack"
        ) from error
    source = parquet.ParquetFile(path)
    if set(source.schema_arrow.names) != set(columns):
        raise ValueError("Parquet columns differ from the pinned source schema")
    for batch in source.iter_batches(batch_size=1024, columns=list(columns)):
        yield from batch.to_pylist()


def helper_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in (root / "src/research/candidate_io.py", root / "src/research/io.py")
    }
