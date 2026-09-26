"""Atomic local artifact replacement without partial destination writes."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def atomic_write(path: str | Path, writer: Callable[[BinaryIO], object]) -> None:
    """Serialize into a sibling file, then replace only after a successful flush."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
        temporary.chmod(mode)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
