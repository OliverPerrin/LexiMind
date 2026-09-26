"""Process-scoped catalogue writer locks for POSIX and Windows."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import BinaryIO


@contextmanager
def _lock_file(handle: BinaryIO, platform: str) -> Iterator[None]:
    """Acquire before yielding; unsupported platforms or failures abort writes."""
    if platform == "posix":
        fcntl = import_module("fcntl")

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif platform == "nt":
        msvcrt = import_module("msvcrt")

        # locking() uses the current offset and allows a range beyond EOF.
        # Always lock byte zero, including when an old lock file is empty.
        # LK_LOCK retries contention ten times, then raises; never proceed unlocked.
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        raise RuntimeError(f"Catalogue writer locking is unsupported on {platform!r}")


@contextmanager
def advisory_lock(path: Path) -> Iterator[None]:
    """Release locks and close descriptors on success, errors, or interruption.

    The persistent lock file must not be unlinked: another process may already have
    opened that inode. Closing its handle also releases a lock if explicit unlock
    fails. Windows contention that outlasts the CRT retry window fails closed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        with _lock_file(handle, os.name):
            yield
