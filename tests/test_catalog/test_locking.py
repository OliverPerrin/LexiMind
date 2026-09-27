"""Controlled backend contracts; Windows behavior is not natively exercised."""

import os
import sys
from types import SimpleNamespace

import pytest

from src.catalog.locking import _lock_file, advisory_lock


def test_posix_lock_releases_after_interruption_and_closes_descriptor(tmp_path, monkeypatch):
    import src.catalog.locking as locking

    calls = []
    original = locking._lock_file
    monkeypatch.setattr(locking, "_lock_file", lambda handle, platform: original(handle, "posix"))
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(
            LOCK_EX=1, LOCK_UN=2, flock=lambda descriptor, mode: calls.append((descriptor, mode))
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        with advisory_lock(tmp_path / "writer.lock"):
            raise KeyboardInterrupt
    assert [mode for _, mode in calls] == [1, 2]
    with pytest.raises(OSError):
        os.fstat(calls[0][0])


def test_windows_backend_locks_and_unlocks_same_byte_after_body_failure(tmp_path, monkeypatch):
    calls = []

    def record(descriptor, mode, size):
        calls.append((mode, size, os.lseek(descriptor, 0, os.SEEK_CUR)))

    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=record)
    )
    with (tmp_path / "writer.lock").open("a+b") as handle:
        with pytest.raises(ValueError, match="body failed"):
            with _lock_file(handle, "nt"):
                handle.seek(20)
                raise ValueError("body failed")
    assert calls == [(1, 1, 0), (2, 1, 0)]


@pytest.mark.parametrize(
    "platform,module,acquire",
    [
        ("posix", "fcntl", "flock"),
        ("nt", "msvcrt", "locking"),
    ],
)
def test_backend_lock_failure_never_enters_writer(tmp_path, monkeypatch, platform, module, acquire):
    calls = []

    def fail(*args):
        calls.append(args)
        raise OSError("lock unavailable")

    backend = SimpleNamespace(LOCK_EX=1, LOCK_UN=2, LK_LOCK=1, LK_UNLCK=2)
    setattr(backend, acquire, fail)
    monkeypatch.setitem(sys.modules, module, backend)
    with (tmp_path / "writer.lock").open("a+b") as handle:
        with pytest.raises(OSError, match="lock unavailable"):
            with _lock_file(handle, platform):
                pytest.fail("Writer entered without its lock")
    assert len(calls) == 1  # No unlock for an acquisition that failed.


def test_unsupported_backend_fails_closed(tmp_path):
    with (tmp_path / "writer.lock").open("a+b") as handle:
        with pytest.raises(RuntimeError, match="unsupported"):
            with _lock_file(handle, "unsupported"):
                pytest.fail("Writer entered without a supported lock backend")
