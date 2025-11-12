"""Small training helpers."""

from __future__ import annotations

import random
import threading
from typing import Optional

import numpy as np
import torch


_seed_sequence: Optional[np.random.SeedSequence] = None
_seed_lock = threading.Lock()
_spawn_counter = 0
_thread_local = threading.local()


def set_seed(seed: int) -> np.random.Generator:
    """Seed stdlib/Torch RNGs and initialise this thread's NumPy generator."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base_seq = np.random.SeedSequence(seed)
    child = base_seq.spawn(1)[0]
    rng = np.random.default_rng(child)

    global _seed_sequence, _spawn_counter
    with _seed_lock:
        _seed_sequence = base_seq
        _spawn_counter = 1
    _thread_local.rng = rng
    return rng


def numpy_generator() -> np.random.Generator:
    """Return the calling thread's NumPy generator, creating one if needed."""

    rng = getattr(_thread_local, "rng", None)
    if rng is not None:
        return rng

    global _seed_sequence, _spawn_counter
    with _seed_lock:
        if _seed_sequence is None:
            _seed_sequence = np.random.SeedSequence()
            _spawn_counter = 0
        child_seq = _seed_sequence.spawn(1)[0]
        _spawn_counter += 1

    rng = np.random.default_rng(child_seq)
    _thread_local.rng = rng
    return rng
