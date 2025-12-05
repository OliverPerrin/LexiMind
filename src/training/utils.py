"""
Training utilities for LexiMind.

Provides reproducibility helpers including seed management for stdlib, PyTorch,
and NumPy random number generators with thread-safe spawning support.

Author: Oliver Perrin
Date: December 2025
"""

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
