"""
Randomness utilities for LexiMind.

Provides seed management for reproducibility.

Author: Oliver Perrin
Date: December 2025
"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
