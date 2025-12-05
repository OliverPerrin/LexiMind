"""
Output postprocessing utilities for LexiMind.

Provides text cleaning helpers for model outputs.

Author: Oliver Perrin
Date: December 2025
"""

from typing import List


def strip_whitespace(texts: List[str]) -> List[str]:
    return [text.strip() for text in texts]
