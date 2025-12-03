"""Output cleaning helpers."""

from typing import List


def strip_whitespace(texts: List[str]) -> List[str]:
    return [text.strip() for text in texts]
