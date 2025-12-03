"""Attention plotting utilities."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_attention(matrix: np.ndarray, tokens: Sequence[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError("Attention matrix must be 2-dimensional")
    token_count = len(tokens)
    if token_count == 0:
        raise ValueError("tokens must contain at least one item")
    if matrix.shape != (token_count, token_count):
        raise ValueError(
            f"Attention matrix shape {matrix.shape} must match (len(tokens), len(tokens)) = ({token_count}, {token_count})"
        )

    fig, ax = plt.subplots()
    heatmap = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(token_count))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(token_count))
    ax.set_yticklabels(tokens)
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Attention Weight")
    fig.tight_layout()
    plt.show()
