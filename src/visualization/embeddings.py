"""Embedding visualization helpers."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(embeddings: np.ndarray, labels: list[str]) -> None:
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not labels:
        raise ValueError("labels must be a non-empty list")
    if embeddings.shape[0] != len(labels):
        raise ValueError("number of samples in embeddings must equal length of labels")
    if embeddings.shape[1] < 2:
        raise ValueError("embeddings must have at least 2 features for t-SNE visualization")
    
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto")
    projection = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": projection[:, 0],
        "y": projection[:, 1],
        "label": labels,
    })
    plt.figure()
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=50)
    plt.legend(title="Labels", loc="best")
    plt.tight_layout()
    plt.show()
