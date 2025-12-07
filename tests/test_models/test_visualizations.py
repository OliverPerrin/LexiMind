import os

import matplotlib
import torch

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.attention import MultiHeadAttention, ScaledDotProductAttention
from src.models.positional_encoding import PositionalEncoding

OUTPUTS_DIR = "outputs"


def ensure_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def test_attention_visualization():
    """Visual test to understand attention patterns."""
    ensure_outputs_dir()
    attention = ScaledDotProductAttention()

    # Create a simple case: 5 tokens, each token attends most to itself
    batch_size = 1
    seq_len = 5
    d_k = 64

    # Create Q, K, V
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.eye(seq_len, d_k).unsqueeze(0)  # Identity-like

    # Compute attention
    _output, weights = attention(Q, K, V, return_attn_weights=True)

    # Plot attention weights
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        weights[0].detach().numpy(),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=[f"Key {i}" for i in range(seq_len)],
        yticklabels=[f"Query {i}" for i in range(seq_len)],
    )
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Keys (What we attend TO)")
    plt.ylabel("Queries (What is attending)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "attention_visualization.png")
    plt.savefig(save_path)
    print(f"✅ Saved visualization to {save_path}")
    plt.close()


def test_visualize_multihead_attention():
    """
    Visual test to see what different attention heads learn.
    Creates a heatmap showing attention patterns for each head.
    """
    ensure_outputs_dir()
    # Setup
    torch.manual_seed(42)
    d_model, num_heads = 512, 8
    batch_size, seq_len = 1, 10

    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    mha.eval()  # No dropout for visualization

    # Create input with some structure
    # Let's make tokens attend to nearby tokens
    X = torch.randn(batch_size, seq_len, d_model)

    # Add positional bias (tokens are more similar to nearby tokens)
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            X[0, i] += 0.5 * X[0, j] / (distance + 1)

    # Forward pass
    output, attn_weights = mha(X, X, X, return_attn_weights=True)

    # attn_weights shape: (1, 8, 10, 10) = batch, heads, query_pos, key_pos
    attn_weights = attn_weights[0].detach().numpy()  # Remove batch dim: (8, 10, 10)

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Multi-Head Attention: What Each Head Learns", fontsize=16, y=1.02)

    for head_idx in range(num_heads):
        row = head_idx // 4
        col = head_idx % 4
        ax = axes[row, col]

        # Plot attention heatmap for this head
        sns.heatmap(
            attn_weights[head_idx],
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            square=True,
            ax=ax,
            vmin=0,
            vmax=attn_weights[head_idx].max(),
            xticklabels=[f"K{i}" for i in range(seq_len)],
            yticklabels=[f"Q{i}" for i in range(seq_len)],
        )
        ax.set_title(f"Head {head_idx}", fontweight="bold")
        ax.set_xlabel("Keys (attend TO)")
        ax.set_ylabel("Queries (attending FROM)")

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "multihead_attention_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved visualization to {save_path}")
    plt.close()


def test_compare_single_vs_multihead():
    """
    Compare single-head vs multi-head attention capacity.
    """
    ensure_outputs_dir()
    torch.manual_seed(42)
    seq_len, d_model = 8, 512

    X = torch.randn(1, seq_len, d_model)

    # Test with 1 head vs 8 heads
    mha_1head = MultiHeadAttention(d_model, num_heads=1, dropout=0.0)
    mha_8heads = MultiHeadAttention(d_model, num_heads=8, dropout=0.0)

    mha_1head.eval()
    mha_8heads.eval()

    _, attn_1head = mha_1head(X, X, X, return_attn_weights=True)
    _, attn_8heads = mha_8heads(X, X, X, return_attn_weights=True)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Single head
    sns.heatmap(
        attn_1head[0, 0].detach().numpy(),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        square=True,
        ax=axes[0],
    )
    axes[0].set_title("Single-Head Attention\n(Limited expressiveness)", fontweight="bold")
    axes[0].set_xlabel("Keys")
    axes[0].set_ylabel("Queries")

    # Multi-head average
    avg_attn = attn_8heads[0].mean(dim=0).detach().numpy()
    sns.heatmap(avg_attn, annot=True, fmt=".2f", cmap="viridis", cbar=True, square=True, ax=axes[1])
    axes[1].set_title("8-Head Attention (Average)\n(Richer patterns)", fontweight="bold")
    axes[1].set_xlabel("Keys")
    axes[1].set_ylabel("Queries")

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "single_vs_multihead.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved comparison to {save_path}")
    plt.close()


def test_visualize_positional_encoding():
    """
    Visualize the positional encoding pattern.
    Creates heatmap showing encoding values.
    """
    ensure_outputs_dir()
    pos_enc = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)

    # Get encoding matrix
    pe = pos_enc.pe.squeeze(0).numpy()  # (max_len, d_model)

    # Plot first 50 positions and 64 dimensions
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pe[:50, :64].T,
        cmap="RdBu_r",
        center=0,
        xticklabels=5,
        yticklabels=8,
        cbar_kws={"label": "Encoding Value"},
    )
    plt.xlabel("Position in Sequence")
    plt.ylabel("Embedding Dimension")
    plt.title("Positional Encoding Pattern\n(Notice the wave patterns with different frequencies)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "positional_encoding_heatmap.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    test_attention_visualization()
    test_visualize_multihead_attention()
    test_compare_single_vs_multihead()
    test_visualize_positional_encoding()
