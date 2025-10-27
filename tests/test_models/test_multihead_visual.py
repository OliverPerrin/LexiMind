# tests/test_models/test_multihead_visual.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.models.attention import MultiHeadAttention

def visualize_multihead_attention():
    """
    Visual test to see what different attention heads learn.
    Creates a heatmap showing attention patterns for each head.
    """
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
    output, attn_weights = mha(X, X, X)
    
    # attn_weights shape: (1, 8, 10, 10) = batch, heads, query_pos, key_pos
    attn_weights = attn_weights[0].detach().numpy()  # Remove batch dim: (8, 10, 10)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Multi-Head Attention: What Each Head Learns', fontsize=16, y=1.02)
    
    for head_idx in range(num_heads):
        row = head_idx // 4
        col = head_idx % 4
        ax = axes[row, col]
        
        # Plot attention heatmap for this head
        sns.heatmap(
            attn_weights[head_idx],
            annot=True,
            fmt='.2f',
            cmap='viridis',
            cbar=True,
            square=True,
            ax=ax,
            vmin=0,
            vmax=attn_weights[head_idx].max(),
            xticklabels=[f'K{i}' for i in range(seq_len)],
            yticklabels=[f'Q{i}' for i in range(seq_len)]
        )
        ax.set_title(f'Head {head_idx}', fontweight='bold')
        ax.set_xlabel('Keys (attend TO)')
        ax.set_ylabel('Queries (attending FROM)')
    
    plt.tight_layout()
    plt.savefig('outputs/multihead_attention_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to outputs/multihead_attention_visualization.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("Multi-Head Attention Analysis")
    print("="*60)
    
    for head_idx in range(num_heads):
        head_attn = attn_weights[head_idx]
        
        # Find dominant pattern
        diagonal_strength = np.trace(head_attn) / seq_len
        off_diagonal = (head_attn.sum() - np.trace(head_attn)) / (seq_len * (seq_len - 1))
        
        print(f"\nHead {head_idx}:")
        print(f"  Self-attention strength: {diagonal_strength:.3f}")
        print(f"  Cross-attention strength: {off_diagonal:.3f}")
        
        # Find which position each query attends to most
        max_attentions = head_attn.argmax(axis=1)
        print(f"  Attention pattern: {max_attentions.tolist()}")


def compare_single_vs_multihead():
    """
    Compare single-head vs multi-head attention capacity.
    """
    torch.manual_seed(42)
    seq_len, d_model = 8, 512
    
    # Create data with two different patterns
    # Pattern 1: Sequential (token i attends to i+1)
    # Pattern 2: Pairwise (tokens 0-1, 2-3, 4-5, 6-7 attend to each other)
    
    X = torch.randn(1, seq_len, d_model)
    
    # Test with 1 head vs 8 heads
    mha_1head = MultiHeadAttention(d_model, num_heads=1, dropout=0.0)
    mha_8heads = MultiHeadAttention(d_model, num_heads=8, dropout=0.0)
    
    mha_1head.eval()
    mha_8heads.eval()
    
    _, attn_1head = mha_1head(X, X, X)
    _, attn_8heads = mha_8heads(X, X, X)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Single head
    sns.heatmap(
        attn_1head[0, 0].detach().numpy(),
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar=True,
        square=True,
        ax=axes[0]
    )
    axes[0].set_title('Single-Head Attention\n(Limited expressiveness)', fontweight='bold')
    axes[0].set_xlabel('Keys')
    axes[0].set_ylabel('Queries')
    
    # Multi-head average
    avg_attn = attn_8heads[0].mean(dim=0).detach().numpy()
    sns.heatmap(
        avg_attn,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar=True,
        square=True,
        ax=axes[1]
    )
    axes[1].set_title('8-Head Attention (Average)\n(Richer patterns)', fontweight='bold')
    axes[1].set_xlabel('Keys')
    axes[1].set_ylabel('Queries')
    
    plt.tight_layout()
    plt.savefig('outputs/single_vs_multihead.png', dpi=150, bbox_inches='tight')
    print("✅ Saved comparison to outputs/single_vs_multihead.png")


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Visualizing multi-head attention patterns...")
    visualize_multihead_attention()
    
    print("\nComparing single-head vs multi-head...")
    compare_single_vs_multihead()
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("="*60)