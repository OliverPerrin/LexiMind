# Create a file: tests/test_models/test_attention_visual.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.attention import ScaledDotProductAttention

def test_attention_visualization():
    """Visual test to understand attention patterns."""
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
    output, weights = attention(Q, K, V)
    
    # Plot attention weights
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        weights[0].detach().numpy(), 
        annot=True, 
        fmt='.2f',
        cmap='viridis',
        xticklabels=[f'Key {i}' for i in range(seq_len)],
        yticklabels=[f'Query {i}' for i in range(seq_len)]
    )
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Keys (What we attend TO)')
    plt.ylabel('Queries (What is attending)')
    plt.tight_layout()
    plt.savefig('outputs/attention_visualization.png')
    print("✅ Saved visualization to outputs/attention_visualization.png")
    
    # Print some analysis
    print("\n" + "="*50)
    print("Attention Analysis")
    print("="*50)
    for i in range(seq_len):
        max_attn_idx = weights[0, i].argmax().item()
        max_attn_val = weights[0, i, max_attn_idx].item()
        print(f"Query {i} attends most to Key {max_attn_idx} (weight: {max_attn_val:.3f})")

if __name__ == "__main__":
    test_attention_visualization()