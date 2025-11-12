# tests/test_models/test_positional_encoding.py

"""
Tests for positional encoding.
"""

import os

import pytest
import torch
import matplotlib

matplotlib.use("Agg")  # use non-interactive backend for test environments
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    """Test suite for PositionalEncoding."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model, max_len = 512, 5000
        batch_size, seq_len = 2, 100
        
        pos_enc = PositionalEncoding(d_model, max_len, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        pos_enc = PositionalEncoding(d_model=256, max_len=1000, dropout=0.0)
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(1, seq_len, 256)
            output = pos_enc(x)
            assert output.shape == (1, seq_len, 256)
    
    def test_dropout_changes_output(self):
        """Test that dropout is applied during training."""
        torch.manual_seed(42)
        pos_enc = PositionalEncoding(d_model=128, dropout=0.5)
        pos_enc.train()
        
        x = torch.randn(2, 10, 128)
        
        output1 = pos_enc(x)
        output2 = pos_enc(x)
        
        # Should be different due to dropout
        assert not torch.allclose(output1, output2)
        
        # In eval mode, should be deterministic
        pos_enc.eval()
        output3 = pos_enc(x)
        output4 = pos_enc(x)
        assert torch.allclose(output3, output4)
    
    def test_encoding_properties(self):
        """Test mathematical properties of encoding."""
        pos_enc = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
        
        # Get the raw encoding (without dropout)
        pe = pos_enc.pe[0]  # Remove batch dimension
        
        # Each row should have values in [-1, 1] (sin/cos range)
        assert (pe >= -1).all() and (pe <= 1).all()
        
        # Different positions should have different encodings
        assert not torch.allclose(pe[0], pe[1])
        assert not torch.allclose(pe[0], pe[50])


def test_visualize_positional_encoding():
    """
    Visualize the positional encoding pattern.
    Creates heatmap showing encoding values.
    """
    pos_enc = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
    
    # Get encoding matrix
    pe = pos_enc.pe.squeeze(0).numpy()  # (max_len, d_model)
    
    # Plot first 50 positions and 64 dimensions
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pe[:50, :64].T,
        cmap='RdBu_r',
        center=0,
        xticklabels=5,
        yticklabels=8,
        cbar_kws={'label': 'Encoding Value'}
    )
    plt.xlabel('Position in Sequence')
    plt.ylabel('Embedding Dimension')
    plt.title('Positional Encoding Pattern\n(Notice the wave patterns with different frequencies)')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/positional_encoding_heatmap.png', dpi=150)
    print("✅ Saved to outputs/positional_encoding_heatmap.png")


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    test_visualize_positional_encoding()