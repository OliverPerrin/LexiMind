"""
Tests for attention mechanisms.

Run with: pytest tests/test_models/test_attention.py -v
"""

import pytest
import torch
from src.models.attention import ScaledDotProductAttention


class TestScaledDotProductAttention:
    """Test suite for ScaledDotProductAttention."""
    
    def test_output_shape(self):
        """Test that output shapes are correct."""
        attention = ScaledDotProductAttention()
        batch_size, seq_len, d_k = 2, 10, 64
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights are a valid probability distribution."""
        attention = ScaledDotProductAttention()
        batch_size, seq_len, d_k = 2, 10, 64
        
        Q = K = V = torch.randn(batch_size, seq_len, d_k)
        _, weights = attention(Q, K, V)
        
        # Each row should sum to 1 (probability distribution over keys)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_masking(self):
        """Test that masking properly zeros out attention to masked positions."""
        attention = ScaledDotProductAttention()
        batch_size, seq_len, d_k = 1, 5, 64
        
        Q = K = V = torch.randn(batch_size, seq_len, d_k)
        
        # Create mask: only attend to first 3 positions
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :3] = True
        
        _, weights = attention(Q, K, V, mask)
        
        # Positions 3 and 4 should have zero attention weight
        assert torch.allclose(weights[:, :, 3:], torch.zeros(batch_size, seq_len, 2), atol=1e-6)
    
    # TODO: Add more tests as you understand the mechanism better


if __name__ == "__main__":
    pytest.main([__file__, "-v"])