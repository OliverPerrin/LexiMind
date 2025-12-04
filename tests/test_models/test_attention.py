"""
Tests for attention mechanisms.

Run with: pytest tests/test_models/test_attention.py -v
"""

import pytest
import torch

from src.models.attention import MultiHeadAttention, ScaledDotProductAttention


class TestScaledDotProductAttention:
    """Test suite for ScaledDotProductAttention.

    Note: ScaledDotProductAttention expects 4D inputs: (batch, num_heads, seq, d_k)
    """

    def test_output_shape(self):
        """Test that output shapes are correct."""
        attention = ScaledDotProductAttention()
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64

        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        output, weights = attention(Q, K, V, return_attn_weights=True)

        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights are a valid probability distribution."""
        attention = ScaledDotProductAttention()
        batch_size, num_heads, seq_len, d_k = 2, 4, 10, 64

        Q = K = V = torch.randn(batch_size, num_heads, seq_len, d_k)
        _, weights = attention(Q, K, V, return_attn_weights=True)

        # Each row should sum to 1 (probability distribution over keys)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size, num_heads, seq_len), atol=1e-6)

    def test_masking(self):
        """Test that masking properly zeros out attention to masked positions."""
        attention = ScaledDotProductAttention()
        batch_size, num_heads, seq_len, d_k = 1, 4, 5, 64

        Q = K = V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Create mask: only attend to first 3 positions (4D mask)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, :3] = True  # Attend to first 3 key positions

        _, weights = attention(Q, K, V, mask, return_attn_weights=True)

        # Key positions 3 and 4 should have zero attention weight
        assert torch.allclose(
            weights[:, :, :, 3:], torch.zeros(batch_size, num_heads, seq_len, 2), atol=1e-6
        )

    # TODO: Add more tests as you understand the mechanism better


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        d_model, num_heads = 512, 8
        batch_size, seq_len = 2, 10

        mha = MultiHeadAttention(d_model, num_heads)

        Q = K = V = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = mha(Q, K, V, return_attn_weights=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_different_qkv(self):
        """Test with different Q, K, V (cross-attention scenario)."""
        d_model, num_heads = 512, 8
        batch_size = 2
        seq_len_q, seq_len_kv = 10, 20

        mha = MultiHeadAttention(d_model, num_heads)

        Q = torch.randn(batch_size, seq_len_q, d_model)
        K = torch.randn(batch_size, seq_len_kv, d_model)
        V = torch.randn(batch_size, seq_len_kv, d_model)

        output, attn_weights = mha(Q, K, V, return_attn_weights=True)

        # Output has same length as query
        assert output.shape == (batch_size, seq_len_q, d_model)
        # Attention is query_len x key_len
        assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)

    def test_masking(self):
        """Test that masking works correctly."""
        d_model, num_heads = 512, 8
        batch_size, seq_len = 2, 5

        mha = MultiHeadAttention(d_model, num_heads)
        Q = K = V = torch.randn(batch_size, seq_len, d_model)

        # Mask out last 2 positions
        mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, -2:] = False

        _, attn_weights = mha(Q, K, V, mask, return_attn_weights=True)

        # Last 2 positions should have near-zero attention
        assert torch.allclose(
            attn_weights[:, :, :, -2:], torch.zeros(batch_size, num_heads, seq_len, 2), atol=1e-6
        )

    def test_parameters_exist(self):
        """Test that learnable parameters are created."""
        mha = MultiHeadAttention(512, 8)

        # Should have 4 linear layers worth of parameters
        param_names = [name for name, _ in mha.named_parameters()]

        assert any("W_Q" in name or "q_linear" in name.lower() for name in param_names)
        assert any("W_K" in name or "k_linear" in name.lower() for name in param_names)
        assert any("W_V" in name or "v_linear" in name.lower() for name in param_names)
        assert any("W_O" in name or "out" in name.lower() for name in param_names)

    def test_dropout_changes_output(self):
        """Test that dropout is actually applied during training."""
        torch.manual_seed(42)
        mha = MultiHeadAttention(512, 8, dropout=0.5)
        mha.train()  # Enable training mode

        Q = K = V = torch.randn(2, 10, 512)

        # Run twice with same input - should get different outputs due to dropout
        output1, _ = mha(Q, K, V)
        output2, _ = mha(Q, K, V)

        assert not torch.allclose(output1, output2)

        # In eval mode, should be deterministic
        mha.eval()
        output3, _ = mha(Q, K, V)
        output4, _ = mha(Q, K, V)

        assert torch.allclose(output3, output4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
