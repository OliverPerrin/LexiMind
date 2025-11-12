import torch
import pytest
from src.models.encoder import TransformerEncoderLayer


def _take_tensor(output):
    """Return the tensor component regardless of (tensor, attn) tuple output."""

    if isinstance(output, tuple):  # modern layers return (output, attention)
        return output[0]
    return output


def test_output_shape_and_grad():
    """
    The encoder layer should preserve the input shape (batch, seq_len, d_model)
    and gradients should flow to parameters.
    """
    d_model, num_heads, d_ff = 64, 8, 256
    batch_size, seq_len = 2, 10

    layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    out = _take_tensor(layer(x))  # should accept mask=None by default
    assert out.shape == (batch_size, seq_len, d_model)

    # simple backward to ensure gradients propagate
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients found on any parameter"


def test_dropout_behavior_train_vs_eval():
    """
    With dropout > 0, the outputs should differ between two forward calls in train mode
    and be identical in eval mode.
    """
    torch.manual_seed(0)
    d_model, num_heads, d_ff = 64, 8, 256
    batch_size, seq_len = 2, 10

    layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.5)
    x = torch.randn(batch_size, seq_len, d_model)

    layer.train()
    out1 = _take_tensor(layer(x))
    out2 = _take_tensor(layer(x))
    # Training mode with dropout: outputs usually differ
    assert not torch.allclose(out1, out2), "Outputs identical in train mode despite dropout"

    layer.eval()
    out3 = _take_tensor(layer(x))
    out4 = _take_tensor(layer(x))
    # Eval mode deterministic: outputs should be identical
    assert torch.allclose(out3, out4), "Outputs differ in eval mode"


def test_mask_broadcasting_accepts_3d_and_4d_mask():
    """
    The encoder layer should accept a 3D mask (batch, seq_q, seq_k) and a 4D mask
    (batch, 1, seq_q, seq_k) and handle broadcasting across heads without error.
    """
    d_model, num_heads, d_ff = 64, 8, 256
    batch_size, seq_len = 2, 7

    layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model)

    # 3D mask: (batch, seq, seq)
    mask3 = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask3[:, :, -2:] = False  # mask out last two key positions
    out3 = _take_tensor(layer(x, mask=mask3))  # should not raise
    assert out3.shape == (batch_size, seq_len, d_model)

    # 4D mask: (batch, 1, seq, seq) already including head dim for broadcasting
    mask4 = mask3.unsqueeze(1)
    out4 = _take_tensor(layer(x, mask=mask4))
    assert out4.shape == (batch_size, seq_len, d_model)


if __name__ == "__main__":
    # Run tests interactively if needed
    pytest.main([__file__, "-q"])