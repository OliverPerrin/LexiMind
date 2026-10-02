import copy

import pytest
import torch
from transformers.activations import ACT2FN

from src.models.feedforward import FeedForward


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flan_tanh_gelu_matches_upstream_outputs_and_gradients(dtype):
    native = FeedForward(8, 16, dropout=0.0, activation="gated-gelu-tanh").to(dtype)
    reference = copy.deepcopy(native)
    reference.activation = ACT2FN["gelu_new"]
    inputs = torch.linspace(-4, 4, 48, dtype=dtype).reshape(2, 3, 8).requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual, expected = native(inputs), reference(reference_inputs)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weights = torch.linspace(-1, 1, actual.numel(), dtype=dtype).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=tolerance, rtol=tolerance)
    for native_parameter, reference_parameter in zip(
        native.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            native_parameter.grad, reference_parameter.grad, atol=tolerance, rtol=tolerance
        )


def test_legacy_gated_gelu_preserves_exact_activation_and_state_layout():
    legacy = FeedForward(2, 4, dropout=0.0, activation="gated-gelu")
    upgraded = FeedForward(2, 4, dropout=0.0, activation="gated-gelu-tanh")
    upgraded.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.activation.approximate == "none"
    assert upgraded.activation.approximate == "tanh"
    values = torch.linspace(-3, 3, 31)
    torch.testing.assert_close(
        legacy.activation(values), torch.nn.functional.gelu(values), atol=0, rtol=0
    )
    assert not torch.equal(legacy.activation(values), upgraded.activation(values))


class TestFeedForward:
    def test_output_shape(self):
        d_model, d_ff = 512, 2048
        batch_size, seq_len = 2, 10

        ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        out = ffn(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_dropout_changes_output(self):
        torch.manual_seed(0)
        d_model, d_ff = 128, 512
        x = torch.randn(2, 8, d_model)

        ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.5)
        ffn.train()
        out1 = ffn(x)
        out2 = ffn(x)
        # With dropout in train mode, outputs should differ (most likely)
        assert not torch.allclose(out1, out2)

        ffn.eval()
        out3 = ffn(x)
        out4 = ffn(x)
        # In eval mode (no dropout), outputs should be identical for same input
        assert torch.allclose(out3, out4)

    def test_parameter_count_and_grads(self):
        d_model, d_ff = 64, 256
        ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)

        # Parameter existence
        param_names = [name for name, _ in ffn.named_parameters()]
        assert any("linear1" in name for name in param_names)
        assert any("linear2" in name for name in param_names)

        # Parameter shapes
        shapes = {name: p.shape for name, p in ffn.named_parameters()}
        assert shapes.get("linear1.weight") == (d_ff, d_model)
        assert shapes.get("linear2.weight") == (d_model, d_ff)
        assert shapes.get("linear1.bias") == (d_ff,)
        assert shapes.get("linear2.bias") == (d_model,)

        # ensure gradients flow
        x = torch.randn(3, 5, d_model)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        for _, p in ffn.named_parameters():
            assert p.grad is not None
