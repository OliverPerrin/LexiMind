import torch

from src.models.feedforward import FeedForward


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
