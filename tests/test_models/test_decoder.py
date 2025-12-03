import pytest
import torch

from src.models.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    create_causal_mask,
)


def test_create_causal_mask_properties():
    mask = create_causal_mask(5)
    assert mask.shape == (5, 5)
    # diagonal and below should be True
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert mask[i, j].item() is True
            else:
                assert mask[i, j].item() is False


def test_decoder_layer_shapes_and_grad():
    torch.manual_seed(0)
    d_model, num_heads, d_ff = 32, 4, 64
    batch_size, tgt_len, src_len = 2, 6, 7

    layer = TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    tgt = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
    memory = torch.randn(batch_size, src_len, d_model)

    # No masks
    out, attn = layer(tgt, memory, tgt_mask=None, memory_mask=None, collect_attn=True)
    assert out.shape == (batch_size, tgt_len, d_model)
    assert isinstance(attn, dict)
    assert "self" in attn and "cross" in attn
    assert attn["self"].shape == (batch_size, num_heads, tgt_len, tgt_len)
    assert attn["cross"].shape == (batch_size, num_heads, tgt_len, src_len)

    # Backprop works
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_decoder_layer_causal_mask_blocks_future():
    torch.manual_seed(1)
    d_model, num_heads, d_ff = 48, 6, 128
    batch_size, tgt_len, src_len = 1, 5, 5

    layer = TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    # create trivial increasing tgt embeddings so attention patterns are deterministic-ish
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    causal = create_causal_mask(tgt_len, device=tgt.device)  # (T, T)
    tgt_mask = causal.unsqueeze(0)  # (1, T, T) -> layer will handle unsqueeze to heads

    out, attn = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=None, collect_attn=True)
    self_attn = attn["self"].detach()
    # Ensure upper triangle of attention weights is zero (no future attention)
    # For each head and query i, keys j>i should be zero
    B, H, Tq, Tk = self_attn.shape
    for i in range(Tq):
        for j in range(i + 1, Tk):
            assert torch.allclose(self_attn[:, :, i, j], torch.zeros(B, H)), (
                f"Found nonzero attention to future position {j} from query {i}"
            )


def test_decoder_stack_and_greedy_decode_shapes():
    torch.manual_seed(2)
    vocab_size = 30
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 128
    batch_size = 2
    src_len = 7
    max_tgt = 6

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=max_tgt,
        pad_token_id=0,
    )

    # Random memory from encoder
    memory = torch.randn(batch_size, src_len, d_model)

    # Greedy decode: should produce (B, <= max_tgt)
    generated = decoder.greedy_decode(memory, max_len=max_tgt, start_token_id=1, end_token_id=None)
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= max_tgt
    assert (generated[:, 0] == 1).all()  # starts with start token

    # Also test forward with embeddings and collect_attn
    embeddings = torch.randn(batch_size, max_tgt, d_model)
    logits, attn_list = decoder(embeddings, memory, collect_attn=True)
    assert logits.shape == (batch_size, max_tgt, vocab_size)
    assert isinstance(attn_list, list)
    assert len(attn_list) == num_layers
    for attn in attn_list:
        assert "self" in attn and "cross" in attn


def test_decoder_train_eval_dropout_behavior():
    torch.manual_seed(3)
    vocab_size = 40
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 128
    batch_size = 2
    src_len = 6
    tgt_len = 5

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.4,
        max_len=tgt_len,
        pad_token_id=0,
    )

    # token ids with padding possible
    input_ids = torch.randint(1, vocab_size, (batch_size, tgt_len), dtype=torch.long)
    input_ids[0, -1] = 0

    memory = torch.randn(batch_size, src_len, d_model)

    decoder.train()
    out1 = decoder(input_ids, memory)
    out2 = decoder(input_ids, memory)
    # With dropout in train mode, outputs should usually differ
    assert not torch.allclose(out1, out2)

    decoder.eval()
    out3 = decoder(input_ids, memory)
    out4 = decoder(input_ids, memory)
    assert torch.allclose(out3, out4)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
