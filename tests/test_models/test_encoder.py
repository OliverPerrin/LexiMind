import math
import torch
import pytest
from src.models.encoder import TransformerEncoder


def test_encoder_token_ids_and_padding_mask_and_grad():
    """
    Test using token ids as input, automatic padding mask creation when pad_token_id is provided,
    output shape, and that gradients flow through the model.
    """
    torch.manual_seed(0)
    vocab_size = 50
    pad_token_id = 0
    d_model = 64
    num_layers = 3
    num_heads = 8
    d_ff = 128
    batch_size = 2
    seq_len = 12

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        max_len=seq_len,
        pad_token_id=pad_token_id,
    )

    # create inputs with some padding at the end
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    input_ids[0, -3:] = pad_token_id  # first sample has last 3 tokens as padding
    input_ids[1, -1:] = pad_token_id  # second sample has last token as padding

    # Forward pass (token ids)
    out = encoder(input_ids)  # default collect_attn=False
    assert out.shape == (batch_size, seq_len, d_model)

    # Check gradients flow
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients found on any parameter"


def test_encoder_embeddings_input_and_collect_attn():
    """
    Test passing pre-computed embeddings to the encoder, collecting attention weights,
    and verify shapes of attention maps per layer.
    """
    torch.manual_seed(1)
    vocab_size = 100  # not used in this test
    d_model = 48
    num_layers = 4
    num_heads = 6
    d_ff = 128
    batch_size = 1
    seq_len = 10

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=seq_len,
        pad_token_id=None,
    )

    # Create random embeddings directly
    embeddings = torch.randn(batch_size, seq_len, d_model)

    out, attn_list = encoder(embeddings, mask=None, collect_attn=True)
    assert out.shape == (batch_size, seq_len, d_model)
    assert isinstance(attn_list, list)
    assert len(attn_list) == num_layers

    # Each attention weight tensor should have shape (batch, num_heads, seq, seq)
    for attn in attn_list:
        assert attn.shape == (batch_size, num_heads, seq_len, seq_len)


def test_mask_accepts_3d_and_4d_and_broadcasts():
    """
    Test that a provided 3D mask (batch, seq, seq) and an equivalent 4D mask
    (batch, 1, seq, seq) produce outputs of the same shape and do not error.
    """
    torch.manual_seed(2)
    vocab_size = 40
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    batch_size = 2
    seq_len = 7

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=seq_len,
        pad_token_id=None,
    )

    # Create dummy embeddings
    embeddings = torch.randn(batch_size, seq_len, d_model)

    # 3D mask: True indicates allowed attention
    mask3 = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask3[:, :, -2:] = False  # mask out last two keys

    # 4D mask equivalent
    mask4 = mask3.unsqueeze(1)  # (B, 1, S, S)

    out3 = encoder(embeddings, mask=mask3)
    out4 = encoder(embeddings, mask=mask4)

    assert out3.shape == (batch_size, seq_len, d_model)
    assert out4.shape == (batch_size, seq_len, d_model)
    # Outputs should be finite and not NaN
    assert torch.isfinite(out3).all()
    assert torch.isfinite(out4).all()


def test_train_eval_determinism_and_dropout_effect():
    """
    Validate that in train mode with dropout enabled, repeated forwards differ,
    and in eval mode they are equal (deterministic).
    """
    torch.manual_seed(3)
    vocab_size = 60
    pad_token_id = 0
    d_model = 64
    num_layers = 2
    num_heads = 8
    d_ff = 128
    batch_size = 2
    seq_len = 9

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.4,
        max_len=seq_len,
        pad_token_id=pad_token_id,
    )

    # token ids with occasional padding
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    input_ids[0, -2:] = pad_token_id

    # Training mode: randomness due to dropout -> outputs should likely differ
    encoder.train()
    out1 = encoder(input_ids)
    out2 = encoder(input_ids)
    assert not torch.allclose(out1, out2), "Outputs identical in train mode despite dropout"

    # Eval mode: deterministic
    encoder.eval()
    out3 = encoder(input_ids)
    out4 = encoder(input_ids)
    assert torch.allclose(out3, out4), "Outputs differ in eval mode"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])