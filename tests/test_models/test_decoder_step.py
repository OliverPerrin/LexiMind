import torch
import pytest
from typing import Any, Dict, cast
from src.models.decoder import TransformerDecoder


def test_step_equivalence_with_greedy_decode():
    torch.manual_seed(7)
    vocab_size = 25
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    batch_size = 2
    src_len = 6
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

    memory = torch.randn(batch_size, src_len, d_model)

    # 1) Get greedy sequence from naive greedy_decode
    greedy = decoder.greedy_decode(memory, max_len=max_tgt, start_token_id=1, end_token_id=None)

    # 2) Reproduce the same sequence with step() using cache
    cache: Dict[str, Any] = {"past_length": 0}
    generated = torch.full((batch_size, 1), 1, dtype=torch.long)
    for _ in range(max_tgt - 1):
        last_token = generated[:, -1:].to(memory.device)
        logits, cache = decoder.step(cast(torch.LongTensor, last_token), memory, cache=cache)
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    # Compare shapes & that sequences are identical
    assert generated.shape == greedy.shape
    assert torch.equal(generated, greedy)


def test_step_cache_growth_and_shapes():
    torch.manual_seed(9)
    vocab_size = 20
    d_model = 24
    num_layers = 3
    num_heads = 4
    d_ff = 64
    batch_size = 1
    src_len = 5
    steps = 4
    max_tgt = 8

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

    memory = torch.randn(batch_size, src_len, d_model)

    cache: Dict[str, Any] = {"past_length": 0}
    last = torch.full((batch_size, 1), 1, dtype=torch.long)
    for step_idx in range(steps):
        logits, cache = decoder.step(cast(torch.LongTensor, last), memory, cache=cache)
        # check updated past_length
        assert cache["past_length"] == step_idx + 1
        # check cached per-layer keys exist and have expected shape (B, H, seq_len, d_k)
        for i in range(num_layers):
            k = cache.get(f"self_k_{i}")
            v = cache.get(f"self_v_{i}")
            assert k is not None and v is not None
            # seq_len should equal past_length
            assert k.shape[2] == cache["past_length"]
            # shapes match
            assert k.shape[0] == batch_size
            assert v.shape[0] == batch_size
        # advance last token for next loop
        last = logits.argmax(dim=-1, keepdim=True)

    # Also ensure memory projections cached
    for i in range(num_layers):
        assert f"mem_k_{i}" in cache and f"mem_v_{i}" in cache
        mem_k = cache[f"mem_k_{i}"]
        mem_v = cache[f"mem_v_{i}"]
        assert mem_k.shape[0] == batch_size
        assert mem_k.shape[2] == src_len  # seq length of memory