from typing import Any, Dict, cast

import torch

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
        assert mem_k.shape[0] == batch_size
        assert mem_k.shape[2] == src_len  # seq length of memory


def test_finished_batch_rows_do_not_generate_after_eos():
    decoder = TransformerDecoder(
        vocab_size=6,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
        max_len=6,
        pad_token_id=0,
    )
    memory = torch.zeros(2, 2, 8)

    def step(last, memory, cache):
        position = cache.get("past_length", 0)
        logits = torch.zeros(2, 6)
        logits[0, 1 if position == 0 else 5] = 10
        logits[1, 1 if position == 2 else 3] = 10
        return logits, {"past_length": position + 1}

    decoder.step = step
    generated = decoder.greedy_decode(memory, max_len=6, start_token_id=0, end_token_id=1)
    assert generated.tolist() == [[0, 1, 1, 1], [0, 3, 3, 1]]


def test_unigram_repeat_constraint_blocks_all_previous_tokens():
    decoder = TransformerDecoder(
        vocab_size=5,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
        max_len=6,
        pad_token_id=0,
    )
    memory = torch.zeros(1, 2, 8)
    decoder.step = lambda last, memory, cache: (torch.tensor([[0.0, 1.0, 3.0, 4.0, 2.0]]), cache)
    generated = decoder.greedy_decode(memory, max_len=4, start_token_id=0, no_repeat_ngram_size=1)
    assert generated.tolist() == [[0, 3, 2, 4]]
