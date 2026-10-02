"""Generation bookkeeping parity on fixed logits, not a model-quality experiment."""

import itertools

import pytest
import torch

from src.models.decoder import TransformerDecoder


def reference(scores, *, eos, ngram, penalty, min_len, length_penalty, banned):
    histories = [[0] for _ in range(scores.shape[1])]
    finished = [False] * len(histories)
    max_len = scores.shape[0]
    for step in range(max_len - 1):
        for b, history in enumerate(histories):
            values = scores[step, b].tolist()
            if not finished[b]:
                for token in set(history):
                    values[token] = (
                        values[token] * penalty if values[token] < 0 else values[token] / penalty
                    )
            if eos is not None and len(history) < max(1, min_len):
                values[eos] = -float("inf")
            for token in banned:
                values[token] = -float("inf")
            if ngram and not finished[b]:
                prefix = tuple(history[-(ngram - 1) :]) if ngram > 1 else ()
                for start in range(len(history) - ngram + 1):
                    if tuple(history[start : start + ngram - 1]) == prefix:
                        values[history[start + ngram - 1]] = -float("inf")
            if length_penalty != 1 and eos is not None and len(history) >= min_len:
                values[eos] += length_penalty * len(history) / max_len
            token = eos if finished[b] else max(range(len(values)), key=values.__getitem__)
            history.append(token)
            finished[b] |= eos is not None and token == eos
        if eos is not None and all(finished) and len(histories[0]) >= max(1, min_len):
            break
    return torch.tensor(histories)


def test_cached_generation_constraints_match_independent_scalar_reference():
    scores = torch.randn(16, 4, 11, generator=torch.Generator().manual_seed(214))

    class Stub:
        def step(self, last, memory, cache):
            assert not torch.is_grad_enabled()
            n = cache["past_length"]
            return scores[n], {"past_length": n + 1}

    untouched = scores.clone()
    for eos, ngram, penalty, minimum, length_penalty in itertools.product(
        (None, 1), (0, 1, 2, 3), (1.0, 1.2), (0, 7), (1.0, 1.3)
    ):
        expected = reference(
            scores,
            eos=eos,
            ngram=ngram,
            penalty=penalty,
            min_len=minimum,
            length_penalty=length_penalty,
            banned=[4, 7],
        )
        actual = TransformerDecoder.greedy_decode(
            Stub(),
            torch.zeros(4, 1, 1),
            16,
            0,
            end_token_id=eos,
            no_repeat_ngram_size=ngram,
            repetition_penalty=penalty,
            min_len=minimum,
            length_penalty=length_penalty,
            ban_token_ids=[4, 7],
        )
        assert torch.equal(actual, expected)
        assert actual.is_contiguous()
    assert torch.equal(scores, untouched)
    assert torch.equal(
        TransformerDecoder.greedy_decode(Stub(), torch.zeros(4, 1, 1), 16, 0),
        reference(scores, eos=None, ngram=0, penalty=1, min_len=0, length_penalty=1, banned=[]),
    )


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_decode_refuses_invalid_capacity(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        TransformerDecoder.greedy_decode(object(), torch.zeros(1, 1, 1), maximum, 0)


def test_capacity_one_returns_only_start_token_without_a_decoder_step():
    assert TransformerDecoder.greedy_decode(object(), torch.zeros(3, 1, 1), 1, 4).tolist() == [
        [4],
        [4],
        [4],
    ]
