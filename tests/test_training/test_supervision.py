"""Protect the format comparison and its exact source-word verification boundary."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

from src.research.io import read_json
from src.training.denoising import INSTRUCTION, _section_windows, denoising_outcomes
from src.training.supervision import (
    format_row,
    supervision_outcomes,
    validate_supervision_config,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tokenizer():
    value = Tokenizer.from_file(str(ROOT / "artifacts/hf_tokenizer/tokenizer.json"))
    value.no_padding()
    value.no_truncation()
    return value


@pytest.fixture
def row(tokenizer):
    value = _section_windows(
        "Happy children visited the forest beside quiet gardens while gentle rabbits watched ancient wooden fences.",
        tokenizer,
    )
    selected = next(r for r in value if r["target_text"] == "forest")
    return {**selected, "record_id": "fixed-example", "source_evidence": {"fixture": True}}


def generated(tokens, mask=None):
    return SimpleNamespace(
        response_ids=torch.tensor([tokens]),
        response_mask=torch.tensor([mask if mask is not None else [True] * len(tokens)]),
    )


def test_formats_preserve_source_target_and_do_not_mutate_original(row, tokenizer):
    original = deepcopy(row)
    word = format_row(row, "instruction_word", tokenizer)
    span = format_row(row, "t5_span", tokenizer)
    assert row == original
    assert word["input_ids"] == row["input_ids"]
    assert word["labels"] == row["labels"]
    assert span["input_text"] == row["input_text"].removeprefix(INSTRUCTION)
    assert span["input_ids"] == tokenizer.encode(
        span["input_text"], add_special_tokens=False
    ).ids + [1]
    assert span["labels"] == [32099] + row["target_ids"] + [32098, 1]
    for variant in (word, span):
        assert variant["record_id"] == row["record_id"]
        assert variant["source_evidence"] == row["source_evidence"]
        assert variant["target_ids"] == row["target_ids"]
        assert [variant["labels"][i] for i in variant["content_positions"]] == row["target_ids"]
    assert word["control_positions"] == []
    assert [span["labels"][i] for i in span["control_positions"]] == [32099, 32098]
    with pytest.raises(ValueError):
        format_row(row, "invented", tokenizer)


def test_word_verifier_preserves_previous_contract(row, tokenizer):
    for tokens in (row["labels"], [1], [32099] + row["labels"], [32127, 1]):
        responses = generated(tokens)
        assert supervision_outcomes([row], responses, tokenizer, "instruction_word") == (
            denoising_outcomes([row], responses, tokenizer)
        )


def test_span_exact_answer_requires_wrappers_and_terminal_eos(row, tokenizer):
    tokens = [32099] + row["target_ids"] + [32098, 1]
    result = supervision_outcomes([row], generated(tokens), tokenizer, "t5_span")[0]
    assert result["correct"] and result["reward"] == 1
    assert not result["invalid"] and not result["truncated"]
    assert result["content_tokens"] == len(row["target_ids"])
    padded = tokens + [0, 32127]
    result = supervision_outcomes(
        [row], generated(padded, [True] * len(tokens) + [False, False]), tokenizer, "t5_span"
    )[0]
    assert result["correct"]
    wrong = [32099] + tokenizer.encode("Forest", add_special_tokens=False).ids + [32098, 1]
    assert not supervision_outcomes([row], generated(wrong), tokenizer, "t5_span")[0]["correct"]


@pytest.mark.parametrize(
    "tokens",
    [
        [1],
        [32099, 32098, 1],
        [32099, 0, 32098, 1],
        [32099, 2, 32098, 1],
        [32099, 32127, 32098, 1],
        [32099, -1, 32098, 1],
        [32099, 999999, 32098, 1],
    ],
)
def test_empty_or_invisible_span_content_never_earns_reward(row, tokenizer, tokens):
    result = supervision_outcomes([row], generated(tokens), tokenizer, "t5_span")[0]
    assert result["invalid"] and result["reward"] == 0


@pytest.mark.parametrize(
    "malformation",
    [
        "no_open",
        "no_close",
        "extra_open",
        "extra_close",
        "extra_eos",
        "no_eos",
        "after_eos",
        "noncanonical",
    ],
)
def test_malformed_native_outputs_never_earn_reward(row, tokenizer, malformation):
    word = row["target_ids"]
    variants = {
        "no_open": word + [32098, 1],
        "no_close": [32099] + word + [1],
        "extra_open": [32099, 32099] + word + [32098, 1],
        "extra_close": [32099] + word + [32098, 32098, 1],
        "extra_eos": [32099] + word + [32098, 1, 1],
        "no_eos": [32099] + word + [32098],
        "after_eos": [32099] + word + [32098, 1] + word,
        "noncanonical": [32099, 3] + word + [32098, 1],
    }
    result = supervision_outcomes([row], generated(variants[malformation]), tokenizer, "t5_span")[0]
    assert not result["correct"] and result["reward"] == 0
    if malformation == "no_eos":
        assert result["truncated"]


def test_span_masks_are_boolean_contiguous_prefixes(row, tokenizer):
    tokens = [32099] + row["target_ids"] + [32098, 1]
    with pytest.raises(ValueError, match="boolean"):
        supervision_outcomes([row], generated(tokens, [1] * len(tokens)), tokenizer, "t5_span")
    mask = [True] * len(tokens)
    mask[1] = False
    with pytest.raises(ValueError, match="contiguous"):
        supervision_outcomes([row], generated(tokens, mask), tokenizer, "t5_span")
    with pytest.raises(ValueError):
        supervision_outcomes([row], generated(tokens, [True]), tokenizer, "t5_span")


@pytest.mark.parametrize(
    "mutation",
    [
        {"arms": ["instruction_word"]},
        {"arms": ["t5_span", "instruction_word"]},
        {"seeds": [17]},
        {"checkpoints": [128, 1024]},
        {"group_size": 8},
        {"coverage_prompts": 64},
        {"response_cap": 16},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.001},
        {"temperature": 0.5},
        {"max_total_seconds": 3600},
        {"promote": True},
        {"paid_spend_authorized": True},
    ],
)
def test_fixed_experiment_rejects_undeclared_changes(mutation):
    config = read_json(ROOT / "configs/research/book_supervision.json")
    validate_supervision_config(config)
    with pytest.raises(ValueError):
        validate_supervision_config({**config, **mutation})
