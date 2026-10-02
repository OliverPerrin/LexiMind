"""Native adapter/merge contracts on tiny synthetic modules, without model artifacts."""

import io
import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from src.models.adapters import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    extract_effective_delta,
    materialize_merge,
    merge_deltas,
)
from src.models.attention import MultiHeadAttention
from src.models.decoder import TransformerDecoder
from src.models.heads import ClassificationHead

SHARED = ("encoder.self_attn.W_Q",)
PRIVATE = ("decoder.layers.0.self_attn.W_V", "decoder.layers.0.cross_attn.W_Q")
CONFIG = LoRAConfig(rank=2, alpha=4, seed=19)


class TinyBase(nn.Module):
    def __init__(self, relative=False):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embedding = nn.Embedding(13, 4)
        self.encoder.norm = nn.LayerNorm(4)
        self.encoder.self_attn = MultiHeadAttention(4, 2, dropout=0)
        self.decoder = TransformerDecoder(
            13, 4, 1, 2, 8, dropout=0, max_len=16, use_relative_position_bias=relative
        )
        self.decoder.output_projection.weight = self.decoder.embedding.weight  # Safe frozen tie.
        self.head_topic = ClassificationHead(4, 2, pooler="attention", dropout=0)
        self.head_emotion = ClassificationHead(4, 3, dropout=0, problem_type="multi_label")


def base(relative=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return TinyBase(relative).eval()


def specialist(pristine, task="topic", *, shared=SHARED, private=(), head=True):
    model = deepcopy(pristine)
    binding = attach_lora(
        model,
        shared_projections=shared,
        private_projections=private,
        private_heads=("head_" + task,) if head else (),
        config=CONFIG,
    )
    return model, binding


def test_zero_delta_init_preserves_outputs_keys_rng_and_freezes_the_base():
    model = base()
    x = torch.arange(12).reshape(1, 3, 4).float() / 10
    expected = model.encoder.self_attn(x, x, x)[0]
    keys = set(model.state_dict())
    state = torch.get_rng_state().clone()
    binding = attach_lora(
        model, shared_projections=SHARED, private_heads=("head_topic",), config=CONFIG
    )
    assert torch.equal(state, torch.get_rng_state())
    assert set(model.state_dict()) - keys == {SHARED[0] + ".lora_A", SHARED[0] + ".lora_B"}
    assert keys <= set(model.state_dict())
    torch.testing.assert_close(model.encoder.self_attn(x, x, x)[0], expected, rtol=0, atol=0)
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {SHARED[0] + ".lora_A", SHARED[0] + ".lora_B", *binding.private_parameters}
    assert model.decoder.embedding.weight is model.decoder.output_projection.weight
    model.head_topic(model.encoder.self_attn(x, x, x)[0]).sum().backward()
    assert model.encoder.self_attn.W_Q.lora_B.grad.abs().sum() > 0
    assert model.encoder.self_attn.W_Q.weight.grad is None
    assert model.head_topic.out_proj.weight.grad is not None


def test_extraction_uses_effective_matrix_and_keeps_private_components_separate():
    model, binding = specialist(base(), private=PRIVATE)
    with torch.no_grad():
        model.encoder.self_attn.W_Q.lora_A.copy_(
            torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 1.0, 0.0]])
        )
        model.encoder.self_attn.W_Q.lora_B.copy_(torch.arange(8).reshape(4, 2) / 10)
        model.decoder.layers[0].self_attn.W_V.lora_B.fill_(0.2)
        model.head_topic.out_proj.bias.add_(3)
    delta = extract_effective_delta(model, binding, task_id="topic")
    projection = model.encoder.self_attn.W_Q
    torch.testing.assert_close(
        delta.shared[SHARED[0] + ".weight"], 2 * projection.lora_B @ projection.lora_A
    )
    assert set(delta.private_deltas) == {name + ".weight" for name in PRIVATE}
    assert set(delta.private_state) == set(binding.private_parameters)
    assert not any(value.requires_grad for value in delta.shared.values())
    metadata = json.loads(json.dumps(delta.metadata()))
    assert metadata["binding"]["base_sha256"] == binding.base_sha256
    assert len(metadata["shared"][SHARED[0] + ".weight"]["sha256"]) == 64


def test_adapter_state_roundtrips_on_a_fresh_identical_attachment():
    pristine = base()
    model, binding = specialist(pristine, private=PRIVATE)
    with torch.no_grad():
        model.encoder.self_attn.W_Q.lora_B.fill_(0.3)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored, restored_binding = specialist(pristine, private=PRIVATE)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert binding == restored_binding
    expected = extract_effective_delta(model, binding, task_id="topic")
    actual = extract_effective_delta(restored, restored_binding, task_id="topic")
    assert actual.metadata() == expected.metadata()


def test_private_initialization_cannot_be_rebound_through_another_attachment():
    pristine = base()
    other = deepcopy(pristine)
    with torch.no_grad():
        other.head_topic.out_proj.bias.add_(1)
    model, binding = specialist(pristine)
    _, other_binding = specialist(other)
    # The private head is intentionally omitted from the frozen fingerprint,
    # but its initial bytes still belong to the complete-base receipt.
    assert binding.frozen_sha256 == other_binding.frozen_sha256
    assert binding.architecture_sha256 == other_binding.architecture_sha256
    assert binding.base_sha256 != other_binding.base_sha256
    with pytest.raises(ValueError, match="exact binding"):
        extract_effective_delta(model, other_binding, task_id="topic")
    assert extract_effective_delta(model, binding, task_id="topic").binding == binding


@pytest.mark.parametrize("relative", [False, True])
def test_nonzero_private_adapters_match_full_and_cached_decoder_logits(relative):
    model, _ = specialist(base(relative), private=PRIVATE)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.fill_(0.2)
        memory = torch.arange(24).reshape(2, 3, 4).float() / 10
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        full = model.decoder(tokens, memory, memory_mask=mask)
        cache, steps = {"memory_mask": mask}, []
        for position in range(3):
            logits, cache = model.decoder.step(tokens[:, position : position + 1], memory, cache)
            steps.append(logits)
    torch.testing.assert_close(torch.stack(steps, 1), full, rtol=1e-5, atol=1e-6)


def test_task_arithmetic_never_averages_factors_and_materialization_preserves_base():
    pristine = base()
    first, binding_a = specialist(pristine, "topic", private=PRIVATE)
    second, binding_b = specialist(pristine, "emotion")
    with torch.no_grad():
        for model, index in ((first, 0), (second, 1)):
            projection = model.encoder.self_attn.W_Q
            projection.lora_A.zero_()
            projection.lora_B.zero_()
            projection.lora_A[0, index] = 1
            projection.lora_B[index, 0] = 1
        first.head_topic.out_proj.bias.fill_(7)
        first.decoder.layers[0].self_attn.W_V.lora_B.fill_(0.1)
        second.head_emotion.out_proj.bias.fill_(9)
    a = extract_effective_delta(first, binding_a, task_id="topic")
    b = extract_effective_delta(second, binding_b, task_id="emotion")
    merged = merge_deltas([a, b], method="task_arithmetic", weights=[0.5, -1], scale=1)
    expected = torch.diag(torch.tensor([1.0, -2.0, 0.0, 0.0]))
    torch.testing.assert_close(merged.shared[SHARED[0] + ".weight"], expected)
    before = deepcopy(pristine.state_dict())
    materialized = materialize_merge(pristine, merged, private_tasks=("topic", "emotion"))
    assert set(materialized.state_dict()) == set(before)
    torch.testing.assert_close(
        materialized.encoder.self_attn.W_Q.weight, before[SHARED[0] + ".weight"] + expected
    )
    assert torch.equal(materialized.head_topic.out_proj.bias, first.head_topic.out_proj.bias)
    assert torch.equal(materialized.head_emotion.out_proj.bias, second.head_emotion.out_proj.bias)
    private_name = PRIVATE[0] + ".weight"
    torch.testing.assert_close(
        materialized.get_parameter(private_name),
        pristine.get_parameter(private_name) + a.private_deltas[private_name],
    )
    for name, value in before.items():
        assert torch.equal(pristine.state_dict()[name], value)
    assert materialized.decoder.embedding.weight is materialized.decoder.output_projection.weight


def test_one_specialist_fold_matches_unmerged_native_projection_and_attention():
    pristine = base()
    model, binding = specialist(pristine)
    with torch.no_grad():
        model.encoder.self_attn.W_Q.lora_B.fill_(0.4)
    delta = extract_effective_delta(model, binding, task_id="topic")
    folded = materialize_merge(
        pristine, merge_deltas([delta], method="task_arithmetic"), private_tasks=("topic",)
    )
    x = torch.arange(12).reshape(1, 3, 4).float() / 10
    torch.testing.assert_close(
        model.encoder.self_attn(x, x, x)[0], folded.encoder.self_attn(x, x, x)[0]
    )


def test_ties_sign_conflicts_zero_votes_and_aligned_nonzero_means():
    pristine = base()
    values = []
    for task, row in (("a", [4.0, 3.0, 2.0, 1.0]), ("b", [-5.0, 3.0, -2.0, 0.5])):
        model, binding = specialist(pristine, head=False)
        delta = extract_effective_delta(model, binding, task_id=task)
        values.append(
            replace(
                delta, shared={SHARED[0] + ".weight": torch.tensor(row + [0.0] * 12).reshape(4, 4)}
            )
        )
    result = merge_deltas(values, method="ties")
    torch.testing.assert_close(
        result.shared[SHARED[0] + ".weight"][0], torch.tensor([-5.0, 3.0, 0.0, 0.75])
    )
    trimmed = merge_deltas(values, method="ties", density=0.125)
    torch.testing.assert_close(
        trimmed.shared[SHARED[0] + ".weight"][0], torch.tensor([-5.0, 3.0, 0.0, 0.0])
    )


def test_ties_trims_globally_across_projections_with_deterministic_magnitude_ties():
    pristine = base()
    shared = SHARED + ("encoder.self_attn.W_V",)
    values = []
    for index in (1, 2):
        model, binding = specialist(pristine, shared=shared, head=False)
        delta = extract_effective_delta(model, binding, task_id=str(index))
        values.append(
            replace(
                delta,
                shared={
                    shared[0] + ".weight": torch.full((4, 4), float(index)),
                    shared[1] + ".weight": torch.full((4, 4), 10.0 * index),
                },
            )
        )
    result = merge_deltas(values, method="ties", density=0.5)
    assert not result.shared[shared[0] + ".weight"].any()
    assert torch.equal(result.shared[shared[1] + ".weight"], torch.full((4, 4), 15.0))
    tied = replace(values[0], shared={name: torch.ones((4, 4)) for name in values[0].shared})
    result = merge_deltas([tied], method="ties", density=1 / 32)
    assert result.shared[shared[0] + ".weight"].flatten()[0] == 1
    assert sum(torch.count_nonzero(tensor).item() for tensor in result.shared.values()) == 1


@pytest.mark.parametrize("problem", ["alias", "quantized_subclass", "legacy", "not_attention"])
def test_invalid_targets_fail_before_replacing_modules_or_freezing_parameters(problem):
    model = base()
    shared = SHARED
    if problem == "alias":
        model.encoder.self_attn.W_Q.weight = model.encoder.self_attn.W_K.weight
    elif problem == "quantized_subclass":

        class UnsupportedLinear(nn.Linear):
            pass

        model.encoder.self_attn.W_Q = UnsupportedLinear(4, 4)
    elif problem == "legacy":
        model.encoder.self_attn = MultiHeadAttention(4, 2, use_lora=True)
    else:
        shared = ("encoder.embedding",)
    before = deepcopy(model.state_dict())
    flags = {name: value.requires_grad for name, value in model.named_parameters()}
    with pytest.raises(ValueError):
        attach_lora(model, shared_projections=shared, config=CONFIG)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert flags == {name: value.requires_grad for name, value in model.named_parameters()}
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)


def test_base_mutation_private_overlap_and_changed_layout_are_rejected():
    pristine = base()
    model, binding = specialist(pristine)
    a = extract_effective_delta(model, binding, task_id="a")
    b = replace(a, task_id="b")
    merged = merge_deltas([a, b], method="task_arithmetic")
    with pytest.raises(ValueError, match="overlap"):
        materialize_merge(pristine, merged, private_tasks=("a", "b"))
    with torch.no_grad():
        model.encoder.norm.weight.add_(0.01)
    with pytest.raises(ValueError, match="frozen base changed"):
        extract_effective_delta(model, binding, task_id="changed")
    changed = deepcopy(pristine)
    changed.encoder.self_attn.num_heads = 1  # Same parameter shapes, different architecture.
    with pytest.raises(ValueError, match="bound full initialization"):
        materialize_merge(changed, merged)
    with torch.no_grad():
        pristine.head_emotion.out_proj.bias.add_(
            1
        )  # Common full-base evidence includes inactive heads.
    with pytest.raises(ValueError, match="bound full initialization"):
        materialize_merge(pristine, merged)


@pytest.mark.parametrize("problem", ["base", "shape", "dtype", "nan", "keys"])
def test_incompatible_effective_deltas_are_rejected(problem):
    model, binding = specialist(base())
    a = extract_effective_delta(model, binding, task_id="a")
    b = replace(a, task_id="b", shared={name: value.clone() for name, value in a.shared.items()})
    name = SHARED[0] + ".weight"
    if problem == "base":
        b = replace(b, binding=replace(binding, base_sha256="f" * 64))
    elif problem == "shape":
        b.shared[name] = torch.zeros((2, 2))
    elif problem == "dtype":
        b.shared[name] = b.shared[name].double()
    elif problem == "nan":
        b.shared[name][0, 0] = float("nan")
    else:
        b.shared["encoder.embedding.weight"] = b.shared.pop(name)
    with pytest.raises(ValueError):
        merge_deltas([a, b], method="ties")
