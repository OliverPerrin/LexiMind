"""Synthetic objective/gradient contracts; no model, rollout, or training run."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from src.training.rl import (
    GroupRelativeConfig,
    RewardProvenance,
    RolloutContract,
    group_relative_loss,
    information_gain_reward,
    offline_preference_loss,
    update_ema_parameters,
)

PROVENANCE = RewardProvenance("synthetic-fixture", "fixture-v1", "a" * 64)
METHODS = ["dr_grpo", "dapo", "gspo", "bpo_experimental", "rlp"]


def inputs(method="dr_grpo"):
    current = torch.tensor([[-1.0, -2.0], [-1.5, float("nan")]], requires_grad=True)
    config = GroupRelativeConfig(
        4,
        method,
        **({"clip_low": 0.2, "clip_high": 0.28} if method in {"bpo_experimental", "rlp"} else {}),
    )
    return dict(
        policy_log_probs=current,
        behavior_log_probs=current.detach().clone().requires_grad_(),
        response_mask=torch.tensor([[True, True], [True, False]]),
        rewards=torch.tensor([1.0, 3.0], requires_grad=True),
        group_ids=torch.tensor([9, 9]),
        config=config,
        rollout=RolloutContract("behavior-1", 0.7, (PROVENANCE,) * 2),
        expected_behavior_revision="behavior-1",
        policy_temperature=0.7,
    )


def test_dr_grpo_fixed_cap_centering_and_detached_behavior_rewards():
    values = inputs()
    result = group_relative_loss(**values)
    torch.testing.assert_close(result.advantages, torch.tensor([-1.0, 1.0]))
    assert result.loss.item() == pytest.approx(1 / 8)
    assert result.has_signal
    result.loss.backward()
    torch.testing.assert_close(
        values["policy_log_probs"].grad, torch.tensor([[1 / 8, 1 / 8], [-1 / 8, 0]])
    )
    assert values["behavior_log_probs"].grad is None
    assert values["rewards"].grad is None
    assert not result.policy_loss.requires_grad and not result.advantages.requires_grad


@pytest.mark.parametrize("method", ["dapo", "gspo"])
def test_standardized_token_and_sequence_objectives_match_direct_gradient(method):
    values = inputs(method)
    ratios = torch.tensor([[0.8, 1.1], [1.2, 1.0]], dtype=torch.float64)
    old = torch.full((2, 2), -3.0, dtype=torch.float64)
    current = (old + ratios.log()).requires_grad_()
    values.update(policy_log_probs=current, behavior_log_probs=old)
    values["config"] = replace(values["config"], clip_low=0.5, clip_high=0.5)
    result = group_relative_loss(**values)
    expected_current = current.detach().clone().requires_grad_()
    advantage = torch.tensor([-1.0, 1.0], dtype=torch.float64) / (2**0.5)
    mask = values["response_mask"]
    log_ratio = torch.where(mask, expected_current - old, 0)
    if method == "gspo":
        expected = -(log_ratio.sum(1).div(mask.sum(1)).exp() * advantage).mean()
    else:
        expected = -(log_ratio.exp() * advantage[:, None] * mask).sum() / mask.sum()
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    expected.backward()
    torch.testing.assert_close(current.grad, expected_current.grad)


@pytest.mark.parametrize("ratios,signal", [([0.4, 1.6], False), ([1.6, 0.4], True)])
def test_clipping_blocks_only_improvements_beyond_the_bounds(ratios, signal):
    values = inputs()
    old = torch.full((2, 1), -3.0)
    values.update(
        policy_log_probs=(old + torch.tensor(ratios)[:, None].log()).requires_grad_(),
        behavior_log_probs=old,
        response_mask=torch.ones((2, 1), dtype=torch.bool),
    )
    result = group_relative_loss(**values)
    assert result.has_signal == signal
    result.loss.backward()
    assert bool(values["policy_log_probs"].grad.any()) == signal


@pytest.mark.parametrize("clipped", [False, True])
def test_bpo_detaches_complement_weights_and_uses_its_own_clipping_gate(clipped):
    values = inputs("bpo_experimental")
    current = torch.tensor([[0.4], [0.1]], dtype=torch.float64).log().requires_grad_()
    old = torch.tensor([[0.2], [0.3]], dtype=torch.float64).log().requires_grad_()
    values.update(
        policy_log_probs=current,
        behavior_log_probs=old,
        response_mask=torch.ones((2, 1), dtype=torch.bool),
        rewards=torch.tensor([1.0, -1.0]),
        config=GroupRelativeConfig(
            4,
            "bpo_experimental",
            clip_low=0.05 if clipped else 0.9,
            clip_high=0.05 if clipped else 1.0,
            bpo_weight_cap=1.05,
        ),
    )
    result = group_relative_loss(**values)
    result.loss.backward()
    expected = (
        torch.zeros_like(current)
        if clipped
        else (
            -torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
            / (2**0.5)
            * torch.tensor([[1.05], [0.8]], dtype=torch.float64)
            / 2
        )
    )
    torch.testing.assert_close(current.grad, expected)
    assert result.has_signal != clipped
    assert old.grad is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("empty", [False, True])
def test_flat_rewards_or_entirely_masked_batches_report_no_update_signal(method, empty):
    values = inputs(method)
    if empty:
        values["response_mask"].zero_()
        values["policy_log_probs"] = torch.full((2, 2), float("nan"), requires_grad=True)
    else:
        values["rewards"] = torch.ones(2)
    result = group_relative_loss(**values)
    assert result.loss.item() == 0
    assert not result.has_signal
    result.loss.backward()
    assert torch.equal(values["policy_log_probs"].grad, torch.zeros((2, 2)))


def test_reference_penalty_is_stable_detached_and_can_supply_signal_to_flat_groups():
    values = inputs()
    current = torch.full((2, 1), -1.0, dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[1e-6], [-1e-6]], dtype=torch.float64)
    reference = (current.detach() + delta).requires_grad_()
    values.update(
        policy_log_probs=current,
        behavior_log_probs=current.detach(),
        response_mask=torch.ones((2, 1), dtype=torch.bool),
        rewards=torch.ones(2),
        config=GroupRelativeConfig(4, kl_coefficient=0.3),
        reference_log_probs=reference,
        reference_temperature=0.7,
    )
    result = group_relative_loss(**values)
    assert result.has_signal and result.loss.item() > 0
    torch.testing.assert_close(
        result.loss, 0.3 * (delta.expm1() - delta).sum() / 8, atol=1e-20, rtol=1e-8
    )
    result.loss.backward()
    torch.testing.assert_close(current.grad, -0.3 * delta.expm1() / 8, atol=1e-16, rtol=1e-8)
    assert reference.grad is None


def test_groups_are_prompt_local_even_when_interleaved():
    values = inputs()
    values.update(
        policy_log_probs=torch.full((4, 1), -1.0, requires_grad=True),
        behavior_log_probs=torch.full((4, 1), -1.0),
        response_mask=torch.ones((4, 1), dtype=torch.bool),
        rewards=torch.tensor([1.0, 7.0, 3.0, 11.0]),
        group_ids=torch.tensor([8, 2, 8, 2]),
        rollout=RolloutContract("behavior-1", 0.7, (PROVENANCE,) * 4),
    )
    torch.testing.assert_close(
        group_relative_loss(**values).advantages, torch.tensor([-1.0, -2.0, 1.0, 2.0])
    )


@pytest.mark.parametrize("method", METHODS)
def test_flat_noninteger_rewards_cannot_manufacture_signal_through_rounding(method):
    values = inputs(method)
    values.update(
        policy_log_probs=torch.full((8, 1), -1.0, requires_grad=True),
        behavior_log_probs=torch.full((8, 1), -1.0),
        response_mask=torch.ones((8, 1), dtype=torch.bool),
        rewards=torch.full((8,), 0.1),
        group_ids=torch.zeros(8, dtype=torch.long),
        rollout=RolloutContract("behavior-1", 0.7, (PROVENANCE,) * 8),
    )
    result = group_relative_loss(**values)
    assert not result.has_signal
    assert torch.equal(result.advantages, torch.zeros(8))
    result.loss.backward()
    assert torch.equal(values["policy_log_probs"].grad, torch.zeros((8, 1)))


@pytest.mark.parametrize(
    "change,error",
    [
        ({"expected_behavior_revision": "different"}, "revision"),
        ({"policy_temperature": 1.0}, "temperature"),
        ({"response_mask": torch.ones((2, 2))}, "bool"),
        ({"response_mask": torch.tensor([[False, True], [True, False]])}, "contiguous"),
        ({"response_mask": torch.tensor([[False, False], [True, False]])}, "Each sampled response"),
        ({"group_ids": torch.tensor([1, 2])}, "equal sizes"),
        ({"group_ids": torch.tensor([1.0, 1.0])}, "integer"),
        ({"group_ids": torch.tensor([-1, -1])}, "nonnegative"),
        ({"rewards": torch.tensor([float("nan"), 1.0])}, "finite"),
        ({"rewards": torch.tensor([1, 2])}, "floating"),
        ({"policy_log_probs": torch.tensor([[0.1, -2.0], [-1.0, 0.0]])}, "positive logits"),
        ({"policy_log_probs": torch.tensor([[float("nan"), -2.0], [-1.0, 0.0]])}, "finite"),
        ({"behavior_log_probs": torch.full((2, 1), -1.0)}, "match B x T"),
        ({"config": GroupRelativeConfig(1)}, "response cap"),
        ({"config": GroupRelativeConfig(4, kl_coefficient=0.1)}, "requires explicit reference"),
        ({"reference_log_probs": torch.full((2, 2), -1.0)}, "Reference scoring temperature"),
        ({"rollout": RolloutContract("behavior-1", 0.7, (PROVENANCE,))}, "Each response"),
    ],
)
def test_invalid_scoring_and_collection_contracts_fail_closed(change, error):
    values = inputs()
    values.update(change)
    with pytest.raises(ValueError, match=error):
        group_relative_loss(**values)


def test_unequal_group_sizes_are_rejected_instead_of_changing_prompt_weights():
    values = inputs()
    values.update(
        policy_log_probs=torch.full((6, 1), -1.0),
        behavior_log_probs=torch.full((6, 1), -1.0),
        response_mask=torch.ones((6, 1), dtype=torch.bool),
        rewards=torch.arange(6).float(),
        group_ids=torch.tensor([0, 0, 1, 1, 1, 1]),
        rollout=RolloutContract("behavior-1", 0.7, (PROVENANCE,) * 6),
    )
    with pytest.raises(ValueError, match="equal sizes"):
        group_relative_loss(**values)


def test_importance_ratio_overflow_is_not_silently_clamped():
    values = inputs()
    values["behavior_log_probs"] = torch.full((2, 2), -1000.0)
    with pytest.raises(ValueError, match="Importance ratios.*finite"):
        group_relative_loss(**values)


def test_importance_ratio_underflow_cannot_silently_discard_gradients():
    values = inputs()
    values["policy_log_probs"] = torch.full((2, 2), -1000.0, requires_grad=True)
    with pytest.raises(ValueError, match="underflowed"):
        group_relative_loss(**values)


def test_recipe_defaults_and_provenance_contracts_are_explicit():
    assert [
        (GroupRelativeConfig(4, method).clip_low, GroupRelativeConfig(4, method).clip_high)
        for method in METHODS[:3]
    ] == [(0.2, 0.2), (0.2, 0.28), (3e-4, 4e-4)]
    with pytest.raises(ValueError, match="explicit clipping bounds"):
        GroupRelativeConfig(4, "bpo_experimental")
    with pytest.raises(ValueError, match="no-KL"):
        GroupRelativeConfig(4, "bpo_experimental", 0.2, 0.28, kl_coefficient=0.01)
    with pytest.raises(ValueError, match="full-support"):
        RolloutContract("behavior", 0.7, (PROVENANCE,), sampling_top_p=0.9)
    with pytest.raises(ValueError, match="SHA256"):
        RewardProvenance("fixture", "v1", "unverified")


def test_offline_dpo_uses_sequence_sums_and_detaches_reference_scores():
    chosen = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    rejected = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
    ref_chosen = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    ref_rejected = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
    loss = offline_preference_loss(
        chosen,
        rejected,
        torch.tensor([[True, True]]),
        torch.tensor([[True, False]]),
        ref_chosen,
        ref_rejected,
        beta=0.5,
        provenance=(PROVENANCE,),
    )
    expected = -F.logsigmoid(torch.tensor(0.5))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    gradient = 0.5 * torch.sigmoid(torch.tensor(-0.5))
    torch.testing.assert_close(chosen.grad, torch.full_like(chosen, -gradient))
    torch.testing.assert_close(rejected.grad, torch.tensor([[gradient, 0]]))
    assert ref_chosen.grad is None and ref_rejected.grad is None


def test_rlp_corrected_group_baseline_and_per_thought_token_means():
    values = inputs("rlp")
    result = group_relative_loss(**values)
    torch.testing.assert_close(result.advantages, torch.tensor([-2.0, 2.0]))
    # The unequal thought lengths have equal response weight, unlike Dr.GRPO.
    assert result.loss.item() == 0
    assert result.has_signal
    result.loss.backward()
    torch.testing.assert_close(
        values["policy_log_probs"].grad, torch.tensor([[0.5, 0.5], [-1.0, 0]])
    )
    with pytest.raises(ValueError, match="explicit clipping bounds"):
        GroupRelativeConfig(4, "rlp")
    with pytest.raises(ValueError, match="no-KL"):
        GroupRelativeConfig(4, "rlp", 0.2, 0.2, kl_coefficient=0.1)


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_standardization_preserves_nonflat_tiny_and_large_reward_scales(scale):
    values = inputs("gspo")
    values["rewards"] = torch.tensor([scale, 2 * scale])
    torch.testing.assert_close(
        group_relative_loss(**values).advantages, torch.tensor([-1.0, 1.0]) / (2**0.5)
    )


def test_bpo_complement_weights_remain_finite_at_probability_one():
    values = inputs("bpo_experimental")
    values.update(
        policy_log_probs=torch.zeros((2, 1), requires_grad=True),
        behavior_log_probs=torch.zeros((2, 1)),
        response_mask=torch.ones((2, 1), dtype=torch.bool),
        config=replace(values["config"], bpo_epsilon=1e-8),
    )
    result = group_relative_loss(**values)
    assert result.has_signal and torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(values["policy_log_probs"].grad).all()


def test_information_gain_is_per_observed_position_and_detached_without_clipping():
    reasoned = torch.tensor([-2.0, -5.0], requires_grad=True)
    baseline = torch.tensor([-3.0, -1.0], requires_grad=True)
    reward = information_gain_reward(reasoned, baseline)
    torch.testing.assert_close(reward, torch.tensor([1.0, -4.0]))
    assert not reward.requires_grad
    with pytest.raises(ValueError, match="per row"):
        information_gain_reward(reasoned[:, None], baseline[:, None])
    with pytest.raises(ValueError, match="match B x T"):
        information_gain_reward(reasoned, baseline[:1])


class TinyEMA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        self.tied_weight = self.weight
        self.register_buffer("counter", torch.tensor(2))


def test_ema_averages_tied_weights_once_copies_buffers_without_grad_or_policy_mutation():
    policy = TinyEMA()
    teacher = deepcopy(policy).eval()
    teacher.weight.data.zero_()
    teacher.counter.zero_()
    teacher.weight.grad = torch.tensor([5.0, 7.0])
    before = deepcopy(policy.state_dict())
    update_ema_parameters(teacher, policy, decay=0.75)
    torch.testing.assert_close(teacher.weight, torch.tensor([0.25, 0.75]))
    assert teacher.weight is teacher.tied_weight
    assert teacher.counter.item() == 2 and not teacher.training and policy.training
    assert teacher.weight.grad_fn is None
    torch.testing.assert_close(teacher.weight.grad, torch.tensor([5.0, 7.0]))
    for key, value in before.items():
        assert torch.equal(policy.state_dict()[key], value)
    update_ema_parameters(teacher, policy, decay=0.0)
    assert torch.equal(teacher.weight, policy.weight)


@pytest.mark.parametrize("failure", ["module", "alias", "ties", "dtype", "nonfinite"])
def test_ema_rejects_incompatible_or_aliasing_state_before_any_update(failure):
    policy, teacher = TinyEMA(), TinyEMA()
    teacher.weight.data.zero_()
    if failure == "module":
        teacher.add_module("extra", torch.nn.ReLU())
    elif failure == "alias":
        teacher.weight = torch.nn.Parameter(policy.weight.detach())
        teacher.tied_weight = teacher.weight
    elif failure == "ties":
        teacher.tied_weight = torch.nn.Parameter(teacher.weight.detach().clone())
    elif failure == "dtype":
        teacher.double()
    else:
        policy.weight.data[0] = float("nan")
    before = deepcopy(teacher.state_dict())
    with pytest.raises(ValueError, match="EMA"):
        update_ema_parameters(teacher, policy, decay=0.9)
    for key, value in before.items():
        assert torch.equal(teacher.state_dict()[key], value)


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan"), True])
def test_ema_requires_explicit_valid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        update_ema_parameters(TinyEMA(), TinyEMA(), decay=decay)
