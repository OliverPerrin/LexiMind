"""Native policy plumbing on tiny random models and synthetic evidence only."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.models.decoder import TransformerDecoder
from src.models.encoder import TransformerEncoder
from src.models.heads import LMHead
from src.models.multitask import MultiTaskModel
from src.training.policy import (
    ContinuationTarget,
    PolicyTask,
    complete_field_reward,
    deterministic_policy,
    policy_precision,
    rpt_prefix_reward,
    sample_responses,
    score_responses,
)
from src.training.rl import GroupRelativeConfig, RewardProvenance
from src.training.trainer import Trainer, TrainerConfig


def model():
    torch.manual_seed(312)
    options = dict(
        vocab_size=7,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        dropout=0.25,
        pad_token_id=0,
        max_len=32,
        use_relative_position_bias=True,
    )
    result = MultiTaskModel(TransformerEncoder(**options), TransformerDecoder(**options))
    result.add_head("summarization", LMHead(8, 7))
    return result


def provenance(size):
    return tuple(RewardProvenance("synthetic", "unit-fixture-v1", "a" * 64) for _ in range(size))


def inputs():
    ids = torch.tensor([[2, 3, 0], [4, 5, 6]])
    return ids, ids != 0


def samples(instance, *, temperature=0.8):
    ids, mask = inputs()
    return sample_responses(
        instance,
        ids,
        mask,
        group_size=2,
        max_response_tokens=5,
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        temperature=temperature,
        behavior_policy_revision="synthetic-step-0",
        tokenizer_revision="synthetic-tokenizer-v1",
        generator=torch.Generator().manual_seed(93),
    )


def objective(instance, **kwargs):
    return PolicyTask(
        instance,
        mode="group_relative",
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        tokenizer_revision="synthetic-tokenizer-v1",
        temperature=0.8,
        config=GroupRelativeConfig(5),
        expected_behavior_revision="synthetic-step-0",
        **kwargs,
    )


def test_collection_scoring_probability_and_modes_match_with_grouped_prompts():
    instance = model().train()
    instance.encoder.layers[0].norm1.eval()
    modes = [m.training for m in instance.modules()]
    generated = samples(instance)
    assert [m.training for m in instance.modules()] == modes
    assert generated.group_ids.tolist() == [0, 0, 1, 1]
    assert not generated.behavior_log_probs.requires_grad
    assert torch.equal(
        generated.response_ids[~generated.response_mask],
        torch.zeros_like(generated.response_ids[~generated.response_mask]),
    )
    scored = score_responses(
        instance,
        generated.source_ids,
        generated.source_mask,
        generated.response_ids,
        generated.response_mask,
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        temperature=0.8,
    )
    torch.testing.assert_close(scored, generated.behavior_log_probs, atol=2e-6, rtol=2e-6)
    assert [m.training for m in instance.modules()] == modes
    assert scored.requires_grad


def test_active_pad_token_keeps_its_causal_context_and_first_eos_is_included():
    instance = model()
    ids, source_mask = inputs()
    responses = torch.tensor([[2, 0, 3, 1, 0], [0, 4, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    with deterministic_policy(instance), torch.no_grad():
        memory = instance.encoder(ids, mask=source_mask[:, None, :] & source_mask[:, :, None])
        cache = {"past_length": 0, "memory_mask": source_mask}
        last = torch.zeros(2, 1, dtype=torch.long)
        expected = []
        for column in range(responses.shape[1]):
            logits, cache = instance.decoder.step(last, memory, cache)
            expected.append(
                (logits / 0.7)
                .log_softmax(-1)
                .gather(1, responses[:, column : column + 1])
                .squeeze(1)
            )
            last = responses[:, column : column + 1]
        expected = torch.stack(expected, dim=1).masked_fill(~mask, 0)
    actual = score_responses(
        instance,
        ids,
        source_mask,
        responses,
        mask,
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        temperature=0.7,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="first EOS"):
        score_responses(
            instance,
            ids,
            source_mask,
            responses,
            torch.ones_like(mask),
            start_token_id=0,
            end_token_id=1,
            pad_token_id=0,
            temperature=0.7,
        )


def test_group_objective_uses_one_encoder_batch_per_distinct_prompt_and_backpropagates():
    instance = model()
    generated = samples(instance)
    batch = generated.training_batch(torch.tensor([1.0, 0.0, 0.0, 1.0]), provenance(4))
    seen = []
    hook = instance.encoder.register_forward_pre_hook(lambda _, args: seen.append(len(args[0])))
    loss, metrics = objective(instance)(batch)
    hook.remove()
    assert seen == [2]
    assert metrics.has_signal
    loss.backward()
    assert any(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in instance.decoder.parameters()
    )
    damaged = dict(batch, src_ids=batch["src_ids"].clone())
    damaged["src_ids"][1, 0] = 6
    with pytest.raises(ValueError, match="identical prompt"):
        objective(instance)(damaged)
    with pytest.raises(ValueError, match="tokenizer"):
        objective(instance)(dict(batch, tokenizer_revision="wrong"))


class Batches:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = range(len(batch["src_ids"]))

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([self.batch])


def trainer(instance, monkeypatch):
    result = Trainer.__new__(Trainer)
    result.model = instance
    result.optimizer = torch.optim.AdamW(instance.parameters(), lr=0.01, weight_decay=0.3)
    result.config = TrainerConfig(task_sampling="round_robin", gradient_clip_norm=1e6)
    result.device = torch.device("cpu")
    result.use_amp = result.use_bfloat16 = False
    result.pcgrad = result.scheduler = None
    result.global_step = 0
    result.policy_objectives = {"policy": objective(instance)}
    monkeypatch.setattr("src.training.trainer.mlflow.log_metric", lambda *a, **k: None)
    return result


def test_shared_trainer_skips_flat_policy_updates_and_refuses_surrogate_selection(monkeypatch):
    instance = model()
    generated = samples(instance)
    runner = trainer(instance, monkeypatch)
    nonflat = generated.training_batch(torch.tensor([1.0, 0.0, 0.0, 1.0]), provenance(4))
    runner._run_epoch({"policy": Batches(nonflat)}, train=True, epoch=1)
    assert runner.global_step == 1
    before = {name: p.detach().clone() for name, p in instance.named_parameters()}
    optimizer = deepcopy(runner.optimizer.state_dict())
    flat = generated.training_batch(torch.full((4,), 0.1), provenance(4))
    metrics = runner._run_epoch({"policy": Batches(flat)}, train=True, epoch=2)
    assert runner.global_step == 1 and metrics["total_loss"] == 0
    for name, parameter in instance.named_parameters():
        assert torch.equal(parameter, before[name])
        assert parameter.grad is None
    for key, values in optimizer["state"].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                assert torch.equal(value, runner.optimizer.state_dict()["state"][key][name])
    with pytest.raises(ValueError, match="surrogate"):
        runner._run_epoch({"policy": Batches(nonflat)}, train=False, epoch=3)


def test_preference_adapter_has_one_shared_prompt_encoding_and_explicit_reference_contract():
    instance = model()
    ids, mask = inputs()
    chosen = torch.tensor([[2, 1], [3, 1]])
    rejected = torch.tensor([[4, 1], [5, 1]])
    observed = torch.ones_like(chosen, dtype=torch.bool)
    with torch.no_grad():
        refs = [
            score_responses(
                instance,
                ids,
                mask,
                x,
                observed,
                start_token_id=0,
                end_token_id=1,
                pad_token_id=0,
                temperature=1.0,
            )
            for x in (chosen, rejected)
        ]
    task = PolicyTask(
        instance,
        mode="dpo",
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        tokenizer_revision="synthetic-tokenizer-v1",
        expected_reference_revision="reference-v1",
    )
    batch = dict(
        src_ids=ids,
        src_mask=mask,
        chosen_ids=chosen,
        rejected_ids=rejected,
        chosen_mask=observed,
        rejected_mask=observed,
        reference_chosen=refs[0],
        reference_rejected=refs[1],
        reference_revision="reference-v1",
        reference_temperature=1.0,
        precision_contract=policy_precision(instance),
        reference_precision_contract=policy_precision(instance),
        tokenizer_revision="synthetic-tokenizer-v1",
        token_contract=(0, 1, 0),
        provenance=provenance(2),
    )
    loss, metrics = task(batch)
    assert loss.item() == pytest.approx(0.69314718)
    assert metrics.has_signal
    loss.backward()
    with pytest.raises(ValueError, match="temperature"):
        task(dict(batch, reference_temperature=0.8))
    with pytest.raises(ValueError, match="identical"):
        task(dict(batch, rejected_ids=chosen))
    with pytest.raises(ValueError, match="Reference scoring precision"):
        task(dict(batch, reference_precision_contract=("autocast", "torch.bfloat16")))


def test_rpt_reward_uses_byte_prefix_and_actual_token_boundaries_without_normalization():
    target = ContinuationTarget(
        " café!".encode(),
        (b" ", b"caf", "é".encode(), b"!"),
        "a" * 64,
        "b" * 64,
        "synthetic-work",
        "train",
    )
    assert rpt_prefix_reward(b"", target) == 0
    assert rpt_prefix_reward(b" ca", target) == 0
    assert rpt_prefix_reward(b" caf", target) == 1
    assert rpt_prefix_reward(" café".encode(), target) == 1
    assert rpt_prefix_reward(" cafe\u0301".encode(), target) == 0
    assert rpt_prefix_reward(b" cafe", target) == 0
    with pytest.raises(ValueError, match="reconstruct"):
        ContinuationTarget(b"ab", (b"a",), "a" * 64, "b" * 64, "work", "train")


def test_field_reward_does_not_promote_unknown_or_positive_only_metadata():
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    known = torch.ones_like(labels, dtype=torch.bool)
    assert complete_field_reward(labels.bool(), labels, known).item() == 1
    assert complete_field_reward(torch.ones_like(known), labels, known).item() == 0.5
    with pytest.raises(ValueError, match="unknown"):
        complete_field_reward(labels.bool(), labels, torch.tensor([[True, False, True, False]]))
    with pytest.raises(ValueError, match="both positive and negative"):
        complete_field_reward(known, torch.ones_like(labels), known)


@pytest.mark.parametrize("missing", [None, torch.ones(1, 2)])
def test_field_reward_requires_an_explicit_boolean_review_mask(missing):
    with pytest.raises(ValueError, match="explicit boolean"):
        complete_field_reward(torch.tensor([[True, False]]), torch.tensor([[1.0, 0.0]]), missing)


@pytest.mark.parametrize("start", [True, -1, 7])
def test_public_scoring_checks_special_token_ids(start):
    instance = model()
    ids, mask = inputs()
    responses = torch.ones(2, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="Special token"):
        score_responses(
            instance,
            ids,
            mask,
            responses,
            torch.ones_like(responses, dtype=torch.bool),
            start_token_id=start,
            end_token_id=1,
            pad_token_id=0,
            temperature=1.0,
        )


def test_policy_scoring_avoids_decoder_work_for_unused_response_capacity():
    instance = model()
    ids, source_mask = inputs()
    tokens = torch.tensor([[2, 1, 0, 0, 0, 0], [3, 4, 1, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    widths = []
    hook = instance.decoder.register_forward_pre_hook(
        lambda _, args: widths.append(args[0].shape[1])
    )
    scores = score_responses(
        instance,
        ids,
        source_mask,
        tokens,
        mask,
        start_token_id=0,
        end_token_id=1,
        pad_token_id=0,
        temperature=1.0,
    )
    hook.remove()
    assert widths == [3] and scores.shape == tokens.shape
    assert torch.equal(scores[~mask], torch.zeros_like(scores[~mask]))


def test_fully_clipped_policy_batch_preserves_existing_optimizer_and_scheduler(monkeypatch):
    from unittest.mock import Mock

    instance = model()
    generated = samples(instance)
    runner = trainer(instance, monkeypatch)
    batch = generated.training_batch(torch.tensor([1.0, 0.0, 0.0, 1.0]), provenance(4))
    runner._run_epoch({"policy": Batches(batch)}, train=True, epoch=1)
    with torch.no_grad():
        current = score_responses(
            instance,
            generated.source_ids,
            generated.source_mask,
            generated.response_ids,
            generated.response_mask,
            start_token_id=0,
            end_token_id=1,
            pad_token_id=0,
            temperature=0.8,
        )
    ratios = torch.tensor([1.3, 0.7, 0.7, 1.3])[:, None]
    batch["behavior_log_probs"] = torch.where(generated.response_mask, current - ratios.log(), 0.0)
    assert bool((batch["behavior_log_probs"][generated.response_mask] <= 0).all())
    assert not runner.policy_objectives["policy"](batch)[1].has_signal
    before = {key: value.detach().clone() for key, value in instance.state_dict().items()}
    state = deepcopy(runner.optimizer.state_dict())
    runner.scheduler = SimpleNamespace(step=Mock(), get_last_lr=lambda: [0.01])
    runner._run_epoch({"policy": Batches(batch)}, train=True, epoch=2)
    assert runner.global_step == 1
    runner.scheduler.step.assert_not_called()
    for key, value in instance.state_dict().items():
        assert torch.equal(value, before[key])
    for key, values in state["state"].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                assert torch.equal(value, runner.optimizer.state_dict()["state"][key][name])


def test_cpu_autocast_cannot_change_collection_or_rescoring_and_contexts_are_restored():
    instance = model().train()
    instance.encoder.layers[0].norm1.eval()
    modes = [module.training for module in instance.modules()]
    expected = samples(instance)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = samples(instance)
        assert torch.is_autocast_enabled("cpu")
        assert torch.equal(actual.response_ids, expected.response_ids)
        assert torch.equal(actual.behavior_log_probs, expected.behavior_log_probs)
        scored = score_responses(
            instance,
            actual.source_ids,
            actual.source_mask,
            actual.response_ids,
            actual.response_mask,
            start_token_id=0,
            end_token_id=1,
            pad_token_id=0,
            temperature=0.8,
        )
        torch.testing.assert_close(scored, expected.behavior_log_probs, atol=2e-6, rtol=2e-6)
        batch = actual.training_batch(torch.tensor([1.0, 0.0, 0.0, 1.0]), provenance(4))
        _, metrics = objective(instance)(batch)
        assert metrics.has_signal and torch.is_autocast_enabled("cpu")
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with deterministic_policy(instance):
                assert not torch.is_autocast_enabled("cpu")
                raise RuntimeError("synthetic failure")
        assert torch.is_autocast_enabled("cpu")
    assert not torch.is_autocast_enabled("cpu")
    assert [module.training for module in instance.modules()] == modes


@pytest.mark.parametrize("mismatch", ["metadata", "changed_weights", "mixed_weights"])
def test_policy_precision_rejects_missing_or_changed_model_dtype_before_scoring(mismatch):
    instance = model()
    generated = samples(instance)
    task = objective(instance)
    batch = generated.training_batch(torch.tensor([1.0, 0.0, 0.0, 1.0]), provenance(4))
    assert batch["precision_contract"] == ("no_autocast_fp32_log_softmax", "torch.float32")
    if mismatch == "metadata":
        batch.pop("precision_contract")
    elif mismatch == "changed_weights":
        instance.double()
    else:
        instance.decoder.double()
    with pytest.raises(ValueError, match="precision"):
        task(batch)
