"""Synthetic software regression checks; no dataset, checkpoint, or model run."""

import numpy as np
import pytest
import torch

from src.training.pcgrad import PCGrad
from src.training.trainer import Trainer, TrainerConfig


class Batches:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = range(sum(len(batch["labels"]) for batch in batches))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def make_trainer(monkeypatch, *, accum=1, pcgrad=False, diagnostics=0):
    # Avoid training setup, tracking databases, and any pretrained artifacts.
    trainer = Trainer.__new__(Trainer)
    trainer.policy_objectives = {}
    trainer.model = torch.nn.Module()
    trainer.model.encoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.encoder.weight.zero_()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.config = TrainerConfig(
        gradient_accumulation_steps=accum,
        gradient_clip_norm=1e6,
        task_sampling="round_robin",
        warmup_steps=0,
        gradient_conflict_frequency=diagnostics,
    )
    trainer.device = torch.device("cpu")
    trainer.use_amp = trainer.use_bfloat16 = False
    trainer.pcgrad = PCGrad() if pcgrad else None
    trainer.scheduler = None
    trainer.global_step = 0
    trainer._forward_task = lambda task, batch, **kwargs: (
        trainer.model.encoder.weight.sum() * batch["coefficient"],
        {},
    )
    monkeypatch.setattr("src.training.trainer.mlflow.log_metric", lambda *args, **kwargs: None)
    return trainer


def batch(coefficient=1.0, size=1):
    return {"coefficient": coefficient, "labels": torch.zeros(size, dtype=torch.long)}


def test_accumulation_flushes_and_normalizes_partial_window(monkeypatch):
    trainer = make_trainer(monkeypatch, accum=2)
    loader = Batches([batch(2.0), batch(4.0), batch(8.0)])
    trainer._run_epoch({"topic": loader}, train=True, epoch=1)
    # First SGD update: mean(2,4)=3. Final one-element window: 8, not 8/2.
    torch.testing.assert_close(trainer.model.encoder.weight, torch.tensor([[-11.0]]))
    assert trainer.global_step == 2
    assert trainer.model.encoder.weight.grad is None
    trainer._run_epoch({"topic": Batches([batch(5.0)])}, train=True, epoch=2)
    torch.testing.assert_close(trainer.model.encoder.weight, torch.tensor([[-16.0]]))


def test_policy_rejects_unweighted_accumulation_before_mutating_state(monkeypatch):
    trainer = make_trainer(monkeypatch, accum=2)
    trainer.policy_objectives = {"policy": object()}
    trainer.model.eval()
    parameter = trainer.model.encoder.weight
    parameter.grad = torch.tensor([[7.0]])
    with pytest.raises(ValueError, match="weighted policy windows"):
        trainer._run_epoch({"policy": Batches([batch()])}, train=True, epoch=1)
    assert not trainer.model.training and trainer.global_step == 0
    torch.testing.assert_close(parameter.grad, torch.tensor([[7.0]]))


def test_generation_metrics_can_be_disabled_without_changing_loss_or_gradients(monkeypatch):
    trainer = make_trainer(monkeypatch)
    logits = torch.tensor([[[0.1, 0.3, -0.2], [0.4, -0.1, 0.2]]], requires_grad=True)
    trainer.model.forward = lambda *args: logits
    trainer.tokenizer = None  # Any decoding would fail.
    trainer.config.generation_metrics = False
    trainer.config.label_smoothing = 0.0
    ids = torch.tensor([[0, 1]])
    labels = torch.tensor([[1, -100]])
    loss, metrics = trainer._forward_summarization(
        {"src_ids": ids, "tgt_ids": ids, "labels": labels}
    )
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
    torch.testing.assert_close(loss, expected)
    assert metrics == {}
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() == 0


def test_scheduler_counts_remainder_updates(monkeypatch):
    trainer = make_trainer(monkeypatch, accum=2)
    trainer.config.max_epochs = 1
    trainer._setup_scheduler({"topic": Batches([batch(), batch(), batch()])}, 1)
    assert trainer.scheduler.lr_lambdas[0](1) == pytest.approx(0.5)
    assert trainer.scheduler.lr_lambdas[0](2) == pytest.approx(0.1)


def test_validation_visits_each_batch_once_and_weights_examples(monkeypatch):
    trainer = make_trainer(monkeypatch)
    calls = []

    def forward(task, value, **kwargs):
        calls.append((task, value["coefficient"]))
        return torch.tensor(value["coefficient"]), {"accuracy": value["coefficient"]}

    trainer._forward_task = forward
    trainer.config.task_weights = {"topic": 2.0, "emotion": 0.5}
    metrics = trainer._run_epoch(
        {
            "topic": Batches([batch(1.0, size=3), batch(0.0, size=1)]),
            "emotion": Batches([batch(0.2, size=2)]),
        },
        train=False,
        epoch=1,
    )
    assert calls.count(("emotion", 0.2)) == 1
    assert len(calls) == 3
    assert metrics["topic_loss"] == pytest.approx(0.75)
    assert metrics["topic_accuracy"] == pytest.approx(0.75)
    assert metrics["emotion_loss"] == pytest.approx(0.2)
    assert metrics["total_loss"] == pytest.approx(2 * 0.75 + 0.5 * 0.2)


def test_summarization_validation_weights_loss_by_nonignored_tokens(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer._forward_task = lambda task, value, **kwargs: (torch.tensor(value["coefficient"]), {})
    loader = Batches(
        [
            {"coefficient": 2.0, "labels": torch.tensor([[1, 2, 3], [4, -100, -100]])},
            {"coefficient": 8.0, "labels": torch.tensor([[5, -100, -100]])},
        ]
    )
    result = trainer._run_epoch({"summarization": loader}, train=False, epoch=1)
    assert result["summarization_loss"] == pytest.approx((4 * 2 + 1 * 8) / 5)


def test_gradient_diagnostics_preserve_accumulated_grads_and_rng(monkeypatch):
    trainer = make_trainer(monkeypatch)
    parameter = trainer.model.encoder.weight
    parameter.grad = torch.tensor([[7.0]])

    def noisy_forward(task, value, **kwargs):
        torch.rand(3)  # Stand-in for dropout draws during a diagnostic probe.
        return parameter.sum() * value["coefficient"], {}

    trainer._forward_task = noisy_forward
    state = torch.get_rng_state().clone()
    stats = trainer._compute_gradient_conflicts({"topic": batch(1.0), "emotion": batch(-1.0)})
    torch.testing.assert_close(parameter.grad, torch.tensor([[7.0]]))
    assert torch.equal(torch.get_rng_state(), state)
    assert stats["cos_sim_topic_emotion"] == pytest.approx(-1)


def test_diagnostics_do_not_change_sgd_updates(monkeypatch):
    loader = Batches([batch(1.0), batch(3.0), batch(2.0)])
    plain = make_trainer(monkeypatch, accum=2)
    diagnostic = make_trainer(monkeypatch, accum=2, diagnostics=1)
    plain._run_epoch({"topic": loader}, train=True, epoch=1)
    diagnostic._run_epoch({"topic": loader}, train=True, epoch=1)
    torch.testing.assert_close(plain.model.encoder.weight, diagnostic.model.encoder.weight)


@pytest.mark.parametrize("raise_in_probe", [False, True])
def test_gradient_diagnostics_restore_mps_rng_without_hardware(monkeypatch, raise_in_probe):
    trainer = make_trainer(monkeypatch)
    trainer.device = torch.device("mps")
    parameter = trainer.model.encoder.weight
    parameter.grad = torch.tensor([[7.0]])
    mps_state = torch.tensor([17], dtype=torch.uint8)
    reads, writes = [], []

    def get_state(device):
        reads.append(device)
        return mps_state.clone()

    def set_state(state, device):
        writes.append(device)
        mps_state.copy_(state)

    monkeypatch.setattr(torch.mps, "get_rng_state", get_state)
    monkeypatch.setattr(torch.mps, "set_rng_state", set_state)

    def noisy_forward(task, value, **kwargs):
        torch.rand(3)
        mps_state.add_(1)
        if raise_in_probe:
            raise RuntimeError("synthetic probe failure")
        return parameter.sum() * value["coefficient"], {}

    trainer._forward_task = noisy_forward
    cpu_state = torch.get_rng_state().clone()
    if raise_in_probe:
        with pytest.raises(RuntimeError, match="synthetic probe failure"):
            trainer._compute_gradient_conflicts({"topic": batch()})
    else:
        stats = trainer._compute_gradient_conflicts({"topic": batch(), "emotion": batch(-1.0)})
        assert stats["cos_sim_topic_emotion"] == pytest.approx(-1)
    assert reads == writes == [0]
    assert mps_state.item() == 17
    assert torch.equal(torch.get_rng_state(), cpu_state)
    torch.testing.assert_close(parameter.grad, torch.tensor([[7.0]]))


def test_pcgrad_preserves_duplicate_temperature_task_draws(monkeypatch):
    trainer = make_trainer(monkeypatch, pcgrad=True)
    trainer.config.task_sampling = "temperature"
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array(["topic", "topic"]))
    trainer._run_epoch(
        {"topic": Batches([batch(2.0)]), "emotion": Batches([batch(9.0)])}, train=True, epoch=1
    )
    torch.testing.assert_close(trainer.model.encoder.weight, torch.tensor([[-4.0]]))


def test_pcgrad_uses_original_reference_gradients_and_private_grads():
    shared = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    private_a = torch.nn.Parameter(torch.tensor(0.0))
    private_b = torch.nn.Parameter(torch.tensor(0.0))
    unused = torch.nn.Parameter(torch.tensor(5.0))
    shared.grad = torch.tensor([7.0, 7.0])
    stats = PCGrad().backward(
        {
            "a": shared[0] + 2 * private_a,
            "b": -shared[0] + shared[1] + 3 * private_b,
        },
        [shared],
        [private_a, private_b, unused],
    )
    # Original gradients (1,0),(-1,1) project to (0.5,0.5),(0,1).
    torch.testing.assert_close(shared.grad, torch.tensor([7.5, 8.5]))
    assert private_a.grad.item() == 2
    assert private_b.grad.item() == 3
    assert unused.grad is None
    assert stats["conflict_a_b"] == 1


def test_pcgrad_handles_no_shared_parameters_and_accumulation():
    private = torch.nn.Parameter(torch.tensor(0.0))
    PCGrad().backward({"a": 6 * private}, [], [private], gradient_accumulation_steps=3)
    assert private.grad.item() == 2


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("tiny_reference", [False, True])
def test_pcgrad_matches_sequential_reference_with_weights_and_accumulation(
    reduction, tiny_reference
):
    import random

    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    gradients = [
        torch.tensor(values, dtype=torch.float64)
        for values in ([1, -2, 3, 0], [-2, 1, 0, 2], [1, 1, -1, 0])
    ]
    if tiny_reference:
        gradients[2] *= 1e-8
    weights = {"0": 0.2, "1": 0.6, "2": 1.4}
    originals = [grad * weights[str(i)] / 3 for i, grad in enumerate(gradients)]
    expected = torch.zeros_like(parameter)
    random.seed(17)
    for i, original in enumerate(originals):
        projected = original.clone()
        others = [j for j in range(3) if j != i]
        random.shuffle(others)
        for j in others:
            reference = originals[j]
            dot, norm = torch.dot(projected, reference), torch.dot(reference, reference)
            if dot < 0 and norm > 1e-12:
                projected = projected - dot / norm * reference
        expected += projected
    if reduction == "mean":
        expected /= 3
    random.seed(17)
    PCGrad(reduction).backward(
        {str(i): (parameter * grad).sum() for i, grad in enumerate(gradients)},
        [parameter],
        task_weights=weights,
        gradient_accumulation_steps=3,
    )
    torch.testing.assert_close(parameter.grad, expected, rtol=1e-12, atol=1e-12)


def test_profiler_uses_real_loop_and_flushes_warmup_before_recording(monkeypatch):
    from scripts.profile_training import profile_epoch

    trainer = make_trainer(monkeypatch, accum=4)
    loaders = {"topic": Batches([batch(2.0)])}
    profile_epoch(trainer, loaders, 3)
    assert trainer.global_step == 1
    torch.testing.assert_close(trainer.model.encoder.weight, torch.tensor([[-2.0]]))
    assert trainer.model.encoder.weight.grad is None
    updates = []
    profile_epoch(trainer, loaders, 5, lambda: updates.append(trainer.global_step))
    assert updates == [1, 1, 1, 2, 3]
    torch.testing.assert_close(trainer.model.encoder.weight, torch.tensor([[-6.0]]))


def test_nonfinite_loss_fails_instead_of_silently_skipping(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer._forward_task = lambda task, value, **kwargs: (torch.tensor(float("inf")), {})
    with pytest.raises(FloatingPointError):
        trainer._run_epoch({"topic": Batches([batch()])}, train=True, epoch=1)
    assert trainer.global_step == 0


def test_stopping_epoch_is_checkpointed(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from src.training.trainer import EarlyStopping

    trainer = make_trainer(monkeypatch)
    trainer.config.max_epochs = 3
    trainer.early_stopping = EarlyStopping(patience=1)
    trainer._setup_scheduler = lambda *args: None
    trainer._log_config = lambda: None
    trainer._log_metrics = lambda *args: None
    trainer._run_epoch = lambda *args, **kwargs: {"total_loss": 1.0}
    monkeypatch.setattr("src.training.trainer.mlflow.start_run", lambda **kwargs: nullcontext())
    checkpoint = Mock()
    result = trainer.fit({"topic": Batches([batch()])}, {"topic": Batches([batch()])}, checkpoint)
    assert [call.args[0] for call in checkpoint.call_args_list] == [1, 2]
    assert "val_epoch_2" in result
