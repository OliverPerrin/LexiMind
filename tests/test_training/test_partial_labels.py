"""Tiny CPU fixtures for observed-label training plumbing, never corpus experiments."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from src.models.heads import ClassificationHead
from src.models.losses import masked_binary_cross_entropy
from src.training.metrics import ObservedMultilabelMetrics
from src.training.pcgrad import PCGrad
from src.training.trainer import Trainer, TrainerConfig


class Batches:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = range(sum(len(batch["labels"]) for batch in batches))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class SyntheticClassifier(torch.nn.Module):
    def __init__(self, problem_type="multi_label"):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.head_topic = ClassificationHead(2, 2, dropout=0.0, problem_type=problem_type)
        self.heads = {"topic": self.head_topic}
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.head_topic.out_proj.weight.copy_(torch.eye(2))
            self.head_topic.out_proj.bias.zero_()

    def forward(self, task, inputs):
        return self.heads[task](self.encoder(inputs["input_ids"]).unsqueeze(1))


def make_trainer(monkeypatch, *, accum=1, pcgrad=False, problem_type="multi_label"):
    trainer = Trainer.__new__(Trainer)
    trainer.policy_objectives = {}
    trainer.model = SyntheticClassifier(problem_type)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.01, weight_decay=0.3)
    trainer.config = TrainerConfig(
        gradient_accumulation_steps=accum,
        gradient_clip_norm=1e6,
        task_sampling="round_robin",
        scheduler_type="constant",
    )
    trainer.device = torch.device("cpu")
    trainer.use_amp = trainer.use_bfloat16 = False
    trainer.pcgrad = PCGrad() if pcgrad else None
    trainer.scheduler = None
    trainer.global_step = 0
    monkeypatch.setattr("src.training.trainer.mlflow.log_metric", lambda *args, **kwargs: None)
    return trainer


def batch(known=True):
    return {
        "input_ids": torch.tensor([[0.2, -0.3]]),
        "labels": torch.tensor([[1.0, 0.0]]) if known else torch.full((1, 2), float("nan")),
        "label_mask": torch.tensor([[known, known]]),
    }


@pytest.mark.parametrize("pcgrad", [False, True])
def test_unknown_accumulation_windows_do_not_step_decay_moments_or_scheduler(monkeypatch, pcgrad):
    trainer = make_trainer(monkeypatch, accum=2, pcgrad=pcgrad)
    # Populate Adam moments first; even an apparent zero-gradient step would
    # update both moments and parameters after this preceding observed batch.
    trainer._run_epoch({"topic": Batches([batch()])}, train=True, epoch=1)
    before = deepcopy(trainer.model.state_dict())
    state_before = deepcopy(trainer.optimizer.state_dict())
    trainer.scheduler = Mock()
    trainer._run_epoch({"topic": Batches([batch(False)] * 3)}, train=True, epoch=2)
    assert trainer.global_step == 1
    trainer.scheduler.step.assert_not_called()
    for key, expected in before.items():
        assert torch.equal(trainer.model.state_dict()[key], expected)
    state_after = trainer.optimizer.state_dict()
    assert state_before["param_groups"] == state_after["param_groups"]
    for param, values in state_before["state"].items():
        for name, expected in values.items():
            assert torch.equal(state_after["state"][param][name], expected)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_unknown_only_validation_cannot_report_a_perfect_zero_loss(monkeypatch):
    trainer = make_trainer(monkeypatch)
    with pytest.raises(ValueError, match="Validation task 'topic' has no observed labels"):
        trainer._run_epoch({"topic": Batches([batch(False)])}, train=False, epoch=1)
    assert trainer.global_step == 0


def test_each_selected_validation_task_requires_observations(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.model.heads["emotion"] = trainer.model.head_topic
    with pytest.raises(ValueError, match="Validation task 'topic' has no observed labels"):
        trainer._run_epoch(
            {"emotion": Batches([batch()]), "topic": Batches([batch(False)])},
            train=False,
            epoch=1,
        )


def test_fit_never_observes_unknown_validation_as_best_checkpoint_or_early_stop(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.config.max_epochs = 1
    trainer.early_stopping = Mock(return_value=False)
    trainer._log_config = Mock()
    trainer._log_metrics = Mock()
    monkeypatch.setattr("src.training.trainer.mlflow.start_run", lambda **kwargs: nullcontext())
    checkpoint = Mock()
    with pytest.raises(ValueError, match="cannot be used for early stopping or checkpoint"):
        trainer.fit(
            {"topic": Batches([batch()])},
            {"topic": Batches([batch(False)])},
            checkpoint_callback=checkpoint,
        )
    trainer.early_stopping.assert_not_called()
    checkpoint.assert_not_called()
    assert all(call.args[1] == "train" for call in trainer._log_metrics.call_args_list)


@pytest.mark.parametrize("pcgrad", [False, True])
def test_unknown_batch_does_not_discard_observed_gradients_in_same_window(monkeypatch, pcgrad):
    trainer = make_trainer(monkeypatch, accum=2, pcgrad=pcgrad)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    reference = deepcopy(trainer.model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    observed = batch()
    # Existing accumulation averages microbatch losses, including the unknown
    # batch's zero contribution; it is not a merged-superbatch observed mean.
    (masked_binary_cross_entropy(reference("topic", observed), observed["labels"]) / 2).backward()
    reference_optimizer.step()
    trainer._run_epoch({"topic": Batches([observed, batch(False)])}, train=True, epoch=1)
    assert trainer.global_step == 1
    for actual, expected in zip(trainer.model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("pcgrad", [False, True])
def test_unknown_private_head_stays_unchanged_when_other_task_is_observed(monkeypatch, pcgrad):
    trainer = make_trainer(monkeypatch, pcgrad=pcgrad)
    trainer.model.head_emotion = ClassificationHead(2, 2, dropout=0.0, problem_type="multi_label")
    trainer.model.heads["emotion"] = trainer.model.head_emotion
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.01, weight_decay=0.3)
    # Warm the topic head's optimizer moments before making its labels unknown.
    trainer._run_epoch({"topic": Batches([batch()])}, train=True, epoch=1)
    before = deepcopy(trainer.model.head_topic.state_dict())
    trainer._run_epoch(
        {"topic": Batches([batch(False)]), "emotion": Batches([batch()])}, train=True, epoch=2
    )
    assert trainer.global_step == 2
    for key, expected in before.items():
        assert torch.equal(trainer.model.head_topic.state_dict()[key], expected)


def test_epoch_metrics_use_observed_counts_not_batch_f1_or_argmax(monkeypatch):
    trainer = make_trainer(monkeypatch)
    batches = [
        {
            "input_ids": torch.tensor([[4.0, -4.0], [4.0, -4.0]]),
            "labels": torch.tensor([[1.0, 0.0], [1.0, float("nan")]]),
            "label_mask": torch.tensor([[True, True], [True, False]]),
        },
        {
            "input_ids": torch.tensor([[-4.0, 4.0]]),
            "labels": torch.tensor([[1.0, 1.0]]),
            "label_mask": torch.tensor([[True, True]]),
        },
        batch(False),
    ]
    result = trainer._run_epoch({"topic": Batches(batches)}, train=False, epoch=1)
    assert result["topic_observed_micro_f1"] == pytest.approx(6 / 7)
    assert result["topic_observed_micro_precision"] == pytest.approx(1.0)
    assert result["topic_observed_micro_recall"] == pytest.approx(0.75)
    assert result["topic_observed_macro_f1"] == pytest.approx(0.9)
    assert result["topic_label_coverage"] == pytest.approx(5 / 8)
    assert result["topic_observed_label_count"] == 5
    assert result["topic_observed_positive_count"] == 4
    assert result["topic_observed_negative_count"] == 1
    assert "topic_accuracy" not in result
    assert "topic_f1" not in result
    logits = torch.cat([trainer.model("topic", value) for value in batches])
    labels = torch.cat([value["labels"] for value in batches])
    mask = torch.cat([value["label_mask"] for value in batches])
    expected = masked_binary_cross_entropy(logits, labels, mask)
    assert result["topic_loss"] == pytest.approx(expected.item())
    assert result["total_loss"] == pytest.approx(expected.item())
    # Split the same examples differently: aggregate loss and metrics agree.
    repartitioned = [
        {
            "input_ids": logits.detach()[i : i + 1],
            "labels": labels[i : i + 1],
            "label_mask": mask[i : i + 1],
        }
        for i in range(4)
    ]
    assert trainer._run_epoch(
        {"topic": Batches(repartitioned)}, train=False, epoch=1
    ) == pytest.approx(result)


def test_observed_metrics_ignore_unknown_predictions_and_report_zero_coverage():
    empty = ObservedMultilabelMetrics(
        torch.tensor([[True, False]]),
        torch.full((1, 2), float("nan")),
        torch.zeros((1, 2), dtype=torch.bool),
    ).compute()
    assert empty["label_coverage"] == 0
    assert empty["observed_label_count"] == 0
    assert empty["observed_micro_f1"] == 0
    assert empty["total_label_count"] == 2
    labels = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, False]])
    a = ObservedMultilabelMetrics(torch.tensor([[True, False]]), labels, mask).compute()
    b = ObservedMultilabelMetrics(torch.tensor([[True, True]]), labels, mask).compute()
    assert a == b


def test_epoch_computes_observed_summary_only_after_merging(monkeypatch):
    trainer = make_trainer(monkeypatch)
    compute = ObservedMultilabelMetrics.compute
    calls = []

    def counted_summary(counts):
        calls.append(counts.total)
        return compute(counts)

    monkeypatch.setattr(ObservedMultilabelMetrics, "compute", counted_summary)
    result = trainer._run_epoch({"topic": Batches([batch()] * 3)}, train=False, epoch=1)
    assert calls == [6]
    assert result["topic_label_coverage"] == 1.0
    # Standalone batch callers retain their complete public metric dictionary.
    _, metrics = trainer._forward_task("topic", batch())
    assert metrics["label_coverage"] == 1.0
    assert calls == [6, 2]


def test_legacy_topic_ce_and_dense_emotion_bce_keep_their_defaults(monkeypatch):
    trainer = make_trainer(monkeypatch, problem_type="single_label")
    values = {"input_ids": torch.tensor([[1.0, -1.0]]), "labels": torch.tensor([0])}
    loss, metrics = trainer._forward_task("topic", values)
    torch.testing.assert_close(
        loss, F.cross_entropy(trainer.model("topic", values), values["labels"])
    )
    assert metrics == {"accuracy": 1.0}
    trainer.model.head_topic.problem_type = "multi_label"
    trainer.model.heads["emotion"] = trainer.model.head_topic
    values["labels"] = torch.tensor([[1.0, 0.0]])
    loss, metrics = trainer._forward_task("emotion", values)
    torch.testing.assert_close(
        loss, F.binary_cross_entropy_with_logits(trainer.model("emotion", values), values["labels"])
    )
    assert metrics["f1"] == 1.0
    assert metrics["label_coverage"] == 1.0


def test_explicit_single_label_rejects_mask_instead_of_silently_ignoring_it(monkeypatch):
    trainer = make_trainer(monkeypatch, problem_type="single_label")
    values = batch()
    values["labels"] = torch.tensor([0])
    with pytest.raises(ValueError, match="requires a multi_label"):
        trainer._forward_task("topic", values)


def test_model_and_trainer_share_masked_loss(monkeypatch):
    from src.models.factory import ModelConfig, build_multitask_model

    trainer = make_trainer(monkeypatch)
    trainer.model = build_multitask_model(
        SimpleNamespace(vocab_size=13, pad_token_id=0, config=SimpleNamespace(max_length=4)),
        num_emotions=0,
        num_topics=2,
        topic_problem_type="multi_label",
        config=ModelConfig(
            d_model=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_attention_heads=1,
            ffn_dim=8,
            dropout=0.0,
        ),
        load_pretrained=False,
    )
    values = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[1.0, float("nan")]]),
        "label_mask": torch.tensor([[True, False]]),
    }
    model_loss, _ = trainer.model("topic", values, return_loss=True)
    trainer_loss, _ = trainer._forward_task("topic", values)
    torch.testing.assert_close(model_loss, trainer_loss)
