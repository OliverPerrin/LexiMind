"""Pure fixtures for baseline plumbing and report formatting, never experiment runs."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts.train_bert_baseline import (
    BertBaselineConfig,
    BertTrainer,
    load_data,
    print_comparison_summary,
)
from scripts.train_multiseed import aggregate_results, generate_latex_table
from src.data.dataset import EmotionExample, split_emotion_val


class Batches:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.dataset = range(len(coefficients))

    def __len__(self):
        return len(self.coefficients)

    def __iter__(self):
        for coefficient in self.coefficients:
            yield {
                "input_ids": torch.tensor([[coefficient]]),
                "attention_mask": torch.ones(1, 1, dtype=torch.bool),
                "labels": torch.zeros(1, dtype=torch.long),
            }


def test_baseline_temperature_exponent_matches_leximind(monkeypatch):
    trainer = BertTrainer.__new__(BertTrainer)
    trainer.config = BertBaselineConfig(task_sampling_alpha=0.5)
    trainer.train_loaders = {"topic": Batches([1.0] * 16), "emotion": Batches([1.0] * 4)}
    captured = {}

    def choices(tasks, *, weights, k):
        captured.update(zip(tasks, weights, strict=True))
        return ["topic"]

    monkeypatch.setattr("scripts.train_bert_baseline.random.choices", choices)
    next(trainer._make_multitask_iterator())
    assert captured == pytest.approx({"topic": 2 / 3, "emotion": 1 / 3})


def test_baseline_partial_accumulation_uses_the_actual_window():
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, task, ids, mask):
            return self.weight * ids

    trainer = BertTrainer.__new__(BertTrainer)
    trainer.model = Scalar()
    trainer.config = BertBaselineConfig(
        use_amp=False, max_epochs=1, gradient_accumulation_steps=2, gradient_clip_norm=1e6
    )
    trainer.mode = "single-topic"
    trainer.train_loaders = {"topic": Batches([2.0, 4.0, 8.0])}
    trainer.device = torch.device("cpu")
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.scheduler = SimpleNamespace(step=Mock(), get_last_lr=lambda: [1.0])
    trainer.global_step = 0
    trainer._compute_loss = lambda task, logits, labels: logits.mean()
    trainer.train_epoch(0)
    assert trainer.model.weight.item() == -11.0
    assert trainer.global_step == 2
    assert trainer.model.weight.grad is None


def test_baseline_model_selection_and_calibration_are_disjoint(tmp_path):
    emotions = [{"text": f"sample-{i}", "emotions": ["joy"]} for i in range(8)]
    topics = [{"text": "example", "topic": "Fiction"}]
    for task, rows in (("emotion", emotions), ("topic", topics)):
        directory = tmp_path / task
        directory.mkdir()
        for split in ("train", "validation", "test"):
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
    result = load_data(BertBaselineConfig(data_dir=tmp_path))
    selected = {e.text for e in result["emotion_val"]}
    calibrated = {e.text for e in result["emotion_calibration"]}
    assert not selected & calibrated
    assert len(selected | calibrated) == 8
    expected, _ = split_emotion_val([EmotionExample(**row) for row in emotions])
    assert selected == {e.text for e in expected}
    test = load_data(BertBaselineConfig(data_dir=tmp_path), eval_split="test")
    assert len(test["emotion_val"]) == 8


def test_no_hardcoded_leximind_scores_or_missing_metric_zeros(capsys):
    print_comparison_summary({"single-topic": {"evaluation": {"topic": {"accuracy": 0.75}}}})
    output = capsys.readouterr().out
    assert "unreported" in output
    assert "0.0000" not in output
    assert "0.8571" not in output


def test_multiseed_counts_only_actual_values_and_single_seed_is_not_zero_variance():
    aggregated = aggregate_results(
        {
            17: {"topic": {"accuracy": 0.8}, "emotion": {"macro_f1": 0.4}},
            42: {"topic": {"accuracy": 0.6}, "_meta": {"seed": 42}},
        }
    )
    topic = aggregated["topic/accuracy"]
    assert topic["n_seeds"] == 2
    assert topic["seeds"] == [17, 42]
    assert topic["mean"] == pytest.approx(0.7)
    assert topic["std"] == pytest.approx(math.sqrt(0.02))
    assert aggregated["emotion/macro_f1"]["std"] is None
    assert aggregated["emotion/macro_f1"]["n_seeds"] == 1
    assert "_meta/seed" not in aggregated
    table = generate_latex_table(aggregated, [17, 42])
    assert "spread unmeasured" in table
    assert "n=1" in table and "n=2" in table


def test_multiseed_refuses_nonfinite_values():
    with pytest.raises(ValueError, match="Non-finite"):
        aggregate_results({17: {"topic": {"accuracy": float("nan")}}})


def test_weights_only_resume_does_not_apply_last_epoch_to_best(tmp_path):
    from scripts.train import resume_start_epoch

    (tmp_path / "last_epoch.json").write_text('{"epoch": 8}')
    assert resume_start_epoch(tmp_path / "last.pt") == 9
    assert resume_start_epoch(tmp_path / "best.pt") == 1
    assert resume_start_epoch(tmp_path / "epoch_3.pt") == 4
    assert resume_start_epoch(tmp_path / "seed_17_best.pt") == 1


def test_training_split_loader_does_not_open_test_data(tmp_path):
    from scripts.train import load_splits

    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text("synthetic fixture")
    calls = []
    result = load_splits(tmp_path, lambda path: calls.append(path) or [])
    assert set(result) == {"train", "val"}
    assert all(not path.endswith("/test.jsonl") for path in calls)
