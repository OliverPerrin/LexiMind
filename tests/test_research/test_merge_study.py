"""Merge-study metrics and bootstrap contracts on synthetic predictions."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from src.research.merge_study import (
    _dev_half,
    group_weights,
    metrics,
    select_threshold,
    summarize,
    targets,
)


@pytest.fixture
def predictions():
    rng = np.random.default_rng(3)
    truth = rng.random((400, 6)) < 0.15
    truth[:, 5] = False  # An unsupported label is skipped, not scored as zero.
    probs = np.clip(truth * 0.3 + rng.random(truth.shape) * 0.7, 0, 1)
    return probs, truth


def test_unweighted_metrics_match_scikit_learn(predictions):
    probs, truth = predictions
    result = metrics(probs, truth, 0.5, np.ones((1, len(truth))))
    supported = truth.any(axis=0)
    assert result["micro_f1"][0] == pytest.approx(
        f1_score(truth[:, supported], probs[:, supported] >= 0.5, average="micro")
    )
    assert result["macro_f1"][0] == pytest.approx(
        f1_score(truth[:, supported], probs[:, supported] >= 0.5, average="macro")
    )
    expected_ap = np.mean([average_precision_score(truth[:, j], probs[:, j]) for j in range(5)])
    assert result["macro_ap"][0] == pytest.approx(expected_ap)


def test_bootstrap_weights_duplicate_rows_like_a_resample(predictions):
    probs, truth = predictions
    groups = [str(i // 2) for i in range(len(truth))]
    weights = group_weights(groups, 20, seed=1)
    assert np.array_equal(weights[0], np.ones(len(truth)))
    assert (weights[1:].sum(axis=1) == len(truth)).all()
    assert (weights[:, ::2] == weights[:, 1::2]).all()  # Group members move together.
    resampled = np.repeat(np.arange(len(truth)), weights[3].astype(int))
    expected = metrics(probs[resampled], truth[resampled], 0.5, np.ones((1, len(resampled))))
    observed = metrics(probs, truth, 0.5, weights[3:4])
    for name in expected:
        assert observed[name][0] == pytest.approx(expected[name][0])


def test_threshold_selection_and_seed_summary(predictions):
    probs, truth = predictions
    threshold = select_threshold(probs, truth, [0.3, 0.5, 0.7])
    scores = {
        t: metrics(probs, truth, t, np.ones((1, len(truth))))["micro_f1"][0]
        for t in (0.3, 0.5, 0.7)
    }
    assert threshold == max(scores, key=scores.get)
    weights = group_weights([str(i) for i in range(len(truth))], 50, seed=0)
    first, second = metrics(probs, truth, 0.5, weights), metrics(probs**2, truth, 0.5, weights)
    summary = summarize([first, second], baseline=second)
    assert summary["micro_f1"]["seeds"] == [first["micro_f1"][0], second["micro_f1"][0]]
    assert summary["micro_f1"]["mean"] == pytest.approx(
        (first["micro_f1"][0] + second["micro_f1"][0]) / 2
    )
    low, high = summary["macro_ap"]["ci95"]
    assert low <= summary["macro_ap"]["mean"] <= high


def test_targets_and_dev_halves_are_deterministic():
    rows = [{"genre": ["romance"]}, {"genre": []}, {"genre": ["fantasy", "romance"]}]
    assert targets(rows, "genre", ["fantasy", "romance"]).tolist() == [
        [False, True],
        [False, False],
        [True, True],
    ]
    halves = {_dev_half(f"bgc-group:{i}") for i in range(20)}
    assert halves == {"dev_select", "dev_report"}
    assert _dev_half("bgc-group:7") == _dev_half("bgc-group:7")


def test_tied_scores_form_one_threshold_like_scikit_learn():
    truth = np.array([[1], [0], [1], [0], [1], [0]], dtype=bool)
    probs = np.array([[0.9], [0.9], [0.5], [0.5], [0.5], [0.1]])
    observed = metrics(probs, truth, 0.5, np.ones((1, 6)))["macro_ap"][0]
    assert observed == pytest.approx(average_precision_score(truth[:, 0], probs[:, 0]))
