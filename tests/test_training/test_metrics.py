import unittest

import numpy as np
import pytest
import torch

from src.training.metrics import (
    accuracy,
    calculate_bleu,
    classification_report_dict,
    get_confusion_matrix,
    multilabel_f1,
    multilabel_macro_f1,
    rouge_like,
    tune_per_class_thresholds,
)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        preds = [1, 0, 1, 1]
        targets = [1, 0, 0, 1]
        acc = accuracy(preds, targets)
        self.assertEqual(acc, 0.75)

    def test_multilabel_f1(self):
        preds = torch.tensor([[1, 0, 1], [0, 1, 0]])
        targets = torch.tensor([[1, 0, 0], [0, 1, 1]])
        f1 = multilabel_f1(preds, targets)
        self.assertAlmostEqual(f1, 0.666666, places=5)

    def test_rouge_like(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello there", "foo bar baz"]
        score = rouge_like(preds, refs)
        self.assertAlmostEqual(score, 0.583333, places=5)

    def test_calculate_bleu(self):
        preds = ["this is a test"]
        refs = ["this is a test"]
        score = calculate_bleu(preds, refs)
        self.assertAlmostEqual(score, 1.0, places=5)

        preds = ["this is a test"]
        refs = ["this is not a test"]
        score = calculate_bleu(preds, refs)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_classification_report_dict(self):
        preds = ["0", "1", "0", "1"]
        targets = ["0", "0", "0", "1"]
        report = classification_report_dict(preds, targets, labels=["0", "1"])

        self.assertIn("0", report)
        self.assertIn("1", report)
        self.assertIn("macro avg", report)

        # Class 0: TP=2, FP=0, FN=1. Prec=2/2=1.0, Rec=2/3=0.666
        self.assertEqual(report["0"]["precision"], 1.0)
        self.assertAlmostEqual(report["0"]["recall"], 0.666666, places=5)

    def test_get_confusion_matrix(self):
        preds = ["0", "1", "0", "1"]
        targets = ["0", "0", "0", "1"]
        cm = get_confusion_matrix(preds, targets, labels=["0", "1"])
        expected = np.array([[2, 1], [0, 1]])
        np.testing.assert_array_equal(cm, expected)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("thresholds", [[0.9, 0.1, 0.5, 0.3], []])
def test_threshold_tuning_matches_per_class_counts_and_first_tie(thresholds):
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(37, 9, generator=generator)
    labels = torch.randint(0, 2, logits.shape, generator=generator).float()
    labels[:, 0] = 0  # A class with no positives must get the first tied threshold.
    probabilities = logits.sigmoid()
    expected = []
    for column in range(logits.shape[1]):
        scores = []
        for threshold in thresholds:
            predicted = probabilities[:, column] >= threshold
            positives = labels[:, column] == 1
            tp = int((predicted & positives).sum())
            scores.append(2 * tp / max(int(predicted.sum() + positives.sum()), 1))
        expected.append(thresholds[scores.index(max(scores))] if scores else 0.5)
    actual, f1 = tune_per_class_thresholds(logits, labels, thresholds)
    assert actual == expected
    assert f1 == pytest.approx(multilabel_macro_f1(probabilities >= torch.tensor(expected), labels))


def test_threshold_tuning_zero_true_positives_keeps_a_candidate_threshold():
    # Zero precision and recall used to produce NaN and leave the default 0.5
    # in place even when it was outside the supplied candidates.
    thresholds, f1 = tune_per_class_thresholds(
        torch.tensor([[8.0, -8.0], [-8.0, 8.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        [0.1, 0.3],
    )
    assert thresholds == [0.1, 0.1]
    assert f1 == 0
