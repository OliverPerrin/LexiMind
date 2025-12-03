import unittest

import numpy as np
import torch

from src.training.metrics import (
    accuracy,
    calculate_bleu,
    classification_report_dict,
    get_confusion_matrix,
    multilabel_f1,
    rouge_like,
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
