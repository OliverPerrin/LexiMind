import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader

from src.training.trainer import Trainer, TrainerConfig


class TestTrainer(unittest.TestCase):
    def setUp(self):
        # Patch mlflow to prevent real logging
        self.mlflow_patcher = patch("src.training.trainer.mlflow")
        self.mock_mlflow = self.mlflow_patcher.start()

        self.model = MagicMock()
        self.model.to.return_value = self.model  # Ensure .to() returns the same mock
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.config = TrainerConfig(max_epochs=1, logging_interval=1)
        self.device = torch.device("cpu")
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.decode_batch.return_value = ["decoded"]

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            device=self.device,
            tokenizer=self.tokenizer,
        )

    def tearDown(self):
        self.mlflow_patcher.stop()

    def test_fit_summarization(self):
        # Mock dataloader
        batch = {
            "src_ids": torch.tensor([[1, 2]]),
            "tgt_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[1, 2]]),
            "src_mask": torch.tensor([[1, 1]]),
        }
        loader = MagicMock()
        loader.__iter__.return_value = iter([batch])
        loader.__len__.return_value = 1

        loaders = {"summarization": cast(DataLoader, loader)}

        # Mock model forward
        self.model.forward.return_value = torch.randn(1, 2, 10, requires_grad=True)  # (B, T, V)

        history = self.trainer.fit(loaders)

        self.assertIn("train_epoch_1", history)
        self.assertIn("summarization_loss", history["train_epoch_1"])
        self.model.forward.assert_called()
        self.optimizer.step.assert_called()  # Scaler calls step

        # Verify mlflow calls
        self.mock_mlflow.start_run.assert_called()
        self.mock_mlflow.log_params.assert_called()
        self.mock_mlflow.log_metric.assert_called()

    def test_fit_emotion(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([[0, 1]]),
        }
        loader = MagicMock()
        loader.__iter__.return_value = iter([batch])
        loader.__len__.return_value = 1

        loaders = {"emotion": cast(DataLoader, loader)}

        # Mock model forward
        self.model.forward.return_value = torch.randn(1, 2, requires_grad=True)  # (B, num_classes)

        history = self.trainer.fit(loaders)

        self.assertIn("train_epoch_1", history)
        self.assertIn("emotion_loss", history["train_epoch_1"])
        self.assertIn("emotion_f1", history["train_epoch_1"])

    def test_fit_topic(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([1]),
        }
        loader = MagicMock()
        loader.__iter__.return_value = iter([batch])
        loader.__len__.return_value = 1

        loaders = {"topic": cast(DataLoader, loader)}

        # Mock model forward
        self.model.forward.return_value = torch.randn(1, 3, requires_grad=True)  # (B, num_classes)

        history = self.trainer.fit(loaders)

        self.assertIn("train_epoch_1", history)
        self.assertIn("topic_loss", history["train_epoch_1"])
        self.assertIn("topic_accuracy", history["train_epoch_1"])

    def test_validation_loop(self):
        batch = {
            "src_ids": torch.tensor([[1, 2]]),
            "tgt_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[1, 2]]),
        }
        loader = MagicMock()
        loader.__iter__.side_effect = lambda: iter([batch])
        loader.__len__.return_value = 1
        train_loaders = {"summarization": cast(DataLoader, loader)}
        val_loaders = {"summarization": cast(DataLoader, loader)}
        self.model.forward.return_value = torch.randn(1, 2, 10, requires_grad=True)
        self.model.forward.return_value = torch.randn(1, 2, 10, requires_grad=True)
        # Mock decoder for validation generation
        self.model.encoder.return_value = torch.randn(1, 2, 10)
        self.model.decoder.greedy_decode_naive.return_value = torch.tensor([[1, 2]])

        history = self.trainer.fit(train_loaders, val_loaders=val_loaders)

        self.assertIn("val_epoch_1", history)
        self.model.decoder.greedy_decode_naive.assert_called()


if __name__ == "__main__":
    unittest.main()
