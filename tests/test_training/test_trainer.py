"""
Tests for the training loop components.

These are unit tests that verify training components work correctly
without running full training loops (which would be too slow for unit tests).
"""

import unittest

import torch
import torch.nn as nn

from src.training.trainer import TrainerConfig


class SimpleModel(nn.Module):
    """Minimal model for testing training components."""

    def __init__(self, vocab_size: int = 100, d_model: int = 32, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, task: str, inputs: dict):
        input_ids = inputs["input_ids"]
        x = self.embedding(input_ids)  # (B, T, D)

        if task in ("emotion", "topic"):
            pooled = x.mean(dim=1)  # (B, D)
            return self.classifier(pooled)  # (B, num_classes)
        elif task == "summarization":
            return self.lm_head(x)  # (B, T, vocab)
        else:
            raise ValueError(f"Unknown task: {task}")


class TestTrainerConfig(unittest.TestCase):
    """Test trainer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainerConfig()
        self.assertEqual(config.max_epochs, 10)
        self.assertGreater(config.warmup_steps, 0)
        self.assertEqual(config.gradient_accumulation_steps, 1)

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainerConfig(
            max_epochs=5,
            warmup_steps=100,
            gradient_accumulation_steps=4,
        )
        self.assertEqual(config.max_epochs, 5)
        self.assertEqual(config.warmup_steps, 100)
        self.assertEqual(config.gradient_accumulation_steps, 4)


class TestModelForwardPass(unittest.TestCase):
    """Test model forward pass for different tasks."""

    def setUp(self):
        self.model = SimpleModel(vocab_size=100, d_model=32, num_classes=5)

    def test_topic_forward(self):
        """Test topic classification forward pass."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        logits = self.model.forward("topic", batch)
        self.assertEqual(logits.shape, (2, 5))

    def test_emotion_forward(self):
        """Test emotion (multi-label) forward pass."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        logits = self.model.forward("emotion", batch)
        self.assertEqual(logits.shape, (2, 5))

    def test_summarization_forward(self):
        """Test summarization forward pass."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
        }

        logits = self.model.forward("summarization", batch)
        self.assertEqual(logits.shape, (2, 10, 100))  # (B, T, vocab)


class TestGradientFlow(unittest.TestCase):
    """Test that gradients flow through the model."""

    def setUp(self):
        self.model = SimpleModel(vocab_size=100, d_model=32, num_classes=5)

    def test_topic_gradients(self):
        """Test gradients flow for topic classification."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
            "labels": torch.randint(0, 5, (2,)),
        }

        self.model.train()
        logits = self.model.forward("topic", batch)
        loss = nn.CrossEntropyLoss()(logits, batch["labels"])
        loss.backward()

        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters()
        )
        self.assertTrue(has_grads, "No gradients found")

    def test_emotion_gradients(self):
        """Test gradients flow for emotion (BCEWithLogits)."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
            "labels": torch.zeros(2, 5),
        }
        batch["labels"][0, 0] = 1.0
        batch["labels"][1, 2] = 1.0

        self.model.train()
        self.model.zero_grad()
        logits = self.model.forward("emotion", batch)
        loss = nn.BCEWithLogitsLoss()(logits, batch["labels"])
        loss.backward()

        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters()
        )
        self.assertTrue(has_grads, "No gradients found")

    def test_summarization_gradients(self):
        """Test gradients flow for summarization (CrossEntropy on tokens)."""
        batch = {
            "input_ids": torch.randint(1, 100, (2, 10)),
            "labels": torch.randint(0, 100, (2, 10)),
        }

        self.model.train()
        self.model.zero_grad()
        logits = self.model.forward("summarization", batch)
        # Flatten for cross entropy: (B*T, vocab) vs (B*T,)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 100), batch["labels"].view(-1))
        loss.backward()

        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters()
        )
        self.assertTrue(has_grads, "No gradients found")


if __name__ == "__main__":
    unittest.main()
