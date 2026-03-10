"""
BERT Baseline Training for LexiMind Comparison.

Fine-tunes bert-base-uncased on topic classification and emotion detection
to provide baselines for comparison with LexiMind (FLAN-T5-based).

Supports three training modes to disentangle architecture vs. MTL effects:
  1. single-topic   — BERT fine-tuned on topic classification only
  2. single-emotion  — BERT fine-tuned on emotion detection only
  3. multitask       — BERT fine-tuned on both tasks jointly

Uses the same datasets, splits, label encoders, and evaluation metrics as the
main LexiMind pipeline for fair comparison.

Usage:
    python scripts/train_bert_baseline.py --mode single-topic
    python scripts/train_bert_baseline.py --mode single-emotion
    python scripts/train_bert_baseline.py --mode multitask
    python scripts/train_bert_baseline.py --mode all  # Run all three sequentially

Author: Oliver Perrin
Date: March 2026
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    EmotionExample,
    TopicExample,
    load_emotion_jsonl,
    load_topic_jsonl,
)
from src.training.metrics import (
    bootstrap_confidence_interval,
    multilabel_f1,
    multilabel_macro_f1,
    multilabel_micro_f1,
    multilabel_per_class_metrics,
    tune_per_class_thresholds,
)

# Configuration


@dataclass
class BertBaselineConfig:
    """Hyperparameters aligned with LexiMind's full.yaml where applicable."""

    # Model
    model_name: str = "bert-base-uncased"
    max_length: int = 256  # Same as LexiMind classification max_len

    # Optimizer (matching LexiMind's full.yaml)
    lr: float = 3e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6

    # Training
    batch_size: int = 10  # Same as LexiMind
    gradient_accumulation_steps: int = 4  # Same effective batch = 40
    max_epochs: int = 8
    warmup_steps: int = 300
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    seed: int = 17  # Same as LexiMind

    # Task weights (for multi-task mode)
    topic_weight: float = 0.3  # Same as LexiMind
    emotion_weight: float = 1.0

    # Temperature sampling (for multi-task mode)
    task_sampling_alpha: float = 0.5

    # Frozen layers: freeze bottom 4 layers (matching LexiMind's encoder strategy)
    freeze_layers: int = 4

    # Precision
    use_amp: bool = True  # BFloat16 mixed precision

    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "bert_baseline")
    checkpoint_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "checkpoints" / "bert_baseline"
    )

    # Emotion threshold
    emotion_threshold: float = 0.3


# Datasets


class BertEmotionDataset(Dataset):
    """Tokenized emotion dataset for BERT."""

    def __init__(
        self,
        examples: List[EmotionExample],
        tokenizer: AutoTokenizer,
        binarizer: MultiLabelBinarizer,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.binarizer = binarizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.binarizer.transform([ex.emotions])[0]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class BertTopicDataset(Dataset):
    """Tokenized topic dataset for BERT."""

    def __init__(
        self,
        examples: List[TopicExample],
        tokenizer: AutoTokenizer,
        encoder: LabelEncoder,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = self.encoder.transform([ex.topic])[0]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Model


class BertClassificationHead(nn.Module):
    """Classification head on top of BERT [CLS] token.

    For emotion: uses attention pooling + 2-layer MLP (matching LexiMind's emotion head)
    For topic: uses [CLS] + single linear (matching LexiMind's mean pool + linear)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        pooling: str = "cls",  # "cls" or "attention"
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            self.attn_query = nn.Linear(hidden_size, 1, bias=False)

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            # Learned attention pooling (same mechanism as LexiMind)
            scores = self.attn_query(hidden_states)  # (B, L, 1)
            mask = attention_mask.unsqueeze(-1).bool()
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1)
            pooled = (weights * hidden_states).sum(dim=1)
        elif self.pooling == "mean":
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # [CLS] token
            pooled = hidden_states[:, 0, :]

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class BertBaseline(nn.Module):
    """BERT baseline model with task-specific heads.

    Supports single-task and multi-task configurations.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 28,
        num_topics: int = 7,
        tasks: Sequence[str] = ("emotion", "topic"),
        freeze_layers: int = 4,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        self.tasks = list(tasks)
        self.heads = nn.ModuleDict()

        if "emotion" in tasks:
            # Attention pooling + 2-layer MLP (matching LexiMind's emotion head)
            self.heads["emotion"] = BertClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_emotions,
                pooling="attention",
                hidden_dim=hidden_size // 2,  # 384, same ratio as LexiMind
                dropout=0.1,
            )

        if "topic" in tasks:
            # Mean pooling + single linear (matching LexiMind's topic head)
            self.heads["topic"] = BertClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_topics,
                pooling="mean",
                hidden_dim=None,
                dropout=0.1,
            )

        # Freeze bottom N encoder layers (matching LexiMind's strategy)
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n: int) -> None:
        """Freeze embedding + bottom n encoder layers."""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze bottom n layers
        for i in range(min(n, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        frozen = sum(1 for p in self.bert.parameters() if not p.requires_grad)
        total = sum(1 for p in self.bert.parameters())
        print(f"  Frozen {frozen}/{total} BERT parameters (bottom {n} layers + embeddings)")

    def forward(
        self,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, 768)
        return self.heads[task](hidden_states, attention_mask)

    def param_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["bert_encoder"] = sum(p.numel() for p in self.bert.parameters())
        counts["bert_trainable"] = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        for name, head in self.heads.items():
            counts[f"head_{name}"] = sum(p.numel() for p in head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# Training


class BertTrainer:
    """Trainer supporting single-task and multi-task BERT training."""

    def __init__(
        self,
        model: BertBaseline,
        config: BertBaselineConfig,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        device: torch.device,
        mode: str,
    ):
        self.model = model
        self.config = config
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.device = device
        self.mode = mode

        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        # Calculate total training steps
        if len(train_loaders) > 1:
            # Multi-task: use temperature-sampled steps
            sizes = {k: len(v) for k, v in train_loaders.items()}
            total_batches = sum(sizes.values())
        else:
            total_batches = sum(len(v) for v in train_loaders.values())
        self.steps_per_epoch = total_batches // config.gradient_accumulation_steps
        self.total_steps = self.steps_per_epoch * config.max_epochs

        # LR scheduler: linear warmup + cosine decay (matching LexiMind)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8 / config.lr,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.total_steps - config.warmup_steps, 1),
            eta_min=config.lr * 0.1,  # Decay to 10% of peak (matching LexiMind)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps],
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)

        # Loss functions
        self.emotion_loss_fn = nn.BCEWithLogitsLoss()
        self.topic_loss_fn = nn.CrossEntropyLoss()

        # Tracking
        self.global_step = 0
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.training_history: List[Dict[str, Any]] = []

    def _compute_loss(self, task: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if task == "emotion":
            return self.emotion_loss_fn(logits, labels)
        else:
            return self.topic_loss_fn(logits, labels)

    def _get_task_weight(self, task: str) -> float:
        if self.mode != "multitask":
            return 1.0
        if task == "topic":
            return self.config.topic_weight
        return self.config.emotion_weight

    def _make_multitask_iterator(self):
        """Temperature-based task sampling (matching LexiMind)."""
        sizes = {k: len(v.dataset) for k, v in self.train_loaders.items()}
        alpha = self.config.task_sampling_alpha

        # Compute sampling probabilities
        raw = {k: s ** (1.0 / alpha) for k, s in sizes.items()}
        total = sum(raw.values())
        probs = {k: v / total for k, v in raw.items()}

        # Create iterators
        iters = {k: iter(v) for k, v in self.train_loaders.items()}
        tasks = list(probs.keys())
        weights = [probs[t] for t in tasks]

        while True:
            task = random.choices(tasks, weights=weights, k=1)[0]
            try:
                batch = next(iters[task])
            except StopIteration:
                iters[task] = iter(self.train_loaders[task])
                batch = next(iters[task])
            yield task, batch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        epoch_losses: Dict[str, List[float]] = {t: [] for t in self.train_loaders}

        if len(self.train_loaders) > 1:
            # Multi-task: temperature sampling
            iterator = self._make_multitask_iterator()
            total_batches = sum(len(v) for v in self.train_loaders.values())
        else:
            # Single-task: iterate normally
            task_name = list(self.train_loaders.keys())[0]
            iterator = ((task_name, batch) for batch in self.train_loaders[task_name])
            total_batches = len(self.train_loaders[task_name])

        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{self.config.max_epochs}")

        for step_in_epoch in range(total_batches):
            task, batch = next(iterator)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with AMP
            with autocast(dtype=torch.bfloat16, enabled=self.config.use_amp):
                logits = self.model(task, input_ids, attention_mask)
                loss = self._compute_loss(task, logits, labels)
                loss = loss * self._get_task_weight(task)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()
            epoch_losses[task].append(loss.item() * self.config.gradient_accumulation_steps)

            # Optimizer step (every N accumulation steps)
            if (step_in_epoch + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            pbar.set_postfix(
                {
                    f"{task}_loss": f"{epoch_losses[task][-1]:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )
            pbar.update(1)

        pbar.close()

        # Aggregate
        results = {}
        for task, losses in epoch_losses.items():
            if losses:
                results[f"train_{task}_loss"] = sum(losses) / len(losses)
        return results

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Run validation across all tasks."""
        self.model.eval()
        results: Dict[str, Any] = {}

        for task, loader in self.val_loaders.items():
            all_logits = []
            all_labels = []
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(dtype=torch.bfloat16, enabled=self.config.use_amp):
                    logits = self.model(task, input_ids, attention_mask)
                    loss = self._compute_loss(task, logits, labels)

                total_loss += loss.item()
                n_batches += 1
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.float().cpu())

            all_logits_t = torch.cat(all_logits, dim=0)
            all_labels_t = torch.cat(all_labels, dim=0)
            results[f"val_{task}_loss"] = total_loss / max(n_batches, 1)

            if task == "emotion":
                preds = (torch.sigmoid(all_logits_t) > self.config.emotion_threshold).int()
                targets = all_labels_t.int()
                results["val_emotion_sample_f1"] = multilabel_f1(preds, targets)
                results["val_emotion_macro_f1"] = multilabel_macro_f1(preds, targets)
                results["val_emotion_micro_f1"] = multilabel_micro_f1(preds, targets)
                # Store raw logits for threshold tuning later
                results["_emotion_logits"] = all_logits_t
                results["_emotion_labels"] = all_labels_t

            elif task == "topic":
                preds = all_logits_t.argmax(dim=1).numpy()
                targets = all_labels_t.long().numpy()
                results["val_topic_accuracy"] = float(accuracy_score(targets, preds))
                results["val_topic_macro_f1"] = float(
                    f1_score(targets, preds, average="macro", zero_division=0)
                )

        # Combined metric for early stopping / checkpointing
        metric_parts = []
        if "val_emotion_sample_f1" in results:
            metric_parts.append(results["val_emotion_sample_f1"])
        if "val_topic_accuracy" in results:
            metric_parts.append(results["val_topic_accuracy"])
        results["val_combined_metric"] = sum(metric_parts) / max(len(metric_parts), 1)

        return results

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Filter out tensors from metrics
        clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": clean_metrics,
                "config": {
                    "mode": self.mode,
                    "tasks": self.model.tasks,
                    "model_name": self.config.model_name,
                },
            },
            path,
        )

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"\n{'=' * 60}")
        print(f"Training BERT Baseline — Mode: {self.mode}")
        print(f"{'=' * 60}")

        param_counts = self.model.param_count()
        print(f"  Total parameters:     {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
        for name, count in param_counts.items():
            if name.startswith("head_"):
                print(f"  {name}: {count:,}")
        print(f"  Steps/epoch: {self.steps_per_epoch}")
        print(f"  Total steps: {self.total_steps}")
        print()

        all_results: Dict[str, Any] = {"mode": self.mode, "epochs": []}
        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Log
            epoch_result = {
                "epoch": epoch + 1,
                "time_seconds": epoch_time,
                **train_metrics,
                **{k: v for k, v in val_metrics.items() if not k.startswith("_")},
            }
            all_results["epochs"].append(epoch_result)
            self.training_history.append(epoch_result)

            # Print summary
            print(f"\n  Epoch {epoch + 1} ({epoch_time:.0f}s):")
            for k, v in sorted(epoch_result.items()):
                if k not in ("epoch", "time_seconds") and isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

            # Checkpointing
            combined = val_metrics["val_combined_metric"]
            if combined > self.best_metric:
                self.best_metric = combined
                self.patience_counter = 0
                self.save_checkpoint(
                    self.config.checkpoint_dir / self.mode / "best.pt",
                    epoch,
                    val_metrics,
                )
                print(f" New best model (combined metric: {combined:.4f})")
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement ({self.patience_counter}/{self.config.early_stopping_patience})"
                )

            # Always save epoch checkpoint
            self.save_checkpoint(
                self.config.checkpoint_dir / self.mode / f"epoch_{epoch + 1}.pt",
                epoch,
                val_metrics,
            )

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                all_results["early_stopped"] = True
                all_results["best_epoch"] = epoch + 1 - self.config.early_stopping_patience
                break

        total_time = time.time() - start_time
        all_results["total_time_seconds"] = total_time
        all_results["total_time_human"] = f"{total_time / 3600:.1f}h"
        if "early_stopped" not in all_results:
            all_results["early_stopped"] = False
            all_results["best_epoch"] = (
                epoch + 1 - self.patience_counter if self.patience_counter > 0 else epoch + 1
            )
        all_results["param_counts"] = param_counts

        print(f"\n  Training complete in {total_time / 3600:.1f}h")
        print(f"  Best combined metric: {self.best_metric:.4f}")

        return all_results


# Evaluation


def evaluate_bert_model(
    model: BertBaseline,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    config: BertBaselineConfig,
    emotion_classes: Optional[List[str]] = None,
    topic_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full evaluation with the same metrics as LexiMind's evaluate.py."""
    model.eval()
    results: Dict[str, Any] = {}

    with torch.no_grad():
        for task, loader in val_loaders.items():
            all_logits = []
            all_labels = []

            for batch in tqdm(loader, desc=f"Evaluating {task}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(dtype=torch.bfloat16, enabled=config.use_amp):
                    logits = model(task, input_ids, attention_mask)

                all_logits.append(logits.float().cpu())
                all_labels.append(labels.float().cpu())

            all_logits_t = torch.cat(all_logits, dim=0)
            all_labels_t = torch.cat(all_labels, dim=0)

            if task == "emotion":
                # Default threshold
                preds_default = (torch.sigmoid(all_logits_t) > config.emotion_threshold).int()
                targets = all_labels_t.int()

                results["emotion"] = {
                    "default_threshold": config.emotion_threshold,
                    "sample_avg_f1": multilabel_f1(preds_default, targets),
                    "macro_f1": multilabel_macro_f1(preds_default, targets),
                    "micro_f1": multilabel_micro_f1(preds_default, targets),
                }

                # Per-class metrics
                if emotion_classes:
                    per_class = multilabel_per_class_metrics(
                        preds_default, targets, emotion_classes
                    )
                    results["emotion"]["per_class"] = per_class

                # Threshold tuning
                best_thresholds, tuned_macro = tune_per_class_thresholds(all_logits_t, all_labels_t)
                tuned_preds = torch.zeros_like(all_logits_t)
                probs = torch.sigmoid(all_logits_t)
                for c in range(all_logits_t.shape[1]):
                    tuned_preds[:, c] = (probs[:, c] >= best_thresholds[c]).float()
                tuned_preds = tuned_preds.int()

                results["emotion"]["tuned_macro_f1"] = tuned_macro
                results["emotion"]["tuned_sample_avg_f1"] = multilabel_f1(tuned_preds, targets)
                results["emotion"]["tuned_micro_f1"] = multilabel_micro_f1(tuned_preds, targets)

                # Bootstrap CI on sample-avg F1
                per_sample_f1 = []
                for i in range(preds_default.shape[0]):
                    p = preds_default[i].float()
                    g = targets[i].float()
                    tp = (p * g).sum()
                    prec = tp / p.sum().clamp(min=1)
                    rec = tp / g.sum().clamp(min=1)
                    f = (2 * prec * rec) / (prec + rec).clamp(min=1e-8)
                    per_sample_f1.append(f.item())
                mean_f1, ci_low, ci_high = bootstrap_confidence_interval(per_sample_f1)
                results["emotion"]["sample_avg_f1_ci"] = [ci_low, ci_high]

            elif task == "topic":
                preds = all_logits_t.argmax(dim=1).numpy()
                targets = all_labels_t.long().numpy()

                acc = float(accuracy_score(targets, preds))
                macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))

                results["topic"] = {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                }

                # Per-class metrics
                if topic_classes:
                    report = classification_report(
                        targets,
                        preds,
                        target_names=topic_classes,
                        output_dict=True,
                        zero_division=0,
                    )
                    results["topic"]["per_class"] = {
                        name: {
                            "precision": report[name]["precision"],
                            "recall": report[name]["recall"],
                            "f1": report[name]["f1-score"],
                            "support": report[name]["support"],
                        }
                        for name in topic_classes
                        if name in report
                    }

                # Bootstrap CI on accuracy
                per_sample_correct = (preds == targets).astype(float).tolist()
                mean_acc, ci_low, ci_high = bootstrap_confidence_interval(per_sample_correct)
                results["topic"]["accuracy_ci"] = [ci_low, ci_high]

    return results


# Main Pipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config: BertBaselineConfig):
    """Load all datasets and create label encoders."""
    data_dir = config.data_dir

    # Load emotion data
    emo_train = load_emotion_jsonl(str(data_dir / "emotion" / "train.jsonl"))
    emo_val_path = data_dir / "emotion" / "validation.jsonl"
    if not emo_val_path.exists():
        emo_val_path = data_dir / "emotion" / "val.jsonl"
    emo_val = load_emotion_jsonl(str(emo_val_path))

    # Load topic data
    top_train = load_topic_jsonl(str(data_dir / "topic" / "train.jsonl"))
    top_val_path = data_dir / "topic" / "validation.jsonl"
    if not top_val_path.exists():
        top_val_path = data_dir / "topic" / "val.jsonl"
    top_val = load_topic_jsonl(str(top_val_path))

    # Fit label encoders on training data (same as LexiMind)
    binarizer = MultiLabelBinarizer()
    binarizer.fit([ex.emotions for ex in emo_train])

    label_encoder = LabelEncoder()
    label_encoder.fit([ex.topic for ex in top_train])

    print(
        f"  Emotion: {len(emo_train)} train, {len(emo_val)} val, {len(binarizer.classes_)} classes"
    )
    print(
        f"  Topic:   {len(top_train)} train, {len(top_val)} val, {len(label_encoder.classes_)} classes"
    )
    print(f"  Emotion classes: {list(binarizer.classes_)[:5]}...")
    print(f"  Topic classes:   {list(label_encoder.classes_)}")

    return {
        "emotion_train": emo_train,
        "emotion_val": emo_val,
        "topic_train": top_train,
        "topic_val": top_val,
        "binarizer": binarizer,
        "label_encoder": label_encoder,
    }


def run_experiment(mode: str, config: BertBaselineConfig) -> Dict[str, Any]:
    """Run a single experiment (single-topic, single-emotion, or multitask)."""
    print(f"\n{'═' * 60}")
    print(f"  BERT BASELINE EXPERIMENT: {mode.upper()}")
    print(f"{'═' * 60}")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True

    # Load tokenizer
    print(f"\n  Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load data
    print("  Loading datasets...")
    data = load_data(config)

    # Determine tasks for this mode
    if mode == "single-topic":
        tasks = ["topic"]
    elif mode == "single-emotion":
        tasks = ["emotion"]
    else:
        tasks = ["emotion", "topic"]

    # Create datasets
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}

    if "emotion" in tasks:
        emo_train_ds = BertEmotionDataset(
            data["emotion_train"], tokenizer, data["binarizer"], config.max_length
        )
        emo_val_ds = BertEmotionDataset(
            data["emotion_val"], tokenizer, data["binarizer"], config.max_length
        )
        train_loaders["emotion"] = DataLoader(
            emo_train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loaders["emotion"] = DataLoader(
            emo_val_ds,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    if "topic" in tasks:
        top_train_ds = BertTopicDataset(
            data["topic_train"], tokenizer, data["label_encoder"], config.max_length
        )
        top_val_ds = BertTopicDataset(
            data["topic_val"], tokenizer, data["label_encoder"], config.max_length
        )
        train_loaders["topic"] = DataLoader(
            top_train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loaders["topic"] = DataLoader(
            top_val_ds,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Create model
    print(f"\n  Creating model with tasks: {tasks}")
    model = BertBaseline(
        model_name=config.model_name,
        num_emotions=len(data["binarizer"].classes_),
        num_topics=len(data["label_encoder"].classes_),
        tasks=tasks,
        freeze_layers=config.freeze_layers,
    ).to(device)

    # Train
    trainer = BertTrainer(model, config, train_loaders, val_loaders, device, mode)
    training_results = trainer.train()

    # Load best checkpoint for final evaluation
    best_path = config.checkpoint_dir / mode / "best.pt"
    if best_path.exists():
        print("\n  Loading best checkpoint for final evaluation...")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Full evaluation
    print("\n  Running final evaluation...")
    eval_results = evaluate_bert_model(
        model,
        val_loaders,
        device,
        config,
        emotion_classes=list(data["binarizer"].classes_) if "emotion" in tasks else None,
        topic_classes=list(data["label_encoder"].classes_) if "topic" in tasks else None,
    )

    # Combine results
    final_results = {
        "mode": mode,
        "model": config.model_name,
        "tasks": tasks,
        "training": training_results,
        "evaluation": eval_results,
    }

    # Save results
    output_path = config.output_dir / f"{mode}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable fields
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(final_results), f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return final_results


def print_comparison_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a side-by-side comparison of all experiments."""
    print(f"\n{'═' * 70}")
    print("  BERT BASELINE COMPARISON SUMMARY")
    print(f"{'═' * 70}")

    # Header
    modes = list(all_results.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>16}" for m in modes) + f"{'LexiMind':>16}"
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")

    # LexiMind reference values
    lexmind = {
        "topic_accuracy": 0.8571,
        "topic_macro_f1": 0.8539,
        "emotion_sample_f1": 0.3523,
        "emotion_macro_f1": 0.1432,
        "emotion_micro_f1": 0.4430,
        "emotion_tuned_macro_f1": 0.2936,
    }

    # Topic metrics
    print(f"\n  {'Topic Classification':}")
    for metric_name, display_name in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
    ]:
        row = f"  {display_name:<30}"
        for mode in modes:
            eval_data = all_results[mode].get("evaluation", {})
            topic = eval_data.get("topic", {})
            val = topic.get(metric_name, None)
            row += f"{val:>16.4f}" if val is not None else f"{'—':>16}"
        lm_key = f"topic_{metric_name}"
        row += f"{lexmind.get(lm_key, 0):>16.4f}"
        print(row)

    # Emotion metrics
    print(f"\n  {'Emotion Detection':}")
    for metric_name, display_name in [
        ("sample_avg_f1", "Sample-avg F1 (τ=0.3)"),
        ("macro_f1", "Macro F1 (τ=0.3)"),
        ("micro_f1", "Micro F1 (τ=0.3)"),
        ("tuned_macro_f1", "Tuned Macro F1"),
        ("tuned_sample_avg_f1", "Tuned Sample-avg F1"),
    ]:
        row = f"  {display_name:<30}"
        for mode in modes:
            eval_data = all_results[mode].get("evaluation", {})
            emo = eval_data.get("emotion", {})
            val = emo.get(metric_name, None)
            row += f"{val:>16.4f}" if val is not None else f"{'—':>16}"
        lm_key = f"emotion_{metric_name}"
        row += f"{lexmind.get(lm_key, 0):>16.4f}"
        print(row)

    # Training time
    print(f"\n  {'Training Time':}")
    row = f"  {'Hours':<30}"
    for mode in modes:
        t = all_results[mode].get("training", {}).get("total_time_seconds", 0) / 3600
        row += f"{t:>15.1f}h"
    row += f"{'~9.0h':>16}"
    print(row)

    print(f"\n{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="BERT Baseline Training for LexiMind")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single-topic", "single-emotion", "multitask", "all"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="HuggingFace model name"
    )
    args = parser.parse_args()

    config = BertBaselineConfig()
    config.model_name = args.model
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.mode == "all":
        modes = ["single-topic", "single-emotion", "multitask"]
    else:
        modes = [args.mode]

    all_results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        results = run_experiment(mode, config)
        all_results[mode] = results

        # Clear GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save combined results
    if len(all_results) > 1:
        combined_path = config.output_dir / "combined_results.json"

        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if not k.startswith("_")}
            if isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(combined_path, "w") as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"  Combined results saved to {combined_path}")

        print_comparison_summary(all_results)


if __name__ == "__main__":
    main()
