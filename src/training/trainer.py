"""
Multi-task Trainer for LexiMind.

Handles training across summarization, emotion, and topic heads with:
- Mixed-precision (bfloat16 on Ampere+)
- Gradient accumulation
- Cosine LR schedule with warmup
- Early stopping
- MLflow logging
- Temperature-based task sampling (configurable alpha)
- Gradient conflict diagnostics

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.tokenization import Tokenizer
from .metrics import accuracy, calculate_bleu, calculate_rouge, multilabel_f1, rouge_like

# --------------- Configuration ---------------


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    max_epochs: int = 10
    gradient_clip_norm: float = 1.0
    task_weights: Dict[str, float] | None = None
    validation_samples: int = 3
    validation_max_length: int = 128
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1

    # LR scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 500

    # Early stopping
    early_stopping_patience: int | None = 5

    # Task sampling strategy: "round_robin" or "temperature"
    # Temperature sampling: p_i ∝ n_i^alpha where n_i = dataset size
    # alpha < 1 reduces dominance of large tasks (recommended: 0.5-0.7)
    task_sampling: str = "temperature"
    task_sampling_alpha: float = 0.5

    # Gradient conflict diagnostics
    # Compute inter-task gradient cosine similarity every N steps (0 = disabled)
    gradient_conflict_frequency: int = 0

    # MLflow
    experiment_name: str = "LexiMind"
    run_name: str | None = None


# --------------- Early Stopping ---------------


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_value - self.min_delta:
            self.best_value = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# --------------- Trainer ---------------


class Trainer:
    """Multi-task trainer with AMP and gradient accumulation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        device: torch.device,
        tokenizer: Tokenizer,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.global_step = 0

        # Task losses
        self.emotion_loss = torch.nn.BCEWithLogitsLoss()
        self.topic_loss = torch.nn.CrossEntropyLoss()

        # AMP: bfloat16 on Ampere+ GPUs
        self.use_amp = device.type == "cuda"
        self.use_bfloat16 = self.use_amp and torch.cuda.is_bf16_supported()

        # Early stopping
        self.early_stopping: EarlyStopping | None = None
        if config.early_stopping_patience:
            self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # MLflow - use SQLite backend to avoid deprecation warning
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(config.experiment_name)

        # CUDA optimizations
        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def fit(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader] | None = None,
        checkpoint_callback: Callable | None = None,
        start_epoch: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Train model across all tasks."""
        history: Dict[str, Dict[str, float]] = {}
        total_start = time.perf_counter()

        # Setup scheduler
        self._setup_scheduler(train_loaders, start_epoch)

        with mlflow.start_run(run_name=self.config.run_name):
            self._log_config()

            pbar = tqdm(
                range(start_epoch, self.config.max_epochs + 1),
                desc="Training",
                unit="epoch",
                file=sys.stderr,
            )

            for epoch in pbar:
                epoch_start = time.perf_counter()

                # Train
                train_metrics = self._run_epoch(train_loaders, train=True, epoch=epoch)
                history[f"train_epoch_{epoch}"] = train_metrics
                self._log_metrics(train_metrics, "train", epoch)

                # Validate
                if val_loaders:
                    val_metrics = self._run_epoch(val_loaders, train=False, epoch=epoch)
                    history[f"val_epoch_{epoch}"] = val_metrics
                    self._log_metrics(val_metrics, "val", epoch)

                    # Sample generations
                    if "summarization" in val_loaders:
                        self._validate_generation(val_loaders["summarization"], epoch)

                    # Early stopping
                    if self.early_stopping:
                        val_loss = val_metrics.get("total_loss", float("inf"))
                        if self.early_stopping(val_loss):
                            tqdm.write(
                                f"\nEarly stopping at epoch {epoch} (best loss: {self.early_stopping.best_value:.4f})"
                            )

                            break

                # Checkpoint
                if checkpoint_callback:
                    checkpoint_callback(epoch, self.model, history)

                # Update progress
                epoch_time = time.perf_counter() - epoch_start
                loss = train_metrics.get("total_loss", 0)
                pbar.set_postfix({"loss": f"{loss:.3f}", "time": f"{epoch_time:.0f}s"})

        total_time = time.perf_counter() - total_start
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        return history

    def _setup_scheduler(self, loaders: Dict[str, DataLoader], start_epoch: int) -> None:
        """Setup cosine LR schedule with warmup."""
        if self.config.scheduler_type == "constant":
            self.scheduler = None
            return

        steps_per_epoch = max(len(loader) for loader in loaders.values()) // max(
            1, self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * (self.config.max_epochs - start_epoch + 1)
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(f"  LR schedule: cosine, {warmup} warmup, {total_steps} total steps")

    def _run_epoch(
        self,
        loaders: Dict[str, DataLoader],
        *,
        train: bool,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one epoch with configurable task sampling strategy."""
        self.model.train(train)
        metrics: Dict[str, List[float]] = defaultdict(list)
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        max_batches = max(len(loader) for loader in loaders.values())
        accum = self.config.gradient_accumulation_steps

        phase = "Train" if train else "Val"
        pbar = tqdm(range(max_batches), desc=f"  {phase}", leave=False, file=sys.stderr)

        # Temperature-based task sampling: p_i ∝ n_i^alpha
        task_names = list(loaders.keys())
        if self.config.task_sampling == "temperature" and len(task_names) > 1:
            sizes = np.array([len(loaders[t].dataset) for t in task_names], dtype=np.float64)  # type: ignore[arg-type]
            alpha = self.config.task_sampling_alpha
            probs = sizes**alpha
            probs = probs / probs.sum()
            tqdm.write(
                f"  Temperature sampling (α={alpha}): "
                + ", ".join(f"{t}={p:.2%}" for t, p in zip(task_names, probs, strict=True))
            )
        else:
            probs = None

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for step in pbar:
                step_loss = 0.0

                # Select tasks for this step
                if probs is not None and train:
                    # Temperature sampling: sample tasks based on dataset size
                    selected_tasks = list(
                        np.random.choice(task_names, size=len(task_names), replace=True, p=probs)
                    )
                else:
                    # Round-robin: all tasks every step
                    selected_tasks = task_names

                for task in selected_tasks:
                    loader = loaders[task]
                    batch = self._get_batch(iterators, loader, task)
                    if batch is None:
                        continue

                    # Forward with AMP
                    dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                    with torch.autocast("cuda", dtype=dtype, enabled=self.use_amp):
                        loss, task_metrics = self._forward_task(task, batch)

                    # Skip NaN
                    if torch.isnan(loss):
                        continue

                    # Record metrics
                    metrics[f"{task}_loss"].append(loss.item())
                    for name, val in task_metrics.items():
                        metrics[f"{task}_{name}"].append(val)

                    # Track step loss for both train and val
                    weight = (self.config.task_weights or {}).get(task, 1.0)
                    step_loss += loss.item() * weight

                    # Backward (train only)
                    if train:
                        scaled = (loss * weight) / accum
                        scaled.backward()

                # Gradient conflict diagnostics
                if (
                    train
                    and self.config.gradient_conflict_frequency > 0
                    and (step + 1) % self.config.gradient_conflict_frequency == 0
                ):
                    conflict_stats = self._compute_gradient_conflicts(loaders, iterators)
                    for k, v in conflict_stats.items():
                        metrics[f"grad_{k}"].append(v)
                        mlflow.log_metric(f"grad_{k}", v, step=self.global_step)

                # Optimizer step
                if train and (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                        # Log learning rate to MLflow
                        current_lr = self.scheduler.get_last_lr()[0]
                        mlflow.log_metric("learning_rate", current_lr, step=self.global_step)
                    self.global_step += 1

                if step_loss > 0:
                    metrics["total_loss"].append(step_loss)
                    if train:
                        pbar.set_postfix({"loss": f"{step_loss:.3f}"})

        # Average metrics
        averaged = {k: sum(v) / len(v) for k, v in metrics.items() if v}
        tqdm.write(
            f"[{phase.lower()}] epoch {epoch}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in averaged.items() if k != "epoch")
        )
        return averaged

    def _get_batch(self, iterators: Dict, loader: DataLoader, task: str) -> Dict | None:
        """Get next batch, cycling if exhausted."""
        try:
            batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(loader)
            try:
                batch = next(iterators[task])
            except StopIteration:
                return None
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _forward_task(self, task: str, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """Route to task-specific forward pass."""
        if task == "summarization":
            return self._forward_summarization(batch)
        elif task == "emotion":
            return self._forward_emotion(batch)
        elif task == "topic":
            return self._forward_topic(batch)
        raise ValueError(f"Unknown task: {task}")

    def _forward_summarization(self, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """Seq2seq forward for summarization."""
        inputs = {"src_ids": batch["src_ids"], "tgt_ids": batch["tgt_ids"]}
        if "src_mask" in batch:
            inputs["src_mask"] = batch["src_mask"]

        logits = self.model.forward("summarization", inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

        # Decode predictions and references
        preds = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
        refs = self._decode_labels(batch["labels"])

        # Calculate comprehensive metrics
        metrics = {"rouge_like": rouge_like(preds, refs)}

        # Proper ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        try:
            rouge_scores = calculate_rouge(preds, refs)
            metrics["rouge1"] = rouge_scores["rouge1"]
            metrics["rouge2"] = rouge_scores["rouge2"]
            metrics["rougeL"] = rouge_scores["rougeL"]
        except Exception:
            pass  # Fall back to rouge_like only if rouge-score not installed

        # BLEU-4 score
        try:
            metrics["bleu4"] = calculate_bleu(preds, refs)
        except Exception:
            pass

        return loss, metrics

    def _forward_emotion(self, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """Multi-label emotion classification."""
        inputs = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"]

        logits = self.model.forward("emotion", inputs)
        loss = self.emotion_loss(logits, batch["labels"].float())
        # Lower threshold (0.3) for multi-label - 28 classes means lower confidence per class
        preds = (torch.sigmoid(logits) > 0.3).int()
        return loss, {"f1": multilabel_f1(preds, batch["labels"].int())}

    def _forward_topic(self, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """Single-label topic classification."""
        inputs = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"]

        logits = self.model.forward("topic", inputs)
        loss = self.topic_loss(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        return loss, {"accuracy": accuracy(preds.tolist(), batch["labels"].tolist())}

    def _decode_labels(self, labels: torch.Tensor) -> List[str]:
        """Decode labels, replacing -100 with pad token."""
        valid = labels.clone()
        valid[valid == -100] = self.tokenizer.pad_token_id
        return self.tokenizer.decode_batch(valid.tolist())

    def _validate_generation(self, val_loader: DataLoader, epoch: int) -> None:
        """Generate sample summaries for quality check."""
        self.model.eval()
        n = self.config.validation_samples

        tqdm.write(f"\n{'=' * 50}")
        tqdm.write(f"[Validation Samples - Epoch {epoch}]")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n:
                    break

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                src_ids = batch["src_ids"][:1]
                src_mask = batch.get("src_mask", None)
                if src_mask is not None:
                    src_mask = src_mask[:1]

                # Generate with anti-repetition
                model: Any = self.model
                enc_mask = (
                    src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
                )
                memory = model.encoder(src_ids, mask=enc_mask)
                generated = model.decoder.greedy_decode(
                    memory=memory,
                    max_len=self.config.validation_max_length,
                    start_token_id=self.tokenizer.bos_token_id,
                    end_token_id=self.tokenizer.eos_token_id,
                    device=self.device,
                    memory_mask=src_mask,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )

                src = self.tokenizer.decode(src_ids[0].tolist())
                out = self.tokenizer.decode(generated[0].tolist())
                ref = self._decode_labels(batch["labels"][:1])[0]

                tqdm.write(f"\nSample {i + 1}:")
                tqdm.write(f"  Source: {src[:100]}...")
                tqdm.write(f"  Generated: {out}")
                tqdm.write(f"  Reference: {ref[:100]}...")

        tqdm.write(f"{'=' * 50}\n")
        self.model.train()

    def _compute_gradient_conflicts(
        self,
        loaders: Dict[str, DataLoader],
        iterators: Dict,
    ) -> Dict[str, float]:
        """Compute inter-task gradient cosine similarity to diagnose conflicts.

        Returns cosine similarity between gradient vectors for each task pair.
        Negative values indicate conflicting gradients (negative transfer risk).
        """
        task_grads: Dict[str, torch.Tensor] = {}

        for task, loader in loaders.items():
            self.optimizer.zero_grad()
            batch = self._get_batch(iterators, loader, task)
            if batch is None:
                continue

            dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
            with torch.autocast("cuda", dtype=dtype, enabled=self.use_amp):
                loss, _ = self._forward_task(task, batch)

            if torch.isnan(loss):
                continue

            loss.backward()

            # Flatten all gradients into a single vector
            grad_vec = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_vec.append(p.grad.detach().clone().flatten())
            if grad_vec:
                task_grads[task] = torch.cat(grad_vec)

        self.optimizer.zero_grad()

        # Compute pairwise cosine similarity
        stats: Dict[str, float] = {}
        tasks = list(task_grads.keys())
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                g1, g2 = task_grads[t1], task_grads[t2]
                cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                stats[f"cos_sim_{t1}_{t2}"] = cos_sim
                stats[f"conflict_{t1}_{t2}"] = 1.0 if cos_sim < 0 else 0.0

        return stats

    def _log_config(self) -> None:
        """Log config to MLflow."""
        mlflow.log_params(
            {
                "max_epochs": self.config.max_epochs,
                "gradient_clip_norm": self.config.gradient_clip_norm,
                "label_smoothing": self.config.label_smoothing,
                "task_weights": str(self.config.task_weights),
                "warmup_steps": self.config.warmup_steps,
                "scheduler_type": self.config.scheduler_type,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )

    def _log_metrics(self, metrics: Dict[str, float], prefix: str, epoch: int) -> None:
        """Log metrics to MLflow."""
        for k, v in metrics.items():
            if k != "epoch":
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)
