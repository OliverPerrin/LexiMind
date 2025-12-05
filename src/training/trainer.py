"""
Multi-task Trainer for LexiMind.

Handles training across summarization, emotion, and topic heads with mixed-precision,
gradient accumulation, and MLflow logging.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.tokenization import Tokenizer
from .metrics import accuracy, multilabel_f1, rouge_like
from .nan_debugger import NaNDetector

# --------------- Configuration ---------------


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    max_epochs: int = 1
    gradient_clip_norm: float = 1.0
    task_weights: Dict[str, float] | None = None
    validation_samples: int = 3
    validation_max_length: int = 128
    label_smoothing: float = 0.0
    experiment_name: str = "LexiMind"
    run_name: str | None = None
    gradient_accumulation_steps: int = 1


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

        # Task losses
        self.emotion_loss = torch.nn.BCEWithLogitsLoss()
        self.topic_loss = torch.nn.CrossEntropyLoss()

        # AMP setup: bfloat16 for Ampere+ GPUs, float16 otherwise
        self.use_amp = device.type == "cuda"
        self.use_bfloat16 = self.use_amp and torch.cuda.is_bf16_supported()
        self.scaler = torch.GradScaler("cuda", enabled=(self.use_amp and not self.use_bfloat16))

        # NaN detection
        self.nan_detector = NaNDetector(model, enabled=True)
        self.nan_skip_count = 0
        self.max_nan_skips = 50

        # Track current step for debugging
        self._current_step = 0

        self._nan_counter = 0
        mlflow.set_experiment(config.experiment_name)

        # CUDA optimizations
        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    # --------------- Training Loop ---------------

    def fit(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader] | None = None,
        checkpoint_callback: Callable | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train model across all tasks with progress tracking."""
        history: Dict[str, Dict[str, float]] = {}
        total_start = time.perf_counter()

        with mlflow.start_run(run_name=self.config.run_name):
            self._log_config()

            # Epoch progress bar
            epoch_pbar = tqdm(
                range(1, self.config.max_epochs + 1),
                desc="Training",
                unit="epoch",
                position=0,
                file=sys.stderr,
                dynamic_ncols=True,
            )

            for epoch in epoch_pbar:
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

                    if "summarization" in val_loaders:
                        self._validate_generation(val_loaders["summarization"], epoch)

                # Checkpoint
                if checkpoint_callback:
                    checkpoint_callback(epoch, self.model, history)

                # Update epoch progress bar with metrics
                epoch_time = time.perf_counter() - epoch_start
                total_time = time.perf_counter() - total_start
                desc = f"Epoch {epoch}/{self.config.max_epochs}"
                if "total_loss" in train_metrics:
                    desc += f" | loss={train_metrics['total_loss']:.3f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.set_postfix(
                    {"time": f"{epoch_time:.1f}s", "total": f"{total_time:.1f}s"}
                )

        total_time = time.perf_counter() - total_start
        print(f"\n✓ Training complete in {total_time:.1f}s")
        return history

    def _log_config(self) -> None:
        """Log config to MLflow."""
        mlflow.log_params(
            {
                "max_epochs": self.config.max_epochs,
                "gradient_clip_norm": self.config.gradient_clip_norm,
                "label_smoothing": self.config.label_smoothing,
                "task_weights": str(self.config.task_weights),
            }
        )

    def _log_metrics(self, metrics: Dict[str, float], prefix: str, epoch: int) -> None:
        """Log metrics to MLflow."""
        for k, v in metrics.items():
            if k != "epoch":
                mlflow.log_metric(f"{prefix}_{k}", v, step=epoch)

    # --------------- Epoch Execution ---------------

    def _run_epoch(
        self,
        loaders: Dict[str, DataLoader],
        *,
        train: bool,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one epoch with progress bar."""
        phase = "Train" if train else "Val"
        self.model.train(train)

        metrics: Dict[str, List[float]] = defaultdict(list)
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        max_batches = max(len(loader) for loader in loaders.values())
        accum_steps = self.config.gradient_accumulation_steps

        # Batch progress bar (nested under epoch bar)
        pbar = tqdm(
            range(max_batches),
            desc=f"  {phase}",
            unit="batch",
            leave=False,
            position=1,
            file=sys.stderr,
            dynamic_ncols=True,
        )

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for step in pbar:
                self._current_step = step
                step_loss = 0.0

                for task, loader in loaders.items():
                    batch = self._get_batch(iterators, loader, task)
                    if batch is None:
                        continue

                    # Forward with AMP
                    amp_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp):
                        loss, task_metrics = self._forward_task(task, batch)

                    # NaN check
                    if torch.isnan(loss):
                        self._nan_counter += 1
                        if self._nan_counter > 10:
                            raise RuntimeError("Training diverging - too many NaN losses")
                        continue
                    self._nan_counter = 0

                    # Record metrics
                    metrics[f"{task}_loss"].append(loss.item())
                    for name, val in task_metrics.items():
                        metrics[f"{task}_{name}"].append(val)

                    # Backward
                    if train:
                        weight = (self.config.task_weights or {}).get(task, 1.0)
                        scaled = (loss * weight) / accum_steps
                        step_loss += scaled.item() * accum_steps

                        if self.use_bfloat16:
                            scaled.backward()
                        else:
                            self.scaler.scale(scaled).backward()

                # Optimizer step
                if train and (step + 1) % accum_steps == 0:
                    self._optimizer_step()

                if step_loss > 0:
                    metrics["total_loss"].append(step_loss)

                # Update progress bar
                if metrics["total_loss"]:
                    pbar.set_postfix({"loss": f"{metrics['total_loss'][-1]:.3f}"})

        # Average and print summary
        averaged = {k: sum(v) / len(v) for k, v in metrics.items() if v}
        averaged["epoch"] = float(epoch)

        summary = f"[{phase.lower()}] epoch {epoch}: "
        summary += ", ".join(f"{k}={v:.4f}" for k, v in averaged.items() if k != "epoch")
        tqdm.write(summary)

        return averaged

    def _optimizer_step(self) -> None:
        """Optimizer step with gradient clipping and NaN detection."""
        # Check gradients for NaN/Inf BEFORE clipping
        nan_grad = self.nan_detector.check_gradients(self._current_step)
        if nan_grad is not None:
            param_name, _ = nan_grad
            print(f"⚠ Skipping optimizer step due to NaN gradient in {param_name}")
            self.optimizer.zero_grad()
            self.nan_skip_count += 1
            if self.nan_skip_count > self.max_nan_skips:
                raise RuntimeError("Too many NaN gradients, stopping")
            return

        # Clip and step
        if self.use_bfloat16:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
        else:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.optimizer.zero_grad()

        # Check parameters for NaN AFTER update
        nan_param = self.nan_detector.check_parameters(self._current_step)
        if nan_param is not None:
            raise RuntimeError(
                f"NaN in parameter {nan_param} after optimizer step at step {self._current_step}!"
            )

    def _get_batch(
        self, iterators: Dict, loader: DataLoader, task: str
    ) -> Dict[str, torch.Tensor] | None:
        """Get next batch, cycling iterator if exhausted."""
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

    # --------------- Task Forward Passes ---------------

    def _forward_task(
        self, task: str, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Route to task-specific forward pass with NaN detection."""
        if task == "summarization":
            loss, task_metrics = self._forward_summarization(batch)
        elif task == "emotion":
            loss, task_metrics = self._forward_emotion(batch)
        elif task == "topic":
            loss, task_metrics = self._forward_topic(batch)
        else:
            raise ValueError(f"Unknown task: {task}")

        # Check for NaN in loss
        if torch.isnan(loss):
            self.nan_skip_count += 1
            print(
                f"⚠ NaN loss detected in {task} at step {self._current_step} (skip {self.nan_skip_count}/{self.max_nan_skips})"
            )
            if self.nan_skip_count > self.max_nan_skips:
                raise RuntimeError(f"Too many NaN batches ({self.nan_skip_count}), stopping")
            # Return zero loss to skip this batch
            return torch.tensor(0.0, device=loss.device, requires_grad=True), task_metrics

        return loss, task_metrics

    def _forward_summarization(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
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

        # Quick ROUGE estimate
        preds = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
        refs = self._decode_labels(batch["labels"])
        return loss, {"rouge_like": rouge_like(preds, refs)}

    def _forward_emotion(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Multi-label emotion classification."""
        inputs = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"]

        logits = self.model.forward("emotion", inputs)
        loss = self.emotion_loss(logits, batch["labels"].float())
        preds = (torch.sigmoid(logits) > 0.5).int()
        return loss, {"f1": multilabel_f1(preds, batch["labels"].int())}

    def _forward_topic(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
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

    # --------------- Validation Generation ---------------

    def _validate_generation(self, val_loader: DataLoader, epoch: int) -> None:
        """Generate sample summaries for quality check."""
        self.model.eval()
        n = self.config.validation_samples

        tqdm.write(f"\n{'=' * 50}")
        tqdm.write(f"[Validation Samples - Epoch {epoch}]")
        tqdm.write(f"{'=' * 50}")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n:
                    break

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                src_ids = batch["src_ids"][:1]
                src_mask = batch.get("src_mask")
                if src_mask is not None:
                    src_mask = src_mask[:1]

                # Encode and generate
                enc_mask = (
                    src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
                )
                model: Any = self.model
                memory = model.encoder(src_ids, mask=enc_mask)
                generated = model.decoder.greedy_decode_naive(
                    memory=memory,
                    max_len=self.config.validation_max_length,
                    start_token_id=self.tokenizer.bos_token_id,
                    end_token_id=self.tokenizer.eos_token_id,
                    device=self.device,
                    memory_mask=src_mask,
                )

                # Decode and display
                src = self.tokenizer.decode(src_ids[0].tolist())
                out = self.tokenizer.decode(generated[0].tolist())
                ref = self._decode_labels(batch["labels"][:1])[0]

                tqdm.write(f"\nSample {i + 1}:")
                tqdm.write(f"  Source: {src[:120]}..." if len(src) > 120 else f"  Source: {src}")
                tqdm.write(f"  Generated: {out}")
                tqdm.write(
                    f"  Reference: {ref[:120]}..." if len(ref) > 120 else f"  Reference: {ref}"
                )

        tqdm.write(f"{'=' * 50}\n")
        self.model.train()
