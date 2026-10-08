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
from ..models.losses import masked_binary_cross_entropy
from .metrics import (
    MultilabelBatchMetrics,
    ObservedMultilabelMetrics,
    calculate_bleu,
    calculate_rouge,
    multilabel_f1,
    rouge_like,
)
from .pcgrad import PCGrad
from .policy import PolicyBatchMetrics, PolicyTask

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
    generation_metrics: bool = True

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

    # PCGrad: Project Conflicting Gradients (Yu et al., NeurIPS 2020)
    # When enabled, computes per-task gradients independently and projects
    # conflicting gradient pairs to reduce negative transfer.
    use_pcgrad: bool = False

    # MLflow
    experiment_name: str = "LexiMind"
    run_name: str | None = None
    tracking_uri: str = "sqlite:///mlruns.db"

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.task_sampling not in {"temperature", "round_robin"}:
            raise ValueError("task_sampling must be temperature or round_robin")


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
        policy_objectives: Dict[str, PolicyTask] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.global_step = 0
        self.policy_objectives = dict(policy_objectives or {})
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name in {"summarization", "emotion", "topic"}
            or objective.model is not self.model
            for name, objective in self.policy_objectives.items()
        ):
            raise ValueError("Policy objectives need distinct task names and this Trainer's model")

        # AMP: bfloat16 on Ampere+ GPUs
        self.use_amp = device.type == "cuda"
        self.use_bfloat16 = False
        if self.use_amp:
            with torch.cuda.device(device):
                self.use_bfloat16 = (
                    torch.cuda.get_device_capability(device)[0] >= 8
                    and torch.cuda.is_bf16_supported()
                )
        if self.use_amp and not self.use_bfloat16:
            raise ValueError(
                "CUDA training requires bfloat16 support; unscaled float16 backward "
                "(including PCGrad) is unsupported. Use a BF16-capable GPU or CPU/MPS float32."
            )

        # Early stopping
        self.early_stopping: EarlyStopping | None = None
        if config.early_stopping_patience:
            self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # MLflow - use SQLite backend to avoid deprecation warning
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        # PCGrad
        self.pcgrad: PCGrad | None = None
        if config.use_pcgrad:
            self.pcgrad = PCGrad(reduction="sum")
            print("  PCGrad: enabled (gradient surgery for conflicting task gradients)")

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
                stop_training = False

                # Reset PCGrad stats per epoch
                if self.pcgrad is not None:
                    self.pcgrad.reset_stats()

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

                            stop_training = True

                # Checkpoint
                if checkpoint_callback:
                    checkpoint_callback(epoch, self.model, history)
                if stop_training:
                    break

                # Update progress
                epoch_time = time.perf_counter() - epoch_start
                loss = train_metrics.get("total_loss", 0)
                pbar.set_postfix({"loss": f"{loss:.3f}", "time": f"{epoch_time:.0f}s"})

        total_time = time.perf_counter() - total_start
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        return history

    def _setup_scheduler(self, loaders: Dict[str, DataLoader], start_epoch: int) -> None:
        """Setup cosine LR schedule with warmup.

        Each outer training step consumes one batch per *selected* task
        (temperature sampling draws ``len(tasks)`` task labels with
        replacement, so functionally each step does ``len(tasks)`` forward
        passes but still counts as one optimizer-ready iteration for
        ``gradient_accumulation_steps``). ``max_batches`` therefore equals
        the size of the longest loader, and the optimizer steps per epoch
        are ``ceil(max_batches / gradient_accumulation_steps)`` including the
        final partial accumulation window.
        """
        if self.config.scheduler_type == "constant":
            self.scheduler = None
            return

        if not loaders or any(len(loader) == 0 for loader in loaders.values()):
            raise ValueError("Every selected task requires a nonempty data loader")

        accum = max(1, self.config.gradient_accumulation_steps)
        max_batches = max(len(loader) for loader in loaders.values())
        steps_per_epoch = math.ceil(max_batches / accum)
        epochs_remaining = max(1, self.config.max_epochs - start_epoch + 1)
        total_steps = steps_per_epoch * epochs_remaining
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(
            f"  LR schedule: cosine, warmup={warmup}, "
            f"{steps_per_epoch} optimizer steps/epoch x {epochs_remaining} epochs "
            f"= {total_steps} total (max_batches={max_batches}, accum={accum})"
        )

    def _run_epoch(
        self,
        loaders: Dict[str, DataLoader],
        *,
        train: bool,
        epoch: int,
        step_callback: Callable[[], None] | None = None,
    ) -> Dict[str, float]:
        """Run one epoch with configurable task sampling strategy."""
        if (
            train
            and self.policy_objectives.keys() & loaders.keys()
            and self.config.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "Policy tasks require gradient_accumulation_steps=1; weighted policy windows are not implemented"
            )
        self.model.train(train)
        if not loaders or any(len(loader) == 0 for loader in loaders.values()):
            raise ValueError("Every selected task requires a nonempty data loader")
        metric_totals: Dict[str, float] = defaultdict(float)
        metric_weights: Dict[str, int] = defaultdict(int)

        def record(name: str, value: float, weight: int = 1) -> None:
            metric_totals[name] += value * weight
            metric_weights[name] += weight

        observed_metrics: Dict[str, ObservedMultilabelMetrics] = {}
        window_has_supervision = False
        diagnostic_batches: Dict[str, Dict] = {}
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        max_batches = max(len(loader) for loader in loaders.values())
        accum = self.config.gradient_accumulation_steps
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        phase = "Train" if train else "Val"
        if not train and self.policy_objectives.keys() & loaders.keys():
            raise ValueError(
                "Policy surrogate loss cannot select checkpoints; use separate held-out outcome evaluation"
            )
        pbar = tqdm(range(max_batches), desc=f"  {phase}", leave=False, file=sys.stderr)

        # Temperature-based task sampling: p_i ∝ n_i^alpha.
        #
        # Each outer training step draws ``len(task_names)`` task labels
        # *with replacement* from this distribution, so every step still
        # runs ``len(task_names)`` forward/backward passes (like
        # round-robin) but the identity of the tasks in each triple is
        # stochastic. A task whose probability is p therefore receives
        # ~p × len(tasks) forward passes per outer step in expectation
        # (e.g. with p=0.45 over 3 tasks, ~1.35 forwards/step or ~45% of
        # all task draws). This matches the 45/43/12% figures reported in
        # the paper while preserving the per-step gradient-accumulation
        # accounting of round-robin training.
        task_names = list(loaders.keys())
        if self.config.task_sampling == "temperature" and len(task_names) > 1:
            sizes = np.array([len(loaders[t].dataset) for t in task_names], dtype=np.float64)  # type: ignore[arg-type]
            alpha = self.config.task_sampling_alpha
            probs = sizes**alpha
            probs = probs / probs.sum()
            tqdm.write(
                f"  Temperature sampling (α={alpha}, {len(task_names)} draws/step): "
                + ", ".join(f"{t}={p:.2%}" for t, p in zip(task_names, probs, strict=True))
            )
        else:
            probs = None

        use_pcgrad = train and self.pcgrad is not None
        if use_pcgrad:
            # Project only encoder gradients. Decoder/head gradients are summed,
            # even when several generative objectives share the same decoder.
            shared_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            shared_ids = {id(p) for p in shared_params}
            head_params = [
                p for p in self.model.parameters() if p.requires_grad and id(p) not in shared_ids
            ]
            tqdm.write("  PCGrad active: computing per-task gradients with projection")

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
                    selected_tasks = (
                        task_names
                        if train
                        else [task for task in task_names if step < len(loaders[task])]
                    )

                # Preserve microbatch-mean accumulation, including a zero for
                # unknown batches. Each masked BCE uses its own observed count;
                # this is not an observed-cell mean over the merged window.
                window_size = min(accum, max_batches - (step // accum) * accum)

                # For PCGrad: collect task losses first, then do joint backward
                pcgrad_losses: Dict[str, torch.Tensor] = {}

                for task in selected_tasks:
                    loader = loaders[task]
                    batch = self._get_batch(iterators, loader, task)
                    if batch is None:
                        continue
                    if train and self.config.gradient_conflict_frequency > 0:
                        diagnostic_batches[task] = batch

                    # Forward with AMP
                    dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                    with torch.autocast("cuda", dtype=dtype, enabled=self.use_amp):
                        loss, task_metrics = self._forward_task(
                            task, batch, summarize_metrics=False
                        )

                    # Inspect the already-needed host scalar once, instead of
                    # synchronizing separately for isfinite and loss.item().
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(
                            f"Non-finite {task} loss at epoch {epoch}, step {step}"
                        )

                    # Record metrics
                    batch_size = (
                        task_metrics.responses
                        if isinstance(task_metrics, PolicyBatchMetrics)
                        else len(batch["labels"])
                    )
                    loss_count = 1
                    has_supervision = True
                    if isinstance(task_metrics, PolicyBatchMetrics):
                        loss_count = task_metrics.loss_weight
                        has_supervision = task_metrics.has_signal
                    elif isinstance(task_metrics, MultilabelBatchMetrics):
                        counts = task_metrics.counts
                        loss_count = int(counts.observed.sum())
                        has_supervision = loss_count > 0
                        if task not in observed_metrics:
                            observed_metrics[task] = counts
                        else:
                            observed_metrics[task].merge(counts)
                        # Keep legacy sample F1 for dense emotion labels only.
                        if "f1" in task_metrics:
                            record(f"{task}_f1", task_metrics["f1"], batch_size)
                    # Validation visits each example once. Loss is token-averaged
                    # for summarization, observed-cell-averaged for multi-label,
                    # and example-averaged for single-label classifiers.
                    elif not train:
                        loss_count = (
                            int((batch["labels"] != -100).sum())
                            if task == "summarization"
                            else batch_size
                        )
                    record(f"{task}_loss", loss_value, loss_count)
                    if not isinstance(task_metrics, MultilabelBatchMetrics):
                        for name, val in task_metrics.items():
                            record(f"{task}_{name}", val, 1 if train else batch_size)

                    # Track step loss for both train and val
                    weight = (self.config.task_weights or {}).get(task, 1.0)
                    step_loss += loss_value * weight

                    # Backward (train only)
                    if train and has_supervision and weight != 0:
                        window_has_supervision = True
                        if use_pcgrad:
                            # Collect losses for PCGrad (backward later)
                            # Temperature sampling can select the same task more
                            # than once. All draws must contribute to its gradient.
                            pcgrad_losses[task] = pcgrad_losses.get(task, 0) + loss
                        else:
                            scaled = (loss * weight) / window_size
                            scaled.backward()

                # PCGrad: single autograd.grad pass per task over shared+head
                # params, projection applied only to the shared portion.
                if use_pcgrad and pcgrad_losses and len(pcgrad_losses) > 1:
                    pcgrad_stats = self.pcgrad.backward(  # type: ignore[union-attr]
                        pcgrad_losses,
                        shared_params=shared_params,
                        head_params=head_params,
                        task_weights=self.config.task_weights,
                        gradient_accumulation_steps=window_size,
                    )

                    for k, v in pcgrad_stats.items():
                        record(f"pcgrad_{k}", v)
                        if self.global_step % 100 == 0:
                            mlflow.log_metric(f"pcgrad_{k}", v, step=self.global_step)

                elif use_pcgrad and pcgrad_losses and len(pcgrad_losses) == 1:
                    # Single task sampled this step, no conflict to resolve.
                    for task_name, loss_val in pcgrad_losses.items():
                        w = (self.config.task_weights or {}).get(task_name, 1.0)
                        scaled = (loss_val * w) / window_size
                        scaled.backward()

                # Gradient conflict diagnostics (non-PCGrad mode)
                if (
                    train
                    and not use_pcgrad
                    and self.config.gradient_conflict_frequency > 0
                    and (step + 1) % self.config.gradient_conflict_frequency == 0
                ):
                    conflict_stats = self._compute_gradient_conflicts(diagnostic_batches)
                    for k, v in conflict_stats.items():
                        record(f"grad_{k}", v)
                        mlflow.log_metric(f"grad_{k}", v, step=self.global_step)

                # Optimizer step
                if train and ((step + 1) % accum == 0 or step + 1 == max_batches):
                    if window_has_supervision:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                            current_lr = self.scheduler.get_last_lr()[0]
                            mlflow.log_metric("learning_rate", current_lr, step=self.global_step)
                        self.global_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    window_has_supervision = False

                if train:
                    record("total_loss", step_loss)
                    pbar.set_postfix({"loss": f"{step_loss:.3f}"})
                if step_callback is not None:
                    step_callback()

        # Constant-memory weighted means preserve each task's reduction policy.
        averaged = {
            key: total / metric_weights[key] if metric_weights[key] else 0.0
            for key, total in metric_totals.items()
        }
        averaged.setdefault("total_loss", 0.0)
        for task, counts in observed_metrics.items():
            if not train and not counts.observed.any():
                raise ValueError(
                    f"Validation task '{task}' has no observed labels; "
                    "its loss cannot be used for early stopping or checkpoint selection"
                )
            averaged.update({f"{task}_{key}": value for key, value in counts.compute().items()})
            if int(counts.observed.sum()) != counts.total:
                averaged.pop(f"{task}_f1", None)
        if not train:
            averaged["total_loss"] = sum(
                averaged[f"{task}_loss"] * (self.config.task_weights or {}).get(task, 1.0)
                for task in task_names
            )
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

    def _forward_task(
        self, task: str, batch: Dict, *, summarize_metrics: bool = True
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Route to task-specific forward pass."""
        if task in self.policy_objectives:
            return self.policy_objectives[task](batch)
        if task == "summarization":
            return self._forward_summarization(batch)
        if task in {"emotion", "topic"}:
            return self._forward_classification(
                task,
                batch,
                threshold=0.3 if task == "emotion" else 0.5,
                summarize_metrics=summarize_metrics,
            )
        raise ValueError(f"Unknown task: {task}")

    def _forward_summarization(self, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """Seq2seq forward for summarization.

        During training, only cheap ``rouge_like`` is reported per-batch; the
        expensive rouge-score / BLEU computations on teacher-forced argmax
        outputs are unreliable as quality signals (they are not the tokens
        the decoder would actually generate) and consume significant CPU
        time per epoch, so they are reserved for validation batches where
        they serve as sanity metrics alongside full generation during
        ``_validate_generation``.
        """
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

        if not self.config.generation_metrics:
            return loss, {}

        preds = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
        refs = self._decode_labels(batch["labels"])

        metrics: Dict[str, float] = {"rouge_like": rouge_like(preds, refs)}

        if not self.model.training:
            rouge_scores = calculate_rouge(preds, refs)
            metrics.update(
                {
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"],
                    "bleu4": calculate_bleu(preds, refs),
                }
            )

        return loss, metrics

    def _forward_classification(
        self, task: str, batch: Dict, *, threshold: float, summarize_metrics: bool = True
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        inputs = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"]
        logits = self.model.forward(task, inputs)
        head = getattr(self.model, "heads", {}).get(task)
        head = getattr(head, "_orig_mod", head)
        # Headless adapters retain established task defaults; labels never select mode.
        problem_type = getattr(
            head, "problem_type", "multi_label" if task == "emotion" else "single_label"
        )
        if problem_type == "multi_label":
            labels = batch["labels"]
            mask = batch.get("label_mask")
            loss = masked_binary_cross_entropy(logits, labels, mask)
            predictions = logits.detach().sigmoid() > threshold
            counts = ObservedMultilabelMetrics(predictions, labels, mask)
            metrics = MultilabelBatchMetrics(counts, summarize=summarize_metrics)
            if task == "emotion" and int(counts.observed.sum()) == counts.total:
                metrics["f1"] = multilabel_f1(predictions, labels)
            return loss, metrics
        if problem_type != "single_label":
            raise ValueError(f"Unknown classification problem_type: {problem_type}")
        if batch.get("label_mask") is not None:
            raise ValueError("label_mask requires a multi_label classification head")
        if batch["labels"].ndim != 1 or batch["labels"].shape[0] != logits.shape[0]:
            raise ValueError("Single-label classification requires labels with shape B")
        loss = F.cross_entropy(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        return loss, {"accuracy": (preds == batch["labels"]).sum().item() / preds.numel()}

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
        batches: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Compute inter-task gradient cosine similarity to diagnose conflicts.

        Compare the same encoder parameter set across tasks. This excludes
        private-head geometry and matches the encoder-only PCGrad policy; it
        does not diagnose decoder conflicts between generative objectives.

        Uses the latest already-consumed batch per task: diagnostics neither
        steal training batches nor clear/replace accumulated parameter gradients.
        CPU and active CUDA/MPS RNG state is restored, including training-mode dropout.

        Returns cosine similarity between encoder-gradient vectors for each
        task pair. Negative values indicate conflicting gradients on the
        shared encoder (negative transfer risk).
        """
        shared_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        if not shared_params:
            return {}
        task_grads: Dict[str, torch.Tensor] = {}
        devices = []
        if self.device.type == "cuda":
            devices = [
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            ]
        elif self.device.type == "mps":
            devices = [0]
        with torch.random.fork_rng(
            devices=devices, device_type="mps" if self.device.type == "mps" else "cuda"
        ):
            for task, batch in batches.items():
                dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                with torch.autocast("cuda", dtype=dtype, enabled=self.use_amp):
                    loss, task_metrics = self._forward_task(task, batch, summarize_metrics=False)
                if (
                    isinstance(task_metrics, MultilabelBatchMetrics)
                    and not task_metrics.counts.observed.any()
                ) or (isinstance(task_metrics, PolicyBatchMetrics) and not task_metrics.has_signal):
                    continue
                if not torch.isfinite(loss):
                    continue
                grads = torch.autograd.grad(loss, shared_params, allow_unused=True)
                task_grads[task] = torch.cat(
                    [
                        grad.detach().flatten().to(torch.float32)
                        if grad is not None
                        else torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                        for p, grad in zip(shared_params, grads, strict=True)
                    ]
                )

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
