"""Multi-task trainer coordinating summarization, emotion, and topic heads."""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.tokenization import Tokenizer
from .metrics import accuracy, multilabel_f1, rouge_like


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    gradient_clip_norm: float = 1.0
    logging_interval: int = 50
    task_weights: Dict[str, float] | None = None
    validation_samples: int = 3
    validation_max_length: int = 128
    label_smoothing: float = 0.0  # Label smoothing for regularization (e.g., 0.1)
    experiment_name: str = "LexiMind"
    run_name: str | None = None
    gradient_accumulation_steps: int = 1


class Trainer:
    """Coordinates multi-task optimisation across task-specific dataloaders."""

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
        self.emotion_loss = torch.nn.BCEWithLogitsLoss()
        self.topic_loss = torch.nn.CrossEntropyLoss()
        # Apply label smoothing to summarization task if configured
        self.label_smoothing = config.label_smoothing
        self._progress_last_len = 0
        self.gradient_accumulation_steps = max(1, config.gradient_accumulation_steps)
        self._nan_counter = 0  # Track consecutive NaNs

        # Mixed Precision Training
        # Initialize GradScaler for float16/bfloat16 training
        # This scales gradients to prevent underflow during backward pass
        # Note: bfloat16 generally doesn't need scaling, but we keep it for safety unless it causes NaNs
        self.scaler = torch.GradScaler("cuda", enabled=(device.type == "cuda"))

        # Initialize MLflow
        mlflow.set_experiment(config.experiment_name)

    def fit(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader] | None = None,
        checkpoint_callback: Callable | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train the model.

        Args:
            train_loaders: Task-specific training dataloaders
            val_loaders: Optional task-specific validation dataloaders
            checkpoint_callback: Optional callback(epoch, model, history) to save checkpoints

        Returns:
            Training history dictionary
        """
        history: Dict[str, Dict[str, float]] = {}
        total_epochs = max(1, self.config.max_epochs)
        start_time = time.perf_counter()

        with mlflow.start_run(run_name=self.config.run_name):
            # Log configuration
            mlflow.log_params(
                {
                    "max_epochs": self.config.max_epochs,
                    "gradient_clip_norm": self.config.gradient_clip_norm,
                    "label_smoothing": self.config.label_smoothing,
                    "task_weights": str(self.config.task_weights),
                    "device": str(self.device),
                }
            )

            for epoch in range(1, total_epochs + 1):
                epoch_start = time.perf_counter()
                train_metrics = self._run_epoch(
                    train_loaders,
                    train=True,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    epoch_start=epoch_start,
                    global_start=start_time,
                )
                history[f"train_epoch_{epoch}"] = train_metrics

                # Log training metrics to MLflow
                for k, v in train_metrics.items():
                    if k != "epoch":
                        mlflow.log_metric(f"train_{k}", v, step=epoch)

                if val_loaders:
                    val_metrics = self._run_epoch(val_loaders, train=False, epoch=epoch)
                    history[f"val_epoch_{epoch}"] = val_metrics

                    # Log validation metrics to MLflow
                    for k, v in val_metrics.items():
                        if k != "epoch":
                            mlflow.log_metric(f"val_{k}", v, step=epoch)

                    # Generate sample summaries for manual quality assessment
                    if "summarization" in val_loaders:
                        self._validate_generation(val_loaders["summarization"], epoch)

                # Save checkpoint after each epoch
                if checkpoint_callback is not None:
                    checkpoint_callback(epoch, self.model, history)

                epoch_duration = time.perf_counter() - epoch_start
                total_elapsed = time.perf_counter() - start_time
                self._print_epoch_progress(epoch, total_epochs, epoch_duration, total_elapsed)

        return history

    def _run_epoch(
        self,
        loaders: Dict[str, DataLoader],
        *,
        train: bool,
        epoch: int,
        total_epochs: int | None = None,
        epoch_start: float | None = None,
        global_start: float | None = None,
    ) -> Dict[str, float]:
        phase = "train" if train else "eval"
        self.model.train(train)
        metrics_accumulator: Dict[str, list[float]] = defaultdict(list)
        iterator_map: Dict[str, Iterator[Dict[str, torch.Tensor]]] = {
            task: iter(loader) for task, loader in loaders.items()
        }
        max_batches = max(len(loader) for loader in loaders.values())
        progress_enabled = (
            train
            and max_batches > 0
            and total_epochs is not None
            and epoch_start is not None
            and global_start is not None
        )

        def emit_progress(step: int, final: bool = False) -> None:
            if not progress_enabled:
                return
            total_epochs_value = total_epochs
            epoch_start_value = epoch_start
            global_start_value = global_start
            assert total_epochs_value is not None
            assert epoch_start_value is not None
            assert global_start_value is not None
            self._update_epoch_progress(
                epoch=epoch,
                total_epochs=total_epochs_value,
                step=step,
                total_steps=max_batches,
                epoch_start=epoch_start_value,
                global_start=global_start_value,
                final=final,
            )

        emit_progress(0)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for step in range(max_batches):
                # Mark step begin for CUDA Graphs (inductor) to handle memory reuse correctly
                if (
                    train
                    and self.device.type == "cuda"
                    and hasattr(torch.compiler, "cudagraph_mark_step_begin")
                ):
                    torch.compiler.cudagraph_mark_step_begin()

                backward_performed = False
                step_total_loss = 0.0

                # Mixed Precision Context
                # Using bfloat16 for my RTX 4070 (Ampere/Ada) - better stability than float16
                # Disable scaler for bfloat16 to prevent NaNs
                use_bfloat16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

                for task, loader in loaders.items():
                    batch = self._next_batch(iterator_map, loader, task)
                    if batch is None:
                        continue

                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
                        enabled=(self.device.type == "cuda"),
                    ):
                        loss, task_metrics = self._forward_task(task, batch, train)

                    if torch.isnan(loss):
                        if train:
                            self._nan_counter += 1
                            print(
                                f"Warning: NaN loss detected for task '{task}'. Skipping update for this task. (Consecutive NaNs: {self._nan_counter})"
                            )
                            if self._nan_counter > 10:
                                raise RuntimeError(
                                    "Too many consecutive NaN losses. Training is diverging."
                                )
                        continue
                    else:
                        if train:
                            self._nan_counter = 0

                    weight = self._task_weight(task)
                    # Scale loss by gradient accumulation steps
                    weighted_loss = (loss * weight) / self.gradient_accumulation_steps
                    step_total_loss += weighted_loss.item() * self.gradient_accumulation_steps

                    metrics_accumulator[f"{task}_loss"].append(loss.item())
                    for metric_name, metric_value in task_metrics.items():
                        metrics_accumulator[f"{task}_{metric_name}"].append(metric_value)

                    if train:
                        # Scale loss before backward to prevent underflow
                        # We accumulate gradients from all tasks before stepping the optimizer
                        # This effectively minimizes the weighted sum of losses: L_total = w1*L1 + w2*L2 + ...
                        if use_bfloat16:
                            # bfloat16 doesn't need scaling and it can cause NaNs
                            weighted_loss.backward()
                        else:
                            self.scaler.scale(weighted_loss).backward()
                        backward_performed = True

                if backward_performed:
                    metrics_accumulator["total_loss"].append(step_total_loss)

                # Perform optimizer step only after accumulating enough gradients
                if (
                    train
                    and backward_performed
                    and (step + 1) % self.gradient_accumulation_steps == 0
                ):
                    # Unscale gradients before clipping
                    if use_bfloat16:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )

                        # Step optimizer using scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                if (
                    train
                    and self.config.logging_interval
                    and (step + 1) % self.config.logging_interval == 0
                ):
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        torch.cuda.empty_cache()
                emit_progress(step + 1)
        emit_progress(max_batches, final=True)

        averaged = {
            name: sum(values) / len(values)
            for name, values in metrics_accumulator.items()
            if values
        }
        averaged["epoch"] = float(epoch)
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in averaged.items() if k != "epoch")
        print(f"[{phase}] epoch {epoch}: {metric_str}")
        return averaged

    def _next_batch(
        self,
        iterator_map: Dict[str, Iterator[Dict[str, torch.Tensor]]],
        loader: DataLoader,
        task: str,
    ) -> Dict[str, torch.Tensor] | None:
        try:
            batch = next(iterator_map[task])
        except StopIteration:
            iterator_map[task] = iter(loader)
            try:
                batch = next(iterator_map[task])
            except StopIteration:
                return None
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _forward_task(
        self, task: str, batch: Dict[str, torch.Tensor], train: bool
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if task == "summarization":
            summarization_inputs = {
                "src_ids": batch["src_ids"],
                "tgt_ids": batch["tgt_ids"],
            }
            if "src_mask" in batch:
                summarization_inputs["src_mask"] = batch["src_mask"]
            logits = self.model.forward("summarization", summarization_inputs)
            vocab_size = logits.size(-1)
            # Apply label smoothing for regularization - prevents overconfident predictions
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                batch["labels"].view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
            summaries = self._decode_predictions(logits)
            references = self._decode_labels(batch["labels"])
            rouge = rouge_like(summaries, references)
            return loss, {"rouge_like": rouge}

        if task == "emotion":
            emotion_inputs = {"input_ids": batch["input_ids"]}
            if "attention_mask" in batch:
                emotion_inputs["attention_mask"] = batch["attention_mask"]
            logits = self.model.forward("emotion", emotion_inputs)
            loss = self.emotion_loss(logits, batch["labels"].float())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            labels = batch["labels"].int()
            f1 = multilabel_f1(preds, labels)
            return loss, {"f1": f1}

        if task == "topic":
            topic_inputs = {"input_ids": batch["input_ids"]}
            if "attention_mask" in batch:
                topic_inputs["attention_mask"] = batch["attention_mask"]
            logits = self.model.forward("topic", topic_inputs)
            loss = self.topic_loss(logits, batch["labels"])
            preds = logits.argmax(dim=-1)
            acc = accuracy(preds.tolist(), batch["labels"].tolist())
            return loss, {"accuracy": acc}

        raise ValueError(f"Unknown task '{task}'")

    def _task_weight(self, task: str) -> float:
        if not self.config.task_weights:
            return 1.0
        return self.config.task_weights.get(task, 1.0)

    def _decode_predictions(self, logits: torch.Tensor) -> List[str]:
        generated = logits.argmax(dim=-1)
        return self.tokenizer.decode_batch(generated.tolist())

    def _decode_labels(self, labels: torch.Tensor) -> List[str]:
        valid = labels.clone()
        valid[valid == -100] = self.tokenizer.pad_token_id
        return self.tokenizer.decode_batch(valid.tolist())

    def _validate_generation(self, val_loader: DataLoader, epoch: int) -> None:
        """Generate and print sample summaries to monitor quality during training."""
        self.model.eval()
        samples_generated = 0
        print(f"\n{'=' * 80}")
        print(f"[Validation Generation - Epoch {epoch}]")
        print(f"{'=' * 80}")

        with torch.no_grad():
            for batch in val_loader:
                if samples_generated >= self.config.validation_samples:
                    break

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                src_ids = batch["src_ids"]
                src_mask = batch.get("src_mask")
                labels = batch["labels"]

                # Only process first item from batch
                src_ids = src_ids[:1]
                if src_mask is not None:
                    src_mask = src_mask[:1]
                labels = labels[:1]

                # Encode source
                encoder_mask = None
                if src_mask is not None:
                    encoder_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)
                memory = self.model.encoder(src_ids, mask=encoder_mask)

                # DEBUG: Check encoder output statistics
                if samples_generated == 0:
                    print("\n[DEBUG] Encoder output stats:")
                    print(f"  Shape: {memory.shape}")
                    print(f"  Mean: {memory.mean().item():.6f}")
                    print(f"  Std: {memory.std().item():.6f}")
                    print(f"  Min: {memory.min().item():.6f}")
                    print(f"  Max: {memory.max().item():.6f}")
                    print(f"  Has NaN: {torch.isnan(memory).any().item()}")
                    print(f"  Has Inf: {torch.isinf(memory).any().item()}")

                    # Check first few positions
                    print(f"  First position norm: {memory[0, 0].norm().item():.4f}")
                    print(f"  Last position norm: {memory[0, -1].norm().item():.4f}")

                # Ban special tokens from generation
                ban_token_ids = [self.tokenizer.bos_token_id, self.tokenizer.pad_token_id]
                unk_id = getattr(self.tokenizer._tokenizer, "unk_token_id", None)
                if isinstance(unk_id, int):
                    ban_token_ids.append(unk_id)
                ban_token_ids = [tid for tid in ban_token_ids if tid is not None]

                # Generate using naive method (full forward, O(N^2)) for debugging
                generated = self.model.decoder.greedy_decode_naive(
                    memory=memory,
                    max_len=self.config.validation_max_length,
                    start_token_id=self.tokenizer.bos_token_id,
                    end_token_id=self.tokenizer.eos_token_id,
                    device=self.device,
                    memory_mask=src_mask,
                )

                # Decode
                source_text = self.tokenizer.decode(src_ids[0].tolist())
                generated_text = self.tokenizer.decode(generated[0].tolist())
                reference_text = self._decode_labels(labels)[0]

                print(f"\nSample {samples_generated + 1}:")
                print(
                    f"Raw token IDs: {generated[0][:20].tolist()}..."
                )  # Debug: show first 20 tokens
                print(
                    f"Source: {source_text[:200]}..."
                    if len(source_text) > 200
                    else f"Source: {source_text}"
                )
                print(f"Generated: {generated_text}")
                print(
                    f"Reference: {reference_text[:200]}..."
                    if len(reference_text) > 200
                    else f"Reference: {reference_text}"
                )
                print("-" * 80)

                samples_generated += 1

        print(f"{'=' * 80}\n")
        self.model.train()

    def _print_epoch_progress(
        self,
        epoch: int,
        total_epochs: int,
        epoch_duration: float,
        total_elapsed: float,
    ) -> None:
        progress = epoch / total_epochs
        percent = progress * 100
        remaining_epochs = total_epochs - epoch
        eta = (total_elapsed / epoch) * remaining_epochs if epoch > 0 else 0.0
        bar = self._format_progress_bar(progress)
        message = (
            f"[progress] {bar} {percent:5.1f}% | epoch {epoch}/{total_epochs} "
            f"| last {epoch_duration:6.2f}s | total {total_elapsed:6.2f}s | ETA {eta:6.2f}s"
        )
        print(message, flush=True)

    @staticmethod
    def _format_progress_bar(progress: float, width: int = 20) -> str:
        clamped = max(0.0, min(1.0, progress))
        filled = int(round(clamped * width))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}]"

    def _update_epoch_progress(
        self,
        *,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        epoch_start: float,
        global_start: float,
        final: bool = False,
    ) -> None:
        if total_steps <= 0 or total_epochs <= 0:
            return
        bounded_step = max(0, min(step, total_steps))
        step_fraction = bounded_step / total_steps
        epochs_completed = (epoch - 1) + step_fraction
        overall_progress = epochs_completed / total_epochs
        percent = overall_progress * 100.0
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - global_start
        if epochs_completed > 0:
            remaining_epochs = max(total_epochs - epochs_completed, 0.0)
            total_eta = (
                (total_elapsed / epochs_completed) * remaining_epochs if total_elapsed > 0 else 0.0
            )
        else:
            total_eta = 0.0

        if step > 0:
            epoch_eta = (epoch_elapsed / step) * (total_steps - step)
        else:
            epoch_eta = 0.0

        bar = self._format_progress_bar(overall_progress, width=self._progress_bar_width())
        message = (
            f"[progress] {bar} {percent:5.1f}% "
            f"e {epoch}/{total_epochs} "
            f"s {bounded_step}/{total_steps} "
            f"ep_eta {self._format_duration(epoch_eta)} "
            f"tot_eta {self._format_duration(total_eta)}"
        )
        display = self._truncate_to_terminal(message)
        padding = " " * max(self._progress_last_len - len(display), 0)
        print(f"\r{display}{padding}", end="", flush=True)
        if final:
            print()
            self._progress_last_len = 0
        else:
            self._progress_last_len = len(display)

    def _truncate_to_terminal(self, text: str) -> str:
        columns = self._terminal_width()
        if columns <= 0:
            return text
        if len(text) >= columns:
            return text[: max(columns - 1, 1)]
        return text

    def _progress_bar_width(self) -> int:
        columns = self._terminal_width()
        reserved = 60
        if columns <= reserved:
            return 10
        return max(10, min(30, columns - reserved))

    @staticmethod
    def _terminal_width() -> int:
        try:
            return shutil.get_terminal_size(fallback=(120, 20)).columns
        except OSError:
            return 120

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02}m"
        if seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02}s"
        return f"{seconds:4.1f}s"
