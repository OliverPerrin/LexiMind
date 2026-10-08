"""
Training script for LexiMind.

Train the retained custom transformer on explicitly supplied, reviewed task
splits. No dataset is selected by default. Research execution remains a separate
authorization step; this entry point does not perform study admission.

Usage (after separate authorization):
    python scripts/train.py training=default 'training.trainer.tasks=[topic]' data.processed.topic=/path/to/splits

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_task_dataloaders
from src.data.dataset import load_splits as load_splits
from src.data.dataset import load_training_datasets
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.training.trainer import Trainer, TrainerConfig
from src.training.utils import (
    build_training_model,
    build_training_optimizer,
    compile_training_model,
    load_training_adapter,
    prepare_training_runtime,
    save_training_checkpoint,
)
from src.training.utils import freeze_encoder_layers as freeze_encoder_layers
from src.utils.io import load_state
from src.utils.labels import LabelMetadata, load_label_metadata, save_label_metadata


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resume_start_epoch(checkpoint: Path) -> int:
    """Infer only epoch metadata that belongs to the chosen weights.

    best.pt may precede last.pt by several epochs; its sibling last_epoch.json
    cannot identify it. This is a weights-only continuation, not exact recovery
    of optimizer, scheduler or RNG state.
    """
    if checkpoint.name == "last.pt":
        metadata = checkpoint.parent / "last_epoch.json"
        if metadata.exists():
            epoch = json.loads(metadata.read_text())["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError("last_epoch.json must record a positive integer epoch")
            return epoch + 1
    match = re.fullmatch(r"epoch_(\d+)", checkpoint.stem)
    return int(match.group(1)) + 1 if match else 1


def validate_resume_labels(
    cfg: DictConfig,
    *,
    emotion: list[str],
    topic: list[str],
    topic_problem_type: str = "single_label",
    topic_input_format: str = "text",
    topic_mapping_sha256: str | None = None,
) -> None:
    """Bind existing classification columns before a weights-only continuation."""
    if not cfg.get("resume_from"):
        return
    labels_path = cfg.get("resume_labels")
    if not isinstance(labels_path, str) or not labels_path.strip():
        raise ValueError(
            "resume_from requires explicit resume_labels=/path/to/checkpoint-labels.json; matching head dimensions do not establish label order"
        )
    saved = load_label_metadata(labels_path)
    paired = Path(cfg.resume_from).parent / "labels.json"
    if paired.exists() and load_label_metadata(paired) != saved:
        raise ValueError("resume_labels differs from the checkpoint directory's label contract")
    if saved.emotion != emotion or saved.topic != topic:
        raise ValueError(
            "Resume label vocabularies/order differ from current datasets; supply the checkpoint's exact ordered labels and compatible dataset labels.json files"
        )
    if (
        saved.topic_problem_type != topic_problem_type
        or saved.topic_input_format != topic_input_format
        or saved.topic_mapping_sha256 != topic_mapping_sha256
    ):
        raise ValueError("Resume topic loss mode, input format or field mapping differs")


def prepare_checkpoint_labels(
    metadata: LabelMetadata, labels_path: Path, checkpoint_dir: Path
) -> None:
    """Publish the label contract before weights; never replace a different contract."""
    paired = checkpoint_dir / "labels.json"
    targets = {labels_path.resolve(), paired.resolve()}
    for path in targets:
        if path.exists() and load_label_metadata(path) != metadata:
            raise ValueError(
                f"Different label metadata already exists at {path}; use a new output location"
            )
    if (
        metadata.topic_problem_type == "multi_label"
        and not paired.exists()
        and any(checkpoint_dir.glob("*.pt"))
    ):
        raise ValueError(
            "Book field checkpoints require a fresh directory or their existing paired labels.json"
        )
    for path in targets:
        if not path.exists():
            save_label_metadata(metadata, path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    start_time = time.perf_counter()

    print("=" * 60)
    print("LexiMind Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device, snapshot = prepare_training_runtime(cfg)

    # --------------- Load Data ---------------

    print("\nLoading datasets...")
    data_cfg = cfg.data
    trainer_cfg = cfg.training.get("trainer", {})

    enabled_tasks = list(trainer_cfg.get("tasks", ["summarization", "emotion", "topic"]))
    train_datasets, val_datasets = load_training_datasets(
        data_cfg.processed,
        enabled_tasks,
        max_train_samples=trainer_cfg.get("max_train_samples"),
        max_val_samples=trainer_cfg.get("max_val_samples"),
        topic_problem_type=data_cfg.get("topic_problem_type", "single_label"),
    )
    for task in enabled_tasks:
        print(
            f"  {task}: {len(train_datasets[task]):,} train, {len(val_datasets.get(task, [])):,} val"
        )
    print(f"  Enabled tasks: {enabled_tasks}")
    emotion_classes = getattr(train_datasets.get("emotion"), "emotion_classes", [])
    topic_classes = getattr(train_datasets.get("topic"), "topic_classes", [])
    topic_dataset = train_datasets.get("topic")
    topic_contract: dict[str, Any] = {
        "topic_problem_type": getattr(topic_dataset, "topic_problem_type", "single_label"),
        "topic_input_format": getattr(topic_dataset, "topic_input_format", "text"),
        "topic_mapping_sha256": getattr(topic_dataset, "topic_mapping_sha256", None),
    }
    validate_resume_labels(cfg, emotion=emotion_classes, topic=topic_classes, **topic_contract)
    label_metadata = LabelMetadata(emotion=emotion_classes, topic=topic_classes, **topic_contract)
    labels_path = Path(cfg.labels_out)
    ckpt_dir = Path(cfg.checkpoint_out).parent
    prepare_checkpoint_labels(label_metadata, labels_path, ckpt_dir)

    # GPU optimizations for Ampere+
    if device.type == "cuda":
        # Optional CUDA backend settings retained from the training recipe.
        torch.backends.cudnn.benchmark = True

        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  TF32 + cudnn.benchmark enabled (Ampere GPU)")
        else:
            print("  cudnn.benchmark enabled")

    # --------------- Tokenizer ---------------

    tok_cfg = data_cfg.get("tokenizer", {})
    max_len = int(cfg.training.get("tokenizer_max_length") or tok_cfg.get("max_length", 512))

    tokenizer = Tokenizer(
        TokenizerConfig(
            pretrained_model_name=str(snapshot)
            if snapshot
            else tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
            max_length=max_len,
        )
    )
    print(f"  Tokenizer: {tokenizer.vocab_size:,} vocab, max_len={max_len}")

    # --------------- DataLoaders ---------------

    dl_cfg = cfg.training.get("dataloader", {})
    loader_options = dict(
        batch_size=int(dl_cfg.get("batch_size", 8)),
        num_workers=int(dl_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        max_length=max_len,
        classification_max_length=min(256, max_len),
    )
    train_loaders = build_task_dataloaders(
        train_datasets, tokenizer, shuffle=True, **loader_options
    )
    val_loaders = build_task_dataloaders(val_datasets, tokenizer, shuffle=False, **loader_options)

    # --------------- Model ---------------

    print("\nBuilding model...")

    model, adapter_binding = build_training_model(
        cfg,
        tokenizer,
        num_emotions=len(emotion_classes),
        num_topics=len(topic_classes),
        topic_problem_type=topic_contract["topic_problem_type"],
        snapshot=snapshot,
        compile_modules=False,
    )

    # Resume from checkpoint?
    start_epoch = 1
    resume_path = cfg.get("resume_from")
    if resume_path:
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")
        print(f"  Loading weights from: {resume_path}")
        print("  Weights-only continuation: optimizer, scheduler and RNG state are not restored.")
        if adapter_binding is not None:
            load_training_adapter(model, resume_path, adapter_binding, label_metadata)
            epoch_path = Path(str(resume_path).replace(".adapter.pt", ".pt"))
        else:
            load_state(model, str(resume_path))
            epoch_path = Path(resume_path)
        start_epoch = resume_start_epoch(epoch_path)

    compile_training_model(model, cfg)

    print("\nStarting training...")
    sched_cfg = cfg.training.get("scheduler", {})
    optimizer = build_training_optimizer(model, cfg, device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=TrainerConfig(
            max_epochs=int(trainer_cfg.get("max_epochs", 10)),
            gradient_clip_norm=float(trainer_cfg.get("gradient_clip_norm", 1.0)),
            task_weights=trainer_cfg.get("task_weights"),
            label_smoothing=float(trainer_cfg.get("label_smoothing", 0.1)),
            validation_max_length=int(trainer_cfg.get("validation_max_length", 128)),
            gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
            scheduler_type=str(sched_cfg.get("name", "cosine")),
            warmup_steps=int(sched_cfg.get("warmup_steps", 500)),
            early_stopping_patience=trainer_cfg.get("early_stopping_patience"),
            task_sampling=str(trainer_cfg.get("task_sampling", "temperature")),
            task_sampling_alpha=float(trainer_cfg.get("task_sampling_alpha", 0.5)),
            gradient_conflict_frequency=int(trainer_cfg.get("gradient_conflict_frequency", 0)),
            use_pcgrad=bool(trainer_cfg.get("use_pcgrad", False)),
            generation_metrics=bool(trainer_cfg.get("generation_metrics", True)),
            tracking_uri=str(trainer_cfg.get("tracking_uri", "sqlite:///mlruns.db")),
        ),
        device=device,
        tokenizer=tokenizer,
    )

    # Checkpoint callback
    best_val_loss = float("inf")

    def save_checkpoint(epoch: int, model: torch.nn.Module, history: Dict) -> None:
        nonlocal best_val_loss
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        prepare_checkpoint_labels(label_metadata, labels_path, ckpt_dir)

        save_training_checkpoint(model, ckpt_dir / "last.pt", adapter_binding, label_metadata)
        with (ckpt_dir / "last_epoch.json").open("w") as f:
            json.dump({"epoch": epoch}, f)

        val_key = f"val_epoch_{epoch}"
        if val_key in history:
            val_loss = history[val_key].get("total_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_training_checkpoint(
                    model, ckpt_dir / "best.pt", adapter_binding, label_metadata
                )
                print(f"  New best model saved (val_loss={val_loss:.4f})")

    history = trainer.fit(
        train_loaders,
        val_loaders if val_loaders else None,
        checkpoint_callback=save_checkpoint,
        start_epoch=start_epoch,
    )

    # --------------- Save Outputs ---------------

    print("\nSaving outputs...")

    # Labels
    prepare_checkpoint_labels(label_metadata, labels_path, ckpt_dir)
    print(f"  Labels: {labels_path}")

    # History
    history_path = Path(cfg.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_path}")

    total_time = time.perf_counter() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} minutes")
    checkpoint = ckpt_dir / "best.pt"
    if checkpoint.exists():
        print(f"  Best checkpoint: {checkpoint}")
    else:
        print(f"  Last checkpoint: {ckpt_dir / 'last.pt'} (no validation-selected best)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pilot":
        from src.training.pilot import main as pilot_main

        raise SystemExit(pilot_main(sys.argv[2:]))
    else:
        main()
