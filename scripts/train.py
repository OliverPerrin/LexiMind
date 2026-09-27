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
from typing import Dict

import hydra
import torch
import torch._functorch.config
from omegaconf import DictConfig, OmegaConf

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_task_dataloaders
from src.data.dataset import load_splits as load_splits
from src.data.dataset import load_training_datasets
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import ModelConfig, build_multitask_model
from src.training.trainer import Trainer, TrainerConfig
from src.utils.io import load_state, save_state
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


def validate_resume_labels(cfg: DictConfig, *, emotion: list[str], topic: list[str]) -> None:
    """Bind existing classification columns before a weights-only continuation."""
    if not cfg.get("resume_from"):
        return
    labels_path = cfg.get("resume_labels")
    if not isinstance(labels_path, str) or not labels_path.strip():
        raise ValueError(
            "resume_from requires explicit resume_labels=/path/to/checkpoint-labels.json; matching head dimensions do not establish label order"
        )
    saved = load_label_metadata(labels_path)
    if saved.emotion != emotion or saved.topic != topic:
        raise ValueError(
            "Resume label vocabularies/order differ from current datasets; supply the checkpoint's exact ordered labels and compatible dataset labels.json files"
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    start_time = time.perf_counter()

    print("=" * 60)
    print("LexiMind Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

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
    )
    for task in enabled_tasks:
        print(
            f"  {task}: {len(train_datasets[task]):,} train, {len(val_datasets.get(task, [])):,} val"
        )
    print(f"  Enabled tasks: {enabled_tasks}")
    emotion_classes = getattr(train_datasets.get("emotion"), "emotion_classes", [])
    topic_classes = getattr(train_datasets.get("topic"), "topic_classes", [])
    validate_resume_labels(cfg, emotion=emotion_classes, topic=topic_classes)

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
            pretrained_model_name=tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
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

    # Check for overrides in training config
    grad_ckpt = cfg.training.get(
        "gradient_checkpointing", cfg.model.get("gradient_checkpointing", False)
    )
    use_rel_pos = cfg.training.get(
        "use_relative_position_bias", cfg.model.get("use_relative_position_bias", False)
    )

    model_cfg = ModelConfig(
        d_model=cfg.model.d_model,
        vocab_size=getattr(cfg.model, "vocab_size", None),
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        use_pretrained=cfg.model.use_pretrained,
        pretrained_model_name=cfg.model.pretrained_model_name,
        activation=getattr(cfg.model, "activation", "gelu"),
        use_relative_position_bias=use_rel_pos,
        gradient_checkpointing=grad_ckpt,
    )

    if grad_ckpt:
        print("  Gradient checkpointing: on")
    if not use_rel_pos:
        print("  FlashAttention: on (no relative position bias)")

    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emotion_classes),
        num_topics=len(topic_classes),
        config=model_cfg,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Freeze lower encoder layers (keeps pretrained language understanding, adapts upper layers)
    freeze_layers = cfg.training.get("freeze_encoder_layers", 0)
    if freeze_layers > 0:
        frozen_params = 0
        # Freeze embedding layer
        if hasattr(model.encoder, "embed_tokens"):
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
        # Freeze specified number of encoder layers
        if hasattr(model.encoder, "layers"):
            for i, layer in enumerate(model.encoder.layers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen layers: 0-{freeze_layers - 1} ({frozen_params / 1e6:.1f}M params)")
        print(f"  Trainable: {trainable:,} ({trainable / 1e6:.1f}M)")

    # Resume from checkpoint?
    start_epoch = 1
    resume_path = cfg.get("resume_from")
    if resume_path:
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")
        print(f"  Loading weights from: {resume_path}")
        print("  Weights-only continuation: optimizer, scheduler and RNG state are not restored.")
        load_state(model, str(resume_path))
        start_epoch = resume_start_epoch(Path(resume_path))

    # Compile model for speed
    # Note: "reduce-overhead" mode uses CUDA graphs which conflicts with gradient checkpointing
    # Use "default" mode when checkpointing is enabled
    use_pcgrad = bool(trainer_cfg.get("use_pcgrad", False))
    if use_pcgrad:
        # PCGrad needs retain_graph=True for multiple backward passes, which is
        # incompatible with torch.compile's donated buffer optimization
        torch._functorch.config.donated_buffer = False
        print("  Donated buffer disabled (required for PCGrad + torch.compile)")

    # dynamic=True produces a single symbolic-shape graph so dynamic padding
    # ("longest" + pad_to_multiple_of=8 in the collators) doesn't trigger
    # recompilation on every new batch length. "reduce-overhead" with
    # dynamic shapes is not supported, so we only use dynamic under "default".
    compile_mode = "default" if grad_ckpt else "reduce-overhead"
    compile_dynamic = compile_mode == "default"
    if cfg.training.get("compile_encoder", True):
        model.encoder = torch.compile(  # type: ignore[assignment]
            model.encoder, mode=compile_mode, dynamic=compile_dynamic
        )
        print(f"  Encoder compiled ({compile_mode}, dynamic={compile_dynamic})")
    if cfg.training.get("compile_decoder", True):
        model.decoder = torch.compile(  # type: ignore[assignment]
            model.decoder, mode=compile_mode, dynamic=compile_dynamic
        )
        print(f"  Decoder compiled ({compile_mode}, dynamic={compile_dynamic})")

    # --------------- Train ---------------

    print("\nStarting training...")
    opt_cfg = cfg.training.get("optimizer", {})
    sched_cfg = cfg.training.get("scheduler", {})

    # Use fused AdamW on CUDA for ~5-10% speedup
    use_fused = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-5)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        eps=float(opt_cfg.get("eps", 1e-8)),
        betas=tuple(float(value) for value in opt_cfg.get("betas", (0.9, 0.999))),
        fused=use_fused,
    )
    if use_fused:
        print("  Fused AdamW: on")

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
        ),
        device=device,
        tokenizer=tokenizer,
    )

    # Checkpoint callback
    ckpt_dir = Path(cfg.checkpoint_out).parent
    best_val_loss = float("inf")

    def save_checkpoint(epoch: int, model: torch.nn.Module, history: Dict) -> None:
        nonlocal best_val_loss
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_state(model, str(ckpt_dir / "last.pt"))
        with (ckpt_dir / "last_epoch.json").open("w") as f:
            json.dump({"epoch": epoch}, f)

        val_key = f"val_epoch_{epoch}"
        if val_key in history:
            val_loss = history[val_key].get("total_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_state(model, str(ckpt_dir / "best.pt"))
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
    labels_path = Path(cfg.labels_out)
    save_label_metadata(
        LabelMetadata(emotion=emotion_classes, topic=topic_classes),
        labels_path,
    )
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
    print(f"  Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
