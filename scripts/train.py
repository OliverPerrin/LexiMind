#!/usr/bin/env python3
"""
Training script for LexiMind.

Simple, clean training with multi-task learning across:
- Summarization (BookSum + arXiv papers)
- Emotion classification (GoEmotions, 28 labels)
- Topic classification (Books + Papers, 7 labels: Arts, Business, Fiction, History, Philosophy, Science, Technology)

Usage:
    python scripts/train.py training=medium
    python scripts/train.py training=full

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    build_emotion_dataloader,
    build_summarization_dataloader,
    build_topic_dataloader,
)
from src.data.dataset import (
    EmotionDataset,
    SummarizationDataset,
    TopicDataset,
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import ModelConfig, build_multitask_model
from src.training.trainer import Trainer, TrainerConfig
from src.utils.io import load_state, save_state
from src.utils.labels import LabelMetadata, save_label_metadata


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: Path, loader_fn) -> Dict[str, list]:
    """Load train/val/test splits from data directory."""
    splits = {}
    for name, aliases in [("train", ["train"]), ("val", ["val", "validation"]), ("test", ["test"])]:
        for alias in aliases:
            path = data_dir / f"{alias}.jsonl"
            if path.exists():
                splits[name] = loader_fn(str(path))
                break
    return splits


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
    
    # GPU optimizations for Ampere+
    if device.type == "cuda":
        # Enable cudnn benchmark for fixed-size inputs (10-20% speedup)
        torch.backends.cudnn.benchmark = True
        
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ TF32 + cudnn.benchmark enabled for Ampere GPU")
        else:
            print("✓ cudnn.benchmark enabled")
    
    # --------------- Load Data ---------------
    
    print("\nLoading datasets...")
    data_cfg = cfg.data
    trainer_cfg = cfg.training.get("trainer", {})
    
    # Load splits
    summ_splits = load_splits(Path(data_cfg.processed.summarization), load_summarization_jsonl)
    emot_splits = load_splits(Path(data_cfg.processed.emotion), load_emotion_jsonl)
    topic_splits = load_splits(Path(data_cfg.processed.topic), load_topic_jsonl)
    
    # Apply sample limits for dev runs
    max_train = trainer_cfg.get("max_train_samples")
    max_val = trainer_cfg.get("max_val_samples")
    if max_train:
        for splits in [summ_splits, emot_splits, topic_splits]:
            splits["train"] = splits["train"][:max_train]
    if max_val:
        for splits in [summ_splits, emot_splits, topic_splits]:
            if "val" in splits:
                splits["val"] = splits["val"][:max_val]
    
    print(f"  Summarization: {len(summ_splits['train']):,} train, {len(summ_splits.get('val', [])):,} val")
    print(f"  Emotion: {len(emot_splits['train']):,} train, {len(emot_splits.get('val', [])):,} val")
    print(f"  Topic: {len(topic_splits['train']):,} train, {len(topic_splits.get('val', [])):,} val")
    
    # --------------- Tokenizer ---------------
    
    tok_cfg = data_cfg.get("tokenizer", {})
    max_len = int(cfg.training.get("tokenizer_max_length") or tok_cfg.get("max_length", 512))
    
    tokenizer = Tokenizer(TokenizerConfig(
        pretrained_model_name=tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
        max_length=max_len,
    ))
    print(f"  Tokenizer: {tokenizer.vocab_size:,} vocab, max_len={max_len}")
    
    # --------------- Datasets ---------------
    
    summ_train = SummarizationDataset(summ_splits["train"])
    summ_val = SummarizationDataset(summ_splits.get("val", []))
    emot_train = EmotionDataset(emot_splits["train"])
    emot_val = EmotionDataset(emot_splits.get("val", []), binarizer=emot_train.binarizer)
    topic_train = TopicDataset(topic_splits["train"])
    topic_val = TopicDataset(topic_splits.get("val", []), encoder=topic_train.encoder)
    
    print(f"  Emotions: {len(emot_train.emotion_classes)} classes")
    print(f"  Topics: {len(topic_train.topic_classes)} classes → {list(map(str, topic_train.topic_classes))}")
    
    # --------------- DataLoaders ---------------
    
    dl_cfg = cfg.training.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 8))
    num_workers = int(dl_cfg.get("num_workers", 4))
    
    # Classification tasks don't need full 512 tokens - 256 is sufficient
    # This speeds up emotion/topic forward passes significantly
    classification_max_len = min(256, max_len)
    
    train_loaders = {
        "summarization": build_summarization_dataloader(
            summ_train, tokenizer, shuffle=True,
            max_source_length=max_len, max_target_length=max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
        "emotion": build_emotion_dataloader(
            emot_train, tokenizer, shuffle=True, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
        "topic": build_topic_dataloader(
            topic_train, tokenizer, shuffle=True, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        ),
    }
    
    val_loaders = {}
    if summ_val:
        val_loaders["summarization"] = build_summarization_dataloader(
            summ_val, tokenizer, shuffle=False,
            max_source_length=max_len, max_target_length=max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        )
    if emot_val:
        val_loaders["emotion"] = build_emotion_dataloader(
            emot_val, tokenizer, shuffle=False, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        )
    if topic_val:
        val_loaders["topic"] = build_topic_dataloader(
            topic_val, tokenizer, shuffle=False, max_length=classification_max_len,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        )
    
    # --------------- Model ---------------
    
    print("\nBuilding model...")
    
    # Check for overrides in training config
    grad_ckpt = cfg.training.get("gradient_checkpointing", cfg.model.get("gradient_checkpointing", False))
    use_rel_pos = cfg.training.get("use_relative_position_bias", cfg.model.get("use_relative_position_bias", False))
    
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
        print("  ✓ Gradient checkpointing enabled")
    if not use_rel_pos:
        print("  ✓ FlashAttention enabled (no relative position bias)")
    
    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emot_train.emotion_classes),
        num_topics=len(topic_train.topic_classes),
        config=model_cfg,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Freeze lower encoder layers (keeps pretrained language understanding, adapts upper layers)
    freeze_layers = cfg.training.get("freeze_encoder_layers", 0)
    if freeze_layers > 0:
        frozen_params = 0
        # Freeze embedding layer
        if hasattr(model.encoder, 'embed_tokens'):
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
        # Freeze specified number of encoder layers
        if hasattr(model.encoder, 'layers'):
            for i, layer in enumerate(model.encoder.layers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                        frozen_params += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Frozen encoder layers 0-{freeze_layers-1} ({frozen_params/1e6:.1f}M params)")
        print(f"  Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    
    # Resume from checkpoint?
    start_epoch = 1
    resume_path = cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        print(f"  Resuming from: {resume_path}")
        load_state(model, str(resume_path))
        import re
        digits = re.findall(r"\d+", Path(resume_path).stem)
        if digits:
            start_epoch = int(digits[-1]) + 1
    
    # Compile model for speed
    # Note: "reduce-overhead" mode uses CUDA graphs which conflicts with gradient checkpointing
    # Use "default" mode when checkpointing is enabled
    compile_mode = "default" if grad_ckpt else "reduce-overhead"
    if cfg.training.get("compile_encoder", True):
        model.encoder = torch.compile(model.encoder, mode=compile_mode)  # type: ignore[assignment]
        print(f"  ✓ Encoder compiled ({compile_mode})")
    if cfg.training.get("compile_decoder", True):
        model.decoder = torch.compile(model.decoder, mode=compile_mode)  # type: ignore[assignment]
        print(f"  ✓ Decoder compiled ({compile_mode})")
    
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
        fused=use_fused,
    )
    if use_fused:
        print("  ✓ Fused AdamW optimizer")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=TrainerConfig(
            max_epochs=int(trainer_cfg.get("max_epochs", 10)),
            gradient_clip_norm=float(trainer_cfg.get("gradient_clip_norm", 1.0)),
            task_weights=trainer_cfg.get("task_weights"),
            label_smoothing=float(trainer_cfg.get("label_smoothing", 0.1)),
            gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
            scheduler_type=str(sched_cfg.get("name", "cosine")),
            warmup_steps=int(sched_cfg.get("warmup_steps", 500)),
            early_stopping_patience=trainer_cfg.get("early_stopping_patience"),
        ),
        device=device,
        tokenizer=tokenizer,
    )
    
    # Checkpoint callback
    ckpt_dir = Path(cfg.checkpoint_out).parent
    best_val_loss = float('inf')
    
    def save_checkpoint(epoch: int, model: torch.nn.Module, history: Dict) -> None:
        nonlocal best_val_loss
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save epoch checkpoint
        save_state(model, str(ckpt_dir / f"epoch_{epoch}.pt"))
        
        # Track best
        val_key = f"val_epoch_{epoch}"
        if val_key in history:
            val_loss = history[val_key].get("total_loss", float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_state(model, str(ckpt_dir / "best.pt"))
                print(f"  💾 New best model (val_loss={val_loss:.4f})")
    
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
        LabelMetadata(emotion=emot_train.emotion_classes, topic=topic_train.topic_classes),
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
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"  Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
