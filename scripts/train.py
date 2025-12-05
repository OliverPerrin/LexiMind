"""
Training script for LexiMind.

Orchestrates dataset loading, model construction, torch.compile optimization,
and multi-task training with checkpoint management.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Sequence, cast

# Suppress torch inductor warnings that mess up progress bars
os.environ.setdefault("TORCH_LOGS", "-all")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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
from src.training.utils import set_seed
from src.utils.io import save_state
from src.utils.labels import LabelMetadata, save_label_metadata

# --------------- Data Loading ---------------

SPLIT_ALIASES: Dict[str, Sequence[str]] = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}


def load_splits(data_dir: Path, loader) -> Dict[str, list]:
    """Load train/val/test splits from data directory."""
    splits = {}
    for name, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            for ext in ("jsonl", "json"):
                path = data_dir / f"{alias}.{ext}"
                if path.exists():
                    splits[name] = loader(str(path))
                    break
            if name in splits:
                break
        if name not in splits:
            raise FileNotFoundError(f"Missing {name} split in {data_dir}")
    return splits


def limit_samples(splits: Dict[str, list], cfg: DictConfig) -> None:
    """Apply sample limits for dev/debug runs."""
    for split, key in [("train", "max_train_samples"), ("val", "max_val_samples")]:
        limit = cfg.get(key)
        if limit and split in splits and len(splits[split]) > limit:
            splits[split] = splits[split][: int(limit)]
            print(f"  {split}: limited to {limit} samples")


# --------------- Model Compilation ---------------


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile model with inductor backend (default mode, no CUDA graphs)."""
    from src.training.safe_compile import apply_safe_config, compile_model_safe

    # Apply safe configuration first
    apply_safe_config()
    # Compile with default mode (inductor without CUDA graphs)
    return compile_model_safe(model, mode="default")


# --------------- Main ---------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Benchmark mode: skip saving checkpoints (for speed testing)
    benchmark_mode = cfg.get("benchmark", False)
    if benchmark_mode:
        print("⚡ BENCHMARK MODE: Checkpoints will NOT be saved")

    # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx) - ~2x matmul speedup
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("✓ TF32 enabled for Ampere GPU")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
        torch.backends.cuda.enable_flash_sdp(True)  # Flash attention if available
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention

    # Disable debug APIs for max speed
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # --------------- Load Data ---------------

    data_cfg = cfg.data
    trainer_cfg = cfg.training.get("trainer", {})

    print("\nLoading datasets...")
    summ_splits = load_splits(Path(data_cfg.processed.summarization), load_summarization_jsonl)
    emot_splits = load_splits(Path(data_cfg.processed.emotion), load_emotion_jsonl)
    topic_splits = load_splits(Path(data_cfg.processed.topic), load_topic_jsonl)

    # Apply dev/debug sample limits
    for splits in [summ_splits, emot_splits, topic_splits]:
        limit_samples(splits, trainer_cfg)

    # --------------- Tokenizer & Datasets ---------------

    tok_cfg = data_cfg.get("tokenizer", {})
    tokenizer = Tokenizer(
        TokenizerConfig(
            pretrained_model_name=tok_cfg.get("pretrained_model_name", "google/flan-t5-base"),
            max_length=int(tok_cfg.get("max_length", 512)),
            lower=bool(tok_cfg.get("lower", False)),
        )
    )

    summ_train = SummarizationDataset(summ_splits["train"])
    summ_val = SummarizationDataset(summ_splits["val"])
    emot_train = EmotionDataset(emot_splits["train"])
    emot_val = EmotionDataset(emot_splits["val"], binarizer=emot_train.binarizer)
    topic_train = TopicDataset(topic_splits["train"])
    topic_val = TopicDataset(topic_splits["val"], encoder=topic_train.encoder)

    # --------------- DataLoaders ---------------

    dl_cfg = cfg.training.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 8))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    max_len = tokenizer.config.max_length

    train_loaders = {
        "summarization": build_summarization_dataloader(
            summ_train,
            tokenizer,
            shuffle=True,
            max_source_length=max_len,
            max_target_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "emotion": build_emotion_dataloader(
            emot_train,
            tokenizer,
            shuffle=True,
            max_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "topic": build_topic_dataloader(
            topic_train,
            tokenizer,
            shuffle=True,
            max_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    val_loaders = {
        "summarization": build_summarization_dataloader(
            summ_val,
            tokenizer,
            shuffle=False,
            max_source_length=max_len,
            max_target_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "emotion": build_emotion_dataloader(
            emot_val,
            tokenizer,
            shuffle=False,
            max_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "topic": build_topic_dataloader(
            topic_val,
            tokenizer,
            shuffle=False,
            max_length=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    # --------------- Model ---------------

    print("\nBuilding model...")
    device = torch.device(cfg.device)
    model_cfg = ModelConfig(
        d_model=cfg.model.d_model,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        use_pretrained=cfg.model.use_pretrained,
        pretrained_model_name=cfg.model.pretrained_model_name,
        activation=getattr(cfg.model, "activation", "gelu"),
        use_relative_position_bias=getattr(cfg.model, "use_relative_position_bias", False),
    )
    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emot_train.emotion_classes),
        num_topics=len(topic_train.topic_classes),
        config=model_cfg,
    ).to(device)

    # Compile encoder/decoder for faster training (skip heads - small overhead)
    if model.encoder is not None:
        from src.models.encoder import TransformerEncoder

        model.encoder = cast(TransformerEncoder, compile_model(model.encoder))
    if model.decoder is not None:
        from src.models.decoder import TransformerDecoder

        model.decoder = cast(TransformerDecoder, compile_model(model.decoder))

    # --------------- Optimizer & Trainer ---------------

    opt_cfg = cfg.training.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-5)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=TrainerConfig(
            max_epochs=int(trainer_cfg.get("max_epochs", 1)),
            gradient_clip_norm=float(trainer_cfg.get("gradient_clip_norm", 1.0)),
            task_weights=trainer_cfg.get("task_weights"),
            label_smoothing=float(trainer_cfg.get("label_smoothing", 0.0)),
            gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        ),
        device=device,
        tokenizer=tokenizer,
    )

    # --------------- Train ---------------

    def save_checkpoint(epoch: int, model: torch.nn.Module, history: Dict) -> None:
        if benchmark_mode:
            return  # Skip saving in benchmark mode
        path = Path(cfg.checkpoint_out).parent / f"epoch_{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(model, str(path))

    print("\nStarting training...")
    history = trainer.fit(train_loaders, val_loaders, checkpoint_callback=save_checkpoint)

    # --------------- Save Outputs ---------------

    if benchmark_mode:
        total_time = time.perf_counter() - start_time
        print(f"\n{'=' * 50}")
        print(f"⚡ Benchmark complete in {total_time:.1f}s")
        print("  (No files saved in benchmark mode)")
        print(f"{'=' * 50}")
        return

    # Best checkpoint
    ckpt_path = Path(cfg.checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_state(model, str(ckpt_path))

    # Labels
    labels_path = Path(cfg.labels_out)
    save_label_metadata(
        LabelMetadata(emotion=emot_train.emotion_classes, topic=topic_train.topic_classes),
        labels_path,
    )

    # History
    history_path = Path(cfg.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    total_time = time.perf_counter() - start_time
    print(f"\n{'=' * 50}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Labels: {labels_path}")
    print(f"  History: {history_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
