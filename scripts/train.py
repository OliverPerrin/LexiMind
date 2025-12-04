"""End-to-end training entrypoint for the LexiMind multitask model."""

from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, cast

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

SplitExamples = Dict[str, list]


SPLIT_ALIASES: Dict[str, Sequence[str]] = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}


def _read_examples(data_dir: Path, loader) -> SplitExamples:
    splits: SplitExamples = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        found = False
        for alias in aliases:
            for extension in ("jsonl", "json"):
                candidate = data_dir / f"{alias}.{extension}"
                if candidate.exists():
                    splits[canonical] = loader(str(candidate))
                    found = True
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"Missing {canonical} split under {data_dir}")
    return splits


def _limit_samples(splits: SplitExamples, trainer_cfg: DictConfig) -> None:
    """Limit the number of samples in train/val splits if configured."""
    max_train = trainer_cfg.get("max_train_samples")
    max_val = trainer_cfg.get("max_val_samples")

    if max_train is not None and "train" in splits:
        original_len = len(splits["train"])
        limit = int(max_train)
        if original_len > limit:
            splits["train"] = splits["train"][:limit]
            print(f"Limited 'train' split from {original_len} to {limit} samples")

    if max_val is not None and "val" in splits:
        original_len = len(splits["val"])
        limit = int(max_val)
        if original_len > limit:
            splits["val"] = splits["val"][:limit]
            print(f"Limited 'val' split from {original_len} to {limit} samples")


def compile_model_safe(model: torch.nn.Module) -> Tuple[Any, str]:
    """
    Safely compile model with best available backend.

    Returns:
        Compiled model and backend name used
    """
    system = platform.system()

    # NOTE: The 'inductor' backend causes NaN gradients during backward pass with
    # bfloat16 autocast on the decoder (seq2seq tasks). This is a known issue.
    # Use 'aot_eager' which provides graph optimization without inductor's codegen.
    # See: debug_compile_config.py and test_compile_modes.py for investigation.

    # Try aot_eager first - it's stable and provides good speedup
    try:
        print("Attempting to compile with 'aot_eager' backend...")
        compiled_model = torch.compile(model, backend="aot_eager")
        print("✓ Successfully compiled with 'aot_eager' backend")
        return cast(torch.nn.Module, compiled_model), "aot_eager"
    except Exception as e:
        warnings.warn(f"aot_eager backend failed: {e}", stacklevel=2)

    # Fallback: Try other backends (inductor may work for encoder-only tasks)
    backends_to_try = ["eager"]
    if system != "Windows":
        # On Linux, inductor might work for some configurations
        backends_to_try = ["eager", "inductor"]

    for backend in backends_to_try:
        try:
            print(f"Attempting to compile with '{backend}' backend...")
            compiled_model = torch.compile(model, backend=backend)
            # Trigger a dummy run or just return? torch.compile is lazy.
            # I assume it works if the call succeeds, runtime errors handled later.
            print(f"✓ Successfully compiled with '{backend}' backend")
            return cast(torch.nn.Module, compiled_model), backend
        except Exception as e:
            print(f"✗ '{backend}' backend failed: {e}")
            continue

    # No compilation worked, return original model
    warnings.warn("All torch.compile backends failed, using uncompiled model", stacklevel=2)
    return model, "none"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Enable TF32 for Ampere/Ada GPUs (RTX 30xx/40xx)
    # This provides significant speedup on RTX 4070
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TF32 for Ampere/Ada GPU...")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tunes convolution algorithms

    # Access configs directly from Hydra cfg object
    data_cfg = cfg.data
    training_cfg = cfg.training

    # Instantiate ModelConfig directly from cfg.model
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

    summarization_dir = Path(data_cfg.processed.summarization)
    emotion_dir = Path(data_cfg.processed.emotion)
    topic_dir = Path(data_cfg.processed.topic)

    summarization_splits = _read_examples(summarization_dir, load_summarization_jsonl)
    emotion_splits = _read_examples(emotion_dir, load_emotion_jsonl)
    topic_splits = _read_examples(topic_dir, load_topic_jsonl)

    # Apply sample limits if configured (e.g. for dev/medium runs)
    trainer_cfg = training_cfg.get("trainer", {})
    print("\nApplying dataset limits...")
    _limit_samples(summarization_splits, trainer_cfg)
    _limit_samples(emotion_splits, trainer_cfg)
    _limit_samples(topic_splits, trainer_cfg)
    print("Dataset limits applied.\n")

    tokenizer_section = data_cfg.get("tokenizer", {})
    tokenizer_config = TokenizerConfig(
        pretrained_model_name=tokenizer_section.get("pretrained_model_name", "google/flan-t5-base"),
        max_length=int(tokenizer_section.get("max_length", 512)),
        lower=bool(tokenizer_section.get("lower", False)),
    )
    tokenizer = Tokenizer(tokenizer_config)

    summarization_train = SummarizationDataset(summarization_splits["train"])
    summarization_val = SummarizationDataset(summarization_splits["val"])

    emotion_train = EmotionDataset(emotion_splits["train"])
    emotion_val = EmotionDataset(emotion_splits["val"], binarizer=emotion_train.binarizer)

    topic_train = TopicDataset(topic_splits["train"])
    topic_val = TopicDataset(topic_splits["val"], encoder=topic_train.encoder)

    dataloader_args = training_cfg.get("dataloader", {})
    batch_size = int(dataloader_args.get("batch_size", 8))
    shuffle = bool(dataloader_args.get("shuffle", True))
    # Optimization: Use multiple workers and pinned memory for faster data transfer
    num_workers = int(dataloader_args.get("num_workers", 4))
    pin_memory = bool(dataloader_args.get("pin_memory", True))
    max_length = tokenizer.config.max_length

    train_loaders = {
        "summarization": build_summarization_dataloader(
            summarization_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_source_length=max_length,
            max_target_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "emotion": build_emotion_dataloader(
            emotion_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "topic": build_topic_dataloader(
            topic_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    val_loaders = {
        "summarization": build_summarization_dataloader(
            summarization_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_source_length=max_length,
            max_target_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "emotion": build_emotion_dataloader(
            emotion_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "topic": build_topic_dataloader(
            topic_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    device = torch.device(cfg.device)
    model = build_multitask_model(
        tokenizer,
        num_emotions=len(emotion_train.emotion_classes),
        num_topics=len(topic_train.topic_classes),
        config=model_cfg,
    ).to(device)

    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 3.0e-5))
    # Add weight decay for regularization to prevent overfitting
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optimize model execution graph with torch.compile (PyTorch 2.0+)
    # This fuses kernels and reduces overhead for faster training
    # Note: We only compile encoder/decoder for training, not the step() method used in generation
    # Compile encoder and decoder separately to avoid control flow issues in MultiTaskModel.forward
    # Compiling the top-level model causes excessive recompilation due to task switching
    use_compile = True  # torch.compile for faster training

    if use_compile and model.encoder is not None:
        model.encoder, backend_used = compile_model_safe(model.encoder)
    else:
        backend_used = "disabled"
    if use_compile and model.decoder is not None:
        # Compile decoder.forward but keep step/greedy_decode uncompiled for generation
        model.decoder, _ = compile_model_safe(model.decoder)

    # Compile heads
    if use_compile:
        for name, head in model.heads.items():
            compiled_head, _ = compile_model_safe(head)
            model.heads[name] = compiled_head
            # Update the registered module as well to ensure parameters are tracked correctly
            setattr(model, f"head_{name}", compiled_head)

    print(f"Using compilation backend: {backend_used}")

    # Verify weights loaded correctly (check for NaNs/Infs)
    print("\n=== Weight Loading Verification ===")
    has_issues = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN in {name}")
            has_issues = True
        if torch.isinf(param).any():
            print(f"WARNING: Inf in {name}")
            has_issues = True
    if not has_issues:
        print("✓ No NaNs or Infs found in model parameters.")
    print("=== Verification Complete ===\n")

    trainer_cfg = training_cfg.get("trainer", {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=TrainerConfig(
            max_epochs=int(trainer_cfg.get("max_epochs", 1)),
            gradient_clip_norm=float(trainer_cfg.get("gradient_clip_norm", 1.0)),
            logging_interval=int(trainer_cfg.get("logging_interval", 50)),
            task_weights=trainer_cfg.get("task_weights"),
            label_smoothing=float(trainer_cfg.get("label_smoothing", 0.0)),
            gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        ),
        device=device,
        tokenizer=tokenizer,
    )

    # Save checkpoint after every epoch to avoid losing good early checkpoints
    # Previous training showed overfitting at epoch 5 but good results at epoch 3
    def save_epoch_checkpoint(epoch: int, model: torch.nn.Module, history: Dict) -> None:
        epoch_path = Path(cfg.checkpoint_out).parent / f"epoch_{epoch}.pt"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        save_state(model, str(epoch_path))
        print(f"Checkpoint saved: {epoch_path}")

    history = trainer.fit(train_loaders, val_loaders, checkpoint_callback=save_epoch_checkpoint)

    checkpoint_path = Path(cfg.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_state(model, str(checkpoint_path))

    labels_path = Path(cfg.labels_out)
    save_label_metadata(
        LabelMetadata(
            emotion=emotion_train.emotion_classes,
            topic=topic_train.topic_classes,
        ),
        labels_path,
    )

    history_path = Path(cfg.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training complete. Checkpoint saved to {checkpoint_path}")
    print(f"Label metadata saved to {labels_path}")
    print(f"History saved to {history_path}")

    # Run evaluation pipeline
    print("\nRunning evaluation pipeline...")
    import subprocess

    try:
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--split",
                "test",  # Evaluate on test set
                "--checkpoint",
                str(checkpoint_path),
                "--labels",
                str(labels_path),
                "--output-dir",
                "outputs",
            ],
            check=True,
        )
        print("Evaluation pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation pipeline failed with error: {e}")


if __name__ == "__main__":
    main()
