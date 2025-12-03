"""End-to-end training entrypoint for the LexiMind multitask model."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Sequence, cast

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

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
    )

    summarization_dir = Path(data_cfg.processed.summarization)
    emotion_dir = Path(data_cfg.processed.emotion)
    topic_dir = Path(data_cfg.processed.topic)

    summarization_splits = _read_examples(summarization_dir, load_summarization_jsonl)
    emotion_splits = _read_examples(emotion_dir, load_emotion_jsonl)
    topic_splits = _read_examples(topic_dir, load_topic_jsonl)

    tokenizer_section = data_cfg.get("tokenizer", {})
    tokenizer_config = TokenizerConfig(
        pretrained_model_name=tokenizer_section.get("pretrained_model_name", "facebook/bart-base"),
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
    max_length = tokenizer.config.max_length

    train_loaders = {
        "summarization": build_summarization_dataloader(
            summarization_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_source_length=max_length,
            max_target_length=max_length,
        ),
        "emotion": build_emotion_dataloader(
            emotion_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
        ),
        "topic": build_topic_dataloader(
            topic_train,
            tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length,
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
        ),
        "emotion": build_emotion_dataloader(
            emotion_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        ),
        "topic": build_topic_dataloader(
            topic_val,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
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
    # This fuses kernels and reduces overhead for faster training on my RTX 4070
    print("Compiling model with torch.compile...")
    model = cast(torch.nn.Module, torch.compile(model))

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
        ),
        device=device,
        tokenizer=tokenizer,
    )

    # Save checkpoint after every epoch to avoid losing good early checkpoints
    # Previous training showed overfitting at epoch 5 but good results at epoch 3
    def save_epoch_checkpoint(epoch: int) -> None:
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
