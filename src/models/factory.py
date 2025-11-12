"""Factory helpers to assemble multitask models for inference/training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..data.tokenization import Tokenizer
from ..utils.config import load_yaml
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .heads import ClassificationHead, LMHead
from .multitask import MultiTaskModel


@dataclass(slots=True)
class ModelConfig:
    """Configuration describing the transformer architecture."""

    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.num_attention_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.d_model <= 0 or self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.num_attention_heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("Model dimensions must be positive")


def load_model_config(path: Optional[str | Path]) -> ModelConfig:
    """Load a model configuration from YAML with sane defaults."""

    if path is None:
        return ModelConfig()

    data = load_yaml(str(path)).data
    return ModelConfig(
        d_model=int(data.get("d_model", 512)),
        num_encoder_layers=int(data.get("num_encoder_layers", 6)),
        num_decoder_layers=int(data.get("num_decoder_layers", 6)),
        num_attention_heads=int(data.get("num_attention_heads", 8)),
        ffn_dim=int(data.get("ffn_dim", 2048)),
        dropout=float(data.get("dropout", 0.1)),
    )


def build_multitask_model(
    tokenizer: Tokenizer,
    *,
    num_emotions: int,
    num_topics: int,
    config: ModelConfig | None = None,
) -> MultiTaskModel:
    """Construct the multitask transformer with heads for the three tasks."""

    cfg = config or ModelConfig()
    if not isinstance(num_emotions, int) or num_emotions <= 0:
        raise ValueError("num_emotions must be a positive integer")
    if not isinstance(num_topics, int) or num_topics <= 0:
        raise ValueError("num_topics must be a positive integer")
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_encoder_layers,
        num_heads=cfg.num_attention_heads,
        d_ff=cfg.ffn_dim,
        dropout=cfg.dropout,
        max_len=tokenizer.config.max_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_attention_heads,
        d_ff=cfg.ffn_dim,
        dropout=cfg.dropout,
        max_len=tokenizer.config.max_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = MultiTaskModel(encoder=encoder, decoder=decoder, decoder_outputs_logits=True)
    model.add_head(
        "summarization",
        LMHead(d_model=cfg.d_model, vocab_size=tokenizer.vocab_size, tie_embedding=decoder.embedding),
    )
    model.add_head(
        "emotion",
        ClassificationHead(d_model=cfg.d_model, num_labels=num_emotions, pooler="mean", dropout=cfg.dropout),
    )
    model.add_head(
        "topic",
        ClassificationHead(d_model=cfg.d_model, num_labels=num_topics, pooler="mean", dropout=cfg.dropout),
    )
    return model
