from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from omegaconf import OmegaConf

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    max_seq_length: int

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    max_grad_norm: float
    mixed_precision: bool

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: Dict[str, Any]
    tasks: Dict[str, Any]

def load_config(config_path: str) -> Config:
    """Load config from YAML and convert to structured dataclass."""
    cfg = OmegaConf.load(config_path)
    
    # Convert to dataclass for type safety
    model_cfg = ModelConfig(**cfg.model)
    training_cfg = TrainingConfig(**cfg.training)
    
    return Config(
        model=model_cfg,
        training=training_cfg,
        data=dict(cfg.data),
        tasks=dict(cfg.tasks)
    )
