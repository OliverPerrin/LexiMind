"""Factory helpers to assemble multitask models for inference/training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import BartModel

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
    use_pretrained: bool = False
    pretrained_model_name: str = "facebook/bart-base"

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
        use_pretrained=bool(data.get("use_pretrained", False)),
        pretrained_model_name=str(data.get("pretrained_model_name", "facebook/bart-base")),
    )


def _load_pretrained_weights(encoder: TransformerEncoder, decoder: TransformerDecoder, model_name: str) -> None:
    """Load pretrained BART weights into custom encoder/decoder."""
    print(f"Loading pretrained weights from {model_name}...")
    bart = BartModel.from_pretrained(model_name)
    
    # Load encoder weights
    print("Transferring encoder weights...")
    encoder.embedding.weight.data.copy_(bart.encoder.embed_tokens.weight.data)
    # Skip positional encoding - BART uses learned positions, I use sinusoidal
    # implementation will work fine with sinusoidal encodings
    
    for i, (custom_layer, bart_layer) in enumerate(zip(encoder.layers, bart.encoder.layers)):
        # Self-attention
        custom_layer.self_attn.W_Q.weight.data.copy_(bart_layer.self_attn.q_proj.weight.data)
        custom_layer.self_attn.W_Q.bias.data.copy_(bart_layer.self_attn.q_proj.bias.data)
        custom_layer.self_attn.W_K.weight.data.copy_(bart_layer.self_attn.k_proj.weight.data)
        custom_layer.self_attn.W_K.bias.data.copy_(bart_layer.self_attn.k_proj.bias.data)
        custom_layer.self_attn.W_V.weight.data.copy_(bart_layer.self_attn.v_proj.weight.data)
        custom_layer.self_attn.W_V.bias.data.copy_(bart_layer.self_attn.v_proj.bias.data)
        custom_layer.self_attn.W_O.weight.data.copy_(bart_layer.self_attn.out_proj.weight.data)
        custom_layer.self_attn.W_O.bias.data.copy_(bart_layer.self_attn.out_proj.bias.data)
        
        # Layer norms
        custom_layer.norm1.weight.data.copy_(bart_layer.self_attn_layer_norm.weight.data)
        custom_layer.norm1.bias.data.copy_(bart_layer.self_attn_layer_norm.bias.data)
        custom_layer.norm2.weight.data.copy_(bart_layer.final_layer_norm.weight.data)
        custom_layer.norm2.bias.data.copy_(bart_layer.final_layer_norm.bias.data)
        
        # FFN - use linear1/linear2 
        custom_layer.ffn.linear1.weight.data.copy_(bart_layer.fc1.weight.data)
        custom_layer.ffn.linear1.bias.data.copy_(bart_layer.fc1.bias.data)
        custom_layer.ffn.linear2.weight.data.copy_(bart_layer.fc2.weight.data)
        custom_layer.ffn.linear2.bias.data.copy_(bart_layer.fc2.bias.data)
    
    # BART has layernorm_embedding at the input, I have final_norm at output
    # Copy it to final_norm - not a perfect match but close enough for transfer learning
    if hasattr(bart.encoder, 'layernorm_embedding'):
        encoder.final_norm.weight.data.copy_(bart.encoder.layernorm_embedding.weight.data)
        encoder.final_norm.bias.data.copy_(bart.encoder.layernorm_embedding.bias.data)
    
    # Load decoder weights
    print("Transferring decoder weights...")
    decoder.embedding.weight.data.copy_(bart.decoder.embed_tokens.weight.data)
    # Skip positional encoding - BART uses learned positions, we use sinusoidal
    
    for i, (custom_layer, bart_layer) in enumerate(zip(decoder.layers, bart.decoder.layers)):
        # Self-attention
        custom_layer.self_attn.W_Q.weight.data.copy_(bart_layer.self_attn.q_proj.weight.data)
        custom_layer.self_attn.W_Q.bias.data.copy_(bart_layer.self_attn.q_proj.bias.data)
        custom_layer.self_attn.W_K.weight.data.copy_(bart_layer.self_attn.k_proj.weight.data)
        custom_layer.self_attn.W_K.bias.data.copy_(bart_layer.self_attn.k_proj.bias.data)
        custom_layer.self_attn.W_V.weight.data.copy_(bart_layer.self_attn.v_proj.weight.data)
        custom_layer.self_attn.W_V.bias.data.copy_(bart_layer.self_attn.v_proj.bias.data)
        custom_layer.self_attn.W_O.weight.data.copy_(bart_layer.self_attn.out_proj.weight.data)
        custom_layer.self_attn.W_O.bias.data.copy_(bart_layer.self_attn.out_proj.bias.data)
        
        # Cross-attention
        custom_layer.cross_attn.W_Q.weight.data.copy_(bart_layer.encoder_attn.q_proj.weight.data)
        custom_layer.cross_attn.W_Q.bias.data.copy_(bart_layer.encoder_attn.q_proj.bias.data)
        custom_layer.cross_attn.W_K.weight.data.copy_(bart_layer.encoder_attn.k_proj.weight.data)
        custom_layer.cross_attn.W_K.bias.data.copy_(bart_layer.encoder_attn.k_proj.bias.data)
        custom_layer.cross_attn.W_V.weight.data.copy_(bart_layer.encoder_attn.v_proj.weight.data)
        custom_layer.cross_attn.W_V.bias.data.copy_(bart_layer.encoder_attn.v_proj.bias.data)
        custom_layer.cross_attn.W_O.weight.data.copy_(bart_layer.encoder_attn.out_proj.weight.data)
        custom_layer.cross_attn.W_O.bias.data.copy_(bart_layer.encoder_attn.out_proj.bias.data)
        
        # Layer norms
        custom_layer.norm1.weight.data.copy_(bart_layer.self_attn_layer_norm.weight.data)
        custom_layer.norm1.bias.data.copy_(bart_layer.self_attn_layer_norm.bias.data)
        custom_layer.norm2.weight.data.copy_(bart_layer.encoder_attn_layer_norm.weight.data)
        custom_layer.norm2.bias.data.copy_(bart_layer.encoder_attn_layer_norm.bias.data)
        custom_layer.norm3.weight.data.copy_(bart_layer.final_layer_norm.weight.data)
        custom_layer.norm3.bias.data.copy_(bart_layer.final_layer_norm.bias.data)
        
        # FFN - use linear1/linear2 (not fc1/fc2)
        custom_layer.ffn.linear1.weight.data.copy_(bart_layer.fc1.weight.data)
        custom_layer.ffn.linear1.bias.data.copy_(bart_layer.fc1.bias.data)
        custom_layer.ffn.linear2.weight.data.copy_(bart_layer.fc2.weight.data)
        custom_layer.ffn.linear2.bias.data.copy_(bart_layer.fc2.bias.data)
    
    # BART has layernorm_embedding at the input, we have final_norm at output
    if hasattr(bart.decoder, 'layernorm_embedding'):
        decoder.final_norm.weight.data.copy_(bart.decoder.layernorm_embedding.weight.data)
        decoder.final_norm.bias.data.copy_(bart.decoder.layernorm_embedding.bias.data)
    
    print("Pretrained weights loaded successfully!")


def build_multitask_model(
    tokenizer: Tokenizer,
    *,
    num_emotions: int,
    num_topics: int,
    config: ModelConfig | None = None,
    load_pretrained: bool | None = None,
) -> MultiTaskModel:
    """Construct the multitask transformer with heads for the three tasks.
    
    Args:
        tokenizer: Tokenizer for vocabulary size and pad token
        num_emotions: Number of emotion classes
        num_topics: Number of topic classes
        config: Model architecture configuration
        load_pretrained: Override config.use_pretrained (for inference to skip loading)
    """

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
    
    # Load pretrained weights if requested (but allow override for inference)
    should_load = cfg.use_pretrained if load_pretrained is None else load_pretrained
    if should_load:
        _load_pretrained_weights(encoder, decoder, cfg.pretrained_model_name)

    # NOTE: Weight tying disabled because the current checkpoint was trained without it
    # For NEW training runs, uncomment this line to enable proper weight tying:
    # decoder.output_projection.weight = decoder.embedding.weight
    
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
