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
    quantization: Optional[str] = None  # "4bit" or "8bit"

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
        if self.quantization not in [None, "4bit", "8bit"]:
            raise ValueError(
                f"quantization must be None, '4bit', or '8bit', got {self.quantization}"
            )


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
        quantization=data.get("quantization", None),
    )


def _load_pretrained_weights(
    encoder: TransformerEncoder, decoder: TransformerDecoder, model_name: str
) -> None:
    """Load pretrained BART weights into custom encoder/decoder."""
    print(f"Loading pretrained weights from {model_name}...")
    bart = BartModel.from_pretrained(model_name)

    # Load encoder weights
    print("Transferring encoder weights...")
    encoder.embedding.weight.data.copy_(bart.encoder.embed_tokens.weight.data)
    # Skip positional encoding - BART uses learned positions, I use sinusoidal
    # implementation will work fine with sinusoidal encodings

    for _i, (custom_layer, bart_layer) in enumerate(zip(encoder.layers, bart.encoder.layers)):
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
    if hasattr(bart.encoder, "layernorm_embedding"):
        encoder.final_norm.weight.data.copy_(bart.encoder.layernorm_embedding.weight.data)
        encoder.final_norm.bias.data.copy_(bart.encoder.layernorm_embedding.bias.data)

    # Load decoder weights
    print("Transferring decoder weights...")
    decoder.embedding.weight.data.copy_(bart.decoder.embed_tokens.weight.data)
    # Skip positional encoding - BART uses learned positions, we use sinusoidal

    for _i, (custom_layer, bart_layer) in enumerate(zip(decoder.layers, bart.decoder.layers)):
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
    if hasattr(bart.decoder, "layernorm_embedding"):
        decoder.final_norm.weight.data.copy_(bart.decoder.layernorm_embedding.weight.data)
        decoder.final_norm.bias.data.copy_(bart.decoder.layernorm_embedding.bias.data)

    print("Pretrained weights loaded successfully!")


def _load_llama_weights(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    model_name: str,
    quantization: Optional[str] = None,
) -> None:
    """
    Load pretrained Llama/Gemma weights into custom encoder/decoder.

    Demonstrates flexibility by mapping Llama's specific architecture
    (RMSNorm, SwiGLU, RoPE) to our custom implementation.
    """
    print(f"Loading pretrained weights from {model_name}...")
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Use device_map='cpu' to avoid OOM during loading, unless quantized (needs GPU)
        device_map = "auto" if quantization else "cpu"

        llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if not quantization else None,
            quantization_config=quantization_config,
            device_map=device_map,
        )
    except Exception as e:
        print(f"Could not load Llama model: {e}")
        return

    # Llama is decoder-only, so we primarily map to our decoder.
    # However, we can also initialize our encoder with the same weights
    # to create a symmetric starting point (common in seq2seq from decoder-only).

    print("Transferring Llama weights to Encoder & Decoder...")

    # 1. Embeddings
    # Llama: model.embed_tokens
    if hasattr(llama.model.embed_tokens, "weight"):
        encoder.embedding.weight.data.copy_(llama.model.embed_tokens.weight.data)
        decoder.embedding.weight.data.copy_(llama.model.embed_tokens.weight.data)

    # 2. Layers
    # Llama layers: model.layers
    # Our layers: encoder.layers, decoder.layers

    # We'll map the first N layers of Llama to our Encoder and Decoder
    num_layers = min(len(encoder.layers), len(llama.model.layers))

    for i in range(num_layers):
        llama_layer = llama.model.layers[i]
        enc_layer = encoder.layers[i]
        dec_layer = decoder.layers[i]

        # --- Self-Attention ---
        # Llama: q_proj, k_proj, v_proj, o_proj
        # Ours: W_Q, W_K, W_V, W_O

        # Encoder Self-Attn
        enc_layer.self_attn.W_Q.weight.data.copy_(llama_layer.self_attn.q_proj.weight.data)
        enc_layer.self_attn.W_K.weight.data.copy_(llama_layer.self_attn.k_proj.weight.data)
        enc_layer.self_attn.W_V.weight.data.copy_(llama_layer.self_attn.v_proj.weight.data)
        enc_layer.self_attn.W_O.weight.data.copy_(llama_layer.self_attn.o_proj.weight.data)

        # Decoder Self-Attn
        dec_layer.self_attn.W_Q.weight.data.copy_(llama_layer.self_attn.q_proj.weight.data)
        dec_layer.self_attn.W_K.weight.data.copy_(llama_layer.self_attn.k_proj.weight.data)
        dec_layer.self_attn.W_V.weight.data.copy_(llama_layer.self_attn.v_proj.weight.data)
        dec_layer.self_attn.W_O.weight.data.copy_(llama_layer.self_attn.o_proj.weight.data)

        # Note: Llama uses RoPE (Rotary Embeddings), so there are no absolute position embeddings to load.
        # Our model should have use_rope=True for this to work best.

        # --- Feed Forward (SwiGLU) ---
        # Llama: gate_proj, up_proj, down_proj
        # Ours (if activation='swiglu'): linear_gate, linear1 (up), linear2 (down)

        if hasattr(enc_layer.ffn, "linear_gate") and hasattr(llama_layer.mlp, "gate_proj"):
            # Encoder FFN
            enc_layer.ffn.linear_gate.weight.data.copy_(llama_layer.mlp.gate_proj.weight.data)
            enc_layer.ffn.linear1.weight.data.copy_(llama_layer.mlp.up_proj.weight.data)
            enc_layer.ffn.linear2.weight.data.copy_(llama_layer.mlp.down_proj.weight.data)

            # Decoder FFN
            dec_layer.ffn.linear_gate.weight.data.copy_(llama_layer.mlp.gate_proj.weight.data)
            dec_layer.ffn.linear1.weight.data.copy_(llama_layer.mlp.up_proj.weight.data)
            dec_layer.ffn.linear2.weight.data.copy_(llama_layer.mlp.down_proj.weight.data)
        else:
            # Fallback for standard FFN if Llama weights are standard (e.g. older models)
            # or if our model is not configured for SwiGLU
            pass

        # --- Normalization (RMSNorm) ---
        # Llama: input_layernorm, post_attention_layernorm
        # Ours: norm1, norm2 (Encoder) / norm1, norm2, norm3 (Decoder)
        # Note: Llama uses RMSNorm, we use LayerNorm. Weights are compatible (scale), but bias is missing in RMSNorm.

        # Encoder Norms
        enc_layer.norm1.weight.data.copy_(llama_layer.input_layernorm.weight.data)
        enc_layer.norm2.weight.data.copy_(llama_layer.post_attention_layernorm.weight.data)

        # Decoder Norms
        dec_layer.norm1.weight.data.copy_(llama_layer.input_layernorm.weight.data)
        # norm2 is cross-attn, we skip or reuse
        dec_layer.norm3.weight.data.copy_(llama_layer.post_attention_layernorm.weight.data)

    # 3. Final Norm
    # Llama: model.norm
    if hasattr(llama.model, "norm"):
        encoder.final_norm.weight.data.copy_(llama.model.norm.weight.data)
        decoder.final_norm.weight.data.copy_(llama.model.norm.weight.data)

    print("Llama weights loaded successfully!")


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
        quantization=cfg.quantization,
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
        quantization=cfg.quantization,
    )

    # Load pretrained weights if requested (but allow override for inference)
    should_load = cfg.use_pretrained if load_pretrained is None else load_pretrained
    if should_load:
        if (
            "llama" in cfg.pretrained_model_name.lower()
            or "gemma" in cfg.pretrained_model_name.lower()
        ):
            _load_llama_weights(
                encoder, decoder, cfg.pretrained_model_name, quantization=cfg.quantization
            )
        else:
            _load_pretrained_weights(encoder, decoder, cfg.pretrained_model_name)

    # NOTE: Weight tying disabled because the current checkpoint was trained without it
    # For NEW training runs, uncomment this line to enable proper weight tying:
    # decoder.output_projection.weight = decoder.embedding.weight

    model = MultiTaskModel(encoder=encoder, decoder=decoder, decoder_outputs_logits=True)
    model.add_head(
        "summarization",
        LMHead(
            d_model=cfg.d_model, vocab_size=tokenizer.vocab_size, tie_embedding=decoder.embedding
        ),
    )
    model.add_head(
        "emotion",
        ClassificationHead(
            d_model=cfg.d_model, num_labels=num_emotions, pooler="mean", dropout=cfg.dropout
        ),
    )
    model.add_head(
        "topic",
        ClassificationHead(
            d_model=cfg.d_model, num_labels=num_topics, pooler="mean", dropout=cfg.dropout
        ),
    )
    return model
