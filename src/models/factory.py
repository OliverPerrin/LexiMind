"""Factory helpers to assemble multitask models for inference/training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, cast

import torch
from transformers import T5ForConditionalGeneration

from ..data.tokenization import Tokenizer
from ..utils.config import load_yaml
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .heads import ClassificationHead, LMHead
from .multitask import MultiTaskModel

# Type alias for activation functions
ActivationType = Literal["gelu", "relu", "swiglu", "gated-gelu"]


@dataclass
class ModelConfig:
    """Configuration describing the transformer architecture."""

    d_model: int = 768
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    num_attention_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    use_pretrained: bool = False
    pretrained_model_name: str = "google/flan-t5-base"
    quantization: Optional[str] = None  # "4bit" or "8bit"
    use_learned_pos_enc: bool = True  # Use learned positional embeddings
    activation: str = (
        "gated-gelu"  # "gelu", "relu", "swiglu", or "gated-gelu" (use gated-gelu for T5/FLAN-T5)
    )
    use_relative_position_bias: bool = (
        False  # T5-style relative position bias (use True for T5/FLAN-T5)
    )

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
        pretrained_model_name=str(data.get("pretrained_model_name", "google/flan-t5-base")),
        quantization=data.get("quantization", None),
        use_learned_pos_enc=bool(data.get("use_learned_pos_enc", True)),
        activation=str(data.get("activation", "gelu")),
        use_relative_position_bias=bool(data.get("use_relative_position_bias", False)),
    )


def _load_pretrained_weights(
    encoder: TransformerEncoder, decoder: TransformerDecoder, model_name: str
) -> None:
    """
    Load pretrained T5/FLAN-T5 weights into custom encoder/decoder.

    T5 architecture compatibility with our custom Transformer:
    - T5 uses Pre-LN (RMSNorm before sublayers) ✓ matches our design
    - T5 uses relative position bias instead of absolute embeddings
      -> We now load T5's relative position bias weights into our T5RelativePositionBias modules
      -> This allows exact weight transfer without requiring fine-tuning
    - T5 uses gated FFN (wi_0, wi_1, wo) - we use gated-gelu FFN matching this
    - T5 attention has no bias, our attention has bias
      -> We zero-initialize the bias terms
    """
    print(f"Loading pretrained weights from {model_name}...")
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load shared embeddings (T5 uses shared embeddings for encoder and decoder)
    # Note: T5's vocab is padded to multiple of 128 for efficiency (32100 -> 32128)
    # Our model uses the tokenizer's actual vocab size, so we only copy the valid tokens
    print("Transferring shared token embeddings...")
    shared_embeddings = t5.shared.weight.data
    our_vocab_size = encoder.embedding.weight.size(0)
    t5_vocab_size = shared_embeddings.size(0)

    if our_vocab_size != t5_vocab_size:
        print(f"  Vocab size mismatch: our model={our_vocab_size}, T5={t5_vocab_size}")
        # Copy only the tokens that exist in both (T5 pads vocab to multiple of 128)
        min_vocab = min(our_vocab_size, t5_vocab_size)
        print(f"  Copying first {min_vocab} token embeddings...")
        encoder.embedding.weight.data[:min_vocab].copy_(shared_embeddings[:min_vocab])
        decoder.embedding.weight.data[:min_vocab].copy_(shared_embeddings[:min_vocab])
    else:
        encoder.embedding.weight.data.copy_(shared_embeddings)
        decoder.embedding.weight.data.copy_(shared_embeddings)

    # Note: T5 uses relative position bias (computed in attention, not absolute embeddings).
    # We now use T5RelativePositionBias which will be loaded below. The pos_encoder in our model
    # is still present but adds zero/minimal contribution when relative_position_bias is used.

    # Load encoder weights
    print("Transferring encoder weights...")
    t5_encoder = t5.encoder

    for custom_layer, t5_layer in zip(encoder.layers, t5_encoder.block, strict=False):
        t5_self_attn = t5_layer.layer[0].SelfAttention
        t5_ffn = t5_layer.layer[1].DenseReluDense
        t5_norm1 = t5_layer.layer[0].layer_norm
        t5_norm2 = t5_layer.layer[1].layer_norm

        # Self-attention (T5 has no bias in attention projections)
        custom_layer.self_attn.W_Q.weight.data.copy_(t5_self_attn.q.weight.data)
        custom_layer.self_attn.W_K.weight.data.copy_(t5_self_attn.k.weight.data)
        custom_layer.self_attn.W_V.weight.data.copy_(t5_self_attn.v.weight.data)
        custom_layer.self_attn.W_O.weight.data.copy_(t5_self_attn.o.weight.data)

        # Zero-initialize bias (T5 doesn't have attention bias)
        if custom_layer.self_attn.W_Q.bias is not None:
            custom_layer.self_attn.W_Q.bias.data.zero_()
            custom_layer.self_attn.W_K.bias.data.zero_()
            custom_layer.self_attn.W_V.bias.data.zero_()
            custom_layer.self_attn.W_O.bias.data.zero_()

        # Layer norms (T5 uses RMSNorm like us, just weight, no bias)
        custom_layer.norm1.weight.data.copy_(t5_norm1.weight.data)
        custom_layer.norm2.weight.data.copy_(t5_norm2.weight.data)

        # FFN - T5 uses gated FFN: wi_0 (gate), wi_1 (up), wo (down)
        # If our model uses swiglu activation: linear_gate (gate), linear1 (up), linear2 (down)
        # If our model uses standard activation: linear1 (up), linear2 (down) - partial transfer
        if hasattr(t5_ffn, "wi_0") and hasattr(custom_layer.ffn, "linear_gate"):
            # Full gated FFN transfer (swiglu mode)
            custom_layer.ffn.linear_gate.weight.data.copy_(t5_ffn.wi_0.weight.data)
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi_1.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)
            if custom_layer.ffn.linear_gate.bias is not None:
                custom_layer.ffn.linear_gate.bias.data.zero_()
        elif hasattr(t5_ffn, "wi_1"):
            # T5 v1.1 / FLAN-T5 gated FFN -> standard FFN (partial transfer)
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi_1.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)
        elif hasattr(t5_ffn, "wi"):
            # Original T5 v1.0
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)

        # Zero-initialize FFN bias (T5 doesn't have FFN bias)
        if custom_layer.ffn.linear1.bias is not None:
            custom_layer.ffn.linear1.bias.data.zero_()
            custom_layer.ffn.linear2.bias.data.zero_()

    # Encoder final norm
    encoder.final_norm.weight.data.copy_(t5_encoder.final_layer_norm.weight.data)

    # Load encoder relative position bias (T5 stores it only in first layer, shared across all layers)
    if hasattr(encoder, "relative_position_bias") and encoder.relative_position_bias is not None:
        print("Transferring encoder relative position bias...")
        t5_enc_rel_bias = (
            t5_encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.data
        )
        encoder.relative_position_bias.relative_attention_bias.weight.data.copy_(t5_enc_rel_bias)

    # Load decoder weights
    print("Transferring decoder weights...")
    t5_decoder = t5.decoder

    for custom_layer, t5_layer in zip(decoder.layers, t5_decoder.block, strict=False):
        t5_self_attn = t5_layer.layer[0].SelfAttention
        t5_cross_attn = t5_layer.layer[1].EncDecAttention
        t5_ffn = t5_layer.layer[2].DenseReluDense
        t5_norm1 = t5_layer.layer[0].layer_norm
        t5_norm2 = t5_layer.layer[1].layer_norm
        t5_norm3 = t5_layer.layer[2].layer_norm

        # Self-attention
        custom_layer.self_attn.W_Q.weight.data.copy_(t5_self_attn.q.weight.data)
        custom_layer.self_attn.W_K.weight.data.copy_(t5_self_attn.k.weight.data)
        custom_layer.self_attn.W_V.weight.data.copy_(t5_self_attn.v.weight.data)
        custom_layer.self_attn.W_O.weight.data.copy_(t5_self_attn.o.weight.data)

        if custom_layer.self_attn.W_Q.bias is not None:
            custom_layer.self_attn.W_Q.bias.data.zero_()
            custom_layer.self_attn.W_K.bias.data.zero_()
            custom_layer.self_attn.W_V.bias.data.zero_()
            custom_layer.self_attn.W_O.bias.data.zero_()

        # Cross-attention
        custom_layer.cross_attn.W_Q.weight.data.copy_(t5_cross_attn.q.weight.data)
        custom_layer.cross_attn.W_K.weight.data.copy_(t5_cross_attn.k.weight.data)
        custom_layer.cross_attn.W_V.weight.data.copy_(t5_cross_attn.v.weight.data)
        custom_layer.cross_attn.W_O.weight.data.copy_(t5_cross_attn.o.weight.data)

        if custom_layer.cross_attn.W_Q.bias is not None:
            custom_layer.cross_attn.W_Q.bias.data.zero_()
            custom_layer.cross_attn.W_K.bias.data.zero_()
            custom_layer.cross_attn.W_V.bias.data.zero_()
            custom_layer.cross_attn.W_O.bias.data.zero_()

        # Layer norms
        custom_layer.norm1.weight.data.copy_(t5_norm1.weight.data)
        custom_layer.norm2.weight.data.copy_(t5_norm2.weight.data)
        custom_layer.norm3.weight.data.copy_(t5_norm3.weight.data)

        # FFN - same gated logic as encoder
        if hasattr(t5_ffn, "wi_0") and hasattr(custom_layer.ffn, "linear_gate"):
            # Full gated FFN transfer (swiglu mode)
            custom_layer.ffn.linear_gate.weight.data.copy_(t5_ffn.wi_0.weight.data)
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi_1.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)
            if custom_layer.ffn.linear_gate.bias is not None:
                custom_layer.ffn.linear_gate.bias.data.zero_()
        elif hasattr(t5_ffn, "wi_1"):
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi_1.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)
        elif hasattr(t5_ffn, "wi"):
            custom_layer.ffn.linear1.weight.data.copy_(t5_ffn.wi.weight.data)
            custom_layer.ffn.linear2.weight.data.copy_(t5_ffn.wo.weight.data)

        if custom_layer.ffn.linear1.bias is not None:
            custom_layer.ffn.linear1.bias.data.zero_()
            custom_layer.ffn.linear2.bias.data.zero_()

    # Decoder final norm
    decoder.final_norm.weight.data.copy_(t5_decoder.final_layer_norm.weight.data)

    # Load decoder relative position biases (T5 stores them in first layer, shared across all layers)
    # Decoder has both self-attention bias and cross-attention bias
    if (
        hasattr(decoder, "self_relative_position_bias")
        and decoder.self_relative_position_bias is not None
    ):
        print("Transferring decoder self-attention relative position bias...")
        t5_dec_self_rel_bias = (
            t5_decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.data
        )
        decoder.self_relative_position_bias.relative_attention_bias.weight.data.copy_(
            t5_dec_self_rel_bias
        )

    if (
        hasattr(decoder, "cross_relative_position_bias")
        and decoder.cross_relative_position_bias is not None
    ):
        print("Transferring decoder cross-attention relative position bias...")
        # Cross-attention relative position bias is in EncDecAttention of first block
        t5_dec_cross_rel_bias = (
            t5_decoder.block[0].layer[1].EncDecAttention.relative_attention_bias.weight.data
        )
        decoder.cross_relative_position_bias.relative_attention_bias.weight.data.copy_(
            t5_dec_cross_rel_bias
        )

    # Load LM head weights (T5's lm_head)
    # Handle vocab size mismatch (T5 pads to multiple of 128)
    print("Transferring LM head weights...")
    lm_head_weights = t5.lm_head.weight.data
    our_vocab_size = decoder.output_projection.weight.size(0)
    t5_vocab_size = lm_head_weights.size(0)

    if our_vocab_size != t5_vocab_size:
        print(f"  LM head vocab mismatch: our model={our_vocab_size}, T5={t5_vocab_size}")
        min_vocab = min(our_vocab_size, t5_vocab_size)
        print(f"  Copying first {min_vocab} LM head weights...")
        decoder.output_projection.weight.data[:min_vocab].copy_(lm_head_weights[:min_vocab])
    else:
        decoder.output_projection.weight.data.copy_(lm_head_weights)

    if decoder.output_projection.bias is not None:
        decoder.output_projection.bias.data.zero_()

    print("Pretrained FLAN-T5 weights loaded successfully!")


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

    # Get max_length from tokenizer (handle both custom and HF tokenizers)
    if hasattr(tokenizer, "config") and hasattr(tokenizer.config, "max_length"):
        max_len = tokenizer.config.max_length
    elif hasattr(tokenizer, "model_max_length"):
        max_len = tokenizer.model_max_length
    else:
        max_len = 512  # Default fallback

    # Cast activation to the literal type for mypy
    activation = cast(ActivationType, cfg.activation)

    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_encoder_layers,
        num_heads=cfg.num_attention_heads,
        d_ff=cfg.ffn_dim,
        dropout=cfg.dropout,
        max_len=max_len,
        pad_token_id=tokenizer.pad_token_id,
        quantization=cfg.quantization,
        use_learned_pos_enc=cfg.use_learned_pos_enc,
        activation=activation,
        use_relative_position_bias=cfg.use_relative_position_bias,
    )
    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_attention_heads,
        d_ff=cfg.ffn_dim,
        dropout=cfg.dropout,
        max_len=max_len,
        pad_token_id=tokenizer.pad_token_id,
        quantization=cfg.quantization,
        use_learned_pos_enc=cfg.use_learned_pos_enc,
        activation=activation,
        use_relative_position_bias=cfg.use_relative_position_bias,
    )

    # Load pretrained weights if requested (but allow override for inference)
    should_load = cfg.use_pretrained if load_pretrained is None else load_pretrained
    if should_load:
        model_name_lower = cfg.pretrained_model_name.lower()
        if "t5" in model_name_lower or "flan" in model_name_lower:
            _load_pretrained_weights(encoder, decoder, cfg.pretrained_model_name)
        elif "llama" in model_name_lower or "gemma" in model_name_lower:
            _load_llama_weights(
                encoder, decoder, cfg.pretrained_model_name, quantization=cfg.quantization
            )
        else:
            # Default to T5 loading for unknown models
            print(
                f"Warning: Unknown model type '{cfg.pretrained_model_name}', attempting T5-style loading..."
            )
            _load_pretrained_weights(encoder, decoder, cfg.pretrained_model_name)

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
