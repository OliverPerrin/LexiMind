"""Minimal inference helpers that rely on the custom transformer stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...data.preprocessing import TextPreprocessor, TransformerTokenizer
from ...models.multitask import MultiTaskModel


def _load_tokenizer(tokenizer_path: Path) -> TransformerTokenizer:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer file '{tokenizer_path}' not found")
    return TransformerTokenizer.load(tokenizer_path)


def load_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load MultiTaskModel together with the tokenizer-driven preprocessor."""

    device = torch.device(config.get("device", "cpu"))
    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path is None:
        raise ValueError("'tokenizer_path' missing in config")

    tokenizer = _load_tokenizer(Path(tokenizer_path))
    preprocessor = TextPreprocessor(
        max_length=int(config.get("max_length", 512)),
        tokenizer=tokenizer,
        min_freq=int(config.get("min_freq", 1)),
        lowercase=bool(config.get("lowercase", True)),
    )

    encoder_kwargs = dict(config.get("encoder", {}))
    decoder_kwargs = dict(config.get("decoder", {}))

    encoder = preprocessor.build_encoder(**encoder_kwargs)
    decoder = preprocessor.build_decoder(**decoder_kwargs)
    model = MultiTaskModel(encoder=encoder, decoder=decoder)

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.to(device)

    return {
        "loaded": True,
        "device": device,
        "mt": model,
        "preprocessor": preprocessor,
    }


def summarize_text(
    text: str,
    compression: float = 0.25,
    collect_attn: bool = False,
    models: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
    if models is None or not models.get("loaded"):
        raise RuntimeError("Models must be loaded via load_models before summarize_text is called")

    model: MultiTaskModel = models["mt"]
    preprocessor: TextPreprocessor = models["preprocessor"]
    device: torch.device = models["device"]

    batch = preprocessor.batch_encode([text])
    tokenizer = preprocessor.tokenizer
    encoder = model.encoder
    decoder = model.decoder
    if tokenizer is None or encoder is None or decoder is None:
        raise RuntimeError("Encoder, decoder, and tokenizer must be configured before summarization")
    input_ids = batch.input_ids.to(device)
    memory = encoder(input_ids)
    src_len = batch.lengths[0]
    max_tgt = max(4, int(src_len * compression))
    generated = decoder.greedy_decode(
        memory,
        max_len=min(preprocessor.max_length, max_tgt),
        start_token_id=tokenizer.bos_id,
        end_token_id=tokenizer.eos_id,
    )
    summary = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return summary.strip(), None if not collect_attn else {}


def classify_emotion(text: str, models: Optional[Dict[str, Any]] = None) -> Tuple[List[float], List[str]]:
    if models is None or not models.get("loaded"):
        raise RuntimeError("Models must be loaded via load_models before classify_emotion is called")

    model: MultiTaskModel = models["mt"]
    preprocessor: TextPreprocessor = models["preprocessor"]
    device: torch.device = models["device"]

    batch = preprocessor.batch_encode([text])
    input_ids = batch.input_ids.to(device)
    result = model.forward("emotion", {"input_ids": input_ids})
    logits = result[1] if isinstance(result, tuple) else result
    scores = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()
    labels = models.get("emotion_labels") or [
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
    ]
    return scores, labels[: len(scores)]


def topic_for_text(text: str, models: Optional[Dict[str, Any]] = None) -> Tuple[int, List[str]]:
    if models is None or not models.get("loaded"):
        raise RuntimeError("Models must be loaded via load_models before topic_for_text is called")

    model: MultiTaskModel = models["mt"]
    preprocessor: TextPreprocessor = models["preprocessor"]
    device: torch.device = models["device"]

    batch = preprocessor.batch_encode([text])
    input_ids = batch.input_ids.to(device)
    encoder = model.encoder
    if encoder is None:
        raise RuntimeError("Encoder must be configured before topic_for_text is called")
    memory = encoder(input_ids)
    embedding = memory.mean(dim=1).detach().cpu()
    _ = embedding  # placeholder for downstream clustering hook
    return 0, ["topic_stub"]