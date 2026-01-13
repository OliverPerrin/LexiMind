"""
FastAPI dependency providers for LexiMind.

Manages lazy initialization and caching of the inference pipeline.

Author: Oliver Perrin
Date: December 2025
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

from ..inference.factory import create_inference_pipeline
from ..inference.pipeline import InferencePipeline


@lru_cache(maxsize=1)
def get_pipeline() -> InferencePipeline:
    """Lazily construct and cache the inference pipeline for the API."""

    checkpoint = Path("checkpoints/best.pt")
    labels = Path("artifacts/labels.json")
    model_config = Path("configs/model/base.yaml")

    try:
        pipeline, _ = create_inference_pipeline(
            checkpoint_path=checkpoint,
            labels_path=labels,
            model_config_path=model_config,
        )
    except FileNotFoundError as exc:
        logger.exception("Pipeline initialization failed: missing artifact")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface initialization issues to the caller
        logger.exception("Pipeline initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",
        ) from exc
    return pipeline
