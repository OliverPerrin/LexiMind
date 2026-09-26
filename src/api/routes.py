"""
API routes for LexiMind.

Defines REST endpoints for text analysis including summarization,
emotion detection, and topic classification.

Author: Oliver Perrin
Date: December 2025
"""

import logging
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, status

from ..inference import EmotionPrediction, InferencePipeline, TopicPrediction
from .dependencies import get_pipeline
from .schemas import SummaryRequest, SummaryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=SummaryResponse)
def summarize(
    payload: SummaryRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),  # noqa: B008
) -> SummaryResponse:
    try:
        outputs = pipeline.batch_predict([payload.text])
    except Exception as exc:  # noqa: BLE001 - retain details in server logs only
        logger.exception("Text analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text analysis temporarily unavailable",
        ) from exc
    summaries = cast(list[str], outputs["summaries"])
    emotion_preds = cast(list[EmotionPrediction], outputs["emotion"])
    topic_preds = cast(list[TopicPrediction], outputs["topic"])

    emotion = emotion_preds[0]
    topic = topic_preds[0]
    return SummaryResponse(
        summary=summaries[0],
        emotion_labels=emotion.labels,
        emotion_scores=emotion.scores,
        topic=topic.label,
        topic_confidence=topic.confidence,
    )
