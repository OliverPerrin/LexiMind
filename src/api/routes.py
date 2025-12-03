"""API routes."""

from typing import cast

from fastapi import APIRouter, Depends, HTTPException, status

from ..inference import EmotionPrediction, InferencePipeline, TopicPrediction
from .dependencies import get_pipeline
from .schemas import SummaryRequest, SummaryResponse

router = APIRouter()


@router.post("/summarize", response_model=SummaryResponse)
def summarize(
    payload: SummaryRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),  # noqa: B008
) -> SummaryResponse:
    try:
        outputs = pipeline.batch_predict([payload.text])
    except Exception as exc:  # noqa: BLE001 - surface inference error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
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
