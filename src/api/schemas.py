"""API schemas."""

from pydantic import BaseModel


class SummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str
    emotion_labels: list[str]
    emotion_scores: list[float]
    topic: str
    topic_confidence: float
