"""
Pydantic schemas for LexiMind API.

Defines request and response models for the REST API.

Author: Oliver Perrin
Date: December 2025
"""

from pydantic import BaseModel


class SummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str
    emotion_labels: list[str]
    emotion_scores: list[float]
    topic: str
    topic_confidence: float
