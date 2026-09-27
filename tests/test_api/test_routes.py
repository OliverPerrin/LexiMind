"""API integration tests for the inference endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_pipeline
from src.inference.pipeline import EmotionPrediction, TopicPrediction


class StubPipeline:
    def batch_predict(self, texts):  # pragma: no cover - simple stub
        return {
            "summaries": [f"summary:{text}" for text in texts],
            "emotion": [EmotionPrediction(labels=["joy"], scores=[0.9]) for _ in texts],
            "topic": [TopicPrediction(label="news", confidence=0.8) for _ in texts],
        }


def test_summarize_route_returns_pipeline_outputs() -> None:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: StubPipeline()
    client = TestClient(app)

    try:
        response = client.post("/summarize", json={"text": "hello world"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"] == "summary:hello world"
        assert payload["emotion_labels"] == ["joy"]
        assert payload["topic"] == "news"
        assert payload["topic_confidence"] == 0.8
    finally:
        app.dependency_overrides.clear()


def test_inference_failure_does_not_expose_internal_paths_or_exception_details() -> None:
    class FailedPipeline:
        def batch_predict(self, texts):
            raise RuntimeError("private checkpoint path: /secret/weights.pt")

    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: FailedPipeline()
    with TestClient(app) as client:
        response = client.post("/summarize", json={"text": "hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Text analysis temporarily unavailable"}
