import json
from types import SimpleNamespace

from scripts.build_discovery_dataset import load_literary, run_inference


class FakePipeline:
    def summarize(self, texts):
        return ["Model summary"]

    def predict_topics(self, texts):
        return [SimpleNamespace(label="Fiction", confidence=0.8)]

    def predict_emotions(self, texts, threshold):
        return [SimpleNamespace(labels=["joy", "fear"], scores=[0.99, 0.98])]


def test_even_high_social_emotion_scores_are_not_book_moods():
    sample = {
        "id": "book",
        "title": "A book",
        "text": "Some text",
        "source_type": "literary",
        "dataset": "gutenberg",
    }
    first = run_inference(FakePipeline(), [sample])
    second = run_inference(FakePipeline(), [sample])
    assert first == second
    assert first[0]["emotion"] == "Unknown"
    assert first[0]["emotion_confidence"] == 0.0
    assert first[0]["emotion_status"] == "unvalidated_domain_abstention"
    assert first[0]["raw_emotion_scores"] == {"joy": 0.99, "fear": 0.98}


def test_legacy_pairs_without_identity_evidence_are_excluded(tmp_path):
    folder = tmp_path / "summarization"
    folder.mkdir()
    old = {"title": "College Girl", "type": "literary", "source": "X" * 500, "summary": "Y" * 100}
    verified = {
        **old,
        "work_id": "title-author:known",
        "identity_status": "title_and_author_matched",
        "description_source": "https://example.org/source",
    }
    (folder / "train.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(verified) + "\n")
    result = load_literary(tmp_path)
    assert len(result) == 1
    assert result[0]["id"] == verified["work_id"]
    assert result[0]["description_source"] == verified["description_source"]
