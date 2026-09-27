import json
from pathlib import Path

from src.catalog.demo import format_book_card, has_validated_tone, load_demo_items

ROOT = Path(__file__).resolve().parents[2]


def test_demo_uses_canonical_books_and_abstains_on_historical_paper_tones(tmp_path):
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {
                    "id": "legacy-book",
                    "title": "College Girl",
                    "source_type": "literary",
                    "reference_summary": "Wrong book",
                    "emotion": "joy",
                },
                {
                    "id": "paper",
                    "title": "Paper",
                    "source_type": "academic",
                    "emotion": "joy",
                    "emotion_confidence": 0.9,
                },
            ]
        )
    )
    items, notices = load_demo_items(ROOT / "web/data/books.json", [legacy])
    assert not notices
    assert not any(item["id"] == "legacy-book" for item in items)
    paper = next(item for item in items if item["id"] == "paper")
    assert paper["emotion"] == "Unknown"
    assert not has_validated_tone(paper)
    book = next(item for item in items if item["source_type"] == "literary")
    card = format_book_card(book)
    assert book["source_url"] in card
    assert book["authors"][0] in card
    assert "Tone:" not in card
    assert "Open Library" in card


def test_missing_catalogue_never_falls_back_to_legacy_literary_pairs(tmp_path):
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"source_type": "literary", "title": "Legacy book"}) + "\n")
    items, notices = load_demo_items(tmp_path / "missing.json", [legacy])
    assert items == []
    assert "unavailable" in notices[0]
    assert "identity was not verified" in notices[0]


def test_validated_tone_requires_explicit_status_and_source():
    assert not has_validated_tone({"emotion": "joy", "emotion_confidence": 0.99})
    assert not has_validated_tone({"emotion": "joy", "emotion_status": "validated_editorial"})
    assert has_validated_tone(
        {
            "emotion": "joy",
            "emotion_status": "validated_editorial",
            "emotion_source": "https://example.org/editorial-labels",
        }
    )
    assert not has_validated_tone(
        {
            "emotion": ["joy"],
            "emotion_status": "validated_editorial",
            "emotion_source": "https://example.org/source",
        }
    )
    assert not has_validated_tone(
        {
            "emotion": "joy",
            "emotion_status": "validated_editorial",
            "emotion_source": "javascript:alert(1)",
        }
    )
    assert not has_validated_tone(
        {
            "emotion": "joy",
            "emotion_status": ["validated_editorial"],
            "emotion_source": "https://example.org/source",
        }
    )
