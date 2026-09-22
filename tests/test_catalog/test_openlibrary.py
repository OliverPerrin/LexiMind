import json
from pathlib import Path

import pytest

from src.catalog.openlibrary import (
    OpenLibraryClient,
    apply_review,
    digest,
    genres_from_subjects,
    make_book,
    validate_book,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture_records():
    search = {
        "key": "/works/OL1W",
        "author_name": ["Jane Smith"],
        "author_key": ["OL2A"],
        "first_publish_year": 1999,
    }
    work = {
        "key": "/works/OL1W",
        "title": "A book",
        "authors": [{"author": {"key": "/authors/OL2A"}}],
        "description": {"value": "The source description."},
        "subjects": ["Science fiction", "Fiction"],
        "covers": [123],
    }
    return search, {
        "data": work,
        "retrievedAt": "2026-09-22T00:00:00+00:00",
        "sha256": digest(work),
    }


def test_book_preserves_source_and_does_not_invent_moods():
    search, response = fixture_records()
    book = make_book(search, response)
    assert book["description"] == response["data"]["description"]["value"]
    assert book["descriptionSource"] == "https://openlibrary.org/works/OL1W"
    assert book["sourceContentHash"] == response["sha256"]
    assert book["genres"] == ["Science fiction"]
    assert book["moods"] == []
    assert book["identifiers"]["isbns"] == []


def test_missing_description_is_missing_not_generated():
    search, response = fixture_records()
    del response["data"]["description"]
    book = make_book(search, response)
    assert book["description"] == ""
    assert book["descriptionSource"] is None


def test_identity_disagreement_is_rejected():
    search, response = fixture_records()
    with pytest.raises(ValueError, match="identity mismatch"):
        make_book({**search, "key": "/works/OL99W"}, response)
    with pytest.raises(ValueError, match="author identity mismatch"):
        make_book({**search, "author_key": ["OL99A"]}, response)


def test_genres_require_an_explicit_subject():
    assert genres_from_subjects(["Science fiction", "Fiction"]) == ["Science fiction"]
    assert genres_from_subjects(["History and criticism of science fiction"]) == []


def test_offline_cache_miss_makes_no_request(tmp_path):
    client = OpenLibraryClient(tmp_path, offline=True)
    with pytest.raises(FileNotFoundError):
        client.get("/works/OL1W.json")
    assert client.requests == 0


def test_source_review_is_pinned_and_keeps_raw_evidence():
    search, response = fixture_records()
    book = make_book(search, response)
    decision = {
        "sourceContentHash": book["sourceContentHash"],
        "reason": "Uninformative source description",
    }
    review = {"withheldDescriptions": {book["id"]: decision}}
    result = apply_review(book, review)
    assert result["description"] == ""
    assert result["descriptionSource"] is None
    assert response["data"]["description"]["value"] == "The source description."
    with pytest.raises(ValueError, match="Source changed"):
        apply_review({**book, "sourceContentHash": "changed"}, review)


def test_shipped_catalogue_has_auditable_sources():
    catalogue = json.loads((ROOT / "web/data/books.json").read_text())
    manifest = json.loads((ROOT / "data/catalog/manifest.json").read_text())
    review = json.loads((ROOT / "data/catalog/selection_review.json").read_text())
    responses = [
        json.loads(path.read_text()) for path in (ROOT / "data/catalog/raw").glob("*.json")
    ]
    by_hash = {response["sha256"]: response for response in responses}
    assert len(catalogue) >= 60
    assert len({book["id"] for book in catalogue}) == len(catalogue)
    assert manifest["catalogueSha256"] == digest(catalogue)
    assert manifest["selectionReviewSha256"] == digest(review)
    assert not {book["id"] for book in catalogue}.intersection(review["excludedWorks"])
    for book in catalogue:
        validate_book(book)
        response = by_hash[book["sourceContentHash"]]
        assert digest(response["data"]) == response["sha256"]
        assert response["data"]["key"] == book["identifiers"]["openLibraryWork"]
        description = response["data"].get("description", "")
        if isinstance(description, dict):
            description = description.get("value", "")
        if book["id"] in review["withheldDescriptions"]:
            assert book["description"] == ""
            assert (
                review["withheldDescriptions"][book["id"]]["sourceContentHash"]
                == response["sha256"]
            )
        else:
            assert book["description"] == description.strip()
