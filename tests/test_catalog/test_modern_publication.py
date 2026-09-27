"""Synthetic source fixtures verify modern selection, never publication facts."""

import copy
import hashlib
from urllib.parse import urlencode

import pytest

from scripts.build_book_catalog import MODERN_SUBJECTS, SEARCH_FIELDS, _add_modern_books
from src.catalog.openlibrary import (
    API_BASE,
    CatalogueIntegrityError,
    OpenLibraryClient,
    digest,
    make_book,
    validate_publication_review,
)
from src.catalog.storage import json_text
from tests.test_catalog.test_openlibrary import fixture_records


@pytest.fixture
def modern_fixture(tmp_path):
    search, work = fixture_records()
    search["first_publish_year"] = 2020
    client = OpenLibraryClient(tmp_path, offline=True)

    def cache(envelope):
        data = json_text(envelope).encode()
        path = tmp_path / (hashlib.sha256(envelope["url"].encode()).hexdigest() + ".json")
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    work_hash = cache(work)
    queries = []
    for index, subject in enumerate(MODERN_SUBJECTS):
        params = {
            "q": f'subject:"{subject}" language:eng first_publish_year:[2000 TO 2025]',
            "limit": 6,
            "fields": SEARCH_FIELDS,
        }
        data = {"docs": [search] if index == 0 else []}
        response = {
            "url": API_BASE + "/search.json?" + urlencode(params),
            "sha256": digest(data),
            "retrievedAt": work["retrievedAt"],
            "data": data,
        }
        queries.append(
            {
                "url": response["url"],
                "sha256": response["sha256"],
                "cacheFileSha256": cache(response),
            }
        )
    baseline_search, baseline_work = fixture_records()
    baseline_search["key"] = "/works/OL9W"
    baseline_work["url"] = API_BASE + "/works/OL9W.json"
    baseline_work["data"]["key"] = "/works/OL9W"
    baseline_work["data"]["title"] = "Synthetic baseline"
    baseline_work["sha256"] = digest(baseline_work["data"])
    baseline = make_book(baseline_search, baseline_work)
    modern = {
        "schemaVersion": 1,
        "yearRange": [2000, 2025],
        "reviewedAt": work["retrievedAt"],
        "baselineCatalogueFileSha256": "a" * 64,
        "preservedWorkIds": ["OL9W"],
        "queries": queries,
        "works": {
            "OL1W": {
                "status": "include",
                "sourceContentHash": work["sha256"],
                "cacheFileSha256": work_hash,
                "searchUrl": queries[0]["url"],
                "searchFirstPublished": 2020,
                "firstPublished": 2019,
                "reason": "Synthetic fixture only: original audio precedes a print reissue.",
                "publicationEvidence": [
                    {
                        "name": "Fixture Publisher",
                        "kind": "publisher",
                        "url": "https://publisher.example/fixture",
                        "locator": "Synthetic release date",
                        "summary": "Fixture-only publication evidence, not a production judgment.",
                    }
                ],
            }
        },
    }
    return client, modern, {"/works/OL9W": baseline}


def admit(fixture):
    client, modern, books = fixture
    result = _add_modern_books(
        client, {"excludedWorks": {}, "withheldDescriptions": {}}, modern, books
    )
    return result, books


def test_original_year_overrides_search_with_attribution_and_preserves_baseline(modern_fixture):
    result, books = admit(modern_fixture)
    assert set(books) == {"/works/OL1W", "/works/OL9W"}
    assert result["addedWorkIds"] == ["OL1W"]
    assert books["/works/OL1W"]["firstPublished"] == 2019
    assert books["/works/OL1W"]["firstPublishedSource"] == {
        "name": "Fixture Publisher",
        "url": "https://publisher.example/fixture",
    }
    assert modern_fixture[0].requests == 0


@pytest.mark.parametrize("year", [1999, 2026, None, 2000.0, True])
def test_out_of_range_or_unreviewed_year_is_rejected(modern_fixture, year):
    modern_fixture[1]["works"]["OL1W"]["firstPublished"] = year
    with pytest.raises(CatalogueIntegrityError, match="original-publication"):
        admit(modern_fixture)


@pytest.mark.parametrize("year", [2000, 2025])
def test_year_range_endpoints_are_inclusive(modern_fixture, year):
    modern_fixture[1]["works"]["OL1W"]["firstPublished"] = year
    _, books = admit(modern_fixture)
    assert books["/works/OL1W"]["firstPublished"] == year


@pytest.mark.parametrize("target", ["query", "work"])
def test_exact_cache_bytes_must_match_review(modern_fixture, target):
    client, review, _ = modern_fixture
    url = review["queries"][0]["url"] if target == "query" else API_BASE + "/works/OL1W.json"
    path = client.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(CatalogueIntegrityError, match="source changed"):
        admit(modern_fixture)


def test_unresolved_candidate_is_explicitly_excluded(modern_fixture):
    decision = modern_fixture[1]["works"]["OL1W"]
    decision["status"] = "exclude"
    del decision["firstPublished"], decision["publicationEvidence"]
    result, books = admit(modern_fixture)
    assert set(books) == {"/works/OL9W"}
    assert result["excludedWorkIds"] == ["OL1W"]


def test_removed_baseline_id_blocks_publication(modern_fixture):
    modern_fixture[1]["preservedWorkIds"].append("OL999W")
    with pytest.raises(CatalogueIntegrityError, match="remove existing IDs"):
        admit(modern_fixture)


def test_search_year_and_evidence_are_not_optional(modern_fixture):
    modern_fixture[1]["works"]["OL1W"]["searchFirstPublished"] = 2021
    with pytest.raises(CatalogueIntegrityError, match="search year"):
        admit(modern_fixture)
    review = copy.deepcopy(modern_fixture[1])
    review["works"]["OL1W"]["publicationEvidence"] = []
    with pytest.raises(CatalogueIntegrityError, match="evidence required"):
        validate_publication_review(review)


def test_unreviewed_source_candidate_blocks_build(modern_fixture):
    decision = modern_fixture[1]["works"].pop("OL1W")
    modern_fixture[1]["works"]["OL2W"] = decision
    with pytest.raises(CatalogueIntegrityError, match="source-pinned review"):
        admit(modern_fixture)
