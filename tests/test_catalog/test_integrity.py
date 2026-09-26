import copy
import hashlib
import json

import pytest

from scripts.build_book_catalog import build_catalogue
from src.catalog.openlibrary import (
    CatalogueIntegrityError,
    OpenLibraryClient,
    SourceRecordError,
    digest,
    load_catalogue,
    make_book,
    validate_book,
    validate_catalogue,
)
from src.catalog.storage import json_text, publish_catalogue, write_text_atomic
from tests.test_catalog.test_openlibrary import fixture_records


@pytest.fixture
def book():
    return make_book(*fixture_records())


@pytest.mark.parametrize(
    "field,value",
    [
        ("authors", "Jane Smith"),
        ("authors", ["Jane Smith", None]),
        ("title", 123),
        ("subjects", "Science"),
        ("genres", [""]),
        ("firstPublished", True),
        ("firstPublished", 1.5),
        ("coverUrl", "https://covers.openlibrary.org.evil.example/b/id/123-L.jpg"),
        ("coverUrl", "javascript:alert(1)"),
        ("sourceContentHash", "not-a-hash"),
        ("descriptionSource", "https://openlibrary.org/works/OL999W"),
    ],
)
def test_book_validation_rejects_malformed_fields(book, field, value):
    book[field] = value
    with pytest.raises(SourceRecordError):
        validate_book(book)


def test_source_timestamp_and_hash_are_verified_at_admission():
    search, response = fixture_records()
    response["data"]["description"] = "Changed without a matching digest"
    with pytest.raises(CatalogueIntegrityError, match="checksum"):
        make_book(search, response)
    response["sha256"] = digest(response["data"])
    response["retrievedAt"] = "2026-09-26T12:00:00"
    with pytest.raises(CatalogueIntegrityError, match="metadata"):
        make_book(search, response)


@pytest.mark.parametrize(
    "path",
    [
        "@evil.example/work.json",
        "//evil.example/a",
        "/works/../search.json",
        "/works/OL1W.json?callback=x",
        "/authors/OL1A.json",
    ],
)
def test_client_rejects_paths_before_any_network_request(tmp_path, path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Must reject before network I/O")

    monkeypatch.setattr("src.catalog.openlibrary.urlopen", unexpected)
    with pytest.raises(SourceRecordError):
        OpenLibraryClient(tmp_path).get(path)


def test_corrupt_cache_never_falls_back_to_a_live_fetch(tmp_path, monkeypatch):
    url = "https://openlibrary.org/works/OL1W.json"
    cache = tmp_path / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    cache.write_text('{"truncated":')
    client = OpenLibraryClient(tmp_path)
    with pytest.raises(CatalogueIntegrityError, match="Corrupt API cache"):
        client.get("/works/OL1W.json")
    assert client.requests == 0


def test_duplicate_catalogue_identities_require_review(book):
    with pytest.raises(CatalogueIntegrityError, match="Duplicate work ID"):
        validate_catalogue([book, book])
    duplicate = copy.deepcopy(book)
    duplicate["id"] = "OL99W"
    duplicate["identifiers"]["openLibraryWork"] = "/works/OL99W"
    duplicate["source"]["url"] = "https://openlibrary.org/works/OL99W"
    duplicate["descriptionSource"] = duplicate["source"]["url"]
    with pytest.raises(CatalogueIntegrityError, match="Duplicate title/author"):
        validate_catalogue([book, duplicate])


def test_changed_review_aborts_build_instead_of_silently_dropping_work():
    search, response = fixture_records()

    class Client:
        def get(self, path, params=None):
            if path == "/search.json":
                return {
                    "data": {"docs": [search]},
                    "url": "https://openlibrary.org/search.json",
                    "sha256": "search",
                }
            return response

    review = {
        "schemaVersion": 1,
        "excludedWorks": {},
        "withheldDescriptions": {"OL1W": {"sourceContentHash": "0" * 64, "reason": "Old review"}},
    }
    with pytest.raises(CatalogueIntegrityError, match="Source changed"):
        build_catalogue(Client(), review)


def test_catalogue_manifest_mismatch_is_rejected(book):
    with pytest.raises(CatalogueIntegrityError, match="disagree"):
        validate_catalogue([book], {"count": 1, "catalogueSha256": "old"})


def test_partial_json_serialization_preserves_previous_file(tmp_path):
    target = tmp_path / "records.jsonl"
    target.write_bytes(b"previous\r\n")

    def chunks():
        yield '{"first": true}\n'
        raise ValueError("serialization failed")

    with pytest.raises(ValueError, match="serialization"):
        write_text_atomic(target, chunks())
    assert target.read_bytes() == b"previous\r\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_paired_publication_restores_previous_files_on_second_replace_failure(
    tmp_path, book, monkeypatch
):
    import src.catalog.storage as storage

    catalogue_path, manifest_path = tmp_path / "books.json", tmp_path / "manifest.json"
    catalogue_path.write_bytes(b"old catalogue\r\n")
    manifest_path.write_bytes(b"old manifest\r\n")
    manifest = {"count": 1, "catalogueSha256": digest([book])}
    replace = storage.os.replace

    def fail_data_replace(source, target):
        if target == catalogue_path:
            raise OSError("simulated disk failure")
        return replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail_data_replace)
    with pytest.raises(OSError, match="disk failure"):
        publish_catalogue(catalogue_path, [book], manifest_path, manifest)
    assert catalogue_path.read_bytes() == b"old catalogue\r\n"
    assert manifest_path.read_bytes() == b"old manifest\r\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_publication_is_deterministic_and_hashes_exact_bytes(tmp_path, book):
    catalogue_path, manifest_path = tmp_path / "books.json", tmp_path / "manifest.json"
    manifest = {
        "count": 1,
        "catalogueSha256": digest([book]),
        "catalogueFileSha256": hashlib.sha256(json_text([book]).encode()).hexdigest(),
    }
    publish_catalogue(catalogue_path, [book], manifest_path, manifest)
    previous = catalogue_path.read_bytes(), manifest_path.read_bytes()
    publish_catalogue(catalogue_path, [book], manifest_path, manifest)
    assert previous == (catalogue_path.read_bytes(), manifest_path.read_bytes())
    assert (
        hashlib.sha256(catalogue_path.read_bytes()).hexdigest() == manifest["catalogueFileSha256"]
    )
    validate_catalogue(
        json.loads(catalogue_path.read_text()), json.loads(manifest_path.read_text())
    )


def test_three_artifact_publication_receipt_and_rollback(tmp_path, book, monkeypatch):
    import src.catalog.storage as storage

    catalogue_path, manifest_path, receipt_path = (
        tmp_path / name for name in ("books.json", "manifest.json", "receipt.json")
    )
    manifest = {"count": 1, "catalogueSha256": digest([book])}
    publish_catalogue(catalogue_path, [book], manifest_path, manifest, receipt_path)
    original = {path: path.read_bytes() for path in (catalogue_path, manifest_path, receipt_path)}
    receipt = json.loads(receipt_path.read_text())
    assert receipt["sourceManifestSha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert load_catalogue(catalogue_path, receipt_path) == [book]
    replace = storage.os.replace

    def fail_final_replace(source, target):
        if target == catalogue_path:
            raise OSError("third replacement failed")
        return replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="third replacement"):
        publish_catalogue(
            catalogue_path, [book], manifest_path, {**manifest, "new": True}, receipt_path
        )
    assert all(path.read_bytes() == content for path, content in original.items())


def test_reader_rejects_interrupted_publication(tmp_path, book):
    catalogue_path, manifest_path, receipt_path = (
        tmp_path / name for name in ("books.json", "manifest.json", "receipt.json")
    )
    publish_catalogue(
        catalogue_path,
        [book],
        manifest_path,
        {"count": 1, "catalogueSha256": digest([book])},
        receipt_path,
    )
    changed = {**book, "title": "Changed book metadata"}
    catalogue_path.write_text(json_text([changed]))
    with pytest.raises(CatalogueIntegrityError, match="publication receipt disagree"):
        load_catalogue(catalogue_path, receipt_path)


def test_failed_rollback_preserves_recovery_bytes(tmp_path, book, monkeypatch):
    import src.catalog.storage as storage

    catalogue_path, manifest_path = tmp_path / "books.json", tmp_path / "manifest.json"
    manifest_path.write_bytes(b"original manifest")
    replace = storage.os.replace
    calls = 0

    def fail_after_first(source, target):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("persistent filesystem failure")
        return replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail_after_first)
    with pytest.raises(RuntimeError, match="recovery files retained"):
        publish_catalogue(
            catalogue_path, [book], manifest_path, {"count": 1, "catalogueSha256": digest([book])}
        )
    assert any(path.read_bytes() == b"original manifest" for path in tmp_path.glob("*.tmp"))
