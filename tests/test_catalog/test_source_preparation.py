"""Future-preparation fixtures only: downloads and historical writes are replaced."""

import json

import pytest

from scripts import download_data
from src.catalog.source_documents import booksum_document_identity, gutenberg_document_identity
from src.catalog.splits import split_source_records, split_summarization_records


def _paragraph(label):
    return (
        f"The {label} was a story of the people in the village. They walked through the fields and spoke of their home. "
        * 5
    )


def test_gutenberg_converter_keeps_all_document_paragraphs_together(tmp_path, monkeypatch):
    source = [
        {
            "TEXT": "\n\n".join(
                _paragraph(f"document {book} chapter {chapter}") for chapter in range(6)
            ),
            "METADATA": json.dumps(
                {"text_id": book, "title": f"Book {book}", "authors": "Jane Smith"}
            ),
        }
        for book in range(1, 15)
    ]
    monkeypatch.setattr(download_data, "load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(download_data, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(download_data, "is_clean_prose", lambda paragraph: True)
    written = {}
    monkeypatch.setattr(
        download_data,
        "write_jsonl",
        lambda records, path, desc: written.setdefault(path.stem, records),
    )
    download_data.download_gutenberg(max_samples=1000)
    assignments = {}
    for split, rows in written.items():
        for row in rows:
            assignments.setdefault(row["document_id"], set()).add(split)
            assert row["identity_scope"] == "provider_document"
            assert row["work_identity_status"] == "unresolved"
            assert "work_id" not in row
            assert row["author"] == "Jane Smith"
    assert len(assignments) == 14
    assert all(len(partitions) == 1 for partitions in assignments.values())
    assert sum(map(len, written.values())) == 84


def test_missing_gutenberg_provider_id_groups_full_document_without_claiming_a_work():
    text = "Full document content"
    identity = gutenberg_document_identity({}, {}, "sedthh/gutenberg_english", text)
    assert identity["document_identity_basis"] == "full_source_text_sha256"
    assert identity["work_identity_status"] == "unresolved"
    records = [{"type": "gutenberg", "text": str(i), **identity} for i in range(10)]
    partitions = split_source_records(records)
    assert len([rows for rows in partitions.values() if rows]) == 1
    assert partitions == split_source_records(records[::-1])


def test_pg19_provider_url_and_title_survive_fallback_converter(tmp_path, monkeypatch):
    item = {
        "url": "http://www.gutenberg.org/ebooks/123",
        "short_book_title": "A Provider Title",
        "text": "\n\n".join(_paragraph(str(i)) for i in range(3)),
    }

    def loader(name, **kwargs):
        if name == "sedthh/gutenberg_english":
            raise RuntimeError("fixture fallback")
        assert name == "pg19"
        return [item]

    monkeypatch.setattr(download_data, "load_dataset", loader)
    monkeypatch.setattr(download_data, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(download_data, "is_clean_prose", lambda paragraph: True)
    rows = []
    monkeypatch.setattr(download_data, "write_jsonl", lambda records, *args: rows.extend(records))
    download_data.download_gutenberg()
    assert len(rows) == 3
    assert all(row["title"] == "A Provider Title" for row in rows)
    assert all(row["provider_document_id"] == item["url"] for row in rows)
    assert all(row["provider_dataset"] == "deepmind/pg19" for row in rows)


def test_booksum_converter_retains_parent_bid_across_chapter_bearing_ids(monkeypatch):
    source = [
        {
            "bid": 27681,
            "book_id": f"Dr. Example.chapter {i}",
            "chapter_path": f"all_chapterized_books/27681-chapters/0{i}.txt",
            "chapter": _paragraph(str(i)),
            "summary_text": f"Summary {i}",
            "summary_id": f"chapter {i}",
            "source": "fixture",
            "summary_url": "https://example.org/summary",
        }
        for i in range(1, 4)
    ]
    monkeypatch.setattr(download_data, "load_dataset", lambda *args, **kwargs: {"train": source})
    monkeypatch.setattr(download_data, "is_english_text", lambda text: True)
    monkeypatch.setattr(download_data, "is_quality_text", lambda text: True)
    monkeypatch.setattr(download_data, "is_play_text", lambda text: False)
    rows = download_data.download_booksum()
    assert len(rows) == 3
    assert {row["document_id"] for row in rows} == {"hf:kmfoda/booksum:bid:27681"}
    assert {row["title"] for row in rows} == {"Dr. Example"}
    assert all(row["provider_ids"]["book_id"].startswith("Dr. Example.chapter") for row in rows)
    assert all(row["summary_source_url"] == "https://example.org/summary" for row in rows)
    assert len(split_summarization_records(rows)["train"]) == 3
    with pytest.raises(ValueError, match="Conflicting source splits"):
        split_summarization_records([rows[0], {**rows[1], "split": "test"}])


def test_booksum_missing_parent_id_is_actionable_and_never_title_matched():
    with pytest.raises(ValueError, match="retain bid"):
        booksum_document_identity({"book_id": "Same Title.chapter 1", "chapter": "text"})
    identity = booksum_document_identity(
        {"chapter_path": "all_chapterized_books/123-chapters/01.txt"}
    )
    assert identity["document_id"] == "hf:kmfoda/booksum:bid:123"
    assert identity["document_identity_basis"] == "chapter_path_parent"
    with pytest.raises(ValueError, match="disagree"):
        booksum_document_identity(
            {"bid": 123, "chapter_path": "all_chapterized_books/456-chapters/01.txt"}
        )


def test_provider_namespaces_do_not_invent_cross_source_work_equivalence():
    gutenberg = gutenberg_document_identity(
        {}, {"text_id": 123}, "sedthh/gutenberg_english", "text"
    )
    booksum = booksum_document_identity({"bid": 123})
    assert gutenberg["document_id"] != booksum["document_id"]
    assert gutenberg["work_identity_status"] == booksum["work_identity_status"] == "unresolved"


def test_pg19_url_variants_keep_the_same_provider_document_group():
    first = gutenberg_document_identity(
        {"url": "http://www.gutenberg.org/ebooks/123"}, {}, "deepmind/pg19", "source"
    )
    second = gutenberg_document_identity(
        {"url": "https://gutenberg.org/ebooks/123/"}, {}, "deepmind/pg19", "source"
    )
    assert first["document_id"] == second["document_id"]
    assert first["provider_document_id"] != second["provider_document_id"]
