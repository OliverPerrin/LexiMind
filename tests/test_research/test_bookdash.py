"""Synthetic source, attribution and whole-work holdout contracts; no downloads."""

import copy
import json
from collections import Counter

import pytest

from src.research.builders import bookdash as prep
from src.research.candidate_io import sha


def fixture():
    book = {
        "slug": "fixture",
        "title": "Fixture",
        "source_heading": "Fixture: A Story! ",
        "creators": ["First Creator", "Second Creator", "Third Creator"],
        "publication_date": "2016-01-01",
        "provider_id": "unique-fixture-id",
        "provider_isbn": "978-1-928318-22-4",
        "expected_words": 120,
    }
    raw = "---\nlayout: book\n---\n# Fixture: A Story! \n\n" + "\n".join(
        f"![]({{{{ site.image-set }}}}/{i:02d}.jpg)\n" + "narrative " * 10 for i in range(1, 13)
    )
    metadata = """titles:
  fixture:
    title: "Fixture"
    creator: "First Creator, Second Creator and Third Creator"
    date: "2016-01-01"
    publisher: "Book Dash"
    identifier: "unique-fixture-id"
    source: "978-1-928318-22-4"
    language: "en"
    rights: "http://creativecommons.org/licenses/by/4.0/"
"""
    return book, raw.encode(), metadata.encode()


def test_explicit_heading_preserves_publisher_title_and_all_creators():
    book, raw, metadata = fixture()
    result = prep.prepare_book(book, raw, metadata)
    assert result["title"] == "Fixture"
    assert result["creators"] == book["creators"]
    assert len(result["sections"]) == 12 and result["whitespace_words"] == 120
    assert result["source_sha256"] == sha(raw)
    assert result["license"]["id"] == "CC-BY-4.0"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.update(title="Fixture changed"),
        lambda b: b.update(source_heading="Fixture: A Story!"),
        lambda b: b.update(creators=b["creators"][:2]),
        lambda b: b.update(publication_date="2017-01-01"),
        lambda b: b.update(provider_isbn="9780763679859"),
        lambda b: b.update(expected_words=121),
        lambda b: b.update(slug="../fixture"),
    ],
)
def test_unreviewed_metadata_or_heading_changes_fail(mutation):
    book, raw, metadata = fixture()
    mutation(book)
    with pytest.raises(ValueError):
        prep.prepare_book(book, raw, metadata)


def test_license_and_page_framing_are_not_inferred():
    book, raw, metadata = fixture()
    with pytest.raises(ValueError, match="license"):
        prep.prepare_book(book, raw, metadata.replace(b"by/4.0", b"by-nc/4.0"))
    with pytest.raises(ValueError, match="page order"):
        prep.prepare_book(book, raw.replace(b"12.jpg", b"13.jpg"), metadata)


def screen_book(work_id, title, text, isbn=""):
    return {"work_id": work_id, "title": title, "provider_isbn": isbn, "sections": [{"text": text}]}


def test_title_only_and_isbn_matches_quarantine_without_assuming_identity():
    books = [
        screen_book("one", "Shared Title", "first source"),
        screen_book("two", "Another Title", "second source", "9781928318224"),
    ]
    matches = prep.screen(
        books,
        [
            {"source": "bgc", "id": "old", "title": "  shared TITLE  ", "isbns": []},
            {
                "source": "catalogue",
                "id": "other",
                "title": "Completely Different",
                "isbns": ["978-1-928318-22-4"],
            },
        ],
    )
    assert matches["one"][0]["matching_keys"] == ["title"]
    assert matches["two"][0]["matching_keys"] == ["isbn"]


def test_duplicate_nonempty_pages_quarantine_both_works():
    books = [
        screen_book("one", "First", "Repeated page"),
        screen_book("two", "Second", "repeated   PAGE"),
    ]
    books[0]["sections"].append({"text": "different remaining story"})
    matches = prep.screen(books, [])
    assert set(matches) == {"one", "two"}
    assert all(row["matching_keys"] == ["page"] for rows in matches.values() for row in rows)
    books[0]["sections"][0]["text"] = ""
    books[1]["sections"][0]["text"] = ""
    assert prep.screen(books, []) == {}


def test_declared_heading_alias_and_prior_text_matches_are_screened():
    book = screen_book("new", "Publisher Title", "A retained page")
    book["source_heading"] = "What if...?"
    external = [
        {"source": "bgc", "id": "alias", "title": "What If...?", "isbns": []},
        {
            "source": "prior_licensed",
            "id": "past",
            "title": "Unrelated",
            "isbns": [],
            "sections": [{"text": "a retained PAGE"}],
        },
    ]
    matches = prep.screen([book], external)["new"]
    assert matches[0]["matching_keys"] == ["title"]
    assert matches[1]["matching_keys"] == ["body", "page"]


def test_split_counts_are_balanced_stable_and_quarantine_excluded():
    books = [{"work_id": f"work-{i}"} for i in range(35)]
    matches = {"work-0": [{"source": "bgc", "id": "prior"}]}
    splits = prep.assign_splits(books, matches)
    assert splits == prep.assign_splits(list(reversed(books)), matches)
    assert "work-0" not in splits
    assert Counter(splits.values()) == {"train": 24, "dev": 5, "test": 5}
    with pytest.raises(ValueError, match="three"):
        prep.assign_splits(books[:2], {})


def test_acquisition_is_offline_and_rejects_changed_source_or_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.urllib.request, "build_opener", lambda *_: pytest.fail("Network used"))
    raw = b"source bytes"
    url = prep.RAW + "fixture/en/index.md"
    path = tmp_path / prep.CACHE / "sources" / sha(raw) / "fixture.md"
    descriptor = {
        "path": str(path.relative_to(tmp_path)),
        "bytes": len(raw),
        "sha256": sha(raw),
        "url": url,
    }
    with pytest.raises(FileNotFoundError, match="fetch"):
        prep.acquire(tmp_path, descriptor, url)
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    receipt = path.with_name(path.name + ".receipt.json")
    receipt.write_text(
        json.dumps({**descriptor, "final_url": url, "retrieved_at": "2026-10-02T00:00:00+00:00"})
    )
    assert prep.acquire(tmp_path, descriptor, url)[0] == raw
    path.write_bytes(b"other bytes!")
    with pytest.raises(ValueError):
        prep.acquire(tmp_path, descriptor, url)
    path.write_bytes(raw)
    event = json.loads(receipt.read_text())
    event["final_url"] = "https://example.test/redirect"
    receipt.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="location"):
        prep.acquire(tmp_path, descriptor, url)
    wrong = copy.deepcopy(descriptor)
    wrong["bytes"] = True
    with pytest.raises(ValueError, match="bounded"):
        prep.acquire(tmp_path, wrong, url)
