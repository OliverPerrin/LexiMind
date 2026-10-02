"""Synthetic BGC archive contracts; no source downloads or models."""

import hashlib
import io
import zipfile

import pytest

from src.research.builders import bgc_source as prep


def book(
    *,
    isbn="9780451457998",
    provider="325356",
    title="Synthetic Title",
    author="A. Writer",
    body="A made-up blurb & literal <b>markup</b>.",
    published="Sep 01, 2000 ",
    labels=((0, "Fiction"), (1, "Fantasy")),
):
    topics = "".join(f"<d{depth}>{label}</d{depth}>" for depth, label in labels)
    return (
        f'<book date="2018-08-18" xml:lang="en"> \n<title>{title}</title>\n<body>{body}</body>\n<copyright>synthetic fixture</copyright>\n<metadata>\n<topics>\n{topics}\n</topics>\n<author>{author}</author>\n<published>{published}</published>\n<page_num> 100 Pages</page_num>\n<isbn>{isbn}</isbn>\n<url>https://www.penguinrandomhouse.com/books/{provider}/synthetic/</url>\n</metadata>\n</book>\n'
    ).encode()


def archive(path, rows=None, *, extra=None):
    rows = rows or {
        "train": [book()],
        "dev": [book(isbn="9780385499187", provider="88210", body="different")],
        "test": [book(isbn="9780452006478", provider="326977", body="last")],
    }
    with zipfile.ZipFile(path, "w") as output:
        for split, records in rows.items():
            output.writestr(prep.MEMBERS[split], b"".join(records))
        output.writestr("README.txt", "Synthetic fixture; CC BY-NC 4.0")
        output.writestr("hierarchy.txt", "Fiction\tFantasy\nPoetry\n")
        if extra:
            output.writestr(*extra)
    return path


def test_literal_fields_and_stream_hash_preserved_without_xml_repair():
    raw = book(
        body="Tea & coffee &amp; <em>literal</em>", labels=((0, "Fiction"), (1, "Fantasy & Magic"))
    )
    digest = hashlib.sha256()
    (row,) = prep.records(io.BytesIO(raw), digest=digest)
    assert row["body"] == "Tea & coffee &amp; <em>literal</em>"
    assert row["labels"] == [(0, "Fiction"), (1, "Fantasy & Magic")]
    assert row["published"] == "Sep 01, 2000 "
    assert digest.hexdigest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "raw",
    [
        book().replace(b"</book>\n", b""),
        book().replace(b"</body>", b"</wrong>"),
        book().replace(b"<d1>Fantasy</d1>", b"<d1>Fantasy</d2>"),
        b"unexpected prefix\n" + book(),
    ],
)
def test_malformed_or_truncated_records_fail_instead_of_skipping(raw):
    with pytest.raises(ValueError, match="record|topic"):
        list(prep.records(io.BytesIO(raw)))


def test_record_bound_is_enforced_before_unbounded_read(monkeypatch):
    monkeypatch.setattr(prep, "MAX_RECORD_BYTES", 100)
    with pytest.raises(ValueError, match="bound"):
        list(prep.records(io.BytesIO(book(body="x" * 1000))))


def test_exact_normalized_and_identity_overlap_are_separate(tmp_path):
    first = book(body="café & tea", author="", published="Jan 01, 2001")
    second = book(isbn="9780385499187", provider="88210", body="cafe\u0301  & tea", title="Other")
    duplicate = book(
        body="café & tea", author="", published="Jan 01, 2001", labels=((0, "Fiction"),)
    )
    path = archive(
        tmp_path / "synthetic.zip",
        {
            "train": [first, second],
            "dev": [duplicate],
            "test": [
                book(
                    isbn="invalid",
                    provider="invalid",
                    body="last",
                    published="not a date",
                    labels=((0, "Poetry"),),
                )
            ],
        },
    )
    before = path.read_bytes()
    result = prep.audit_archive(path)
    assert result["total_records"] == 4
    assert result["unique_label_names"] == 3
    assert result["label_record_counts"] == {"Fantasy": 2, "Fiction": 3, "Poetry": 1}
    assert result["duplicates"]["isbn13"]["cross_split_groups"] == 1
    assert result["duplicates"]["isbn13"]["different_label_set_groups"] == 1
    assert result["duplicates"]["exact_blurb"]["records_in_duplicate_groups"] == 2
    assert result["duplicates"]["normalized_blurb"]["records_in_duplicate_groups"] == 3
    assert result["identity"]["invalid_or_missing_isbn13"] == 1
    assert result["identity"]["unparsed_or_missing_provider_url"] == 1
    assert result["splits"]["train"]["empty_fields"] == {"author": 1}
    assert result["splits"]["train"]["edition_dates"]["minimum"] == "2000-09-01"
    assert result["splits"]["test"]["edition_dates"]["unparsed"] == 1
    assert result["hierarchy"]["standalone_roots"] == ["Poetry"]
    assert path.read_bytes() == before
    assert "café" not in str(result)  # No source text is exported in the report.
    assert list(tmp_path.iterdir()) == [path]  # Members are streamed, never extracted.


def test_archive_size_member_and_missing_split_guards(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unexpected"):
        prep.audit_archive(archive(tmp_path / "unexpected.zip", extra=("../outside.txt", "bad")))
    with pytest.raises(ValueError, match="Missing"):
        prep.audit_archive(archive(tmp_path / "missing.zip", {"train": [book()]}))
    path = archive(tmp_path / "large.zip")
    monkeypatch.setattr(prep, "MAX_UNCOMPRESSED_BYTES", 100)
    with pytest.raises(ValueError, match="Uncompressed"):
        prep.audit_archive(path)
    monkeypatch.setattr(prep, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ValueError, match="100 MB"):
        prep.audit_archive(path)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_duplicate_archive_members_rejected(tmp_path):
    path = archive(tmp_path / "duplicate.zip")
    with zipfile.ZipFile(path, "a") as output, pytest.warns(UserWarning, match="Duplicate name"):
        output.writestr(prep.MEMBERS["train"], book())
    with pytest.raises(ValueError, match="duplicate archive"):
        prep.audit_archive(path)


def test_cached_archive_is_create_only_and_bad_import_never_publishes(tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    source.write_bytes(b"synthetic source")
    destination = tmp_path / "cache" / "source.zip"
    monkeypatch.setattr(prep, "SOURCE_BYTES", source.stat().st_size)
    monkeypatch.setattr(prep, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    reference = prep.preserve_source(source, destination)
    assert prep.preserve_source(source, destination) == reference
    source.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="differs"):
        prep.preserve_source(source, destination)
    assert destination.read_bytes() == b"synthetic source"
    with pytest.raises(ValueError, match="differs"):
        prep.preserve_source(source, tmp_path / "new.zip")
    assert not (tmp_path / "new.zip").exists()
    assert not list(tmp_path.glob(".*"))


def test_identifiers_and_edition_dates_are_validated_without_inventing_work_ids():
    assert prep.isbn13("978-0-451-45799-8") == "9780451457998"
    assert prep.isbn13("9780451457999") is None
    assert prep.isbn13("0000000000000") is None
    assert prep.isbn13("1234567890128") is None
    assert prep.provider_id("https://www.penguinrandomhouse.com/books/123/title/?isbn=1") == "123"
    assert prep.provider_id("https://other.example/books/123/title/") is None
    assert prep.provider_id("https://www.penguinrandomhouse.com:garbage/books/12/title/") is None
    assert prep.provider_id("https://user@www.penguinrandomhouse.com/books/12/title/") is None
    assert prep.provider_id("https://www.penguinrandomhouse.com:80/books/12/title/") is None
    assert prep.provider_id("https://[invalid/books/12/title/") is None
    assert prep.edition_date("Feb 29, 2000 ") == "2000-02-29"
    assert prep.edition_date("Feb 29, 2001") is None
    assert prep.edition_date("January 2000") is None
