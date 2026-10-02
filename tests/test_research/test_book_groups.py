"""Leakage components preserve source evidence without adjudicating works or labels."""

import json
from dataclasses import replace

import pytest

from src.research.book_groups import BookGroupRecord, prepare_book_groups, text_keys
from src.research.builders.bgc_groups import grouping_record


def record(
    number,
    *,
    split="train",
    title=None,
    author="Writer",
    body=None,
    isbn=None,
    provider=None,
    labels=((0, "Fiction"),),
):
    keys = text_keys(
        title if title is not None else f"Title {number}",
        author,
        body if body is not None else f"Blurb {number}",
    )
    if isbn:
        keys["isbn13"] = isbn
    if provider:
        keys["provider_book_id"] = provider
    return BookGroupRecord(f"bgc:{split}:{number}", split, number, keys, labels)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_transitive_identity_text_and_title_links_stay_in_one_split(tmp_path):
    inputs = [
        record(1, isbn="9780451457998"),
        record(2, split="dev", isbn="9780451457998", body="café tea", labels=((0, "Poetry"),)),
        record(3, split="test", body="cafe\u0301  tea", title="Linked title"),
        record(4, title="Linked title"),
        record(5),
    ]
    report = prepare_book_groups(inputs, tmp_path, "pinned-source")
    assigned = {row["record_id"]: row for row in rows(tmp_path / "assignments.jsonl")}
    component = [assigned[item.record_id] for item in inputs[:4]]
    assert len({row["group_id"] for row in component}) == 1
    assert len({row["proposed_split"] for row in component}) == 1
    assert all(row["review_required"] for row in component)
    assert assigned[inputs[4].record_id]["group_id"] != component[0]["group_id"]
    assert not assigned[inputs[4].record_id]["review_required"]
    assert {(row["source_split"], row["source_row"]) for row in assigned.values()} == {
        (row.source_split, row.source_row) for row in inputs
    }
    assert report["counts"]["groups"] == 2
    assert report["counts"]["original_cross_split_groups"] == 1
    assert report["counts"]["groups_with_identity_label_conflicts"] == 1
    assert all(
        value["proposed_cross_split_keys"] == 0 for value in report["matching_key_overlap"].values()
    )
    (review,) = rows(tmp_path / "review_groups.jsonl")
    assert review["identity_label_conflicts"] == [{"kind": "isbn13", "value": "9780451457998"}]
    assert review["status"] == "unadjudicated_leakage_constraint"
    assert {tuple(map(tuple, row["source_labels"])) for row in review["members"]} == {
        ((0, "Fiction"),),
        ((0, "Poetry"),),
    }


def test_output_is_identical_with_reversed_record_and_key_order(tmp_path):
    inputs = [record(1, body="shared"), record(2, split="test", body="shared"), record(3)]
    first = prepare_book_groups(inputs, tmp_path / "first", "same-source")
    second = prepare_book_groups(
        [replace(row, keys=dict(reversed(list(row.keys.items())))) for row in reversed(inputs)],
        tmp_path / "second",
        "same-source",
    )
    assert first == second
    for name in ("assignments", "review_groups"):
        assert (tmp_path / "first" / f"{name}.jsonl").read_bytes() == (
            tmp_path / "second" / f"{name}.jsonl"
        ).read_bytes()
    assert prepare_book_groups(inputs, tmp_path / "first", "same-source") == first


def test_empty_blurbs_and_incomplete_title_author_pairs_do_not_join(tmp_path):
    inputs = [
        record(1, title="Shared", author="", body=""),
        record(2, title="Shared", author=" \t", body="\n"),
        record(3, title="", author="Writer", body=""),
        record(4, title="", author="Writer", body=""),
    ]
    report = prepare_book_groups(inputs, tmp_path, "source")
    assert report["counts"]["groups"] == 4
    assert not rows(tmp_path / "review_groups.jsonl")
    assert report["records_without_matching_key"]["normalized_title_author_candidate"] == 4
    assert report["records_without_matching_key"]["exact_blurb"] == 4


def test_case_and_punctuation_are_not_silently_fuzzy_matched():
    assert text_keys("A Title", "Author", "Some text") != text_keys(
        "a title", "Author", "some text"
    )
    assert text_keys("A Title", "Author", "Some text") != text_keys(
        "A Title!", "Author", "Some text!"
    )
    assert text_keys("cafe\u0301  title", "A.  Writer", "x") == text_keys(
        "café title", "A. Writer", "x"
    )


def test_provider_identity_links_and_conflicts_do_not_change_source_labels(tmp_path):
    report = prepare_book_groups(
        [
            record(1, provider="325356", labels=((0, "Fiction"), (1, "Fantasy"))),
            record(2, split="test", provider="325356", labels=((0, "Fiction"),)),
        ],
        tmp_path,
        "source",
    )
    assert report["matching_key_overlap"]["provider_book_id"]["different_label_set_keys"] == 1
    assert report["counts"]["groups_with_identity_label_conflicts"] == 1
    (review,) = rows(tmp_path / "review_groups.jsonl")
    assert [len(member["source_labels"]) for member in review["members"]] == [1, 2]


def test_rare_source_label_support_reports_missing_proposed_split_labels(tmp_path):
    report = prepare_book_groups([record(1, labels=((0, "Rare"),))], tmp_path, "source")
    support = report["proposed_split_source_label_support"]
    assert sum(item["unique_labels"] for item in support.values()) == 1
    assert sum(item["missing_labels"] == ["Rare"] for item in support.values()) == 2
    assert sorted(
        item["labels_with_fewer_than_five_records"]["Rare"] for item in support.values()
    ) == [0, 0, 1]


@pytest.mark.parametrize(
    "inputs,limit,match",
    [
        ([], 100, "empty"),
        ([record(1), record(1)], 100, "Repeated"),
        ([record(1), record(2)], 1, "bounded"),
    ],
)
def test_bad_or_unbounded_source_fails_without_publishing(tmp_path, inputs, limit, match):
    with pytest.raises(ValueError, match=match):
        prepare_book_groups(inputs, tmp_path, "source", max_rows=limit)
    assert not (tmp_path / "assignments.jsonl").exists()


def test_changed_assignment_requires_separate_output_location(tmp_path):
    prepare_book_groups([record(1)], tmp_path, "source")
    before = (tmp_path / "assignments.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Existing candidate artifact differs"):
        prepare_book_groups([record(2)], tmp_path, "source")
    assert (tmp_path / "assignments.jsonl").read_bytes() == before


def test_source_adapter_excludes_invalid_isbn_and_provider_identity():
    row = {
        "title": "Title",
        "author": "Writer",
        "body": "blurb",
        "isbn": "9780451457990",
        "url": "https://other.example/books/325356/",
        "labels": [(0, "Fiction")],
    }
    adapted = grouping_record(row, "dev", 2)
    assert adapted.record_id == "bgc:dev:2"
    assert "isbn13" not in adapted.keys and "provider_book_id" not in adapted.keys
    adapted = grouping_record(
        {
            **row,
            "isbn": "978-0451457998",
            "url": "https://www.penguinrandomhouse.com/books/325356/name/",
        },
        "train",
        1,
    )
    assert adapted.keys["isbn13"] == "9780451457998"
    assert adapted.keys["provider_book_id"] == "325356"
