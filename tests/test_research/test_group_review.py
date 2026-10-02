"""Review evidence must not turn collisions, source omissions or editions into gold."""

import json

import pytest

from src.research.builders.book_groups_review import (
    external_records,
    identity_keys,
    lines,
    scan,
    select_groups,
)


def row(title="Title", author="Writer", body="Original source text", **changes):
    return {
        "title": title,
        "author": author,
        "body": body,
        "isbn": "9780451457998",
        "url": "https://www.penguinrandomhouse.com/books/123/example/",
        "labels": [(0, "Fiction")],
        **changes,
    }


def assignment(number, group="group:a", split="train"):
    return {
        "record_id": f"bgc:{split}:{number}",
        "source_split": split,
        "source_row": number,
        "group_id": group,
        "proposed_split": "dev",
        "review_required": True,
    }


def review(assignments, *, identity=False, keys=None, label_sets=None):
    return {
        "group_id": assignments[0]["group_id"],
        "proposed_split": "dev",
        "records": len(assignments),
        "identity_label_conflicts": [{"kind": "isbn13", "value": "9780451457998"}]
        if identity
        else [],
        "shared_key_counts": keys or {"normalized_title_author_candidate": 1},
        "members": [
            {**assigned, "source_labels": (label_sets or [[[0, "Fiction"]]] * len(assignments))[i]}
            for i, assigned in enumerate(assignments)
        ],
    }


def external(title="Title", creators=None, isbns=None, urls=None):
    return {
        "source": "catalogue",
        "id": "OL1W",
        "title": title,
        "creators": creators or ["Writer"],
        "isbns": isbns or [],
        "urls": urls or ["https://openlibrary.org/works/OL1W"],
    }


def run(rows, assignments, reviews=None, inventory=None):
    return scan(
        ((assigned["record_id"], item) for assigned, item in zip(assignments, rows, strict=True)),
        {assigned["record_id"]: assigned for assigned in assignments},
        reviews or [],
        inventory or [],
    )


def test_distinct_titles_with_shared_text_keep_leakage_constraint_and_source_labels():
    assigned = [assignment(1), assignment(2, split="test")]
    labels = [[[0, "Fiction"]], [[0, "Nonfiction"]]]
    groups, cross = run(
        [row(title="Henry V"), row(title="Coriolanus", labels=[(0, "Nonfiction")])],
        assigned,
        [review(assigned, keys={"normalized_blurb": 1}, label_sets=labels)],
    )
    (group,) = groups
    assert group["dispositions"] == [
        "keep_constraint",
        "review_identity",
        "distinct_titles_share_text",
    ]
    assert group["distinct_title_shared_blurb_clusters"] == 1
    assert group["distinct_titles"] == 2
    assert group["distinct_source_label_sets"] == 2
    assert [member["source_labels"] for member in group["packet"]["members"]] == list(
        reversed(labels)
    )
    assert {member["proposed_split"] for member in group["packet"]["members"]} == {"dev"}
    assert "Original source text" not in json.dumps(group)
    assert not cross


def test_direct_identity_conflict_preserves_omission_without_label_union():
    assigned = [assignment(1), assignment(2)]
    labels = [[[0, "Nonfiction"]], [[0, "Nonfiction"], [2, "Bibles"]]]
    (group,), _ = run(
        [row(labels=[tuple(pair) for pair in values]) for values in labels],
        assigned,
        [review(assigned, identity=True, label_sets=labels)],
    )
    assert group["labels_varying_between_members"] == [[2, "Bibles"]]
    assert group["identity_label_conflicts"]
    assert [member["source_labels"] for member in group["packet"]["members"]] == labels
    assert "distinct_titles_share_text" not in group["dispositions"]


def test_same_title_different_author_is_only_a_review_candidate():
    _, (match,) = run(
        [row(title="The Raven", author="Different Writer")],
        [assignment(1)],
        inventory=[external(title="The Raven")],
    )
    assert match["bgc_candidates"][0]["disposition"] == "title_only_review_do_not_merge"
    assert match["bgc_candidates"][0]["matching_keys"] == ["normalized_title_only"]


def test_cross_source_title_creator_candidate_exposes_group_and_original_location():
    _, (match,) = run(
        [row(title="Little Brother", author="Cory Doctorow")],
        [assignment(33694)],
        inventory=[external(title="Little Brother", creators=["Cory Doctorow"])],
    )
    candidate = match["bgc_candidates"][0]
    assert candidate["matching_keys"] == ["normalized_title_creator_candidate"]
    assert candidate["disposition"] == "cross_source_identity_candidate"
    assert candidate["archive_member"] == "BlurbGenreCollection_EN_train.txt"
    assert candidate["source_row"] == 33694
    assert candidate["group_id"] == "group:a"


@pytest.mark.parametrize(
    "inventory,key",
    [
        (external(title="Another edition", isbns=["978-0-451-45799-8"]), "isbn13"),
        (
            external(title="Another edition", urls=["http://penguinrandomhouse.com/books/123/"]),
            "provider_book_id",
        ),
    ],
)
def test_identity_keys_find_candidates_despite_title_differences(inventory, key):
    _, (match,) = run([row()], [assignment(1)], inventory=[inventory])
    assert match["bgc_candidates"][0]["matching_keys"] == [key]


def test_normalization_retains_case_punctuation_and_creator_boundaries():
    first = identity_keys("cafe\u0301  book", ["A.  Writer"], [], [])
    assert first == identity_keys("café book", ["A. Writer"], [], [])
    assert first != identity_keys("Café book", ["A. Writer"], [], [])
    assert first != identity_keys("café book", ["A Writer"], [], [])
    assert ("normalized_title_creator_candidate", ("Title", "One, Two")) in identity_keys(
        "Title", ["One", "Two"], [], []
    )
    assert not identity_keys("", [""], ["9780000000000"], ["https://example.com/books/123/"])


def test_selection_is_bounded_deterministic_and_never_drops_direct_conflicts():
    reviews = [
        review([assignment(n, group=f"group:{i}") for n in range(1, i + 2)], identity=i == 0)
        for i in range(8)
    ]
    selected = select_groups(reviews)
    assert selected == select_groups(list(reversed(reviews)))
    assert set(selected) == {"group:0", "group:5", "group:6", "group:7"}
    assert selected["group:0"] == ["direct_identity_label_conflict"]


def test_missing_repeated_and_unassigned_rows_fail():
    assigned = assignment(1)
    for inputs, message in [
        ([], "completely cover"),
        ([("missing", row())], "absent"),
        ([(assigned["record_id"], row())] * 2, "repeated"),
    ]:
        with pytest.raises(ValueError, match=message):
            scan(inputs, {assigned["record_id"]: assigned}, [], [])


def test_selected_member_changes_fail_instead_of_reporting_stale_evidence():
    assigned = [assignment(1)]
    pinned_review = review(assigned)
    with pytest.raises(ValueError, match="labels or assignment"):
        run([row(labels=[(0, "Nonfiction")])], assigned, [pinned_review])
    pinned_review["members"].append({**assignment(2), "source_labels": [[0, "Fiction"]]})
    with pytest.raises(ValueError, match="members differ"):
        run([row()], assigned, [pinned_review])


def test_external_creators_and_provider_namespaces_remain_explicit():
    inventory = external_records(
        [
            {
                "id": "OL1W",
                "title": "Title",
                "authors": ["Writer"],
                "source": {"url": "https://openlibrary.org/works/OL1W"},
                "identifiers": {"isbns": []},
            }
        ],
        {
            "books": [
                {
                    "work_id": "licensed",
                    "title": "Title",
                    "creators": ["Writer", "Illustrator"],
                    "source_page": "https://bookdash.org/book/title/",
                    "provider_id": "123",
                }
            ]
        },
    )
    assert inventory[1]["creator_basis"] == "source_creators_may_include_non_authors"
    assert not any(
        kind == "provider_book_id"
        for kind, _ in identity_keys(
            inventory[1]["title"], inventory[1]["creators"], [], inventory[1]["urls"]
        )
    )
    with pytest.raises(ValueError, match="repeats"):
        external_records(
            [],
            {"books": [{"work_id": "a", "title": "Title", "creators": [], "source_page": "x"}] * 2},
        )


def test_jsonl_reader_rejects_overlong_duplicate_keys_and_excess_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    for content, limit, message in [
        (b" " * 256_001, 2, "bound"),
        (b"{}\n{}\n", 1, "bound"),
        (b'{"a":1,"a":2}\n', 2, "Duplicate"),
    ]:
        path.write_bytes(content)
        with pytest.raises(ValueError, match=message):
            list(lines(path, limit))
