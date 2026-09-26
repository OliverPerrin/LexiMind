import pytest

from src.catalog.identity import author_identity, matched_work_id, normalize_title


def test_subtitles_are_retained_for_catalogue_duplicate_keys():
    assert normalize_title("The Island: A Memoir") != normalize_title("Island: A Novel")


def test_author_order_and_gutenberg_name_format_have_stable_keys():
    assert matched_work_id(
        {"title": "Pride and Prejudice", "author": "Austen, Jane, 1775-1817"}
    ) == matched_work_id({"title": "Pride and Prejudice", "authors": ["Jane Austen"]})
    assert author_identity({"authors": ["Jane Austen", "Mary Shelley"]}) == author_identity(
        {"authors": ["Mary Shelley", "Jane Austen"]}
    )


@pytest.mark.parametrize(
    "authors", [None, [], ["Unknown"], ["Jane Austen", ""], ["Jane Austen", None]]
)
def test_missing_or_partial_authorship_cannot_establish_identity(authors):
    with pytest.raises(ValueError, match="author evidence"):
        matched_work_id({"title": "A book", "authors": authors})
