from src.catalog.identity import match_description, matched_work_id, normalize_title


def test_same_title_different_authors_are_not_joined():
    book = {"title": "College Girl", "author": "Smith, Jane"}
    wrong = {"title": "College Girl", "authors": ["John Green"], "description": "Wrong work"}
    right = {"title": "College Girl", "authors": ["Jane Smith"], "description": "Right work"}
    assert match_description(book, [wrong]) is None
    assert match_description(book, [wrong, right]) == right


def test_missing_authors_and_ambiguous_descriptions_fail_closed():
    book = {"title": "War Brides", "authors": ["Jane Smith"]}
    candidate = {**book, "description": "A description"}
    assert match_description({"title": "War Brides"}, [candidate]) is None
    assert (
        match_description(book, [{"title": "War Brides", "description": "Unknown author"}]) is None
    )
    assert (
        match_description(
            book, [candidate, {**candidate, "description": "A conflicting description"}]
        )
        is None
    )


def test_subtitles_and_articles_remain_part_of_identity():
    assert normalize_title("The Island: A Memoir") != normalize_title("Island: A Novel")
    book = {"title": "Island: A Memoir", "authors": ["Jane Smith"]}
    assert match_description(book, [{**book, "title": "Island: A Novel"}]) is None


def test_work_grouping_is_stable_across_gutenberg_author_format():
    assert matched_work_id(
        {"title": "Pride and Prejudice", "author": "Austen, Jane, 1775-1817"}
    ) == matched_work_id({"title": "Pride and Prejudice", "authors": ["Jane Austen"]})
