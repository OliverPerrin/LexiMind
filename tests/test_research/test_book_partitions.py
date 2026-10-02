"""Synthetic source constraints and immutable overlay checks, with no model or network."""

import gzip
import json

import pytest

from src.research.book_partitions import build_components, group_overrides
from src.research.builders.book_groups_review import external_records, file_ref
from src.research.builders.book_partitions import prepare


def test_transitive_components_and_ambiguous_endpoints_never_leak():
    groups = {
        "bgc-group:a": "train",
        "bgc-group:b": "test",
        "bgc-group:c": "dev",
        "bgc-group:untouched": "train",
    }
    external = ["catalogue:a", "licensed_text:a", "licensed_text:separate"]
    strong = [
        ("catalogue:a", "bgc-group:a"),
        ("catalogue:a", "bgc-group:b"),
        ("catalogue:a", "licensed_text:a"),
    ]
    ambiguous = [("licensed_text:a", "bgc-group:c")]
    components = build_components(groups, external, strong, ambiguous)
    overrides = group_overrides(components)
    assert overrides["bgc-group:a"] is overrides["bgc-group:b"]
    assert overrides["bgc-group:a"] is not overrides["bgc-group:c"]
    assert all(overrides[node]["proposed_split"] is None for node in groups if node in overrides)
    assert "bgc-group:untouched" not in overrides
    assert len(components) == 3
    assert any(
        row["members"] == ["licensed_text:separate"]
        and row["proposed_split"] in {"train", "dev", "test"}
        for row in components
    )
    assert all(row["training_eligible"] is False for row in components)
    assert components == build_components(
        dict(reversed(list(groups.items()))),
        reversed(external),
        reversed(strong),
        reversed(ambiguous),
    )


@pytest.mark.parametrize(
    "pairs", [[("catalogue:a", "missing:x")], [("catalogue:a", "catalogue:a")]]
)
def test_constraints_cannot_invent_or_self_join_identities(pairs):
    with pytest.raises(ValueError, match="distinct known"):
        build_components({}, ["catalogue:a"], pairs, [])


def test_component_hash_does_not_depend_on_legacy_split_labels():
    first = build_components(
        {"bgc-group:a": "train"}, ["catalogue:a"], [("catalogue:a", "bgc-group:a")], []
    )
    second = build_components(
        {"bgc-group:a": "test"}, ["catalogue:a"], [("catalogue:a", "bgc-group:a")], []
    )
    assert first[0]["component_id"] == second[0]["component_id"]
    assert first[0]["proposed_split"] == second[0]["proposed_split"]


def fixture(root):
    def write(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return file_ref(path, root)

    def jsonl(relative, values):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
        return file_ref(path, root)

    assignments = [
        {
            "record_id": f"bgc:train:{i}",
            "source_split": "train",
            "source_row": i,
            "group_id": f"bgc-group:{gid}",
            "proposed_split": split,
        }
        for i, (gid, split) in enumerate(
            (("a", "train"), ("a", "train"), ("b", "test"), ("c", "dev"), ("d", "train")), 1
        )
    ]
    assigned = jsonl("data/research_candidates/base/assignments.jsonl", assignments)
    groups = write(
        "research/preparation/bgc_group_manifest.json",
        {"counts": {"records": 5, "groups": 4}, "assignments": assigned},
    )
    catalogue = [
        {
            "id": "a",
            "title": "Book Alpha",
            "authors": ["Writer"],
            "identifiers": {"isbns": []},
            "source": {"url": "https://example.test/alpha"},
        }
    ]
    licensed = {
        "books": [
            {
                "work_id": "alpha",
                "title": "Book Alpha",
                "creators": ["Writer"],
                "source_page": "https://example.test/alpha",
            },
            {
                "work_id": "other",
                "title": "Unrelated Work",
                "creators": ["Other"],
                "source_page": "https://example.test/other",
            },
        ]
    }
    licensed_ref = write("research/preparation/licensed_books_sources.json", licensed)
    licensed_manifest = write(
        "research/preparation/licensed_books_manifest.json",
        {
            "source_inventory": licensed_ref,
            "books": [{"work_id": book["work_id"]} for book in licensed["books"]],
        },
    )
    external = external_records(catalogue, licensed)
    matches = [
        {
            **assignments[index],
            "matching_keys": ["normalized_title_creator_candidate"]
            if index != 3
            else ["normalized_title_only"],
            "disposition": "cross_source_identity_candidate"
            if index != 3
            else "title_only_review_do_not_merge",
        }
        for index in (0, 2, 3)
    ]
    packet = jsonl(
        "data/research_candidates/review/review.jsonl",
        [{"kind": "cross_source_candidates", "external": external[0], "bgc_candidates": matches}],
    )
    review = {
        "policy": "book-group-evidence-review-v1",
        "inputs": {
            "groups": groups,
            "assignments": assigned,
            "catalogue": write("web/data/books.json", catalogue),
            "licensed_sources": licensed_ref,
            "licensed_manifest": licensed_manifest,
        },
        "packet": packet,
        "counts": {"identity_candidate_pairs": 2, "title_only_pairs": 1},
    }
    write("research/preparation/book_group_review.json", review)
    mapping = write(
        "research/preparation/book_field_mapping.json",
        {
            "facets": {
                "genre": {"labels": ["fantasy", "mystery"]},
                "audience": {"labels": ["children"]},
            }
        },
    )
    field_rows = [
        {
            "record_id": row["record_id"],
            "group": {"group_id": row["group_id"], "proposed_split": row["proposed_split"]},
            "fields": {
                "genre": {"positive": ["fantasy"], "negative": ["mystery"]}
                if index % 2 == 0
                else {"positive": [], "negative": []},
                "audience": {"positive": [], "negative": []},
            },
        }
        for index, row in enumerate(assignments)
    ]
    path = root / "data/research_candidates/field_references.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in field_rows))
    write(
        "research/preparation/book_field_manifest.json",
        {
            "assignments": assigned,
            "grouping_manifest": groups,
            "mapping": mapping,
            "field_references": file_ref(path, root),
        },
    )
    for name in (
        "src/research/builders/book_partitions.py",
        "src/research/builders/book_groups_review.py",
        "src/research/builders/bgc_source.py",
        "src/research/book_partitions.py",
        "src/catalog/storage.py",
        "src/research/candidate_io.py",
        "src/research/io.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic implementation reference\n")
    return review, root / assigned["path"]


def test_overlay_reuses_base_refs_quarantines_and_counts_all_field_states(tmp_path):
    _, base = fixture(tmp_path)
    before = base.read_bytes()
    output = tmp_path / "data/research_candidates/overlay"
    report = prepare(output, root=tmp_path)
    assert report == prepare(output, root=tmp_path)
    assert base.read_bytes() == before
    assert report["counts"]["effective_bgc_split_records"] == {"quarantine": 4, "train": 1}
    assert report["counts"]["overridden_bgc_groups"] == 3
    assert report["counts"]["quarantined_components"] == 2
    assert report["counts"]["quarantined_external_records"] == 2
    assert report["counts"]["external_external_constraints"] == {"identity_candidate": 1}
    assert report["counts"]["cross_split_identity_constraints"] == 0
    support = report["field_support"]["by_proposed_split"]
    assert report["field_support"]["columns"] == 3
    assert support["quarantine"]["labels"]["genre:fantasy"] == {
        "positive": 2,
        "negative": 0,
        "unknown": 2,
        "positive_change_from_base": 2,
    }
    assert support["train"]["labels"]["genre:mystery"]["negative"] == 1
    assert support["quarantine"]["labels"]["audience:children"]["unknown"] == 4
    assert support["train"]["labels"]["genre:fantasy"]["positive_change_from_base"] == -1
    for reference in (report["components"], report["constraints"]):
        assert check_reference(tmp_path, reference)
    (output / "components.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Existing candidate artifact differs"):
        prepare(output, root=tmp_path)


def check_reference(root, reference):
    return file_ref(root / reference["path"], root) == reference


def test_changed_pinned_source_is_rejected_before_writing(tmp_path):
    _, base = fixture(tmp_path)
    base.write_bytes(base.read_bytes() + b"\n")
    output = tmp_path / "data/research_candidates/overlay"
    with pytest.raises(ValueError, match="changed"):
        prepare(output, root=tmp_path)
    assert not output.exists()


def test_overlay_cannot_overwrite_or_escape_source_locations(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot contain or replace"):
        prepare(tmp_path / "data/research_candidates/base", root=tmp_path)
    with pytest.raises(ValueError, match="ignored"):
        prepare(tmp_path / "research", root=tmp_path)
