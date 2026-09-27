import pytest

from src.catalog.splits import split_summarization_records


def test_provided_small_validation_and_test_sets_are_not_discarded():
    records = [
        {"type": "academic", "source": "train", "split": "train"},
        {"type": "academic", "source": "validation", "split": "validation"},
        {"type": "academic", "source": "test", "split": "test"},
    ]
    result = split_summarization_records(records)
    assert {name: [row["source"] for row in rows] for name, rows in result.items()} == {
        "train": ["train"],
        "validation": ["validation"],
        "test": ["test"],
    }
    assert records[0]["split"] == "train"  # Input provenance is not mutated.


def test_literary_chapters_share_one_partition_independent_of_input_order():
    records = [{"type": "literary", "work_id": "work-1", "source": str(i)} for i in range(20)]
    first = split_summarization_records(records)
    assert len([rows for rows in first.values() if rows]) == 1
    assert first == split_summarization_records(records[::-1])


def test_pinned_partition_applies_to_all_records_of_a_work():
    records = [
        {"type": "literary", "work_id": "work-1", "source": "chapter one", "split": "test"},
        {"type": "literary", "work_id": "work-1", "source": "chapter two"},
    ]
    assert len(split_summarization_records(records)["test"]) == 2


def test_conflicting_source_partitions_and_missing_identity_fail_closed():
    first = {"type": "literary", "work_id": "work-1", "source": "chapter one", "split": "train"}
    with pytest.raises(ValueError, match="Conflicting source splits"):
        split_summarization_records([first, {**first, "split": "test"}])
    with pytest.raises(ValueError, match="verified work_id"):
        split_summarization_records([{"type": "literary", "title": "Same title", "source": "text"}])
