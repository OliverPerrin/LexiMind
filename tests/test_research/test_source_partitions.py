import hashlib
import json

import pytest

from src.research.io import read_json
from src.research.partitions import prepare_partitions, role_for, text_group


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root, repo="fancyzhx/ag_news"):
    directory = root / "data/research_candidates/source"
    files = {}
    splits = {"train": 250, "test": 10}
    if repo.endswith("go_emotions"):
        splits["validation"] = 100
    for split, count in splits.items():
        rows = []
        for index in range(count):
            row = {
                "text": f"Fixture passage {index}",
                "source_row": index + 1,
                "provider_split": split,
            }
            if repo.endswith("go_emotions"):
                row.update(document_id=f"comment:{split}:{index}", emotions=["curiosity"])
            else:
                row.update(record_id=f"source-row:{split}:{index}", topic="World")
            rows.append(json.dumps(row))
        path = directory / f"prepared/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n")
        raw = path.read_bytes()
        files[split] = {
            "path": f"prepared/{split}.jsonl",
            "rows": count,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = root / "research/preparation/candidate.json"
    put_json(
        manifest,
        {
            "repo": repo,
            "schema_version": 1,
            "revision": "a" * 40,
            "label_names": ["curiosity"] if repo.endswith("go_emotions") else ["World"],
            "local_candidate_directory": str(directory.relative_to(root)),
            "prepared_files": files,
        },
    )
    return manifest, directory


@pytest.mark.parametrize("repo", ["fancyzhx/ag_news", "google-research-datasets/go_emotions"])
def test_official_test_retained_and_repeated_development_text_stays_together(tmp_path, repo):
    manifest, directory = fixture(tmp_path, repo)
    before = {p: p.read_bytes() for p in directory.rglob("*.jsonl")}
    first = prepare_partitions(tmp_path, manifest, directory / "partitions")
    rows = [
        json.loads(line)
        for line in (directory / "partitions/assignments.jsonl").read_text().splitlines()
    ]
    assert all(row["partition"] == "test" for row in rows if row["source_split"] == "test")
    if repo.endswith("go_emotions"):
        assert all(row["partition"] == "train" for row in rows if row["source_split"] == "train")
    assert (
        first["cross_partition_normalized_text_groups"] > 0
    )  # Official source overlap stays visible.
    assert first["within_original_split_cross_partition_text_groups"] == 0
    assert sum(first["counts"].values()) == len(rows)
    assert all(first["counts"].values())
    assert not first["training_authorized"]
    assert not any("text" in row for row in rows)
    assert prepare_partitions(tmp_path, manifest, directory / "partitions") == first
    assert all(path.read_bytes() == content for path, content in before.items())


def test_group_policy_retains_case_and_punctuation_but_normalizes_unicode_and_spaces():
    assert text_group("caf\u00e9\n book") == text_group("cafe\u0301   book")
    assert text_group("Book") != text_group("book")
    assert text_group("Book!") != text_group("Book")
    assert role_for("fancyzhx/ag_news", "train", text_group("a  b")) == role_for(
        "fancyzhx/ag_news", "train", text_group("a b")
    )


def test_corrupt_candidate_does_not_create_assignments(tmp_path):
    manifest, directory = fixture(tmp_path)
    path = directory / "prepared/train.jsonl"
    path.write_text(path.read_text().replace("World", "world"))
    with pytest.raises(ValueError, match="SHA-256 changed"):
        prepare_partitions(tmp_path, manifest, directory / "partitions")
    assert not (directory / "partitions/assignments.jsonl").exists()


def test_missing_original_split_cannot_be_silently_ignored(tmp_path):
    manifest, directory = fixture(tmp_path)
    value = read_json(manifest)
    del value["prepared_files"]["test"]
    put_json(manifest, value)
    with pytest.raises(ValueError, match="exactly the original"):
        prepare_partitions(tmp_path, manifest, directory / "partitions")


def test_conflicting_existing_assignment_is_preserved(tmp_path):
    manifest, directory = fixture(tmp_path)
    output = directory / "partitions"
    output.mkdir()
    (output / "assignments.jsonl").write_text("previous reviewed assignment\n")
    with pytest.raises(ValueError, match="differs"):
        prepare_partitions(tmp_path, manifest, output)
    assert (output / "assignments.jsonl").read_text() == "previous reviewed assignment\n"


def test_duplicate_source_identity_is_rejected_even_with_rehashed_candidate(tmp_path):
    manifest, directory = fixture(tmp_path)
    path = directory / "prepared/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["record_id"] = rows[0]["record_id"]
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    path.write_bytes(raw)
    value = read_json(manifest)
    value["prepared_files"]["train"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    put_json(manifest, value)
    with pytest.raises(ValueError, match="identities repeat"):
        prepare_partitions(tmp_path, manifest, directory / "partitions")
    assert not (directory / "partitions/assignments.jsonl").exists()


def test_missing_revision_cannot_publish_an_assignment(tmp_path):
    manifest, directory = fixture(tmp_path)
    value = read_json(manifest)
    del value["revision"]
    put_json(manifest, value)
    with pytest.raises(ValueError, match="pinned provider revision"):
        prepare_partitions(tmp_path, manifest, directory / "partitions")
    assert not (directory / "partitions/assignments.jsonl").exists()
