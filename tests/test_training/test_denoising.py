"""Source reconstruction and exact reward checks, without model execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

from src.research.builders import bookdash
from src.research.candidate_io import json_bytes, sha
from src.research.io import read_json
from src.training.denoising import (
    INSTRUCTION,
    SENTINEL,
    _section_windows,
    denoising_outcomes,
    prepare_denoising_data,
)


def write(root, path, data):
    content = data if isinstance(data, bytes) else json_bytes(data)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {"path": path, "bytes": len(content), "sha256": sha(content)}


@pytest.fixture
def tokenizer():
    root = Path(__file__).resolve().parents[2]
    result = Tokenizer.from_file(str(root / "artifacts/hf_tokenizer/tokenizer.json"))
    result.no_padding()
    result.no_truncation()
    return result


@pytest.fixture
def corpus(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    write(tmp_path, "tokenizer.json", (repo / "artifacts/hf_tokenizer/tokenizer.json").read_bytes())
    books, metadata, sources = [], [], {}
    text = (
        "Bright children carried purple lanterns through quiet gardens while gentle rabbits "
        "watched beside ancient wooden fences."
    )
    for slug in ("first", "second", "third"):
        raw = (
            "---\nlayout: book\n---\n# Fixture\n\n"
            + "\n".join(f"![]({{{{ site.image-set }}}}/{i:02d}.jpg)\n{text}" for i in range(1, 13))
        ).encode()
        descriptor = write(tmp_path, f"{bookdash.CACHE}/sources/{sha(raw)}/{slug}.md", raw)
        descriptor["url"] = bookdash.RAW + slug + "/en/index.md"
        sources[slug] = descriptor
        books.append(
            {
                "slug": slug,
                "title": "Fixture",
                "source_heading": "Fixture",
                "creators": ["First Creator", "Second Creator", "Third Creator"],
                "publication_date": "2016-01-01",
                "provider_id": "unique-" + slug,
                "provider_isbn": "978-1-928318-22-4",
                "expected_words": len(text.split()) * 12,
                "source": descriptor,
            }
        )
        metadata.append(
            f'  {slug}:\n    title: "Fixture"\n'
            '    creator: "First Creator, Second Creator and Third Creator"\n'
            '    date: "2016-01-01"\n    publisher: "Book Dash"\n'
            f'    identifier: "unique-{slug}"\n'
            '    source: "978-1-928318-22-4"\n    language: "en"\n'
            '    rights: "http://creativecommons.org/licenses/by/4.0/"\n'
        )
    meta = ("titles:\n" + "".join(metadata)).encode()
    shared = {
        "metadata": write(tmp_path, "metadata.yml", meta),
        "tree": write(tmp_path, "tree.json", {"synthetic": True}),
    }
    registry = {"books": books, **shared}
    inputs = {"registry": write(tmp_path, "registry.json", registry), **shared}
    records = [{"work_id": "bookdash-" + book["slug"]} for book in books]
    splits = bookdash.assign_splits(records, {})
    for book, record in zip(books, records, strict=True):
        slug, work_id = book["slug"], record["work_id"]
        split = splits[work_id]
        group_id = "bookdash-work:" + sha(work_id)
        source = sources[slug]
        inputs["source:" + slug] = source
        inputs["receipt:" + slug] = write(
            tmp_path,
            source["path"] + ".receipt.json",
            {**source, "final_url": source["url"], "retrieved_at": "2026-10-02T00:00:00+00:00"},
        )
        raw = (tmp_path / source["path"]).read_bytes()
        work = {
            "schema_version": 1,
            **bookdash.prepare_book(book, raw, meta),
            "group_id": group_id,
            "proposed_split": split,
            "candidate_status": "unadmitted",
            "training_authorized": False,
        }
        record.update(
            group_id=group_id,
            proposed_split=split,
            license="CC-BY-4.0",
            matches=[],
            artifact=write(tmp_path, f"works/{work_id}.json", work),
        )
    manifest = {
        "schema_version": 1,
        "policy": bookdash.POLICY,
        "inputs": inputs,
        "implementation_sha256": {},
        "books": records,
    }
    write(tmp_path, "manifest.json", manifest)
    return tmp_path, manifest


def prepare(root):
    return prepare_denoising_data(root, Path("manifest.json"), Path("tokenizer.json"))


def test_only_used_works_are_opened_and_windows_are_source_bound(corpus):
    root, manifest = corpus
    held = next(book for book in manifest["books"] if book["proposed_split"] == "test")
    slug = held["work_id"].removeprefix("bookdash-")
    for reference in (
        held["artifact"],
        manifest["inputs"]["source:" + slug],
        manifest["inputs"]["receipt:" + slug],
    ):
        (root / reference["path"]).unlink()
    data = prepare(root)
    assert data == prepare(root)
    assert data["provenance"]["counts"] == {"train": 8, "dev": 8}
    assert data["provenance"]["shortfalls_from_eight"] == {}
    assert not data["provenance"]["global_test_used"]
    assert "source:" + slug not in data["provenance"]["inputs"]
    for split in ("train", "dev"):
        intervals = {}
        for row in data[split]:
            evidence = row["source_evidence"]
            work = read_json(root / evidence["artifact"]["path"])
            section = next(s for s in work["sections"] if s["section_id"] == evidence["section_id"])
            start, end = evidence["target_char_span"]
            assert section["text"][start:end] == row["target_text"]
            assert row["labels"] == row["target_ids"] + [1]
            assert row["source_split"] == split and row["input_ids"][-1] == 1
            assert len(row["input_ids"]) <= 160 and row["context_tokens"] >= 12
            assert row["source_evidence_sha256"] == sha(json_bytes(evidence))
            left, right = row["window_char_span"]
            assert row["input_text"] == (
                INSTRUCTION + section["text"][left:start] + SENTINEL + section["text"][end:right]
            )
            assert evidence["window_sha256"] == sha(section["text"][left:right])
            assert evidence["source_section_sha256"] == sha(section["text"])
            key = (row["work_id"], evidence["section_id"])
            assert all(max(left, a) >= min(right, b) for a, b in intervals.get(key, []))
            intervals.setdefault(key, []).append((left, right))


@pytest.mark.parametrize("mutation", ["source", "rehash_artifact", "split", "group", "receipt"])
def test_changed_receipts_and_rehashed_forged_content_fail(corpus, mutation):
    root, manifest = corpus
    book = next(book for book in manifest["books"] if book["proposed_split"] == "train")
    slug = book["work_id"].removeprefix("bookdash-")
    if mutation == "source":
        (root / manifest["inputs"]["source:" + slug]["path"]).write_bytes(b"changed")
    elif mutation == "rehash_artifact":
        payload = read_json(root / book["artifact"]["path"])
        payload["sections"][0]["text"] = "Fabricated source sentence."
        book["artifact"] = write(root, book["artifact"]["path"], payload)
    elif mutation == "split":
        book["proposed_split"] = "dev"
    elif mutation == "group":
        book["group_id"] = "forged"
    else:
        ref = manifest["inputs"]["receipt:" + slug]
        payload = read_json(root / ref["path"])
        payload["final_url"] = "https://example.com/changed"
        manifest["inputs"]["receipt:" + slug] = write(root, ref["path"], payload)
    write(root, "manifest.json", manifest)
    with pytest.raises(ValueError):
        prepare(root)


def test_short_contexts_and_visible_answers_are_not_relaxed(tokenizer):
    assert _section_windows("Tiny story ends.", tokenizer) == []
    assert _section_windows("A " + SENTINEL + " already present.", tokenizer) == []
    text = (
        "Purple children carried purple lanterns through quiet gardens while gentle rabbits "
        "watched beside ancient wooden fences and a book."
    )
    windows = _section_windows(text, tokenizer)
    assert windows
    assert all(
        row["target_text"].casefold() not in {"purple", "book", "word", "return"} for row in windows
    )
    assert all(row["input_text"].startswith(INSTRUCTION) for row in windows)


def test_unicode_targets_keep_source_codepoint_offsets(tokenizer):
    text = "Happy children visited the café beside quiet gardens while gentle rabbits watched ancient wooden fences."
    windows = _section_windows(text, tokenizer)
    target = next(row for row in windows if row["target_text"] == "café")
    left, right = target["target_char_span"]
    assert text[left:right] == "café"


def outcome(tokenizer, tokens, *, mask=None, target="forest"):
    generated = SimpleNamespace(
        response_ids=torch.tensor([tokens]),
        response_mask=torch.tensor([mask if mask is not None else [True] * len(tokens)]),
    )
    return denoising_outcomes([{"target_text": target}], generated, tokenizer)[0]


def test_only_complete_canonical_answer_followed_by_eos_is_rewarded(tokenizer):
    ids = tokenizer.encode("forest", add_special_tokens=False).ids
    assert outcome(tokenizer, ids + [1]) == {
        "correct": True,
        "reward": 1.0,
        "invalid": False,
        "truncated": False,
        "content_tokens": len(ids),
    }
    assert outcome(tokenizer, ids)["truncated"]
    assert outcome(tokenizer, ids)["reward"] == 0
    wrong = tokenizer.encode("Forest", add_special_tokens=False).ids + [1]
    assert not outcome(tokenizer, wrong)["correct"]
    padded = ids + [1, 0, 32127]
    assert outcome(tokenizer, padded, mask=[True] * (len(ids) + 1) + [False, False])["correct"]


@pytest.mark.parametrize("bad", [0, 2, 32127, -1, 999999, 32099])
def test_special_unknown_and_padded_vocabulary_ids_are_invalid(tokenizer, bad):
    ids = tokenizer.encode("forest", add_special_tokens=False).ids
    assert outcome(tokenizer, ids + [bad, 1])["invalid"]
    assert outcome(tokenizer, ids + [bad, 1])["reward"] == 0


def test_empty_duplicate_eos_noncanonical_and_nonprefix_masks_fail(tokenizer):
    ids = tokenizer.encode("forest", add_special_tokens=False).ids
    assert outcome(tokenizer, [1])["invalid"]
    assert outcome(tokenizer, ids + [1, 1])["invalid"]
    assert outcome(tokenizer, [3] + ids + [1])["invalid"]
    with pytest.raises(ValueError, match="contiguous"):
        outcome(tokenizer, [ids[0], 0, 1], mask=[True, False, True])
    with pytest.raises(ValueError, match="boolean"):
        outcome(tokenizer, ids + [1], mask=[1] * (len(ids) + 1))


def test_balanced_schedule_is_order_independent_and_work_balanced():
    from collections import Counter

    from src.training.denoising import balanced_order

    rows = [
        {"work_id": work, "record_id": f"{work}-{i}"}
        for work, size in (("a", 2), ("b", 7), ("c", 1))
        for i in range(size)
    ]
    selected = balanced_order(rows, 30, 17)
    assert selected == balanced_order(list(reversed(rows)), 30, 17)
    assert Counter(row["work_id"] for row in selected) == {"a": 10, "b": 10, "c": 10}
    assert len({row["record_id"] for row in selected if row["work_id"] == "b"}) == 7


def test_warm_adapter_restore_is_exact_and_rejects_partial_bad_payload():
    import torch

    from src.training.denoising import adapter_digest, adapter_snapshot, restore_adapters

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    warm = adapter_snapshot(model)
    expected = adapter_digest(warm)
    with torch.no_grad():
        model[1].weight.add_(4)
    changed = adapter_snapshot(model)
    bad = dict(warm, **{"1.bias": torch.full_like(warm["1.bias"], float("nan"))})
    with pytest.raises(ValueError, match="Warm-start"):
        restore_adapters(model, bad)
    assert adapter_digest(adapter_snapshot(model)) == adapter_digest(changed)
    restore_adapters(model, warm)
    assert adapter_digest(adapter_snapshot(model)) == expected
    assert "0.weight" not in warm
