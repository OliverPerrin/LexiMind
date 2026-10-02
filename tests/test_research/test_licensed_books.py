"""Synthetic licensed-text contracts; never download books or run models."""

import json
from pathlib import Path

import pytest

from src.research.builders import licensed_books as prep
from src.research.candidate_io import create_or_verify, json_bytes, sha


def rpt_fixture(root):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(payload))
        return {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha(path.read_bytes()),
        }

    inventory = write(root / "sources.json", {"status": "synthetic_sources"})
    books, components = [], []
    cache = root / "data/research_candidates/licensed_books"
    for split in ("train", "dev", "test", "quarantine"):
        work = "synthetic-" + split
        body = ("alpha beta gamma delta\n" * 40).strip()
        content = {
            "work_id": work,
            "work_group_id": work,
            "title": work,
            "creators": ["Fixture Author"],
            "source_page": "https://example.test/fixture",
            "source_sha256": sha(body),
            "license": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "sections": [{"section_id": "one", "text": body, "source_lines": [1, 40]}],
        }
        artifact = write(cache / (work + ".json"), content)
        artifact["path"] = Path(artifact["path"]).name
        books.append(
            {"work_id": work, "work_group_id": work, "license": "CC-BY-4.0", "artifact": artifact}
        )
        components.append(
            {
                "component_id": "book-component:" + sha(work),
                "members": ["licensed_text:" + work],
                "proposed_split": None if split == "quarantine" else split,
                "status": "quarantined_title_only_ambiguity"
                if split == "quarantine"
                else "candidate_not_admitted",
            }
        )
    manifest = write(
        root / "research/preparation/licensed_books_manifest.json",
        {"source_inventory": inventory, "cache_root": str(cache.relative_to(root)), "books": books},
    )
    path = root / "components.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in components))
    component_ref = {
        "path": "components.jsonl",
        "bytes": path.stat().st_size,
        "sha256": sha(path.read_bytes()),
    }
    write(
        root / "research/preparation/book_partition_manifest.json",
        {
            "policy": "book-component-overlay-v1",
            "inputs": {"licensed_manifest": manifest, "licensed_sources": inventory},
            "components": component_ref,
        },
    )
    return tokenizer_path, tokenizer


def test_rpt_candidates_retain_partitions_and_exact_token_byte_boundaries(tmp_path, monkeypatch):
    tokenizer_path, tokenizer = rpt_fixture(tmp_path)
    monkeypatch.setattr(prep.urllib.request, "build_opener", lambda *_: pytest.fail("Network used"))
    manifest_path = tmp_path / "research/preparation/licensed_books_manifest.json"
    source_before = manifest_path.read_bytes()
    result = prep.prepare_rpt(tokenizer_path, root=tmp_path)
    assert result == prep.prepare_rpt(tokenizer_path, root=tmp_path)
    assert manifest_path.read_bytes() == source_before
    assert result["counts"]["by_split"] == {"train": 8, "dev": 8, "test": 8}
    assert result["counts"]["works_by_split"]["quarantine"] == 1
    assert result["training_performed"] is False and result["model_inference_performed"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / result["continuations"]["path"]).read_text().splitlines()
    ]
    assert len(rows) == 24
    for row in rows:
        target = row["continuation_target"]
        text = target["observed_bytes"]
        pieces = [piece.encode() for piece in target["token_bytes"]]
        assert all(pieces) and b"".join(pieces) == text.encode()
        assert len(row["target_ids"]) == len(pieces)
        assert tokenizer.encode(text, add_special_tokens=False).ids == row["target_ids"]
        assert (
            tokenizer.encode(row["prompt_normalized_text"], add_special_tokens=False).ids
            == row["prompt_ids"]
        )
        for end in range(1, len(pieces) + 1):
            assert tokenizer.decode(
                row["target_ids"][:end], skip_special_tokens=False
            ).encode() == b"".join(pieces[:end])
        assert target["source_evidence_sha256"] == sha(json_bytes(row["source_evidence"]))
        assert target["tokenizer_sha256"] == sha(tokenizer_path.read_bytes())
        assert target["split"] in {"train", "dev", "test"}
        assert row["attribution"]["creators"] == ["Fixture Author"]
        assert row["candidate_status"] == "unadmitted" and row["training_authorized"] is False
    assert not any(row["work_id"].endswith("quarantine") for row in rows)
    assert all(work["records"] == 0 for work in result["works"] if work["split"] == "quarantine")
    assert "alpha beta" not in json.dumps(result)


def test_rpt_rejects_changed_or_unlinked_partition_evidence(tmp_path):
    tokenizer_path, _ = rpt_fixture(tmp_path)
    path = tmp_path / "components.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        prep.prepare_rpt(tokenizer_path, root=tmp_path)
    assert not (tmp_path / "data/research_candidates/licensed_books/rpt").exists()
    rpt_fixture(tmp_path)
    path = tmp_path / "research/preparation/licensed_books_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="exact licensed manifest"):
        prep.prepare_rpt(tokenizer_path, root=tmp_path)


@pytest.mark.parametrize("positions", [0, 7, 17, True, 8.5])
def test_rpt_position_count_is_bounded_before_tokenizer_load(tmp_path, positions):
    with pytest.raises(ValueError, match="integer from 8 to 16"):
        prep.prepare_rpt(tmp_path / "missing.json", positions_per_work=positions, root=tmp_path)


def test_rpt_skips_unknown_tokens_even_without_added_special_registration(tmp_path):
    tokenizer_path, _ = rpt_fixture(tmp_path)
    manifest_path = tmp_path / "research/preparation/licensed_books_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for book in manifest["books"]:
        path = tmp_path / manifest["cache_root"] / book["artifact"]["path"]
        artifact = json.loads(path.read_text())
        artifact["sections"][0]["text"] = "unrecognized " * 80
        path.write_bytes(json_bytes(artifact))
        book["artifact"].update(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    manifest_path.write_bytes(json_bytes(manifest))
    partition_path = tmp_path / "research/preparation/book_partition_manifest.json"
    partition = json.loads(partition_path.read_text())
    partition["inputs"]["licensed_manifest"].update(
        bytes=manifest_path.stat().st_size, sha256=sha(manifest_path.read_bytes())
    )
    partition_path.write_bytes(json_bytes(partition))
    result = prep.prepare_rpt(tokenizer_path, root=tmp_path)
    assert result["counts"]["records"] == 0
    assert result["counts"]["skipped_non_roundtripping_or_unstable_windows"] > 0


def test_rpt_rejects_decoder_prefix_rewrites():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return type("Encoding", (), {"ids": [int(item) for item in text.split()]})()

        def decode(self, ids, **kwargs):
            # A partial target rewrites an earlier byte when another token arrives.
            text = " ".join(map(str, ids))
            return "rewritten" if len(ids) == 2 else text

    assert prep._rpt_window(Tokenizer(), list(range(40)), 8, set()) is None


def test_rpt_refuses_stochastic_tokenizer_configuration(tmp_path):
    tokenizer_path, _ = rpt_fixture(tmp_path)
    configuration = json.loads(tokenizer_path.read_text())
    configuration["model"]["dropout"] = 0.1
    tokenizer_path.write_bytes(json_bytes(configuration))
    with pytest.raises(ValueError, match="stochastic BPE dropout"):
        prep.prepare_rpt(tokenizer_path, root=tmp_path)


def test_rpt_cli_uses_existing_sources_without_base_repreparation(tmp_path, monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    prep.configure_parser(parser)
    args = parser.parse_args(["--rpt-tokenizer", "tokenizer.json"])
    monkeypatch.setattr(prep, "prepare", lambda **_: pytest.fail("Base source preparation ran"))
    monkeypatch.setattr(prep, "prepare_rpt", lambda *_, **__: {"counts": {"records": 0}})
    monkeypatch.setattr(prep, "RPT_MANIFEST", tmp_path / "rpt.json")
    assert prep.run(args, parser) == 0
    assert json.loads((tmp_path / "rpt.json").read_text()) == {"counts": {"records": 0}}
    with pytest.raises(SystemExit):
        prep.run(parser.parse_args(["--fetch", "--rpt-tokenizer", "tokenizer.json"]), parser)


def novel():
    front = (
        "Little Brother\n\nCory Doctorow\n\nREAD THIS FIRST\n"
        "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license.\n"
        "FRONTMATTER NOT NARRATIVE\n"
    )
    sections = []
    for number in range(1, 23):
        title = f"Chapter {number}" if number < 22 else "Epilogue"
        sections.append(
            f"{title}\n\n[[Shop note\ncontinued]]\n\n[[Shop address]]\n\n"
            f"Synthetic narrative {number}.  \n\nAnother paragraph.\n\n &&&\n\n"
        )
    return (front + "".join(sections) + "Afterword by Bruce Schneier\nNOT NARRATIVE\n").encode()


def picture_book(title="Synthetic title"):
    return (
        f"---\ntitle: Synthetic\n---\n\n# {title}\n\n"
        + "".join(
            f"![IMAGE DESCRIPTION {n}]({{{{ site.image-set }}}}/{n:02d}.jpg)\n\n"
            + (f"Synthetic page {n}.\n\n" if n != 5 else "")
            for n in range(1, 13)
        )
    ).encode()


def source(raw, url=prep.LITTLE_BROTHER_URL):
    return {"url": url, "bytes": len(raw), "sha256": sha(raw)}


def cached(cache, name, raw, descriptor=None):
    descriptor = descriptor or source(raw)
    path = cache / "sources" / descriptor["sha256"] / name
    create_or_verify(path, [raw])
    create_or_verify(
        path.with_name(name + ".receipt.json"),
        [
            json_bytes(
                {
                    **descriptor,
                    "final_url": descriptor["url"],
                    "retrieved_at": "2026-09-27T00:00:00+00:00",
                }
            )
        ],
    )
    return path


def test_novel_retains_all_chapters_and_epilogue_without_non_narrative_text():
    sections = prep.little_brother(novel())
    assert len(sections) == 22
    assert sections[0]["section_id"] == "chapter-01"
    assert sections[-1]["section_id"] == "epilogue"
    assert sections[0]["text"] == "Synthetic narrative 1.\n\nAnother paragraph."
    lines = novel().decode().splitlines()
    for section in sections:
        start, end = section["source_lines"]
        assert prep.clean_lines(lines[start - 1 : end]) == section["text"]
        assert not any(s in section["text"] for s in ("Shop", "&&&", "NOT NARRATIVE", "license"))


@pytest.mark.parametrize(
    "before,after",
    [
        (b"Chapter 2\n", b"Chapter 1\n"),
        (b"[[Shop address]]", b"[[Shop address"),
        (b" &&&", b"missing delimiter"),
        (b"Afterword by Bruce Schneier", b"Unknown afterword"),
    ],
)
def test_novel_rejects_unreviewed_framing(before, after):
    with pytest.raises(ValueError):
        prep.little_brother(novel().replace(before, after, 1))


def test_picture_book_preserves_empty_page_without_image_alt_or_frontmatter():
    sections = prep.bookdash(picture_book())
    assert len(sections) == 12
    assert sections[4] == {
        "section_id": "page-05",
        "title": "Page 5",
        "source_lines": [23, 24],
        "text": "",
    }
    assert [s["section_id"] for s in sections] == [f"page-{i:02d}" for i in range(1, 13)]
    assert not any("IMAGE" in s["text"] or "title" in s["text"] for s in sections)


@pytest.mark.parametrize(
    "before,after",
    [
        (b"/02.jpg", b"/01.jpg"),
        (b"# Synthetic title", b"unexpected navigation"),
        (b"Synthetic page 1.", b"<script>unexpected</script>"),
        (b"Synthetic page 1.", b"[unexpected](https://example.org)"),
    ],
)
def test_picture_book_rejects_missing_pages_and_unexpected_markup(before, after):
    with pytest.raises(ValueError):
        prep.bookdash(picture_book().replace(before, after, 1))


def test_acquisition_defaults_offline_and_cached_bytes_and_receipts_are_verified(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prep.urllib.request, "build_opener", lambda *_: pytest.fail("Network used"))
    raw = novel()
    descriptor = source(raw)
    with pytest.raises(FileNotFoundError, match="--fetch"):
        prep.acquire("little-brother.txt", descriptor, tmp_path)
    path = cached(tmp_path, "little-brother.txt", raw)
    assert prep.acquire("little-brother.txt", descriptor, tmp_path)[0] == raw
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="bytes differ"):
        prep.acquire("little-brother.txt", descriptor, tmp_path)
    path.write_bytes(raw)
    receipt_path = path.with_name(path.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"] = "0" * 64
    receipt_path.write_bytes(json_bytes(receipt))
    with pytest.raises(ValueError, match="receipt differs"):
        prep.acquire("little-brother.txt", descriptor, tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://craphound.com/littlebrother/Cory_Doctorow_-_Little_Brother.txt",
        "https://craphound.com.evil.test/littlebrother/Cory_Doctorow_-_Little_Brother.txt",
        prep.LITTLE_BROTHER_URL + "?new-edition=yes",
        prep.BOOKDASH_BASE + "../../other-repository/book.md",
    ],
)
def test_acquisition_refuses_unpinned_hosts_paths_and_editions(tmp_path, url):
    with pytest.raises(ValueError, match="Invalid pinned"):
        prep.acquire("little-brother.txt", source(b"body", url), tmp_path, fetch=True)


def test_fetch_is_bounded_and_changed_remote_body_is_not_published(tmp_path, monkeypatch):
    descriptor = source(b"expected")

    class Response:
        url = descriptor["url"]
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, bound):
            assert bound == descriptor["bytes"] + 1
            return b"different"

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == descriptor["url"] and timeout == 30
            return Response()

    monkeypatch.setattr(prep.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="bytes differ"):
        prep.acquire("little-brother.txt", descriptor, tmp_path, fetch=True)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="redirect"):
        prep.PinnedRedirects().redirect_request(
            prep.urllib.request.Request(descriptor["url"]),
            None,
            302,
            "",
            {},
            "https://elsewhere.test/",
        )


def fixture_inventory(tmp_path):
    inventory = json.loads(prep.REGISTRY.read_text())
    texts = {"little-brother.txt": novel()}
    metadata = "titles:\n"
    for book in inventory["books"][1:]:
        texts[book["source"]] = picture_book(book["title"])
        metadata += (
            "  "
            + book["work_id"].removeprefix("bookdash-")
            + ":\n"
            + "    identifier: "
            + book["provider_id"]
            + "\n"
            + "    isbn: "
            + book["provider_isbn"]
            + "\n"
            + '    language: "en"\n'
            + '    creator: "'
            + ", ".join(book["creators"])
            + '"\n'
            + "    rights: http://creativecommons.org/licenses/by/4.0/\n"
        )
    texts["bookdash-meta.yml"] = metadata.encode()
    for name, descriptor in inventory["sources"].items():
        inventory["sources"][name] = source(texts[name], descriptor["url"])
        cached(tmp_path / "cache", name, texts[name], inventory["sources"][name])
    registry = tmp_path / "sources.json"
    registry.write_bytes(json_bytes(inventory))
    return registry, inventory, texts


def test_complete_preparation_has_work_groups_attribution_and_no_gold_or_splits(
    tmp_path, monkeypatch
):
    registry, _, _ = fixture_inventory(tmp_path)
    monkeypatch.setattr(prep.urllib.request, "build_opener", lambda *_: pytest.fail("Network used"))
    result = prep.prepare(registry, tmp_path / "cache")
    assert result == prep.prepare(registry, tmp_path / "cache")
    assert result["totals"]["works"] == 4 and result["totals"]["sections"] == 58
    assert result["candidate_status"] == "unadmitted" and result["training_performed"] is False
    assert result["preparation_script_sha256"] == prep.file_hash(Path(prep.__file__))
    assert result["source_inventory"]["sha256"] == prep.file_hash(registry)
    assert result["source_inventory"]["bytes"] == registry.stat().st_size
    assert result["training_authorized"] is False
    assert result["preparation_helper_sha256"] == prep.helper_hashes(prep.ROOT)
    assert "Synthetic narrative" not in json.dumps(result)
    for book in result["books"]:
        path = tmp_path / "cache" / book["artifact"]["path"]
        assert prep.file_hash(path) == book["artifact"]["sha256"]
        output = json.loads(path.read_text())
        assert output["work_group_id"] == output["work_id"]
        assert output["split"] is None and output["labels"] == {}
        assert output["creators"] and output["license"]["obligations"]
        assert output["source_page"].startswith("https://")
    first_path = tmp_path / "cache" / result["books"][0]["artifact"]["path"]
    first_path.write_text("do not overwrite")
    with pytest.raises(ValueError, match="Existing candidate artifact differs"):
        prep.prepare(registry, tmp_path / "cache")
    assert first_path.read_text() == "do not overwrite"


def test_license_evidence_and_work_ids_fail_closed(tmp_path):
    registry, inventory, texts = fixture_inventory(tmp_path)
    book = inventory["books"][1]
    book["license"]["id"] = "MIT"
    with pytest.raises(ValueError, match="license evidence"):
        prep.validate_license(book, texts)
    book["license"]["id"] = "CC-BY-4.0"
    texts["bookdash-meta.yml"] = texts["bookdash-meta.yml"].replace(b"by/4.0/", b"by-nc/4.0/")
    with pytest.raises(ValueError, match="license evidence"):
        prep.validate_license(book, texts)
    inventory["books"].append(inventory["books"][0])
    registry.write_bytes(json_bytes(inventory))
    with pytest.raises(ValueError, match="duplicate work"):
        prep.prepare(registry, tmp_path / "cache")


def test_title_license_and_source_remain_bound_to_the_same_work(tmp_path):
    _, inventory, texts = fixture_inventory(tmp_path)
    first, second = inventory["books"][1:3]
    first["source"] = second["source"]
    with pytest.raises(ValueError, match="edition/attribution/license"):
        prep.validate_license(first, texts)
    first["source"] = "little-ants-big-plan.md"
    texts[first["source"]] = texts[second["source"]]
    with pytest.raises(ValueError, match="edition/attribution/license"):
        prep.validate_license(first, texts)
    with pytest.raises(ValueError, match="Invalid pinned"):
        prep.acquire(first["source"], inventory["sources"][second["source"]], tmp_path)


def test_receipt_timestamp_and_size_are_bounded(tmp_path):
    raw = novel()
    path = cached(tmp_path, "little-brother.txt", raw)
    receipt_path = path.with_name(path.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["retrieved_at"] = "2026-09-27T00:00:00"
    receipt_path.write_bytes(json_bytes(receipt))
    with pytest.raises(ValueError, match="timezone"):
        prep.acquire(path.name, source(raw), tmp_path)
    receipt_path.write_text(" " * 32_001)
    with pytest.raises(ValueError, match="receipt exceeds"):
        prep.acquire(path.name, source(raw), tmp_path)


@pytest.mark.parametrize("replacement", [[], ["Candice Dingwall"], ["Candice Dingwall"] * 3])
def test_complete_creator_attribution_is_required(tmp_path, replacement):
    _, inventory, texts = fixture_inventory(tmp_path)
    book = inventory["books"][1]
    book["creators"] = replacement
    with pytest.raises(ValueError, match="attribution/license"):
        prep.validate_license(book, texts)


def test_duplicate_json_keys_and_boolean_schema_rejected(tmp_path):
    registry, inventory, _ = fixture_inventory(tmp_path)
    inventory["schema_version"] = True
    registry.write_bytes(json_bytes(inventory))
    with pytest.raises(ValueError, match="schema"):
        prep.prepare(registry, tmp_path / "cache")
    registry.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        prep.prepare(registry, tmp_path / "cache")
    raw = novel()
    path = cached(tmp_path / "other-cache", "little-brother.txt", raw)
    path.with_name(path.name + ".receipt.json").write_text('{"url": "a", "url": "b"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        prep.acquire(path.name, source(raw), tmp_path / "other-cache")
