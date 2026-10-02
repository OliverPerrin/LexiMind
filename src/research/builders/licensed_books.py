"""Prepare four pinned, attributed book texts locally; offline unless --fetch.

The extractors deliberately support only the reviewed source editions. Full text,
license evidence and acquisition receipts stay in ignored candidate storage.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.catalog.storage import write_json_atomic
from src.research.candidate_io import create_or_verify, helper_hashes, json_bytes, sha
from src.research.io import check_file, file_hash, read_json, safe_path

from . import ROOT

REGISTRY = ROOT / "research/preparation/licensed_books_sources.json"
CACHE = ROOT / "data/research_candidates/licensed_books"
MANIFEST = ROOT / "research/preparation/licensed_books_manifest.json"
RPT_MANIFEST = ROOT / "research/preparation/rpt_candidate_manifest.json"
RPT_POLICY = "licensed-book-normalized-continuation-v1"
MAX_SOURCE_BYTES = 2_000_000
BOOKDASH_COMMIT = "54916310c5c06a5282e2d5d4add18d64f90e5fde"
BOOKDASH_BASE = f"https://raw.githubusercontent.com/bookdash/bookdash-books/{BOOKDASH_COMMIT}/"
LITTLE_BROTHER_URL = "https://craphound.com/littlebrother/Cory_Doctorow_-_Little_Brother.txt"


SOURCE_URLS = {
    "little-brother.txt": LITTLE_BROTHER_URL,
    "bookdash-meta.yml": BOOKDASH_BASE + "_data/meta.yml",
    **{
        slug + ".md": BOOKDASH_BASE + slug + "/en/index.md"
        for slug in ("little-ants-big-plan", "sizwes-smile", "why-is-nita-upside-down")
    },
}


class PinnedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if newurl != req.full_url:
            raise ValueError("Pinned sources must not redirect; review the new location")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verified_bytes(raw, source):
    if len(raw) != source["bytes"] or sha(raw) != source["sha256"]:
        raise ValueError("Source bytes differ from the reviewed edition")
    return raw


def acquire(name, source, cache, *, fetch=False):
    """Verify existing bytes and their receipt, or acquire exactly one pinned file."""
    if (
        Path(name).name != name
        or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        or type(source["bytes"]) is not int
        or not 0 < source["bytes"] <= MAX_SOURCE_BYTES
        or source["url"] != SOURCE_URLS.get(name)
    ):
        raise ValueError("Invalid pinned source descriptor")
    path = cache / "sources" / source["sha256"] / name
    receipt_path = path.with_name(name + ".receipt.json")
    if path.exists():
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("Cached source exceeds byte bound")
        raw = verified_bytes(path.read_bytes(), source)
        if receipt_path.stat().st_size > 32_000:
            raise ValueError("Acquisition receipt exceeds byte bound")
        receipt = read_json(receipt_path)
        if any(receipt.get(k) != source[k] for k in ("url", "bytes", "sha256")):
            raise ValueError("Acquisition receipt differs from pinned source")
        if receipt.get("final_url") != source["url"] or not receipt.get("retrieved_at"):
            raise ValueError("Acquisition receipt lacks source or time evidence")
        if datetime.fromisoformat(receipt["retrieved_at"]).tzinfo is None:
            raise ValueError("Acquisition receipt timestamp must include a timezone")
    else:
        if not fetch:
            raise FileNotFoundError(f"Missing pinned source {name}; use --fetch to acquire it")
        request = urllib.request.Request(
            source["url"],
            headers={
                "User-Agent": "LexiMind research preparation (https://github.com/OliverPerrin/LexiMind)",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.build_opener(PinnedRedirects()).open(request, timeout=30) as response:
            raw = verified_bytes(response.read(source["bytes"] + 1), source)
            if response.url != source["url"]:
                raise ValueError("Source URL changed")
            receipt = {
                **source,
                "final_url": response.url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
        create_or_verify(path, [raw])
        create_or_verify(receipt_path, [json_bytes(receipt)])
    return raw, {
        "path": str(path.relative_to(cache)),
        **source,
        "receipt": {
            "path": str(receipt_path.relative_to(cache)),
            "sha256": file_hash(receipt_path),
        },
    }


def clean_lines(lines):
    """Normalize line endings/trailing spaces and excess blank lines, not prose."""
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in lines)).strip()


def little_brother(raw):
    lines = raw.decode("utf-8").splitlines()
    headings = [
        (i, line) for i, line in enumerate(lines) if re.fullmatch(r"Chapter [0-9]+|Epilogue", line)
    ]
    if [title for _, title in headings] != [*(f"Chapter {i}" for i in range(1, 22)), "Epilogue"]:
        raise ValueError("Little Brother chapter order differs from reviewed edition")
    if lines.count("Afterword by Bruce Schneier") != 1:
        raise ValueError("Missing unique afterword boundary")
    stop = lines.index("Afterword by Bruce Schneier")
    if stop <= headings[-1][0]:
        raise ValueError("Afterword precedes narrative")
    sections = []
    for index, (start, title) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else stop
        first, last = start + 1, end
        for _ in range(2):
            while first < last and not lines[first].strip():
                first += 1
            if first == last or not lines[first].startswith("[["):
                raise ValueError("Missing expected bookstore dedication")
            while first < last and not lines[first].endswith("]]"):
                first += 1
            if first == last:
                raise ValueError("Unclosed bookstore dedication")
            first += 1
        while first < last and not lines[first].strip():
            first += 1
        while last > first and not lines[last - 1].strip():
            last -= 1
        if last == first or lines[last - 1].strip() != "&&&":
            raise ValueError("Missing narrative section end delimiter")
        last -= 1
        while last > first and not lines[last - 1].strip():
            last -= 1
        text = clean_lines(lines[first:last])
        if not text or "[[" in text or "]]" in text or "&&&" in text:
            raise ValueError("Unexpected narrative framing")
        sections.append(
            {
                "section_id": f"chapter-{index + 1:02d}" if index < 21 else "epilogue",
                "title": title,
                "source_lines": [first + 1, last],
                "text": text,
            }
        )
    return sections


def bookdash(raw):
    lines = raw.decode("utf-8").splitlines()
    if not lines or lines[0] != "---" or lines.count("---") != 2:
        raise ValueError("Unexpected Markdown frontmatter framing")
    front_end = lines.index("---", 1)
    image = re.compile(r"!\[[^\n]*\]\(\{\{ site.image-set \}\}/([0-9]{2})\.jpg\)")
    pages = [(i, match[1]) for i, line in enumerate(lines) if (match := image.fullmatch(line))]
    if [page for _, page in pages] != [f"{i:02d}" for i in range(1, 13)]:
        raise ValueError("Book Dash page order differs from reviewed edition")
    prefix = clean_lines(lines[front_end + 1 : pages[0][0]])
    if not re.fullmatch(r"# [^\n]+", prefix):
        raise ValueError("Unexpected content before first narrative page")
    sections = []
    for index, (start, number) in enumerate(pages):
        end = pages[index + 1][0] if index + 1 < len(pages) else len(lines)
        text = clean_lines(lines[start + 1 : end])
        if any(marker in text for marker in ("![", "{{", "}}", "<", "](", "{%", "# ", "---")):
            raise ValueError("Unsupported markup in pinned Book Dash narrative")
        sections.append(
            {
                "section_id": f"page-{number}",
                "title": f"Page {int(number)}",
                "source_lines": [start + 1, end],
                "text": text,
            }
        )
    return sections


def validate_license(book, sources):
    evidence = sources[book["license"]["evidence_source"]].decode("utf-8")
    if book["extractor"] == "little_brother_txt":
        if (
            book["source"] != "little-brother.txt"
            or book["work_id"] != "cory-doctorow-little-brother"
            or book["title"] != "Little Brother"
            or book["creators"] != ["Cory Doctorow"]
            or book["license"]["evidence_source"] != "little-brother.txt"
            or book["license"]["id"] != "CC-BY-NC-SA-3.0"
            or book["license"]["url"] != "https://creativecommons.org/licenses/by-nc-sa/3.0/"
            or not evidence.startswith("Little Brother\n\nCory Doctorow\n")
            or (
                "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license." not in evidence
            )
        ):
            raise ValueError("Little Brother title/attribution/license evidence changed")
    elif book["extractor"] == "bookdash_markdown":
        slug = book["work_id"].removeprefix("bookdash-")
        headings = re.findall(r"^# (.+)$", sources[book["source"]].decode("utf-8"), re.M)
        expected_title = book["title"].replace("’", "'").casefold()
        match = re.search(
            r"^  " + re.escape(slug) + r":\n(.*?)(?=^  [a-z]|\Z)", evidence, re.M | re.S
        )
        credit_fields = re.findall(r'^    creator: "([^"\n]+)"', match[1], re.M) if match else []
        # These three reviewed source fields use comma/"and" separators. Bind
        # the complete list, so dropping a creator cannot silently lose credit.
        credits = re.split(r", | and ", credit_fields[0]) if len(credit_fields) == 1 else []
        if (
            book["source"] != slug + ".md"
            or book["license"]["evidence_source"] != "bookdash-meta.yml"
            or [h.replace("’", "'").casefold() for h in headings] != [expected_title]
            or book["license"]["id"] != "CC-BY-4.0"
            or book["license"]["url"] != "https://creativecommons.org/licenses/by/4.0/"
            or not credits
            or len(set(credits)) != len(credits)
            or book["creators"] != credits
            or match is None
            or not all(
                token in match[1]
                for token in (
                    book["provider_id"],
                    book["provider_isbn"],
                    'language: "en"',
                    "http://creativecommons.org/licenses/by/4.0/",
                )
            )
        ):
            raise ValueError("Book Dash edition/attribution/license evidence changed")
    else:
        raise ValueError("Unsupported source extractor")


def prepare(registry=REGISTRY, cache=CACHE, *, fetch=False):
    inventory = read_json(registry)
    if type(inventory["schema_version"]) is not int or inventory["schema_version"] != 1:
        raise ValueError("Unsupported inventory schema")
    ids = [book["work_id"] for book in inventory["books"]]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9-]+", i) for i in ids):
        raise ValueError("Invalid or duplicate work identities")
    raw, references = {}, {}
    for name, source in inventory["sources"].items():
        raw[name], references[name] = acquire(name, source, cache, fetch=fetch)
    script_hash, helpers = file_hash(Path(__file__)), helper_hashes(ROOT)
    build_key = sha(
        json_bytes({"inventory": file_hash(registry), "script": script_hash, "helpers": helpers})
    )
    output_root = cache / "prepared" / build_key
    books = []
    for book in inventory["books"]:
        validate_license(book, raw)
        extractor = little_brother if book["extractor"] == "little_brother_txt" else bookdash
        sections = extractor(raw[book["source"]])
        if len(sections) != book["expected_sections"]:
            raise ValueError("Unexpected section count")
        record = {
            "schema_version": 1,
            **book,
            "work_group_id": book["work_id"],
            "split": None,
            "labels": {},
            "candidate_status": "unadmitted",
            "changes": book["excluded"]
            + "; normalize line endings/trailing whitespace and collapse excess blank lines",
            "source_sha256": inventory["sources"][book["source"]]["sha256"],
            "sections": sections,
        }
        path = output_root / (book["work_id"] + ".json")
        artifact = create_or_verify(path, [json_bytes(record)])
        books.append(
            {
                "work_id": book["work_id"],
                "title": book["title"],
                "domain": book["domain"],
                "license": book["license"]["id"],
                "work_group_id": book["work_id"],
                "source": book["source"],
                "sections": len(sections),
                "empty_sections": [s["section_id"] for s in sections if not s["text"]],
                "whitespace_words": sum(len(s["text"].split()) for s in sections),
                "body_utf8_bytes": sum(len(s["text"].encode()) for s in sections),
                "artifact": {"path": str(path.relative_to(cache)), **artifact},
            }
        )
    return {
        "schema_version": 1,
        "candidate_status": "unadmitted",
        "training_performed": False,
        "training_authorized": False,
        "preparation_script": "src/research/builders/licensed_books.py",
        "preparation_script_sha256": script_hash,
        "preparation_helper_sha256": helpers,
        "source_inventory": {
            "path": str(registry.relative_to(ROOT))
            if registry.is_relative_to(ROOT)
            else str(registry),
            "bytes": registry.stat().st_size,
            "sha256": file_hash(registry),
        },
        "cache_root": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
        "build_key": build_key,
        "sources": references,
        "books": books,
        "totals": {
            "works": len(books),
            "sections": sum(b["sections"] for b in books),
            "source_bytes": sum(s["bytes"] for s in inventory["sources"].values()),
            "body_utf8_bytes": sum(b["body_utf8_bytes"] for b in books),
            "whitespace_words": sum(b["whitespace_words"] for b in books),
            "domains": dict(Counter(b["domain"] for b in books)),
        },
        "split_policy": "No split assigned. Keep every chapter/page and future edition of a work in one group; match catalogue/BGC identities before admission.",
        "limitations": [
            "Four selected English works are preparation examples, not a representative corpus or evaluation set.",
            "One long-form YA novel dominates text volume; three children's picture books rely on omitted images. No adult-fiction or general-nonfiction coverage.",
            "Source spelling and punctuation are retained, including source typos; no OCR, inferred text, summaries, topic/genre labels or mood gold.",
            "Licenses stay per work. Little Brother has NC and SA conditions; MIT repository licensing does not replace source licenses or clear every future training use.",
            "Book Dash sections follow image/page markers (including an empty page), not inferred semantic chunks; source line spans include page markers and whitespace.",
        ],
    }


def _rpt_window(tokenizer, ids, cut, special_ids):
    """Accept only independently round-tripping prompts and stable target prefixes."""
    first = max(0, cut - 128)
    prompt = None
    for start in range(first, min(first + 16, cut - 8 + 1)):
        prompt_ids = ids[start:cut]
        text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if (
            not special_ids.intersection(prompt_ids)
            and tokenizer.encode(text, add_special_tokens=False).ids == prompt_ids
        ):
            prompt = (start, prompt_ids, text)
            break
    if prompt is None:
        return None
    for end in range(min(len(ids), cut + 32), cut + 3, -1):
        target_ids = ids[cut:end]
        if special_ids.intersection(target_ids):
            continue
        text = tokenizer.decode(target_ids, skip_special_tokens=False)
        if (
            not text
            or "\ufffd" in text
            or tokenizer.encode(text, add_special_tokens=False).ids != target_ids
        ):
            continue
        previous, pieces = "", []
        for count in range(1, len(target_ids) + 1):
            prefix = tokenizer.decode(target_ids[:count], skip_special_tokens=False)
            if (
                not prefix.startswith(previous)
                or len(prefix) <= len(previous)
                or "\ufffd" in prefix
            ):
                break
            pieces.append(prefix[len(previous) :])
            previous = prefix
        else:
            if previous == text:
                return {
                    "prompt_token_span": [prompt[0], cut],
                    "target_token_span": [cut, end],
                    "prompt_ids": prompt[1],
                    "target_ids": target_ids,
                    "prompt_normalized_text": prompt[2],
                    "observed_bytes": text,
                    "token_bytes": pieces,
                }
    return None


def prepare_rpt(tokenizer_path: Path, *, positions_per_work=8, root=ROOT):
    """Tokenize existing licensed candidates offline; never repin base preparations."""
    if type(positions_per_work) is not int or not 8 <= positions_per_work <= 16:
        raise ValueError("RPT positions per work must be an integer from 8 to 16")
    root, tokenizer_path = root.resolve(), tokenizer_path.resolve()
    if not tokenizer_path.is_relative_to(root) or tokenizer_path.stat().st_size > 10_000_000:
        raise ValueError("RPT tokenizer must be a bounded local repository artifact")

    def reference(path):
        return {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }

    licensed_path = root / "research/preparation/licensed_books_manifest.json"
    partition_path = root / "research/preparation/book_partition_manifest.json"
    inputs = {
        "licensed_manifest": reference(licensed_path),
        "partition_manifest": reference(partition_path),
        "tokenizer": reference(tokenizer_path),
    }
    licensed, partitions = read_json(licensed_path), read_json(partition_path)
    if (
        partitions.get("policy") != "book-component-overlay-v1"
        or partitions["inputs"]["licensed_manifest"] != inputs["licensed_manifest"]
    ):
        raise ValueError("RPT requires the effective overlay for the exact licensed manifest")
    if partitions["inputs"]["licensed_sources"] != licensed["source_inventory"]:
        raise ValueError("RPT source inventory differs from the effective partition evidence")
    inputs.update(
        {"components": partitions["components"], "source_inventory": licensed["source_inventory"]}
    )
    cache = safe_path(root, licensed["cache_root"])
    for book in licensed["books"]:
        artifact = dict(book["artifact"])
        artifact["path"] = str(safe_path(cache, artifact["path"]).relative_to(root))
        key = "work:" + book["work_id"]
        if key in inputs:
            raise ValueError("RPT licensed work IDs repeat")
        inputs[key] = artifact
    if not 1 <= len(licensed["books"]) <= 4:
        raise ValueError("RPT preparation is bounded to the four reviewed licensed works")
    for item in inputs.values():
        if errors := check_file(root, item):
            raise ValueError("; ".join(errors))
    components = {}
    with safe_path(root, inputs["components"]["path"]).open() as stream:
        for number, line in enumerate(stream, 1):
            if number > 10_000 or len(line) > 256_000:
                raise ValueError("RPT component overlay exceeds bounds")
            component = json.loads(line)
            for member in component["members"]:
                if member in components:
                    raise ValueError("RPT identity appears in multiple components")
                components[member] = component
    # Lazy import keeps source extraction and the research CLI independent of ML stacks.
    import tokenizers

    model_config = read_json(tokenizer_path)["model"]
    if model_config.get("dropout") not in (None, 0, 0.0):
        raise ValueError("RPT tokenizer must disable stochastic BPE dropout")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    special_ids = {
        identifier
        for identifier, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    unknown_id = model_config.get("unk_id")
    if unknown_id is None and model_config.get("unk_token"):
        unknown_id = tokenizer.token_to_id(model_config["unk_token"])
    if unknown_id is not None:
        special_ids.add(unknown_id)
    configuration = {
        "policy": RPT_POLICY,
        "positions_per_work": positions_per_work,
        "context_tokens": 128,
        "continuation_tokens": 32,
        "minimum_context_tokens": 8,
        "minimum_continuation_tokens": 4,
        "attempts_per_position": 16,
        "tokenizers_version": tokenizers.__version__,
        "special_tokens_added": False,
        "tokenizer_padding_and_truncation": "disabled_for_source_boundary_detection",
    }
    implementation = {
        "src/research/builders/licensed_books.py": file_hash(Path(__file__)),
        **helper_hashes(ROOT),
    }
    build_key = sha(
        json_bytes(
            {"inputs": inputs, "configuration": configuration, "implementation": implementation}
        )
    )
    records, works = [], []
    for book in licensed["books"]:
        work_id = book["work_id"]
        component = components.get("licensed_text:" + work_id)
        if component is None:
            raise ValueError("RPT licensed work has no effective component assignment")
        split = component["proposed_split"]
        if split not in {"train", "dev", "test", None} or (
            split is None and component["status"] != "quarantined_title_only_ambiguity"
        ):
            raise ValueError("RPT requires an explicit work split or quarantine")
        work = {
            "work_id": work_id,
            "work_group": component["component_id"],
            "split": split or "quarantine",
            "records": 0,
            "skipped_windows": 0,
            "sections": [],
        }
        works.append(work)
        if split is None:
            continue
        artifact = inputs["work:" + work_id]
        prepared = read_json(safe_path(root, artifact["path"]))
        if (
            prepared["work_id"] != work_id
            or prepared["work_group_id"] != book["work_group_id"]
            or prepared["license"]["id"] != book["license"]
        ):
            raise ValueError("RPT prepared work identity or license differs from its manifest")
        sections, total = [], 0
        for section in prepared["sections"]:
            ids = tokenizer.encode(section["text"], add_special_tokens=False).ids
            normalized = tokenizer.decode(ids, skip_special_tokens=False)
            count = max(0, len(ids) - 8 - 4 + 1)
            sections.append((section, ids, total, count, sha(normalized)))
            total += count
            work["sections"].append(
                {
                    "section_id": section["section_id"],
                    "source_text_sha256": sha(section["text"]),
                    "normalized_text_sha256": sha(normalized),
                    "source_bytes": len(section["text"].encode()),
                    "normalized_bytes": len(normalized.encode()),
                    "tokens": len(ids),
                }
            )
        selected, attempted = set(), set()
        for anchor in range(positions_per_work):
            for position in range(
                total * anchor // positions_per_work,
                min(total, total * anchor // positions_per_work + 16),
            ):
                if position in attempted:
                    continue
                attempted.add(position)
                section, ids, offset, _, normalized_hash = next(
                    row for row in sections if row[2] <= position < row[2] + row[3]
                )
                cut = position - offset + 8
                window = _rpt_window(tokenizer, ids, cut, special_ids)
                if window is None:
                    work["skipped_windows"] += 1
                    continue
                identity = (section["section_id"], tuple(window["target_token_span"]))
                if identity in selected:
                    continue
                selected.add(identity)
                evidence = {
                    "artifact": artifact,
                    "source_sha256": prepared["source_sha256"],
                    "section_id": section["section_id"],
                    "source_lines": section["source_lines"],
                    "source_section_sha256": sha(section["text"]),
                    "normalized_section_sha256": normalized_hash,
                    "prompt_token_span": window["prompt_token_span"],
                    "target_token_span": window["target_token_span"],
                    "partition_manifest_sha256": inputs["partition_manifest"]["sha256"],
                }
                evidence_hash = sha(json_bytes(evidence))
                records.append(
                    {
                        "schema_version": 1,
                        "record_id": "book-rpt:" + sha(f"{build_key}:{work_id}:{evidence_hash}"),
                        "candidate_status": "unadmitted",
                        "training_authorized": False,
                        "work_id": work_id,
                        "source_evidence": evidence,
                        "attribution": {
                            key: prepared[key]
                            for key in ("title", "creators", "source_page", "license")
                        },
                        "prompt_ids": window["prompt_ids"],
                        "prompt_normalized_text": window["prompt_normalized_text"],
                        "target_ids": window["target_ids"],
                        "continuation_target": {
                            "byte_encoding": "utf-8",
                            "observed_bytes": window["observed_bytes"],
                            "token_bytes": window["token_bytes"],
                            "tokenizer_sha256": inputs["tokenizer"]["sha256"],
                            "source_evidence_sha256": evidence_hash,
                            "work_group": component["component_id"],
                            "split": split,
                        },
                    }
                )
                work["records"] += 1
                break
    for item in inputs.values():
        if errors := check_file(root, item):
            raise ValueError("RPT inputs changed during tokenization: " + "; ".join(errors))
    path = root / "data/research_candidates/licensed_books/rpt" / build_key / "continuations.jsonl"
    artifact = create_or_verify(
        path,
        (
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            for row in records
        ),
    )
    return {
        "schema_version": 1,
        "candidate_status": "unadmitted",
        "training_authorized": False,
        "training_performed": False,
        "model_inference_performed": False,
        "configuration": configuration,
        "inputs": inputs,
        "implementation_sha256": implementation,
        "build_key": build_key,
        "continuations": {"path": str(path.relative_to(root)), **artifact},
        "counts": {
            "records": len(records),
            "by_split": dict(Counter(row["continuation_target"]["split"] for row in records)),
            "works_by_split": dict(Counter(work["split"] for work in works)),
            "skipped_non_roundtripping_or_unstable_windows": sum(
                work["skipped_windows"] for work in works
            ),
        },
        "works": works,
        "contract": {
            "text_basis": "Independent prompt and continuation decode of original section token IDs, without special tokens; encode(decode(ids)) must equal ids. This is explicitly tokenizer-normalized round-trip text, not raw source bytes or guessed raw byte offsets.",
            "byte_encoding": "UTF-8: encode observed_bytes and each token_bytes string before constructing ContinuationTarget. Concatenated token bytes exactly reconstruct observed bytes. Every nonempty token piece is verified by cumulative decoder prefix extension.",
            "partition_policy": "Use effective whole-work components only; quarantine emits zero rows. Train/dev/test remain explicit and cannot be pooled for training. All rows still require separate dataset/run admission.",
            "sampling": "Up to the requested deterministic evenly spaced anchors per work, with at most sixteen subsequent token positions per anchor. Context stays in one section and at most 128 tokens; observed continuation is 4 to 32 tokens. Non-round-tripping, special/unknown-token or unstable-prefix windows are skipped.",
            "attribution": "Each row retains original creators/source/license. Repository MIT terms do not replace per-work source restrictions.",
            "scope": "Small book-domain adaptation for future RPT-style prefix verification, not a reproduction of the paper's 14B OmniMATH experiment or evidence of quality gains. No checkpoints, teachers, inference, rollout, optimizer or training is invoked.",
        },
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fetch", action="store_true", help="Fetch missing pinned files only")
    parser.add_argument(
        "--rpt-tokenizer",
        type=Path,
        help="Prepare a separate offline RPT candidate from existing licensed/partition manifests with this local tokenizer JSON",
    )
    parser.add_argument(
        "--rpt-positions-per-work",
        type=int,
        default=8,
        help="RPT candidate positions per work, 8 to 16 (default: 8)",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.rpt_tokenizer is not None:
        if args.fetch:
            parser.error("RPT preparation uses existing pinned candidates only; omit --fetch")
        manifest = prepare_rpt(args.rpt_tokenizer, positions_per_work=args.rpt_positions_per_work)
        write_json_atomic(RPT_MANIFEST, manifest)
        print(json.dumps(manifest["counts"], indent=2))
        return 0
    if args.rpt_positions_per_work != 8:
        parser.error("--rpt-positions-per-work requires --rpt-tokenizer")
    manifest = prepare(fetch=args.fetch)
    write_json_atomic(MANIFEST, manifest)
    print(json.dumps(manifest["totals"], indent=2))
    return 0
