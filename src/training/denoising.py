"""Source-bound short-word reconstruction; no semantic grading or model selection."""

from __future__ import annotations

import re
from pathlib import Path

from tokenizers import Tokenizer

from src.research.candidate_io import json_bytes, sha
from src.research.io import check_file, file_hash, read_json, safe_path

INSTRUCTION = "Restore the missing word in this book passage. Return only that word.\n\n"
SENTINEL = "<extra_id_0>"
POLICY = "bookdash-exact-word-infill-v1"
MAX_ROWS_PER_WORK = 8
_WORD = re.compile(r"(?<![\w'’])[^\W\d_]+(?![\w'’])")


def _reference(root: Path, path: Path) -> dict:
    path = path.resolve() if path.is_absolute() else safe_path(root, str(path))
    if not path.is_relative_to(root):
        raise ValueError("Denoising inputs must remain inside the repository")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def _checked(root: Path, reference: dict) -> Path:
    if errors := check_file(root, reference):
        raise ValueError("Denoising source changed: " + "; ".join(errors))
    return safe_path(root, reference["path"])


def _tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if (
        tokenizer.token_to_id("<pad>") != 0
        or tokenizer.token_to_id("</s>") != 1
        or tokenizer.token_to_id("<unk>") != 2
        or tokenizer.token_to_id(SENTINEL) is None
        or read_json(path).get("model", {}).get("dropout")
    ):
        raise ValueError("Denoising requires deterministic FLAN tokenization and special IDs")
    return tokenizer


def _section_windows(text: str, tokenizer: Tokenizer) -> list[dict]:
    """Enumerate deterministic source candidates; selection never sees model outputs."""
    words = list(_WORD.finditer(text))
    special = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    special.add(2)
    candidates: list[dict] = []
    if SENTINEL in text:
        return candidates
    for index, word in enumerate(words):
        target = word.group()
        target_ids = tokenizer.encode(target, add_special_tokens=False).ids
        if (
            len(target) < 4
            or not target.isalpha()
            or not 1 <= len(target_ids) <= 3
            or special.intersection(target_ids)
            or tokenizer.decode(target_ids, skip_special_tokens=False) != target
        ):
            continue
        # Both sides contain source context. Shrink long contexts at word boundaries,
        # never by cutting the answer or including any of it in the masked input.
        first, last = max(0, index - 16), min(len(words) - 1, index + 16)
        while first < index < last:
            left, right = words[first].start(), words[last].end()
            visible = text[left : word.start()] + SENTINEL + text[word.end() : right]
            input_text = INSTRUCTION + visible
            context_ids = tokenizer.encode(visible, add_special_tokens=False).ids
            input_ids = tokenizer.encode(input_text, add_special_tokens=False).ids + [1]
            if len(context_ids) <= 128 and len(input_ids) <= 160:
                break
            if index - first >= last - index:
                first += 1
            else:
                last -= 1
        else:
            continue
        content_ids = tokenizer.encode(
            text[left : word.start()] + text[word.end() : right], add_special_tokens=False
        ).ids
        if len(content_ids) < 12 or special.intersection(content_ids):
            continue
        # Case folding is only an exclusion filter. The verifier never normalizes
        # an answer, and words from the fixed instruction cannot reveal the target.
        if target.casefold() in {match.group().casefold() for match in _WORD.finditer(input_text)}:
            continue
        candidates.append(
            {
                "input_text": input_text,
                "input_ids": input_ids,
                "target_text": target,
                "target_ids": target_ids,
                "labels": target_ids + [1],
                "window_char_span": [left, right],
                "target_char_span": [word.start(), word.end()],
                "context_tokens": len(content_ids),
            }
        )
    return candidates


def prepare_denoising_data(root: Path, manifest_path: Path, tokenizer_path: Path) -> dict:
    """Reconstruct used source artifacts; never open test/quarantine book content."""
    from src.research.builders.bookdash import POLICY as SOURCE_POLICY
    from src.research.builders.bookdash import RAW, acquire, assign_splits, prepare_book

    root = root.resolve()
    references = {
        "manifest": _reference(root, Path(manifest_path)),
        "tokenizer": _reference(root, Path(tokenizer_path)),
    }
    manifest = read_json(_checked(root, references["manifest"]))
    if manifest.get("policy") != SOURCE_POLICY or manifest.get("schema_version") != 1:
        raise ValueError("Denoising requires the independent Book Dash cohort")
    tokenizer = _tokenizer(_checked(root, references["tokenizer"]))
    for path, expected in manifest["implementation_sha256"].items():
        if file_hash(safe_path(root, path)) != expected:
            raise ValueError("Denoising source implementation changed")
    # Verify shared provenance without reading the held-out books' source files,
    # acquisition receipts, or prepared narrative artifacts.
    for name, reference in manifest["inputs"].items():
        if not name.startswith(("source:", "receipt:")):
            _checked(root, reference)
            references[name] = reference
    registry = read_json(safe_path(root, references["registry"]["path"]))
    metadata = safe_path(root, references["metadata"]["path"]).read_bytes()
    if any(registry[key] != references[key] for key in ("metadata", "tree")):
        raise ValueError("Denoising metadata differs from the pinned registry")
    registered = {"bookdash-" + book["slug"]: book for book in registry["books"]}
    if len(registered) != len(registry["books"]):
        raise ValueError("Duplicate source work in the Book Dash registry")
    expected_splits = assign_splits(
        manifest["books"],
        {book["work_id"]: book["matches"] for book in manifest["books"] if book["matches"]},
    )
    roles: dict[str, str] = {}
    groups: dict[str, str] = {}
    for book in manifest["books"]:
        work_id, group_id, split = (book[key] for key in ("work_id", "group_id", "proposed_split"))
        if (
            work_id in roles
            or work_id not in registered
            or not isinstance(group_id, str)
            or group_id != "bookdash-work:" + sha(work_id)
            or split not in {"train", "dev", "test", "quarantine"}
            or split != expected_splits.get(work_id, "quarantine")
            or group_id in groups
            and groups[group_id] != split
        ):
            raise ValueError("Denoising work identities or group splits conflict")
        roles[work_id], groups[group_id] = split, split
    if set(roles) != set(registered):
        raise ValueError("Denoising manifest does not cover its source registry")
    rows: dict[str, list[dict]] = {"train": [], "dev": []}
    counts, shortfalls = {}, {}
    for book in sorted(manifest["books"], key=lambda book: book["work_id"]):
        work_id, group_id, split = (book[key] for key in ("work_id", "group_id", "proposed_split"))
        if split not in rows:
            continue
        source_book = registered[work_id]
        slug = source_book["slug"]
        source = manifest["inputs"]["source:" + slug]
        if any(source[key] != source_book["source"][key] for key in ("bytes", "sha256")):
            raise ValueError("Denoising work source differs from the source registry")
        receipt = manifest["inputs"]["receipt:" + slug]
        _checked(root, receipt)
        raw, observed_receipt = acquire(root, source, RAW + slug + "/en/index.md")
        if observed_receipt != receipt:
            raise ValueError("Denoising source acquisition receipt changed")
        references.update({"source:" + slug: source, "receipt:" + slug: receipt})
        work = read_json(_checked(root, book["artifact"]))
        expected = {
            "schema_version": 1,
            **prepare_book(source_book, raw, metadata),
            "group_id": group_id,
            "proposed_split": split,
            "candidate_status": "unadmitted",
            "training_authorized": False,
        }
        if work != expected or book["license"] != work["license"]["id"]:
            raise ValueError("Denoising artifact does not reconstruct its pinned source")
        references["work:" + work_id] = book["artifact"]
        candidates = []
        section_ids = set()
        for section in work["sections"]:
            section_id = section["section_id"]
            if section_id in section_ids:
                raise ValueError("Denoising work repeats a source section ID")
            section_ids.add(section_id)
            for window in _section_windows(section["text"], tokenizer):
                left, right = window["window_char_span"]
                evidence = {
                    "manifest": references["manifest"],
                    "artifact": book["artifact"],
                    "source": source,
                    "source_section_sha256": sha(section["text"]),
                    "section_id": section_id,
                    "source_lines": section["source_lines"],
                    "window_char_span": window["window_char_span"],
                    "target_char_span": window["target_char_span"],
                    "window_sha256": sha(section["text"][left:right]),
                    "target_sha256": sha(window["target_text"]),
                    "offset_unit": "unicode_codepoint_in_prepared_section",
                    "tokenizer": references["tokenizer"],
                }
                identity = sha(
                    json_bytes(
                        [POLICY, work_id, section_id, source["sha256"], window["target_char_span"]]
                    )
                )
                candidates.append(
                    {
                        **window,
                        "record_id": identity,
                        "work_id": work_id,
                        "group_id": group_id,
                        "source_split": split,
                        "source_evidence": evidence,
                        "source_evidence_sha256": sha(json_bytes(evidence)),
                        "attribution": {
                            key: work[key]
                            for key in ("title", "creators", "source_page", "license")
                        },
                    }
                )
        selected: list[dict] = []
        for candidate in sorted(candidates, key=lambda row: row["record_id"]):
            evidence = candidate["source_evidence"]
            left, right = evidence["window_char_span"]
            if any(
                previous["source_evidence"]["section_id"] == evidence["section_id"]
                and max(left, previous["window_char_span"][0])
                < min(right, previous["window_char_span"][1])
                for previous in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == MAX_ROWS_PER_WORK:
                break
        rows[split].extend(selected)
        counts[work_id] = len(selected)
        if len(selected) < MAX_ROWS_PER_WORK:
            shortfalls[work_id] = MAX_ROWS_PER_WORK - len(selected)
    return {
        **rows,
        "provenance": {
            "policy": POLICY,
            "inputs": references,
            "instruction": INSTRUCTION,
            "work_roles": roles,
            "counts": {role: len(values) for role, values in rows.items()},
            "records_per_work": counts,
            "shortfalls_from_eight": shortfalls,
            "global_test_used": False,
            "study_admission": False,
            "configuration": {
                "max_rows_per_work": MAX_ROWS_PER_WORK,
                "min_target_letters": 4,
                "target_tokens": [1, 3],
                "min_context_tokens": 12,
                "max_context_tokens_including_sentinel": 128,
                "max_input_tokens_including_eos": 160,
                "selection": "ascending deterministic record SHA-256; disjoint source windows",
            },
        },
    }


def denoising_outcomes(rows, generated, tokenizer: Tokenizer) -> list[dict]:
    """Exact full-word reward, requiring EOS and canonical valid content tokens."""
    special = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    special.update({0, 1, 2})
    vocabulary = set(tokenizer.get_vocab().values())
    outcomes = []
    for row, tokens, mask in zip(
        rows,
        generated.response_ids.cpu().tolist(),
        generated.response_mask.cpu().tolist(),
        strict=True,
    ):
        if len(tokens) != len(mask) or any(type(active) is not bool for active in mask):
            raise ValueError("Denoising requires matching boolean response masks")
        active = [token for token, included in zip(tokens, mask, strict=True) if included]
        if mask != [True] * len(active) + [False] * (len(mask) - len(active)):
            raise ValueError("Denoising response masks must be contiguous prefixes")
        terminated = bool(active and active[-1] == 1)
        content = active[:-1] if terminated else active
        invalid = not content or any(
            type(token) is not int or token not in vocabulary or token in special
            for token in content
        )
        prediction = tokenizer.decode(content, skip_special_tokens=False) if not invalid else ""
        if not invalid:
            invalid = tokenizer.encode(prediction, add_special_tokens=False).ids != content
        correct = bool(not invalid and terminated and prediction == row["target_text"])
        outcomes.append(
            {
                "correct": correct,
                "reward": float(correct),
                "invalid": bool(invalid),
                "truncated": not terminated,
                "content_tokens": len(content),
            }
        )
    return outcomes


def balanced_order(rows, count: int, seed: int) -> list[dict]:
    """Cycle works uniformly, shuffling examples without replacement within each work."""
    import random
    from collections import defaultdict

    if type(count) is not int or count < 1 or not rows:
        raise ValueError("A balanced schedule requires examples and a positive count")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["work_id"]].append(row)
    rng = random.Random(seed)
    queues: dict[str, list[dict]] = {}
    result: list[dict] = []
    while len(result) < count:
        works = sorted(grouped)
        rng.shuffle(works)
        for work in works:
            if not queues.get(work):
                queues[work] = sorted(grouped[work], key=lambda row: row["record_id"])
                rng.shuffle(queues[work])
            result.append(queues[work].pop())
            if len(result) == count:
                break
    return result


def adapter_snapshot(model):
    return {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }


def adapter_digest(state) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json_bytes([name, list(value.shape), str(value.dtype)]))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_adapters(model, state) -> None:
    """Validate every destination before changing any tensor or optimizer state."""
    import torch

    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state) or any(
        p.shape != state[name].shape
        or p.dtype != state[name].dtype
        or not bool(torch.isfinite(state[name]).all())
        for name, p in parameters.items()
    ):
        raise ValueError("Warm-start adapter keys, shapes, dtypes or values differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device))


def run_comparison(root: Path, config_path: Path, output: Path, *, prepare_only=False) -> dict:
    """Fixed paired local conditions; Trainer owns all backward/optimizer updates."""
    import gc
    import json
    import math
    import platform
    import subprocess
    import time
    from dataclasses import asdict
    from types import SimpleNamespace

    from src.catalog.storage import write_json_atomic

    config = read_json(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("kind") != "book_denoising_comparison"
        or config.get("seeds") != [17, 29]
        or config.get("warmup_steps") != 128
        or config.get("comparison_prompts") != 64
        or config.get("group_size") != 4
        or config.get("response_cap") != 8
        or config.get("max_total_seconds") != 1800
        or config.get("promote") is not False
        or config.get("paid_spend_authorized") is not False
    ):
        raise ValueError("Unsupported fixed local comparison contract")
    for name in (
        "warmup_learning_rate",
        "adaptation_learning_rate",
        "temperature",
        "clip_low",
        "clip_high",
    ):
        value = config.get(name)
        maximum = 0.001 if "learning_rate" in name else (2 if name == "temperature" else 0.5)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"Invalid comparison {name}")
    for key in ("base_runtime", "data_manifest", "tokenizer"):
        _checked(root, config[key])
    if output.exists():
        raise FileExistsError("Existing comparison output is preserved; use a fresh directory")
    output = output.resolve()
    if not output.is_relative_to(root.resolve() / "outputs"):
        raise ValueError("Comparison output must remain in ignored outputs")
    data = prepare_denoising_data(
        root, Path(config["data_manifest"]["path"]), Path(config["tokenizer"]["path"])
    )
    if data["provenance"]["inputs"]["manifest"] != config["data_manifest"]:
        raise ValueError("Comparison examples differ from the selected source manifest")
    if not data["train"] or not data["dev"]:
        raise ValueError("Comparison needs nonempty train and validation examples")
    output.mkdir(parents=True)
    write_json_atomic(output / "examples.json", data)
    report = {
        "schema_version": 1,
        "kind": config["kind"],
        "status": "prepared",
        "config": config,
        "config_sha256": file_hash(config_path),
        "data": data["provenance"],
        "examples": _reference(root, output / "examples.json"),
        "python": platform.python_version(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_sha256": {
            str(p.relative_to(root)): file_hash(p)
            for folder in ("src/training", "src/models", "src/data")
            for p in sorted((root / folder).glob("*.py"))
        },
        "seeds": [],
        "global_test_used": False,
        "promoted": False,
    }

    def save():
        write_json_atomic(output / "report.json", report)

    save()
    if prepare_only:
        return report
    import torch

    from src.models.adapters import extract_effective_delta
    from src.training.pilot import (
        Batches,
        build_local_model,
        make_local_trainer,
        validate_pilot_config,
    )
    from src.training.policy import (
        PolicyTask,
        deterministic_policy,
        sample_responses,
        score_responses,
    )
    from src.training.rl import GroupRelativeConfig, RewardProvenance

    runtime = read_json(_checked(root, config["base_runtime"]))
    validate_pilot_config(runtime)
    tokenizer = _tokenizer(_checked(root, config["tokenizer"]))
    started = time.perf_counter()
    report.update(
        status="running",
        torch=torch.__version__,
        timing_boundary="After task preparation; includes each model initialization and all training, evaluation and saving. Cooperative bound may overshoot by one operation.",
    )

    def synchronize():
        if runtime["device"] == "mps":
            torch.mps.synchronize()
            report["max_observed_driver_bytes"] = max(
                report.get("max_observed_driver_bytes", 0), torch.mps.driver_allocated_memory()
            )

    def budget():
        synchronize()
        if time.perf_counter() - started >= config["max_total_seconds"]:
            raise TimeoutError("Fixed comparison wall-time budget reached")

    def batch(row, *, device="cpu"):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
        return {
            "src_ids": ids,
            "src_mask": torch.ones_like(ids, dtype=torch.bool),
            "labels": labels,
            "tgt_ids": torch.cat((torch.zeros_like(labels[:, :1]), labels[:, :-1]), dim=1),
        }

    def run_seed(seed):
        budget()
        local = {**runtime, "seed": seed, "lora": {**runtime["lora"], "seed": seed}}
        model, facade, binding, device = build_local_model(local)
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir()
        result = {
            "seed": seed,
            "binding": asdict(binding),
            "phases": {},
            "evaluations": {},
            "checkpoints": {},
        }
        report["seeds"].append(result)
        runner = make_local_trainer(
            model, facade, device, seed_output, config["warmup_learning_rate"]
        )
        parameters = [p for p in model.parameters() if p.requires_grad]

        def checkpoint(name):
            state = adapter_snapshot(model)
            path = seed_output / f"{name}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "binding": asdict(binding),
                    "config_sha256": report["config_sha256"],
                    "adapter_state": state,
                },
                path,
            )
            result["checkpoints"][name] = {
                **_reference(root, path),
                "tensor_sha256": adapter_digest(state),
            }
            save()
            return state

        def evaluate():
            records = []
            works: dict[str, list[dict]] = {}
            with torch.no_grad(), deterministic_policy(model):
                for row in data["dev"]:
                    budget()
                    current = batch(row, device=device)
                    labels = current["labels"]
                    scores = score_responses(
                        model,
                        current["src_ids"],
                        current["src_mask"],
                        labels,
                        torch.ones_like(labels, dtype=torch.bool),
                        start_token_id=0,
                        end_token_id=1,
                        pad_token_id=0,
                        temperature=1.0,
                    )
                    memory = model.encoder(
                        current["src_ids"],
                        mask=current["src_mask"][:, None, :] & current["src_mask"][:, :, None],
                    )
                    generated = model.decoder.greedy_decode(
                        memory,
                        config["response_cap"] + 1,
                        0,
                        end_token_id=1,
                        memory_mask=current["src_mask"],
                    )
                    response = generated[:, 1:]
                    outcome = denoising_outcomes(
                        [row],
                        SimpleNamespace(
                            response_ids=response,
                            response_mask=torch.ones_like(response, dtype=torch.bool),
                        ),
                        tokenizer,
                    )[0]
                    value = {
                        "record_id": row["record_id"],
                        "work_id": row["work_id"],
                        **outcome,
                        "content_nll": float(-scores[0, :-1].mean()),
                        "eos_nll": float(-scores[0, -1]),
                        "response_ids": response.cpu().tolist()[0],
                    }
                    records.append(value)
                    works.setdefault(row["work_id"], []).append(value)
            per_work = {
                name: {
                    "records": len(rows),
                    **{
                        key: sum(float(r[key]) for r in rows) / len(rows)
                        for key in ("correct", "content_nll", "eos_nll", "invalid", "truncated")
                    },
                }
                for name, rows in works.items()
            }
            return {
                "macro": {
                    key: sum(w[key] for w in per_work.values()) / len(per_work)
                    for key in ("correct", "content_nll", "eos_nll", "invalid", "truncated")
                },
                "per_work": per_work,
                "records": records,
            }

        def reset(rate, state=None):
            if state is not None:
                restore_adapters(model, state)
                if adapter_digest(adapter_snapshot(model)) != adapter_digest(state):
                    raise ValueError("Branch does not start from the exact warm adapter tensors")
            runner.optimizer = torch.optim.AdamW(
                parameters, lr=rate, weight_decay=0.0, foreach=False
            )
            runner.global_step, runner.scheduler, runner.policy_objectives = 0, None, {}

        def execute(name, loader, task):
            times: list[float] = []
            phase_started = previous = time.perf_counter()
            result["phases"][name] = phase = {"step_seconds": times, "optimizer_updates": 0}

            def completed():
                nonlocal previous
                synchronize()
                now = time.perf_counter()
                times.append(now - previous)
                previous = now
                phase["optimizer_updates"] = runner.global_step
                if hasattr(loader, "events"):
                    phase["rollouts"] = loader.events
                    loader.events[-1]["optimizer_updated"] = (
                        runner.global_step > loader.events[-1]["optimizer_step_before"]
                    )
                save()
                budget()

            phase["metrics"] = runner._run_epoch(
                {task: loader}, train=True, epoch=1, step_callback=completed
            )
            synchronize()
            phase["wall_seconds"] = time.perf_counter() - phase_started
            checkpoint(name)
            result["evaluations"][name] = evaluate()
            save()

        result["evaluations"]["base"] = evaluate()
        reset(config["warmup_learning_rate"])
        warm_order = balanced_order(data["train"], config["warmup_steps"], seed)
        result["warmup_order"] = [row["record_id"] for row in warm_order]
        execute("warm", Batches([batch(row) for row in warm_order]), "summarization")
        warm = adapter_snapshot(model)
        order = balanced_order(data["train"], config["comparison_prompts"], seed + 1000)
        result["comparison_order"] = [row["record_id"] for row in order]
        result["warm_start_tensor_sha256"] = adapter_digest(warm)
        reset(config["adaptation_learning_rate"], warm)
        result["ce_start_tensor_sha256"] = adapter_digest(adapter_snapshot(model))
        execute("ce", Batches([batch(row) for row in order]), "summarization")
        reset(config["adaptation_learning_rate"], warm)
        result["rl_start_tensor_sha256"] = adapter_digest(adapter_snapshot(model))
        objective = PolicyTask(
            model,
            mode="group_relative",
            start_token_id=0,
            end_token_id=1,
            pad_token_id=0,
            tokenizer_revision=config["tokenizer"]["sha256"],
            temperature=config["temperature"],
            config=GroupRelativeConfig(
                config["response_cap"],
                clip_low=config["clip_low"],
                clip_high=config["clip_high"],
            ),
            expected_behavior_revision=f"seed-{seed}:rl-step-0",
        )
        runner.policy_objectives = {"denoising_rl": objective}

        class OnPolicy:
            def __init__(self):
                self.dataset, self.events = range(len(order)), []

            def __len__(self):
                return len(order)

            def __iter__(self):
                for row in order:
                    budget()
                    current = batch(row, device=device)
                    revision = f"seed-{seed}:rl-step-{runner.global_step}"
                    objective.behavior_revision = revision
                    tick = time.perf_counter()
                    generated = sample_responses(
                        model,
                        current["src_ids"],
                        current["src_mask"],
                        group_size=config["group_size"],
                        max_response_tokens=config["response_cap"],
                        start_token_id=0,
                        end_token_id=1,
                        pad_token_id=0,
                        temperature=config["temperature"],
                        behavior_policy_revision=revision,
                        tokenizer_revision=config["tokenizer"]["sha256"],
                    )
                    outcomes = denoising_outcomes(
                        [row] * config["group_size"], generated, tokenizer
                    )
                    synchronize()
                    self.events.append(
                        {
                            "record_id": row["record_id"],
                            "work_id": row["work_id"],
                            "behavior_revision": revision,
                            "optimizer_step_before": runner.global_step,
                            "rollout_and_verification_seconds": time.perf_counter() - tick,
                            "outcomes": outcomes,
                            "response_ids": generated.response_ids.cpu().tolist(),
                            "response_mask": generated.response_mask.cpu().tolist(),
                            "behavior_log_probs": generated.behavior_log_probs.cpu().tolist(),
                        }
                    )
                    provenance = tuple(
                        RewardProvenance(
                            POLICY,
                            report["code_sha256"]["src/training/denoising.py"],
                            row["source_evidence_sha256"],
                        )
                        for _ in outcomes
                    )
                    yield generated.training_batch(
                        torch.tensor([o["reward"] for o in outcomes], device=device), provenance
                    )

        torch.manual_seed(seed + 2000)
        execute("rl", OnPolicy(), "denoising_rl")
        delta = extract_effective_delta(model, binding, task_id="book_word_reconstruction")
        result["frozen_base_verified"] = True
        result["effective_delta_metadata_sha256"] = sha(json_bytes(delta.metadata()))

    try:
        for seed in config["seeds"]:
            run_seed(seed)
            gc.collect()
            if runtime["device"] == "mps":
                torch.mps.empty_cache()
            save()
        report["status"] = "completed"
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        report["elapsed_execution_seconds"] = time.perf_counter() - started
        save()
    print(
        json.dumps({"status": report["status"], "seconds": report["elapsed_execution_seconds"]}),
        flush=True,
    )
    return report
