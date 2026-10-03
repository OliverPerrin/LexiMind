"""Fixed book-word supervision formats and a no-update reward-coverage probe.

This developmental comparison retains the original source windows and work roles.
It neither opens global test narratives nor selects or promotes a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer

from src.research.candidate_io import json_bytes, sha
from src.research.io import file_hash, read_json
from src.training.denoising import (
    INSTRUCTION,
    SENTINEL,
    _checked,
    _reference,
    _tokenizer,
    adapter_digest,
    adapter_snapshot,
    balanced_order,
    denoising_outcomes,
    prepare_denoising_data,
    restore_adapters,
)

ARMS = ("instruction_word", "t5_span")
POLICY = "bookdash-supervision-formats-v1"
CLOSING_SENTINEL = "<extra_id_1>"


def _sentinels(tokenizer: Tokenizer) -> tuple[int, int]:
    opening = tokenizer.token_to_id(SENTINEL)
    closing = tokenizer.token_to_id(CLOSING_SENTINEL)
    if opening is None or closing is None or opening == closing:
        raise ValueError("Supervision formats require distinct T5 sentinel tokens")
    return opening, closing


def format_row(row: dict, arm: str, tokenizer: Tokenizer) -> dict:
    """Wrap an already source-verified row without selecting different examples."""
    if arm not in ARMS:
        raise ValueError("Unknown supervision format")
    opening, closing = _sentinels(tokenizer)
    if (
        not row["input_text"].startswith(INSTRUCTION)
        or row["input_text"].count(SENTINEL) != 1
        or CLOSING_SENTINEL in row["input_text"]
        or row["labels"] != row["target_ids"] + [1]
        or tokenizer.encode(row["target_text"], add_special_tokens=False).ids != row["target_ids"]
    ):
        raise ValueError("Supervision formatting requires an unchanged denoising row")
    if arm == "instruction_word":
        input_text = row["input_text"]
        labels = list(row["labels"])
        content_positions = list(range(len(row["target_ids"])))
        control_positions = []
    else:
        input_text = row["input_text"][len(INSTRUCTION) :]
        labels = [opening, *row["target_ids"], closing, 1]
        content_positions = list(range(1, 1 + len(row["target_ids"])))
        control_positions = [0, len(labels) - 2]
    input_ids = tokenizer.encode(input_text, add_special_tokens=False).ids + [1]
    if arm == "instruction_word" and input_ids != row["input_ids"]:
        raise ValueError("Instruction-word input tokenization changed")
    if len(input_ids) > 160 or len(labels) > 8:
        raise ValueError("Supervision row exceeds the fixed token limits")
    return {
        **row,
        "arm": arm,
        "input_text": input_text,
        "input_ids": input_ids,
        "labels": labels,
        "content_positions": content_positions,
        "control_positions": control_positions,
    }


def supervision_outcomes(rows, generated, tokenizer: Tokenizer, arm: str) -> list[dict]:
    """Require canonical exact content, the declared output grammar and final EOS."""
    if arm not in ARMS:
        raise ValueError("Unknown supervision format")
    if arm == "instruction_word":
        return denoising_outcomes(rows, generated, tokenizer)
    opening, closing = _sentinels(tokenizer)
    outcomes = []
    special = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    special.update({0, 1, 2, opening, closing})
    vocabulary = set(tokenizer.get_vocab().values())
    for row, tokens, mask in zip(
        rows,
        generated.response_ids.cpu().tolist(),
        generated.response_mask.cpu().tolist(),
        strict=True,
    ):
        if len(tokens) != len(mask) or any(type(active) is not bool for active in mask):
            raise ValueError("Supervision requires matching boolean response masks")
        active = [token for token, included in zip(tokens, mask, strict=True) if included]
        if mask != [True] * len(active) + [False] * (len(mask) - len(active)):
            raise ValueError("Supervision response masks must be contiguous prefixes")
        terminated = bool(active and active[-1] == 1)
        wrapped = bool(
            terminated and len(active) >= 4 and active[0] == opening and active[-2] == closing
        )
        content = active[1:-2] if wrapped else []
        invalid = (
            not wrapped
            or not content
            or any(
                type(token) is not int or token not in vocabulary or token in special
                for token in content
            )
        )
        prediction = tokenizer.decode(content, skip_special_tokens=False) if not invalid else ""
        if not invalid:
            invalid = tokenizer.encode(prediction, add_special_tokens=False).ids != content
        correct = bool(not invalid and prediction == row["target_text"])
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


def validate_supervision_config(config: dict) -> None:
    """Reject unplanned arms, searches, steps, budgets and generation settings."""
    fixed = {
        "schema_version": 1,
        "kind": "book_supervision_comparison",
        "seeds": [17, 29],
        "arms": list(ARMS),
        "checkpoints": [128, 512],
        "learning_rate": 0.0005,
        "response_cap": 8,
        "group_size": 4,
        "coverage_prompts": 46,
        "temperature": 1.0,
        "max_total_seconds": 1800,
        "promote": False,
        "paid_spend_authorized": False,
    }
    for key, expected in fixed.items():
        value = config.get(key)
        if value != expected or isinstance(value, bool) != isinstance(expected, bool):
            raise ValueError(f"Unsupported fixed supervision contract: {key}")
        if isinstance(expected, int) and type(value) is not type(expected):
            raise ValueError(f"Supervision {key} requires an integer")
        if isinstance(expected, list) and (
            not isinstance(value, list)
            or any(
                type(actual) is not type(required)
                for actual, required in zip(value, expected, strict=True)
            )
        ):
            raise ValueError(f"Supervision {key} requires the declared element types")
    for key in ("base_runtime", "data_manifest", "tokenizer"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Supervision requires a pinned {key}")


def _aggregate(records: list[dict]) -> dict:
    keys = ("correct", "content_nll", "control_nll", "eos_nll", "invalid", "truncated")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["work_id"], []).append(record)
    if not grouped:
        raise ValueError("Supervision evaluation requires nonempty work groups")

    def mean(rows, key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None

    per_work = {
        work: {"records": len(rows), **{key: mean(rows, key) for key in keys}}
        for work, rows in grouped.items()
    }
    return {
        "macro": {key: mean(list(per_work.values()), key) for key in keys},
        "per_work": per_work,
        "records": records,
    }


def run_supervision_comparison(
    root: Path, config_path: Path, output: Path, *, prepare_only=False
) -> dict:
    """Run a fixed, local CE contrast and measure reward coverage without RL."""
    import gc
    import platform
    import subprocess
    import time
    from collections import Counter
    from dataclasses import asdict

    from src.catalog.storage import write_json_atomic

    root = root.resolve()
    config = read_json(config_path)
    validate_supervision_config(config)
    for key in ("base_runtime", "data_manifest", "tokenizer"):
        _checked(root, config[key])
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Existing supervision evidence is preserved; use a fresh directory")
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("Supervision output must remain in ignored outputs")
    data = prepare_denoising_data(
        root, Path(config["data_manifest"]["path"]), Path(config["tokenizer"]["path"])
    )
    if data["provenance"]["inputs"]["manifest"] != config["data_manifest"]:
        raise ValueError("Supervision rows differ from the selected source manifest")
    if {role: len(data[role]) for role in ("train", "dev")} != {"train": 179, "dev": 36}:
        raise ValueError("Supervision requires the original 179/36 denoising examples")
    work_counts = {role: len({r["work_id"] for r in data[role]}) for role in ("train", "dev")}
    if work_counts != {"train": 23, "dev": 5}:
        raise ValueError("Supervision requires the original 23/5 work roles")
    if {r["work_id"] for r in data["train"]} & {r["work_id"] for r in data["dev"]}:
        raise ValueError("Supervision train and development works overlap")
    if {r["input_text"] for r in data["train"]} & {r["input_text"] for r in data["dev"]}:
        raise ValueError("Supervision train and development masked inputs overlap")
    tokenizer = _tokenizer(_checked(root, config["tokenizer"]))
    formatted = {
        arm: {
            role: [format_row(row, arm, tokenizer) for row in data[role]]
            for role in ("train", "dev")
        }
        for arm in ARMS
    }
    output.mkdir(parents=True)
    write_json_atomic(output / "examples.json", {**data, "formatted": formatted})
    report = {
        "schema_version": 1,
        "kind": config["kind"],
        "policy": POLICY,
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
            str(path.relative_to(root)): file_hash(path)
            for folder in ("src/training", "src/models", "src/data")
            for path in sorted((root / folder).glob("*.py"))
        },
        "seeds": [],
        "global_test_used": False,
        "promoted": False,
        "rl_optimizer_updates": 0,
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
    from src.training.policy import deterministic_policy, sample_responses, score_responses

    started = time.perf_counter()
    runtime = read_json(_checked(root, config["base_runtime"]))
    validate_pilot_config(runtime)
    report.update(
        status="running",
        torch=torch.__version__,
        timing_boundary="After task preparation; includes model initialization, all training, evaluation, sampling and evidence saving. Cooperative bound may overshoot one operation.",
    )
    save()

    def synchronize():
        if runtime["device"] == "mps":
            torch.mps.synchronize()
            report["max_observed_driver_bytes"] = max(
                report.get("max_observed_driver_bytes", 0), torch.mps.driver_allocated_memory()
            )

    def budget():
        synchronize()
        if time.perf_counter() - started >= config["max_total_seconds"]:
            raise TimeoutError("Fixed supervision wall-time budget reached")

    def batch(row, device="cpu"):
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
        initial = adapter_snapshot(model)
        initial_digest = adapter_digest(initial)
        training_order = balanced_order(data["train"], config["checkpoints"][-1], seed)
        coverage_order = balanced_order(data["train"], config["coverage_prompts"], seed + 4000)
        if set(Counter(row["work_id"] for row in coverage_order).values()) != {2}:
            raise ValueError("Coverage schedule must have exactly two prompts per train work")
        result = {
            "seed": seed,
            "binding": asdict(binding),
            "initial_adapter_sha256": initial_digest,
            "training_order": [r["record_id"] for r in training_order],
            "coverage_order": [r["record_id"] for r in coverage_order],
            "arms": {},
        }
        report["seeds"].append(result)
        save()

        def evaluate(arm, rows, stage):
            tick = time.perf_counter()
            records = []
            with torch.no_grad(), deterministic_policy(model):
                for row in rows:
                    budget()
                    current = batch(row, device)
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
                    response = model.decoder.greedy_decode(
                        memory,
                        config["response_cap"] + 1,
                        0,
                        end_token_id=1,
                        memory_mask=current["src_mask"],
                    )[:, 1:]
                    mask = torch.ones_like(response, dtype=torch.bool)
                    outcome = supervision_outcomes(
                        [row],
                        SimpleNamespace(response_ids=response, response_mask=mask),
                        tokenizer,
                        arm,
                    )[0]
                    records.append(
                        {
                            "record_id": row["record_id"],
                            "work_id": row["work_id"],
                            "arm": arm,
                            "stage": stage,
                            **outcome,
                            "content_nll": float(-scores[0, row["content_positions"]].mean()),
                            "control_nll": (
                                float(-scores[0, row["control_positions"]].mean())
                                if row["control_positions"]
                                else None
                            ),
                            "eos_nll": float(-scores[0, -1]),
                            "target_log_probs": scores.cpu().tolist()[0],
                            "response_ids": response.cpu().tolist()[0],
                            "response_mask": mask.cpu().tolist()[0],
                        }
                    )
            synchronize()
            return {
                "arm": arm,
                "stage": stage,
                "wall_seconds": time.perf_counter() - tick,
                "generated_tokens": sum(len(r["response_ids"]) for r in records),
                **_aggregate(records),
            }

        for arm in ARMS:
            budget()
            restore_adapters(model, initial)
            if adapter_digest(adapter_snapshot(model)) != initial_digest:
                raise ValueError("Supervision arm changed its exact initialized adapter tensors")
            torch.manual_seed(seed)
            arm_output = output / f"seed_{seed}" / arm
            arm_output.mkdir(parents=True)
            arm_result: dict = {
                "arm": arm,
                "initial_adapter_sha256": adapter_digest(adapter_snapshot(model)),
                "phases": {},
                "evaluations": {},
                "checkpoints": {},
            }
            result["arms"][arm] = arm_result
            runner = make_local_trainer(model, facade, device, arm_output, config["learning_rate"])
            runner.global_step = 0
            runner.policy_objectives = {}
            by_id = {row["record_id"]: row for row in formatted[arm]["train"]}
            arm_result["evaluations"]["base"] = evaluate(arm, formatted[arm]["dev"], "base")
            save()
            previous_endpoint = 0
            for endpoint in config["checkpoints"]:
                stage = str(endpoint)
                tick = previous = time.perf_counter()
                stage_rows = [
                    by_id[r["record_id"]] for r in training_order[previous_endpoint:endpoint]
                ]
                phase = {
                    "arm": arm,
                    "from_update": previous_endpoint,
                    "to_update": endpoint,
                    "optimizer_updates": 0,
                    "step_seconds": [],
                    "input_tokens": 0,
                    "content_target_tokens": 0,
                    "supervised_tokens_including_controls": 0,
                }
                arm_result["phases"][stage] = phase

                class CheckedBatches(Batches):
                    def __iter__(self):
                        for row in self.rows:
                            budget()
                            yield batch(row)

                def completed(
                    phase=phase, stage_rows=stage_rows, runner=runner, start=previous_endpoint
                ):
                    nonlocal previous
                    synchronize()
                    now = time.perf_counter()
                    phase["step_seconds"].append(now - previous)
                    previous = now
                    row = stage_rows[len(phase["step_seconds"]) - 1]
                    phase["optimizer_updates"] = runner.global_step - start
                    phase["cumulative_optimizer_updates"] = runner.global_step
                    phase["input_tokens"] += len(row["input_ids"])
                    phase["content_target_tokens"] += len(row["target_ids"])
                    phase["supervised_tokens_including_controls"] += len(row["labels"])
                    save()
                    budget()

                phase["metrics"] = runner._run_epoch(
                    {"summarization": CheckedBatches(stage_rows)},
                    train=True,
                    epoch=1,
                    step_callback=completed,
                )
                synchronize()
                phase["wall_seconds"] = time.perf_counter() - tick
                if runner.global_step != endpoint:
                    raise ValueError(
                        "Supervision did not complete its fixed optimizer update count"
                    )
                state = adapter_snapshot(model)
                path = arm_output / f"step_{endpoint}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "seed": seed,
                        "arm": arm,
                        "optimizer_updates": endpoint,
                        "binding": asdict(binding),
                        "config_sha256": report["config_sha256"],
                        "adapter_state": state,
                    },
                    path,
                )
                arm_result["checkpoints"][stage] = {
                    **_reference(root, path),
                    "arm": arm,
                    "optimizer_updates": endpoint,
                    "tensor_sha256": adapter_digest(state),
                }
                save()
                arm_result["evaluations"][stage] = evaluate(arm, formatted[arm]["dev"], stage)
                save()
                previous_endpoint = endpoint

            arm_result["evaluations"]["train_final"] = evaluate(
                arm, formatted[arm]["train"], "train_final"
            )
            before = adapter_digest(adapter_snapshot(model))
            coverage = {
                "arm": arm,
                "stage": "512",
                "schedule_seed": seed + 4000,
                "sampling_seed": seed + 5000,
                "adapter_before_sha256": before,
                "records": [],
            }
            arm_result["coverage"] = coverage
            save()
            torch.manual_seed(seed + 5000)
            tick = time.perf_counter()
            for source in coverage_order:
                budget()
                row = by_id[source["record_id"]]
                current = batch(row, device)
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
                    behavior_policy_revision=f"seed-{seed}:{arm}:step-512:{before}",
                    tokenizer_revision=config["tokenizer"]["sha256"],
                )
                outcomes = supervision_outcomes(
                    [row] * config["group_size"], generated, tokenizer, arm
                )
                correct = sum(o["correct"] for o in outcomes)
                coverage["records"].append(
                    {
                        "record_id": row["record_id"],
                        "work_id": row["work_id"],
                        "arm": arm,
                        "outcomes": outcomes,
                        "correct_responses": correct,
                        "mixed_reward_group": 0 < correct < config["group_size"],
                        "response_ids": generated.response_ids.cpu().tolist(),
                        "response_mask": generated.response_mask.cpu().tolist(),
                        "behavior_log_probs": generated.behavior_log_probs.cpu().tolist(),
                    }
                )
                save()
            synchronize()
            coverage["wall_seconds"] = time.perf_counter() - tick
            coverage["adapter_after_sha256"] = adapter_digest(adapter_snapshot(model))
            if coverage["adapter_after_sha256"] != before or runner.global_step != 512:
                raise ValueError("No-update coverage probe changed the trained policy")
            coverage["no_updates"] = True

            def counts(records):
                outcomes = [o for r in records for o in r["outcomes"]]
                return {
                    "groups": len(records),
                    "sampled_responses": len(outcomes),
                    "correct_responses": sum(o["correct"] for o in outcomes),
                    "invalid_responses": sum(o["invalid"] for o in outcomes),
                    "truncated_responses": sum(o["truncated"] for o in outcomes),
                    "mixed_reward_groups": sum(r["mixed_reward_group"] for r in records),
                    "generated_tokens": sum(
                        sum(sum(mask) for mask in r["response_mask"]) for r in records
                    ),
                }

            coverage["totals"] = counts(coverage["records"])
            coverage["per_work"] = {
                work: counts([r for r in coverage["records"] if r["work_id"] == work])
                for work in sorted({r["work_id"] for r in coverage["records"]})
            }
            delta = extract_effective_delta(model, binding, task_id=f"book_word_{arm}")
            arm_result["frozen_base_verified"] = True
            arm_result["effective_delta_metadata_sha256"] = sha(json_bytes(delta.metadata()))
            save()
            del runner

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
    return report
