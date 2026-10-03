"""Bounded local feasibility pilots; no global study or dataset admission."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from src.research.candidate_io import json_bytes, sha
from src.research.io import check_file, file_hash, parse_json, read_json, safe_path

PILOT_INSTRUCTION = (
    "Continue this book passage with the next words. Return only the continuation.\n\n"
)
PILOT_WORKS = {
    "cory-doctorow-little-brother": "train",
    "bookdash-why-is-nita-upside-down": "train",
    "bookdash-sizwes-smile": "diagnostic",
}


def prepare_pilot_data(root: Path) -> dict:
    """Verify and select existing train-only windows without changing any candidate.

    The whole Sizwe work is reserved for this pilot's diagnostics, while retaining
    its original global train assignment. No test example reaches the return value.
    All rows are source-observed continuations, not human field labels/preferences.
    """
    root = root.resolve()
    path = root / "research/preparation/rpt_candidate_manifest.json"
    manifest = read_json(path)
    if (
        manifest.get("configuration", {}).get("policy")
        != "licensed-book-normalized-continuation-v1"
    ):
        raise ValueError("Local pilot requires the prepared licensed continuation policy")
    inputs = manifest["inputs"]
    references = {
        "rpt_manifest": {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        },
        **inputs,
        "continuations": manifest["continuations"],
    }
    for reference in references.values():
        if errors := check_file(root, reference):
            raise ValueError("Pilot source changed: " + "; ".join(errors))
    for name, expected in manifest["implementation_sha256"].items():
        if file_hash(safe_path(root, name)) != expected:
            raise ValueError("Pilot continuation implementation changed")
    licensed = read_json(safe_path(root, inputs["licensed_manifest"]["path"]))
    partitions = read_json(safe_path(root, inputs["partition_manifest"]["path"]))
    if (
        partitions["inputs"]["licensed_manifest"] != inputs["licensed_manifest"]
        or partitions["components"] != inputs["components"]
        or licensed["source_inventory"] != inputs["source_inventory"]
    ):
        raise ValueError("Pilot inputs do not bind one consistent licensed work partition")
    components = {}
    for line in safe_path(root, inputs["components"]["path"]).read_text().splitlines():
        component = parse_json(line)
        for member in component["members"]:
            if member in components:
                raise ValueError("Pilot work occurs in multiple components")
            components[member] = component
    tokenizer = Tokenizer.from_file(str(safe_path(root, inputs["tokenizer"]["path"])))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.token_to_id("<pad>") != 0 or tokenizer.token_to_id("</s>") != 1:
        raise ValueError("Local pilot requires FLAN PAD/start=0 and EOS=1")
    special = {key for key, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    unknown = tokenizer.token_to_id("<unk>")
    if unknown is not None:
        special.add(unknown)
    works = {}
    rows: dict[str, list[dict]] = {"train": [], "diagnostic": []}
    identities = set()
    seen_spans: list[tuple[tuple[str, str], int, int]] = []
    for line in safe_path(root, manifest["continuations"]["path"]).read_text().splitlines():
        record = parse_json(line)
        target = record["continuation_target"]
        if target["split"] != "train":
            continue
        work_id = record["work_id"]
        if work_id not in PILOT_WORKS or record["record_id"] in identities:
            raise ValueError("Unexpected or duplicate pilot train record")
        identities.add(record["record_id"])
        role = PILOT_WORKS[work_id]
        component = components["licensed_text:" + work_id]
        if (
            component["proposed_split"] != "train"
            or component["component_id"] != target["work_group"]
        ):
            raise ValueError("Pilot work is not in the effective train partition")
        evidence = record["source_evidence"]
        artifact = inputs["work:" + work_id]
        if (
            evidence["artifact"] != artifact
            or evidence["partition_manifest_sha256"] != inputs["partition_manifest"]["sha256"]
            or target["source_evidence_sha256"] != sha(json_bytes(evidence))
            or target["tokenizer_sha256"] != inputs["tokenizer"]["sha256"]
        ):
            raise ValueError("Pilot record differs from its source bindings")
        if work_id not in works:
            works[work_id] = read_json(safe_path(root, artifact["path"]))
        work = works[work_id]
        attribution = {key: work[key] for key in ("title", "creators", "source_page", "license")}
        if work["work_id"] != work_id or record["attribution"] != attribution:
            raise ValueError("Pilot source identity or attribution changed")
        section = next(
            row for row in work["sections"] if row["section_id"] == evidence["section_id"]
        )
        section_ids = tokenizer.encode(section["text"], add_special_tokens=False).ids
        left, cut = evidence["prompt_token_span"]
        target_start, right = evidence["target_token_span"]
        if (
            not 0 <= left < cut == target_start < right <= len(section_ids)
            or section_ids[left:cut] != record["prompt_ids"]
            or section_ids[cut:right] != record["target_ids"]
            or sha(section["text"]) != evidence["source_section_sha256"]
            or work["source_sha256"] != evidence["source_sha256"]
            or section["source_lines"] != evidence["source_lines"]
        ):
            raise ValueError("Pilot window does not reproduce its observed source tokens")
        prompt_ids, target_ids = record["prompt_ids"], record["target_ids"]
        if not 8 <= len(prompt_ids) <= 128 or not 4 <= len(target_ids) <= 32:
            raise ValueError("Pilot source window exceeds the prepared token bounds")
        if any(token in special for token in (*prompt_ids, *target_ids)):
            raise ValueError("Pilot source contains special or unknown tokens")
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        answer = tokenizer.decode(target_ids, skip_special_tokens=False)
        if (
            prompt != record["prompt_normalized_text"]
            or answer != target["observed_bytes"]
            or tokenizer.encode(answer, add_special_tokens=False).ids != target_ids
            or "".join(target["token_bytes"]) != answer
            or any(
                tokenizer.decode(target_ids[:index], skip_special_tokens=False)
                != "".join(target["token_bytes"][:index])
                for index in range(1, len(target_ids) + 1)
            )
        ):
            raise ValueError("Pilot continuation round trip or byte boundaries changed")
        if role == "diagnostic":
            key = (work_id, evidence["section_id"])
            if any(
                key == previous and max(left, a) < min(right, b) for previous, a, b in seen_spans
            ):
                raise ValueError("Pilot diagnostic windows overlap")
            seen_spans.append((key, left, right))
        input_ids = tokenizer.encode(PILOT_INSTRUCTION + prompt, add_special_tokens=False).ids + [1]
        if len(input_ids) > 160:
            raise ValueError("Pilot instruction plus source context exceeds 160 tokens")
        rows[role].append(
            {
                **record,
                "source_split": "train",
                "pilot_role": role,
                "input_ids": input_ids,
                "labels": target_ids + [1],
            }
        )
    counts = Counter(row["work_id"] for group in rows.values() for row in group)
    if counts != {work: 8 for work in PILOT_WORKS}:
        raise ValueError("Local pilot requires exactly eight prepared rows from each train work")
    groups = [{row["continuation_target"]["work_group"] for row in rows[role]} for role in rows]
    if groups[0] & groups[1]:
        raise ValueError("Pilot optimization and diagnostic works overlap")
    return {
        **rows,
        "provenance": {
            "policy": "licensed-continuation-local-feasibility-v1",
            "inputs": references,
            "instruction": PILOT_INSTRUCTION,
            "work_roles": PILOT_WORKS.copy(),
            "counts": {role: len(values) for role, values in rows.items()},
            "token_counts_including_eos": {
                role: sum(len(row["labels"]) for row in values) for role, values in rows.items()
            },
            "encoder_token_cap": 160,
            "target_token_cap_including_eos": 33,
            "source_split": "train",
            "global_test_used": False,
            "study_admission": False,
            "limitations": [
                "Feasibility/overfit pilot from two optimization works and one diagnostic work; no representative quality estimate.",
                "Nita training windows overlap within pages; diagnostic windows are disjoint and the complete diagnostic work is excluded from optimization.",
                "Original global train assignments and candidate admission status are unchanged; diagnostic is a pilot-only role.",
                "EOS marks the requested continuation boundary, not the end of the source work. Targets are tokenizer-normalized source text.",
                "Preserve each work's attribution and license; Little Brother retains noncommercial/share-alike terms.",
            ],
        },
    }


def validate_pilot_config(config: dict) -> None:
    """Keep this local pilot bounded; it is not a general experiment launcher."""
    if (
        config.get("schema_version") != 1
        or config.get("kind") != "local_continuation_feasibility_pilot"
        or config.get("base_repo") != "google/flan-t5-base"
        or config.get("base_revision") != "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
        or config.get("activation") != "gated-gelu-tanh"
        or config.get("device") not in {"cpu", "mps"}
        or config.get("dtype") != "float32"
        or config.get("batch_size") != 1
        or config.get("group_size") != 2
        or config.get("rl_method") != "dr_grpo"
        or config.get("kl_coefficient") != 0
        or config.get("formal_study_admitted") is not False
        or config.get("paid_spend_authorized") is not False
    ):
        raise ValueError("Unsupported local pilot contract")
    import math

    for key, minimum, maximum in (
        ("sft_steps", 1, 256),
        ("rl_steps", 0, 32),
        ("max_wall_seconds", 1, 1800),
        ("threads", 1, 8),
        ("max_response_tokens", 4, 32),
        ("minimum_rewarded_content_tokens", 4, 8),
        ("seed", 0, 2**31 - 1),
    ):
        if type(config.get(key)) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"Pilot {key} must be an integer in [{minimum}, {maximum}]")
    for key, low, high in (
        ("sft_learning_rate", 0, 0.001),
        ("rl_learning_rate", 0, 0.001),
        ("mps_memory_fraction", 0, 0.4),
        ("temperature", 0, 2),
        ("clip_low", 0, 0.5),
        ("clip_high", 0, 0.5),
    ):
        value: Any = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
            raise ValueError(f"Pilot {key} must be finite and in ({low}, {high}]")
    if config.get("lora") != {"rank": 4, "alpha": 8, "dropout": 0.0, "seed": config["seed"]}:
        raise ValueError("Local pilot uses explicitly bounded rank-four LoRA")


def continuation_rewards(rows, generated, tokenizer, *, minimum_tokens):
    """Exact prefix reward with an explicit length floor; not original RPT scoring."""
    from .policy import ContinuationTarget, rpt_prefix_reward

    special = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    valid_ids = set(tokenizer.get_vocab().values())
    results = []
    for row, tokens, mask in zip(
        rows,
        generated.response_ids.cpu().tolist(),
        generated.response_mask.cpu().tolist(),
        strict=True,
    ):
        ids = [token for token, active in zip(tokens, mask, strict=True) if active]
        terminated = bool(ids and ids[-1] == 1)
        content = ids[:-1] if terminated else ids
        target = dict(row["continuation_target"])
        target.pop("byte_encoding")
        target["observed_bytes"] = target["observed_bytes"].encode("utf-8")
        target["token_bytes"] = tuple(value.encode("utf-8") for value in target["token_bytes"])
        valid = (
            bool(content)
            and all(token in valid_ids for token in content)
            and not special.intersection(content)
        )
        prediction = tokenizer.decode(content, skip_special_tokens=False) if valid else ""
        canonical = tokenizer.encode(prediction, add_special_tokens=False).ids
        prefix = rpt_prefix_reward(prediction.encode("utf-8"), ContinuationTarget(**target))
        # FLAN's padded vocabulary IDs decode to nothing. Canonical token counts
        # also prevent empty-token stuffing from satisfying the length floor.
        score = float(prefix and len(canonical) >= minimum_tokens)
        results.append(
            {
                "reward": score,
                "raw_prefix_reward": prefix,
                "content_tokens": len(content),
                "canonical_content_tokens": len(canonical),
                "valid_token_ids": bool(valid),
                "terminated": terminated,
            }
        )
    return results


class Batches:
    def __init__(self, rows):
        self.rows, self.dataset = rows, range(len(rows))

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def make_local_trainer(model, tokenizer, device, output, learning_rate):
    """Use the same optimizer/trainer boundary for each local pilot condition."""
    import torch

    from src.training.trainer import Trainer, TrainerConfig

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0, foreach=False)
    runner = Trainer(
        model,
        optimizer,
        TrainerConfig(
            max_epochs=1,
            gradient_clip_norm=1.0,
            task_sampling="round_robin",
            scheduler_type="constant",
            early_stopping_patience=None,
            label_smoothing=0.0,
            generation_metrics=False,
            tracking_uri="sqlite:///" + str(output / "tracking.db"),
            experiment_name="LexiMind-local-pilot",
        ),
        device,
        tokenizer,
    )
    runner.scheduler = None
    return runner


def build_local_model(config):
    """Shared offline native FLAN/LoRA setup for bounded local experiments."""
    import gc
    import os

    import torch
    from huggingface_hub import snapshot_download

    from src.data.tokenization import Tokenizer as ModelTokenizer
    from src.data.tokenization import TokenizerConfig
    from src.models.adapters import LoRAConfig, attach_lora
    from src.models.factory import ModelConfig, build_multitask_model

    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise ValueError("Measured MPS pilots require CPU operator fallback disabled")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no automatic device fallback")
        torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
    snapshot = Path(
        snapshot_download(
            config["base_repo"], revision=config["base_revision"], local_files_only=True
        )
    )
    if file_hash(snapshot / "model.safetensors") != config["base_weight_sha256"]:
        raise ValueError("Cached base weights differ from the pinned checkpoint")
    tokenizer = ModelTokenizer(TokenizerConfig(str(snapshot), max_length=160))
    model_config = ModelConfig(
        d_model=768,
        vocab_size=32128,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_attention_heads=12,
        ffn_dim=2048,
        dropout=0.0,
        use_pretrained=True,
        pretrained_model_name=str(snapshot),
        activation=config["activation"],
        use_relative_position_bias=True,
        use_learned_pos_enc=False,
    )
    model = build_multitask_model(tokenizer, num_emotions=0, num_topics=0, config=model_config)
    shared = [f"encoder.layers.{i}.self_attn.W_{p}" for i in range(12) for p in ("Q", "V")]
    private = [
        f"decoder.layers.{i}.{kind}.W_{p}"
        for i in range(12)
        for kind in ("self_attn", "cross_attn")
        for p in ("Q", "V")
    ]
    binding = attach_lora(
        model,
        shared_projections=shared,
        private_projections=private,
        config=LoRAConfig(**config["lora"]),
    )
    model.to(device)
    gc.collect()
    return model, tokenizer, binding, device


def _run_pilot(root: Path, config_path: Path, output: Path, *, prepare_only=False) -> dict:
    """Run bounded local stages through Trainer; save evidence and adapter factors."""
    import json
    import platform
    import random
    import subprocess
    import time
    from dataclasses import asdict

    from src.catalog.storage import write_json_atomic

    config = read_json(config_path)
    if config.get("kind") == "book_supervision_comparison":
        from .supervision import run_supervision_comparison

        return run_supervision_comparison(root, config_path, output, prepare_only=prepare_only)
    if config.get("kind") == "book_denoising_comparison":
        from .denoising import run_comparison

        return run_comparison(root, config_path, output, prepare_only=prepare_only)
    validate_pilot_config(config)
    if errors := check_file(root, config["continuation_manifest"]):
        raise ValueError("; ".join(errors))
    if output.exists():
        raise FileExistsError("Use a fresh pilot output directory; existing runs are preserved")
    output = output.resolve()
    if not output.is_relative_to(root.resolve() / "outputs"):
        raise ValueError("Pilot outputs must remain inside the ignored outputs directory")
    data = prepare_pilot_data(root)
    if data["provenance"]["inputs"]["rpt_manifest"] != config["continuation_manifest"]:
        raise ValueError("Pilot data must bind the explicitly selected continuation manifest")
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "kind": config["kind"],
        "status": "prepared",
        "config": config,
        "config_sha256": file_hash(config_path),
        "data": data["provenance"],
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_sha256": {
            str(path.relative_to(root)): file_hash(path)
            for folder in ("src/models", "src/training", "src/data")
            for path in sorted((root / folder).glob("*.py"))
        },
        "python": platform.python_version(),
        "system": platform.platform(),
        "stages": {},
        "updates": [],
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
    from src.training.policy import PolicyTask, sample_responses, score_responses
    from src.training.rl import GroupRelativeConfig, RewardProvenance

    model, tokenizer, binding, device = build_local_model(config)
    raw_tokenizer = Tokenizer.from_file(
        str(safe_path(root, data["provenance"]["inputs"]["tokenizer"]["path"]))
    )
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    report.update(
        status="running",
        torch=torch.__version__,
        binding=asdict(binding),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    save()
    started = time.perf_counter()
    deadline = started + config["max_wall_seconds"]

    def synchronize():
        if device.type == "mps":
            torch.mps.synchronize()
            report["max_observed_driver_bytes"] = max(
                report.get("max_observed_driver_bytes", 0), torch.mps.driver_allocated_memory()
            )

    def budget():
        synchronize()
        if time.perf_counter() >= deadline:
            raise TimeoutError("Local pilot wall-time budget reached")

    def batch(row, *, on_device=False):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long)
        labels = torch.tensor([row["labels"]], dtype=torch.long)
        values = {
            "src_ids": ids,
            "src_mask": torch.ones_like(ids, dtype=torch.bool),
            "labels": labels,
            "tgt_ids": torch.cat((torch.zeros_like(labels[:, :1]), labels[:, :-1]), dim=1),
        }
        return {key: value.to(device) for key, value in values.items()} if on_device else values

    def evaluate():
        values = {}
        with torch.no_grad():
            for role in ("train", "diagnostic"):
                works: dict[str, Any] = {}
                for row in data[role]:
                    budget()
                    current = batch(row, on_device=True)
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
                    stats = works.setdefault(
                        row["work_id"], {"nll_sum": 0.0, "tokens": 0, "records": 0}
                    )
                    stats["nll_sum"] += float(-scores.sum())
                    stats["tokens"] += labels.numel()
                    stats["records"] += 1
                values[role] = {
                    work: {**stats, "nll_per_token": stats["nll_sum"] / stats["tokens"]}
                    for work, stats in works.items()
                }
        return values

    def checkpoint(name):
        path = output / f"{name}.pt"
        torch.save(
            {
                "schema_version": 1,
                "binding": asdict(binding),
                "config_sha256": report["config_sha256"],
                "adapter_state": {
                    n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
                },
            },
            path,
        )
        report.setdefault("checkpoints", {})[name] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        save()

    trainable = [p for p in model.parameters() if p.requires_grad]
    runner = make_local_trainer(model, tokenizer, device, output, config["sft_learning_rate"])
    times = []
    previous = time.perf_counter()

    def completed():
        nonlocal previous
        synchronize()
        now = time.perf_counter()
        times.append(now - previous)
        previous = now
        report["sft_completed_steps"] = runner.global_step
        if device.type == "mps":
            report["max_observed_driver_bytes"] = max(
                report.get("max_observed_driver_bytes", 0), torch.mps.driver_allocated_memory()
            )
        save()
        budget()

    try:
        report["stages"]["before"] = evaluate()
        save()
        order: list[dict] = []
        rng = random.Random(config["seed"])
        while len(order) < config["sft_steps"]:
            rows = list(data["train"])
            rng.shuffle(rows)
            order.extend(rows)
        previous = time.perf_counter()
        metrics = runner._run_epoch(
            {"summarization": Batches([batch(row) for row in order[: config["sft_steps"]]])},
            train=True,
            epoch=1,
            step_callback=completed,
        )
        report["sft"] = {"metrics": metrics, "step_seconds": times}
        checkpoint("after_sft")
        report["stages"]["after_sft"] = evaluate()
        save()
        # Fresh optimizer state: this is a separate bounded objective stage.
        runner.optimizer = torch.optim.AdamW(
            trainable, lr=config["rl_learning_rate"], weight_decay=0.0, foreach=False
        )
        for index in range(config["rl_steps"]):
            budget()
            row = order[index % len(data["train"])]
            current = batch(row, on_device=True)
            revision = f"{config['pilot_id']}:optimizer-step-{runner.global_step}"
            tick = time.perf_counter()
            sampled = sample_responses(
                model,
                current["src_ids"],
                current["src_mask"],
                group_size=2,
                max_response_tokens=config["max_response_tokens"],
                start_token_id=0,
                end_token_id=1,
                pad_token_id=0,
                temperature=config["temperature"],
                behavior_policy_revision=revision,
                tokenizer_revision=data["provenance"]["inputs"]["tokenizer"]["sha256"],
            )
            outcomes = continuation_rewards(
                [row, row],
                sampled,
                raw_tokenizer,
                minimum_tokens=config["minimum_rewarded_content_tokens"],
            )
            rewards = [item["reward"] for item in outcomes]
            event = {
                "record_id": row["record_id"],
                "work_id": row["work_id"],
                "behavior_revision": revision,
                "outcomes": outcomes,
                "optimizer_step": False,
                "sampling_seconds": time.perf_counter() - tick,
                "response_ids": sampled.response_ids.cpu().tolist(),
                "response_mask": sampled.response_mask.cpu().tolist(),
                "behavior_log_probs": sampled.behavior_log_probs.cpu().tolist(),
            }
            if len(set(rewards)) > 1:
                provenance = tuple(
                    RewardProvenance(
                        "verified_continuation_prefix_min4",
                        report["code_sha256"]["src/training/pilot.py"],
                        row["continuation_target"]["source_evidence_sha256"],
                    )
                    for _ in rewards
                )
                objective = PolicyTask(
                    model,
                    mode="group_relative",
                    start_token_id=0,
                    end_token_id=1,
                    pad_token_id=0,
                    tokenizer_revision=sampled.tokenizer_revision,
                    temperature=config["temperature"],
                    config=GroupRelativeConfig(
                        config["max_response_tokens"],
                        clip_low=config["clip_low"],
                        clip_high=config["clip_high"],
                    ),
                    expected_behavior_revision=revision,
                )
                runner.policy_objectives = {"continuation_rl": objective}
                before = runner.global_step
                policy_batch = sampled.training_batch(
                    torch.tensor(rewards, device=device), provenance
                )
                event["metrics"] = runner._run_epoch(
                    {"continuation_rl": Batches([policy_batch])}, train=True, epoch=index + 1
                )
                event["optimizer_step"] = runner.global_step > before
            else:
                event["skip_reason"] = "flat_group_rewards"
            synchronize()
            event["total_seconds"] = time.perf_counter() - tick
            report["updates"].append(event)
            save()
        checkpoint("after_rl_probe")
        report["stages"]["after_rl_probe"] = evaluate()
        delta = extract_effective_delta(model, binding, task_id="local_continuation")
        report["frozen_base_verified"] = True
        report["effective_delta_metadata_sha256"] = sha(json_bytes(delta.metadata()))
        report["status"] = "completed"
    except (TimeoutError, RuntimeError, ValueError, FloatingPointError) as exc:
        report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        report["elapsed_execution_seconds"] = time.perf_counter() - started
        report["optimizer_steps"] = runner.global_step
        save()
    print(
        json.dumps(
            {key: report[key] for key in ("status", "optimizer_steps", "elapsed_execution_seconds")}
        ),
        flush=True,
    )
    return report


def run_pilot(root: Path, config_path: Path, output: Path, *, prepare_only=False) -> dict:
    """Record failures, including initialization, without touching existing runs."""
    from src.catalog.storage import write_json_atomic

    existed = output.exists()
    try:
        return _run_pilot(root, config_path, output, prepare_only=prepare_only)
    except BaseException as exc:
        report_path = output.resolve() / "report.json"
        if (
            not existed
            and report_path.is_relative_to(root.resolve() / "outputs")
            and report_path.is_file()
        ):
            report = read_json(report_path)
            report.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped",
                error=f"{type(exc).__name__}: {exc}",
            )
            write_json_atomic(report_path, report)
        raise


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run_pilot(root, args.config, args.output, prepare_only=args.prepare_only)
    return 0
