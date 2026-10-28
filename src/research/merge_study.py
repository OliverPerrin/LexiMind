"""Budget-matched joint, specialist and merged LoRA study on book classification.

Question: once the cost of training the experts is counted, do specialists merged
with task arithmetic or TIES match one jointly trained adapter? The protocol and
fixed settings live in ``configs/research/merge_study.json``.

Publisher labels are treated as complete, so absent labels are negatives and the
metrics are standard micro/macro-F1 and macro average precision. Thresholds and
lexical regularisation are chosen on ``dev_select``; results are reported on
``dev_report`` with book-group bootstrap intervals. The test split stays sealed.

    python scripts/research.py merge-study prepare
    python scripts/research.py merge-study lexical --output outputs/merge-study/lexical
    python scripts/research.py merge-study train --tasks genre --seed 17 \\
        --device cuda --output outputs/merge-study/genre-17
    python scripts/research.py merge-study report outputs/merge-study/* \\
        --output research/results/merge_study_baseline.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from src.utils.labels import format_book_input

from .book_fields import map_source_labels
from .book_partitions import group_overrides
from .builders.bgc_source import MEMBERS, records
from .builders.book_fields import jsonl_rows
from .io import file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/research/merge_study.json"
SPLITS = ("train", "dev_select", "dev_report", "test")


# --------------------------------------------------------------------- data


def _dev_half(group: str) -> str:
    return "dev_report" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 else "dev_select"


def prepare(config: dict, root: Path = ROOT) -> dict:
    """Resolve every BGC record to its duplicate-aware split and closed-world labels."""
    data = config["data"]
    archive = root / data["archive"]
    if file_hash(archive) != data["archive_sha256"]:
        raise ValueError("BGC archive differs from the pinned source")
    assignments = {row["record_id"]: row for row in jsonl_rows(root / data["assignments"])}
    overrides = group_overrides(jsonl_rows(root / data["components"]))
    mapping = read_json(root / data["mapping"])
    vocab = {
        task: sorted(mapping["facets"][spec["facet"]]["labels"])
        for task, spec in config["tasks"].items()
    }
    rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    counts = {"train": 0, "dev": 0, "test": 0, "quarantine": 0}
    with zipfile.ZipFile(archive) as source:
        for member_split, member in MEMBERS.items():
            with source.open(member) as stream:
                for number, record in enumerate(records(stream), 1):
                    assigned = assignments[f"bgc:{member_split}:{number}"]
                    component = overrides.get(assigned["group_id"])
                    split = component["proposed_split"] if component else assigned["proposed_split"]
                    group = component["component_id"] if component else assigned["group_id"]
                    if split is None:
                        counts["quarantine"] += 1
                        continue
                    counts[split] += 1
                    fields = map_source_labels(record["labels"], mapping)
                    row = {
                        "id": f"bgc:{member_split}:{number}",
                        "group": group,
                        "text": format_book_input(record["title"], record["body"]),
                    }
                    for task, spec in config["tasks"].items():
                        row[task] = fields[spec["facet"]]["positive"]
                    rows[_dev_half(group) if split == "dev" else split].append(row)
    if counts != data["expected_rows"]:
        raise ValueError(f"Split counts {counts} differ from {data['expected_rows']}")
    cache = root / data["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "counts": counts,
        "splits": {},
    }
    for split, values in rows.items():
        path = cache / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["splits"][split] = {
            "rows": len(values),
            "groups": len({row["group"] for row in values}),
            "sha256": file_hash(path),
            "label_support": {
                task: {label: sum(label in row[task] for row in values) for label in labels}
                for task, labels in vocab.items()
            },
        }
    summary["vocabulary"] = vocab
    (cache / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def load(config: dict, split: str, root: Path = ROOT) -> tuple[list[dict], dict[str, list[str]]]:
    if split == "test" and config["evaluation"]["test_sealed"]:
        raise ValueError("The test split is sealed until the final study run")
    cache = root / config["data"]["cache"]
    summary = read_json(cache / "summary.json")
    path = cache / f"{split}.jsonl.gz"
    if file_hash(path) != summary["splits"][split]["sha256"]:
        raise ValueError(f"Prepared {split} rows changed; rerun prepare")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle], summary["vocabulary"]


def targets(rows: list[dict], task: str, labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    result: np.ndarray = np.zeros((len(rows), len(labels)), dtype=bool)
    for i, row in enumerate(rows):
        result[i, [index[label] for label in row[task]]] = True
    return result


def _save(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


# ------------------------------------------------------------------ metrics


def group_weights(groups: list[str], resamples: int, seed: int) -> np.ndarray:
    """Row weights for a bootstrap over book groups; row 0 is the unweighted sample."""
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=resamples)
    weights: np.ndarray = np.vstack([np.ones(len(groups)), counts[:, inverse]]).astype(np.float64)
    return weights


def metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    weights: np.ndarray,
    min_support: int = 1,
) -> dict:
    """Micro/macro-F1 and macro-AP for each weight row over labels with enough positives."""
    supported = truth.sum(axis=0) >= min_support
    probs, truth = probs[:, supported], truth[:, supported]
    predicted = probs >= threshold
    with np.errstate(all="ignore"):  # macOS Accelerate raises spurious matmul FP flags.
        tp = weights @ (predicted & truth).astype(np.float64)
        fp = weights @ (predicted & ~truth).astype(np.float64)
        fn = weights @ (~predicted & truth).astype(np.float64)
        if not (np.isfinite(tp).all() and np.isfinite(fp).all() and np.isfinite(fn).all()):
            raise FloatingPointError("Nonfinite weighted confusion counts")
        micro = 2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1))
        macro = np.nanmean(2 * tp / (2 * tp + fp + fn), axis=1)
        precision = []
        for label in range(truth.shape[1]):
            order = np.argsort(-probs[:, label], kind="stable")
            scores, hits = probs[order, label], truth[order, label]
            weighted = weights[:, order]
            # Tied scores form one threshold, as in scikit-learn's average precision.
            ends = np.r_[scores[1:] != scores[:-1], True]
            found = np.cumsum(weighted * hits, axis=1)[:, ends]
            seen = np.cumsum(weighted, axis=1)[:, ends]
            gained = np.diff(found, axis=1, prepend=0)
            at_ends = np.divide(gained * found, seen, out=np.zeros_like(seen), where=seen > 0)
            precision.append(at_ends.sum(1) / found[:, -1])
        average_precision = np.nanmean(np.stack(precision, axis=1), axis=1)
    return {"micro_f1": micro, "macro_f1": macro, "macro_ap": average_precision}


def select_threshold(
    probs: np.ndarray, truth: np.ndarray, grid: list[float], min_support: int = 1
) -> float:
    ones = np.ones((1, len(truth)))
    scores = [metrics(probs, truth, t, ones, min_support)["micro_f1"][0] for t in grid]
    return float(grid[int(np.argmax(scores))])


def _interval(values: np.ndarray) -> list[float]:
    bounds = np.asarray(np.nanpercentile(values, [2.5, 97.5]), dtype=np.float64)
    return [float(bounds[0]), float(bounds[1])]


def summarize(per_seed: list[dict], baseline: dict | None = None) -> dict:
    """Seed-mean point estimate, 95% bootstrap interval and seed spread per metric."""
    result: dict = {}
    for name in per_seed[0]:
        values = np.mean([seed[name] for seed in per_seed], axis=0)
        entry = {
            "mean": float(values[0]),
            "ci95": _interval(values[1:]),
            "seed_sd": float(np.std([seed[name][0] for seed in per_seed], ddof=1))
            if len(per_seed) > 1
            else 0.0,
            "seeds": [float(seed[name][0]) for seed in per_seed],
        }
        if baseline is not None:
            difference = values - baseline[name]
            entry["minus_baseline"] = {
                "mean": float(difference[0]),
                "ci95": _interval(difference[1:]),
            }
        result[name] = entry
    return result


# ------------------------------------------------------------------ lexical


def lexical(config: dict, output: Path, root: Path = ROOT) -> dict:
    """TF-IDF one-vs-rest logistic regression; C is chosen by dev_select macro-AP."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    settings, evaluation = config["lexical"], config["evaluation"]
    train, vocab = load(config, "train", root)
    select, _ = load(config, evaluation["selection_split"], root)
    report, _ = load(config, evaluation["report_split"], root)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=tuple(settings["ngram_range"]),
        min_df=settings["min_df"],
        max_features=settings["max_features"],
    )
    features = {
        "train": vectorizer.fit_transform([row["text"] for row in train]),
        "select": vectorizer.transform([row["text"] for row in select]),
        "report": vectorizer.transform([row["text"] for row in report]),
    }
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {"kind": "lexical", "tasks": {}}
    probabilities: dict[str, dict] = {"select": {}, "report": {}}
    for task, labels in vocab.items():
        truth = {
            name: targets(rows, task, labels)
            for name, rows in (("train", train), ("select", select))
        }
        best = None
        for c in settings["C_grid"]:
            model = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear"), n_jobs=-1)
            model.fit(features["train"], truth["train"])
            probs = model.predict_proba(features["select"])
            score = metrics(
                probs,
                truth["select"],
                0.5,
                np.ones((1, len(select))),
                evaluation["min_label_support"],
            )["macro_ap"][0]
            if best is None or score > best[0]:
                best = (score, c, model, probs)
        assert best is not None
        _, c, model, select_probs = best
        probabilities["select"][task] = select_probs.astype(np.float32)
        probabilities["report"][task] = model.predict_proba(features["report"]).astype(np.float32)
        result["tasks"][task] = {"C": c}
    _save(output / "select_probs.npz", probabilities["select"])
    _save(output / "report_probs.npz", probabilities["report"])
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------------- neural


def _runtime(device: str, root: Path):
    from hydra import compose, initialize_config_dir

    from src.data.tokenization import Tokenizer, TokenizerConfig
    from src.training.utils import prepare_training_runtime

    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["training=book_lora", f"device={device}"])
    torch_device, snapshot = prepare_training_runtime(cfg)
    tokenizer = Tokenizer(TokenizerConfig(pretrained_model_name=str(snapshot), max_length=256))
    return cfg, torch_device, snapshot, tokenizer


def build_model(
    config: dict, cfg, tokenizer, snapshot, seed: int, train_tasks: list[str], vocab: dict
):
    """Identical base and heads for every arm of one seed; only ``train_tasks`` heads adapt."""
    import torch

    from src.models.adapters import LoRAConfig, attach_lora
    from src.models.heads import ClassificationHead
    from src.training.utils import build_native_model

    model = build_native_model(cfg, tokenizer, num_emotions=0, num_topics=0, snapshot=snapshot)
    head = config["model"]["head"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for task in config["tasks"]:
            model.add_head(
                task,
                ClassificationHead(
                    cfg.model.d_model,
                    len(vocab[task]),
                    pooler=head["pooler"],
                    dropout=head["dropout"],
                    problem_type="multi_label",
                ),
            )
    lora = config["model"]["lora"]
    paths = [
        f"encoder.layers.{layer}.{'self_attn' if target.startswith('W_') else 'ffn'}.{target}"
        for layer in range(cfg.model.num_encoder_layers)
        for target in lora["targets"]
    ]
    binding = attach_lora(
        model,
        shared_projections=paths,
        private_heads=[f"head_{task}" for task in train_tasks],
        config=LoRAConfig(
            rank=lora["rank"], alpha=lora["alpha"], dropout=lora["dropout"], seed=seed
        ),
    )
    return model, binding


def _encode(tokenizer, rows: list[dict], max_length: int) -> list[list[int]]:
    ids = []
    for begin in range(0, len(rows), 1024):
        batch = tokenizer._tokenizer(
            [row["text"] for row in rows[begin : begin + 1024]],
            truncation=True,
            max_length=max_length,
        )
        ids.extend(batch["input_ids"])
    return ids


def _batch(ids: list[list[int]], indices, pad: int, device):
    import torch

    width = -(-max(len(ids[i]) for i in indices) // 8) * 8
    tokens = torch.full((len(indices), width), pad, dtype=torch.long)
    for row, i in enumerate(indices):
        tokens[row, : len(ids[i])] = torch.tensor(ids[i])
    tokens = tokens.to(device)
    return tokens, tokens != pad


def predict(model, ids, tasks, pad, device, batch_size=128) -> dict[str, np.ndarray]:
    import torch

    from src.models.multitask import MultiTaskModel

    model.eval()
    outputs: dict[str, list] = {task: [] for task in tasks}
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        for begin in range(0, len(ids), batch_size):
            tokens, mask = _batch(ids, range(begin, min(begin + batch_size, len(ids))), pad, device)
            encoded = model.encoder(
                tokens, mask=MultiTaskModel._expand_attention_mask(mask, device)
            )
            for task in tasks:
                logits = model.classify_encoded(task, encoded, mask)
                outputs[task].append(torch.sigmoid(logits.float()).cpu().numpy())
    return {task: np.concatenate(values) for task, values in outputs.items()}


def train(
    config: dict,
    tasks: list[str],
    seed: int,
    updates: int,
    device: str,
    output: Path,
    root: Path = ROOT,
    microbatch: int | None = None,
) -> dict:
    """One adapter trained on ``tasks`` for a fixed number of updates; no checkpoint selection."""
    import math

    import torch
    from torch.nn import functional as F

    if not tasks or set(tasks) - set(config["tasks"]):
        raise ValueError(f"Tasks must be drawn from {sorted(config['tasks'])}")
    if {config["tasks"][task]["source"] for task in tasks} != {"bgc"}:
        raise NotImplementedError("Mixed-source joint batches arrive with the emotion task")
    output.mkdir(parents=True, exist_ok=False)
    cfg, torch_device, snapshot, tokenizer = _runtime(device, root)
    training, evaluation = config["training"], config["evaluation"]
    train_rows, vocab = load(config, "train", root)
    evaluation_rows = {
        split: load(config, evaluation[split], root)[0]
        for split in ("selection_split", "report_split")
    }
    max_length = config["model"]["max_length"]
    ids = _encode(tokenizer, train_rows, max_length)
    truth = {
        task: torch.from_numpy(targets(train_rows, task, vocab[task])).float() for task in tasks
    }
    model, binding = build_model(config, cfg, tokenizer, snapshot, seed, tasks, vocab)
    model.to(torch_device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training["lr"],
        weight_decay=training["weight_decay"],
        fused=torch_device.type == "cuda",
    )
    warmup = max(1, round(training["warmup_fraction"] * updates))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup, max(0.0, (updates - step) / max(1, updates - warmup))
        ),
    )
    batch_size, pad = training["batch_size"], tokenizer.pad_token_id
    microbatch = microbatch or batch_size
    if batch_size % microbatch:
        raise ValueError("The microbatch must divide the protocol batch size")
    generator = torch.Generator().manual_seed(seed)
    order: list[int] = []
    curve = []
    from src.models.multitask import MultiTaskModel

    def synchronize():
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        elif torch_device.type == "mps":
            torch.mps.synchronize()

    torch.manual_seed(seed)  # Dropout draws; LoRA and head initialisation are seeded above.
    model.train()
    synchronize()
    started = time.perf_counter()
    running = 0.0
    for step in range(updates):
        if len(order) < batch_size:
            order.extend(torch.randperm(len(ids), generator=generator).tolist())
        indices, order = order[:batch_size], order[batch_size:]
        loss_value = 0.0
        for begin in range(0, batch_size, microbatch):
            chunk = indices[begin : begin + microbatch]
            tokens, mask = _batch(ids, chunk, pad, torch_device)
            with torch.autocast(
                torch_device.type, dtype=torch.bfloat16, enabled=torch_device.type == "cuda"
            ):
                encoded = model.encoder(
                    tokens, mask=MultiTaskModel._expand_attention_mask(mask, torch_device)
                )
                losses = [
                    F.binary_cross_entropy_with_logits(
                        model.classify_encoded(task, encoded, mask).float(),
                        truth[task][chunk].to(torch_device),
                    )
                    for task in tasks
                ]
            # Chunk-size weighting keeps the batch-32 mean; only dropout draws differ.
            loss = torch.stack(losses).mean() * (len(chunk) / batch_size)
            loss.backward()
            loss_value += loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        torch.nn.utils.clip_grad_norm_(parameters, training["max_grad_norm"])
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss_value
        if (step + 1) % 50 == 0 or step + 1 == updates:
            curve.append({"update": step + 1, "train_loss": running / ((step % 50) + 1)})
            running = 0.0
            print(f"  update {step + 1}/{updates} loss {curve[-1]['train_loss']:.4f}", flush=True)
    synchronize()
    seconds = time.perf_counter() - started
    probabilities = {
        split: predict(
            model, _encode(tokenizer, rows, max_length), tasks, pad, torch_device, 4 * microbatch
        )
        for split, rows in evaluation_rows.items()
    }
    result: dict = {
        "kind": "neural",
        "tasks": {},
        "seed": seed,
        "updates": updates,
        "examples": updates * batch_size,
        "microbatch": microbatch,
        "train_seconds": seconds,
        "device": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else torch_device.type,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "curve": curve,
    }
    result["tasks"] = {task: {} for task in tasks}
    _save(output / "report_probs.npz", probabilities["report_split"])
    _save(output / "select_probs.npz", probabilities["selection_split"])
    torch.save(
        {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        output / "adapter.pt",
    )
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------------- report


def _score(select_probs, report_probs, select_truth, truth, weights, evaluation):
    """Threshold from dev_select by micro-F1, then bootstrap metrics on dev_report."""
    support = evaluation["min_label_support"]
    threshold = select_threshold(select_probs, select_truth, evaluation["threshold_grid"], support)
    return threshold, metrics(report_probs, truth, threshold, weights, support)


def report(config: dict, runs: list[Path], root: Path = ROOT) -> dict:
    """Seed-averaged neural results per task against the lexical baseline, on dev_report."""
    evaluation = config["evaluation"]
    rows, vocab = load(config, evaluation["report_split"], root)
    weights = group_weights(
        [row["group"] for row in rows],
        evaluation["bootstrap_resamples"],
        evaluation["bootstrap_seed"],
    )
    select_rows, _ = load(config, evaluation["selection_split"], root)
    loaded = [
        (
            read_json(path / "run.json"),
            np.load(path / "select_probs.npz"),
            np.load(path / "report_probs.npz"),
        )
        for path in runs
    ]
    lexical_run = next((item for item in loaded if item[0]["kind"] == "lexical"), None)
    summary: dict = {
        "study_id": config["study_id"],
        "split": evaluation["report_split"],
        "rows": len(rows),
        "tasks": {},
    }
    support = evaluation["min_label_support"]
    for task, labels in vocab.items():
        truth, select_truth = targets(rows, task, labels), targets(select_rows, task, labels)

        summary["tasks"][task] = {
            "excluded_labels": [
                label for label, n in zip(labels, truth.sum(0), strict=True) if n < support
            ]
        }
        baseline = None
        if lexical_run is not None:
            run, select_probs, report_probs = lexical_run
            threshold, baseline = _score(
                select_probs[task], report_probs[task], select_truth, truth, weights, evaluation
            )
            summary["tasks"][task]["lexical"] = summarize([baseline]) | {
                "C": run["tasks"][task]["C"],
                "thresholds": [threshold],
            }
        arms: dict[str, list] = {}
        for run, select_probs, report_probs in loaded:
            if run["kind"] == "neural" and task in run["tasks"]:
                arm = "+".join(sorted(run["tasks"])) + f"@{run['updates']}"
                scored = _score(
                    select_probs[task], report_probs[task], select_truth, truth, weights, evaluation
                )
                arms.setdefault(arm, []).append((run["seed"], *scored))
        for arm, values in arms.items():
            values.sort(key=lambda item: item[0])
            summary["tasks"][task][arm] = summarize([v[2] for v in values], baseline) | {
                "seeds_run": [v[0] for v in values],
                "thresholds": [v[1] for v in values],
            }
    return summary


# ---------------------------------------------------------------------- CLI


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CONFIG)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prepare", help="Resolve duplicate-aware splits and closed-world labels")
    lexical_parser = actions.add_parser("lexical", help="Fit the tuned TF-IDF baseline")
    lexical_parser.add_argument("--output", type=Path, required=True)
    train_parser = actions.add_parser("train", help="Train one LoRA adapter for a fixed budget")
    train_parser.add_argument(
        "--tasks", required=True, help="Comma-separated, e.g. genre or genre,topic"
    )
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument(
        "--updates", type=int, help="Defaults to the protocol baseline budget"
    )
    train_parser.add_argument("--device", choices=["cuda", "mps"], required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument(
        "--microbatch", type=int, help="Gradient-accumulation chunk; must divide the batch"
    )
    report_parser = actions.add_parser("report", help="Summarize run directories")
    report_parser.add_argument("runs", type=Path, nargs="+")
    report_parser.add_argument("--output", type=Path, required=True)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config = read_json(args.config)
    if args.action == "prepare":
        summary = prepare(config)
        print(json.dumps({split: value["rows"] for split, value in summary["splits"].items()}))
    elif args.action == "lexical":
        print(json.dumps(lexical(config, args.output), indent=2))
    elif args.action == "train":
        updates = args.updates or config["training"]["baseline_updates"]
        result = train(
            config,
            args.tasks.split(","),
            args.seed,
            updates,
            args.device,
            args.output,
            microbatch=args.microbatch,
        )
        print(
            json.dumps(
                {key: result[key] for key in ("tasks", "examples", "train_seconds")}, indent=2
            )
        )
    else:
        summary = report(config, args.runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        for task, arms in summary["tasks"].items():
            print(f"{task}: excluded {arms['excluded_labels']}")
            for arm, values in arms.items():
                if arm == "excluded_labels":
                    continue
                cells = "  ".join(
                    f"{name} {values[name]['mean']:.4f} [{values[name]['ci95'][0]:.4f}, {values[name]['ci95'][1]:.4f}]"
                    for name in ("micro_f1", "macro_f1", "macro_ap")
                )
                print(f"{task:6} {arm:14} {cells}")
    return 0
