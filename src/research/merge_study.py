"""Budget-matched joint, specialist and merged LoRA study on book classification.

Question: once the cost of training the experts is counted, do specialists merged
with task arithmetic or TIES match one jointly trained adapter? The protocol and
fixed settings live in ``configs/research/merge_study.json``.

Tasks: BGC genre and topic (multi-label; publisher labels treated as complete, so
absent labels are negatives) and CR4 narrative emotion (a distribution over eight
categories from the annotators' answers). Thresholds, lexical regularisation and
merge scales are chosen on ``dev_select``; results are reported on ``dev_report``
with group bootstrap intervals. The test split stays sealed.

    python scripts/research.py merge-study prepare
    python scripts/research.py merge-study lexical --output outputs/merge-study/lexical
    python scripts/research.py merge-study train --tasks genre --seed 17 \\
        --device cuda --output outputs/merge-study/genre-17
    python scripts/research.py merge-study merge --method ties --device cuda \\
        --output outputs/merge-study/ties-17 outputs/merge-study/{genre,topic,emotion}-17
    python scripts/research.py merge-study report outputs/merge-study/* \\
        --output research/results/merge_study.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
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


def _unit(group: str) -> float:
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) / 2**256


def _dev_half(group: str) -> str:
    return "dev_report" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 else "dev_select"


def _bgc_rows(config: dict, root: Path) -> tuple[dict[str, list[dict]], dict]:
    """Every BGC record in its duplicate-aware split, with closed-world genre/topic labels."""
    data = config["data"]
    archive = root / data["archive"]
    if file_hash(archive) != data["archive_sha256"]:
        raise ValueError("BGC archive differs from the pinned source")
    assignments = {row["record_id"]: row for row in jsonl_rows(root / data["assignments"])}
    overrides = group_overrides(jsonl_rows(root / data["components"]))
    mapping = read_json(root / data["mapping"])
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
                        if spec["source"] == "bgc":
                            row[task] = fields[spec["facet"]]["positive"]
                    rows[_dev_half(group) if split == "dev" else split].append(row)
    if counts != data["expected_rows"]:
        raise ValueError(f"Split counts {counts} differ from {data['expected_rows']}")
    vocab = {
        task: sorted(mapping["facets"][spec["facet"]]["labels"])
        for task, spec in config["tasks"].items()
        if spec["source"] == "bgc"
    }
    return rows, vocab


def _cr4_rows(config: dict, root: Path) -> tuple[dict[str, list[dict]], dict]:
    """One row per passage and highlighted character, grouped by linked source documents."""
    spec = config["data"]["cr4"]
    path = root / spec["csv"]
    if hashlib.md5(path.read_bytes()).hexdigest() != spec["md5"]:
        raise ValueError(f"CR4 file differs from the provider MD5; download {spec['url']}")
    tasks = {t: s for t, s in config["tasks"].items() if s["source"] == "cr4"}
    conditions: dict[tuple[str, str], dict] = {}
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    passages: dict[str, str] = {}
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["t0"] != "Yes":
                continue
            document = f"{record['workflow_name']}:{record['file_id']}"
            key = (record["subject_ids"], record["highlighted_char"])
            condition = conditions.setdefault(
                key,
                {
                    "passage": record["passage"],
                    "document": document,
                    **{task: Counter() for task in tasks},
                },
            )
            if condition["passage"] != record["passage"]:
                raise ValueError(f"CR4 condition {key} has more than one passage")
            for task, task_spec in tasks.items():
                label = record[task_spec["column"]]
                if label not in task_spec["labels"]:
                    raise ValueError(f"Unexpected {task_spec['column']} value: {label!r}")
                condition[task][label] += 1
            # Documents that share a passage or a subject belong to one group.
            for linked in (passages.setdefault(record["passage"], document), condition["document"]):
                parent[find(document)] = find(linked)
    rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    fractions = spec["split"]
    for (subject, character), condition in sorted(conditions.items()):
        group = "cr4-documents:" + find(condition["document"])
        position = _unit(group)
        if position < fractions["train"]:
            split = "train"
        elif position < fractions["train"] + fractions["dev"]:
            split = _dev_half(group)
        else:
            split = "test"
        row: dict = {
            "id": f"cr4:{subject}:{hashlib.sha256(character.encode()).hexdigest()[:12]}",
            "group": group,
            "text": f"Character: {character}\nPassage: {condition['passage']}",
        }
        for task, task_spec in tasks.items():
            row[task] = [condition[task][label] for label in task_spec["labels"]]
        rows[split].append(row)
    return rows, {task: list(task_spec["labels"]) for task, task_spec in tasks.items()}


def _support(rows: list[dict], task: str, spec: dict, labels: list[str]) -> dict:
    if spec["type"] == "multi_label":
        return {label: sum(label in row[task] for row in rows) for label in labels}
    majority = Counter(labels[int(np.argmax(row[task]))] for row in rows)
    return {label: majority[label] for label in labels}


def prepare(config: dict, root: Path = ROOT) -> dict:
    """Write each source's split rows and a summary with hashes and label support."""
    cache = root / config["data"]["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    summary: dict = {"schema_version": 2, "study_id": config["study_id"], "sources": {}}
    vocabulary: dict[str, list[str]] = {}
    for source, builder in (("bgc", _bgc_rows), ("cr4", _cr4_rows)):
        rows, vocab = builder(config, root)
        vocabulary |= vocab
        summary["sources"][source] = {}
        for split, values in rows.items():
            path = cache / f"{source}_{split}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in values:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary["sources"][source][split] = {
                "rows": len(values),
                "groups": len({row["group"] for row in values}),
                "sha256": file_hash(path),
                "support": {
                    task: _support(values, task, config["tasks"][task], labels)
                    for task, labels in vocab.items()
                },
            }
    summary["vocabulary"] = vocabulary
    (cache / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def load(config: dict, source: str, split: str, root: Path = ROOT) -> list[dict]:
    if split == "test" and config["evaluation"]["test_sealed"]:
        raise ValueError("The test split is sealed until the final study run")
    cache = root / config["data"]["cache"]
    path = cache / f"{source}_{split}.jsonl.gz"
    if file_hash(path) != read_json(cache / "summary.json")["sources"][source][split]["sha256"]:
        raise ValueError(f"Prepared {source} {split} rows changed; rerun prepare")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def vocabulary(config: dict, root: Path = ROOT) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = read_json(root / config["data"]["cache"] / "summary.json")[
        "vocabulary"
    ]
    return vocab


def targets(rows: list[dict], task: str, labels: list[str]) -> np.ndarray:
    """Boolean multi-label matrix, or annotator counts for a distribution task."""
    if rows and len(rows[0][task]) == len(labels) and all(type(v) is int for v in rows[0][task]):
        counts: np.ndarray = np.asarray([row[task] for row in rows], dtype=np.float64)
        return counts
    index = {label: i for i, label in enumerate(labels)}
    result: np.ndarray = np.zeros((len(rows), len(labels)), dtype=bool)
    for i, row in enumerate(rows):
        result[i, [index[label] for label in row[task]]] = True
    return result


def _save(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


# ------------------------------------------------------------------ metrics


def group_weights(groups: list[str], resamples: int, seed: int) -> np.ndarray:
    """Row weights for a bootstrap over groups; row 0 is the unweighted sample."""
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


def distribution_metrics(probs: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> dict:
    """Accuracy and macro-F1 against a unique majority label, plus annotator agreement.

    Agreement is the share of annotators who chose the predicted category, over all rows.
    """
    predicted = probs.argmax(axis=1)
    majority = counts.argmax(axis=1)
    unique = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) == 1
    share = counts[np.arange(len(counts)), predicted] / counts.sum(axis=1)
    scoped = weights * unique
    classes = np.arange(counts.shape[1])
    hit = (predicted[:, None] == classes) & (majority[:, None] == classes)
    with np.errstate(all="ignore"):
        tp = scoped @ hit.astype(np.float64)
        fp = scoped @ ((predicted[:, None] == classes) & ~hit).astype(np.float64)
        fn = scoped @ ((majority[:, None] == classes) & ~hit).astype(np.float64)
        return {
            "accuracy": scoped @ (predicted == majority).astype(np.float64) / scoped.sum(axis=1),
            "macro_f1": np.nanmean(2 * tp / (2 * tp + fp + fn), axis=1),
            "agreement": weights @ share / weights.sum(axis=1),
        }


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


def selection_score(config: dict, probs: dict, rows: dict, vocab: dict) -> float:
    """Mean dev_select score used to pick merge scales: macro-AP or distribution macro-F1."""
    scores = []
    support = config["evaluation"]["min_label_support"]
    for task, spec in config["tasks"].items():
        truth = targets(rows[spec["source"]], task, vocab[task])
        ones = np.ones((1, len(truth)))
        if spec["type"] == "multi_label":
            scores.append(metrics(probs[task], truth, 0.5, ones, support)["macro_ap"][0])
        else:
            scores.append(distribution_metrics(probs[task], truth, ones)["macro_f1"][0])
    return float(np.mean(scores))


# ------------------------------------------------------------------ lexical


def lexical(config: dict, output: Path, root: Path = ROOT) -> dict:
    """TF-IDF logistic regression per task; C is chosen on dev_select."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    settings, evaluation = config["lexical"], config["evaluation"]
    vocab = vocabulary(config, root)
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {"kind": "lexical", "tasks": {}}
    probabilities: dict[str, dict] = {"select": {}, "report": {}}
    for source in sorted({spec["source"] for spec in config["tasks"].values()}):
        rows = {
            name: load(config, source, split, root)
            for name, split in (
                ("train", "train"),
                ("select", evaluation["selection_split"]),
                ("report", evaluation["report_split"]),
            )
        }
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=tuple(settings["ngram_range"]),
            min_df=settings["min_df"],
            max_features=settings["max_features"],
        )
        features = {"train": vectorizer.fit_transform([row["text"] for row in rows["train"]])}
        for name in ("select", "report"):
            features[name] = vectorizer.transform([row["text"] for row in rows[name]])
        for task, spec in config["tasks"].items():
            if spec["source"] != source:
                continue
            truth = {name: targets(rows[name], task, vocab[task]) for name in ("train", "select")}
            ones = np.ones((1, len(rows["select"])))
            best = None
            for c in settings["C_grid"]:
                if spec["type"] == "multi_label":
                    model = OneVsRestClassifier(
                        LogisticRegression(C=c, solver="liblinear"), n_jobs=-1
                    ).fit(features["train"], truth["train"])
                    probs = model.predict_proba(features["select"])
                    score = metrics(
                        probs, truth["select"], 0.5, ones, evaluation["min_label_support"]
                    )["macro_ap"][0]
                else:
                    # Annotator counts become sample weights: a soft-label cross-entropy.
                    row_index, label_index = np.nonzero(truth["train"])
                    model = LogisticRegression(C=c, max_iter=2000).fit(
                        features["train"][row_index],
                        label_index,
                        sample_weight=truth["train"][row_index, label_index],
                    )
                    probs = model.predict_proba(features["select"])
                    score = distribution_metrics(probs, truth["select"], ones)["macro_f1"][0]
                if best is None or score > best[0]:
                    best = (score, c, model, probs)
            assert best is not None
            _, c, model, select_probs = best
            probabilities["select"][task] = select_probs.astype(np.float32)
            probabilities["report"][task] = model.predict_proba(features["report"]).astype(
                np.float32
            )
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


def build_base(config: dict, cfg, tokenizer, snapshot, seed: int, vocab: dict):
    """The pristine model and every task head, identical for all arms of one seed."""
    import torch

    from src.models.heads import ClassificationHead
    from src.training.utils import build_native_model

    model = build_native_model(cfg, tokenizer, num_emotions=0, num_topics=0, snapshot=snapshot)
    head = config["model"]["head"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for task, spec in config["tasks"].items():
            model.add_head(
                task,
                ClassificationHead(
                    cfg.model.d_model,
                    len(vocab[task]),
                    pooler=head["pooler"],
                    dropout=head["dropout"],
                    problem_type="multi_label" if spec["type"] == "multi_label" else "single_label",
                ),
            )
    return model


def attach(config: dict, cfg, model, seed: int, train_tasks: list[str]):
    """Shared LoRA on every listed encoder projection; only ``train_tasks`` heads adapt."""
    from src.models.adapters import LoRAConfig, attach_lora

    lora = config["model"]["lora"]
    paths = [
        f"encoder.layers.{layer}.{'self_attn' if target.startswith('W_') else 'ffn'}.{target}"
        for layer in range(cfg.model.num_encoder_layers)
        for target in lora["targets"]
    ]
    return attach_lora(
        model,
        shared_projections=paths,
        private_heads=[f"head_{task}" for task in train_tasks],
        config=LoRAConfig(
            rank=lora["rank"], alpha=lora["alpha"], dropout=lora["dropout"], seed=seed
        ),
    )


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


def predict(config, model, ids, tasks, pad, device, batch_size=128) -> dict[str, np.ndarray]:
    """Sigmoid probabilities for multi-label tasks and softmax for distribution tasks."""
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
                logits = model.classify_encoded(task, encoded, mask).float()
                if config["tasks"][task]["type"] == "multi_label":
                    outputs[task].append(torch.sigmoid(logits).cpu().numpy())
                else:
                    outputs[task].append(torch.softmax(logits, dim=-1).cpu().numpy())
    return {task: np.concatenate(values) for task, values in outputs.items()}


def _evaluate(config, model, tokenizer, tasks, device, batch_size, root) -> dict:
    """Probabilities on the selection and report splits, keyed by split then task."""
    evaluation, max_length = config["evaluation"], config["model"]["max_length"]
    result: dict[str, dict] = {"select": {}, "report": {}}
    for source in sorted({config["tasks"][task]["source"] for task in tasks}):
        source_tasks = [task for task in tasks if config["tasks"][task]["source"] == source]
        for name, split in (
            ("select", evaluation["selection_split"]),
            ("report", evaluation["report_split"]),
        ):
            ids = _encode(tokenizer, load(config, source, split, root), max_length)
            result[name] |= predict(
                config, model, ids, source_tasks, tokenizer.pad_token_id, device, batch_size
            )
    return result


def _schedule(config: dict, sources: list[str], updates: int) -> list[str]:
    """Which source each update draws from: the fixed joint pattern, or the one source."""
    if len(sources) == 1:
        return sources * updates
    pattern = [source for source in config["budget"]["joint_batch_pattern"] if source in sources]
    if set(pattern) != set(sources):
        raise ValueError(f"The joint batch pattern must cover {sources}")
    return [pattern[step % len(pattern)] for step in range(updates)]


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

    from src.models.multitask import MultiTaskModel

    if not tasks or set(tasks) - set(config["tasks"]) or len(set(tasks)) != len(tasks):
        raise ValueError(f"Tasks must be distinct names from {sorted(config['tasks'])}")
    tasks = [task for task in config["tasks"] if task in tasks]
    output.mkdir(parents=True, exist_ok=False)
    cfg, torch_device, snapshot, tokenizer = _runtime(device, root)
    training = config["training"]
    vocab = vocabulary(config, root)
    sources = sorted({config["tasks"][task]["source"] for task in tasks})
    streams: dict[str, dict] = {}
    for source in sources:
        rows = load(config, source, "train", root)
        streams[source] = {
            "ids": _encode(tokenizer, rows, config["model"]["max_length"]),
            "tasks": [task for task in tasks if config["tasks"][task]["source"] == source],
            "order": [],
        }
        streams[source]["truth"] = {
            task: torch.from_numpy(targets(rows, task, vocab[task])).float()
            for task in streams[source]["tasks"]
        }
    model = build_base(config, cfg, tokenizer, snapshot, seed, vocab)
    attach(config, cfg, model, seed, tasks)
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
    plan = _schedule(config, sources, updates)
    curve = []

    def synchronize():
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        elif torch_device.type == "mps":
            torch.mps.synchronize()

    def task_loss(task, logits, truth):
        if config["tasks"][task]["type"] == "multi_label":
            return F.binary_cross_entropy_with_logits(logits, truth)
        target = truth / truth.sum(dim=1, keepdim=True)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()

    torch.manual_seed(seed)  # Dropout draws; LoRA and head initialisation are seeded above.
    model.train()
    synchronize()
    started = time.perf_counter()
    running = 0.0
    for step, source in enumerate(plan):
        stream = streams[source]
        if len(stream["order"]) < batch_size:
            stream["order"].extend(torch.randperm(len(stream["ids"]), generator=generator).tolist())
        indices, stream["order"] = stream["order"][:batch_size], stream["order"][batch_size:]
        loss_value = 0.0
        for begin in range(0, batch_size, microbatch):
            chunk = indices[begin : begin + microbatch]
            tokens, mask = _batch(stream["ids"], chunk, pad, torch_device)
            with torch.autocast(
                torch_device.type, dtype=torch.bfloat16, enabled=torch_device.type == "cuda"
            ):
                encoded = model.encoder(
                    tokens, mask=MultiTaskModel._expand_attention_mask(mask, torch_device)
                )
                losses = [
                    task_loss(
                        task,
                        model.classify_encoded(task, encoded, mask).float(),
                        stream["truth"][task][chunk].to(torch_device),
                    )
                    for task in stream["tasks"]
                ]
            # Chunk-size weighting keeps the batch mean; only dropout draws differ.
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
    probabilities = _evaluate(config, model, tokenizer, tasks, torch_device, 4 * microbatch, root)
    result: dict = {
        "kind": "neural",
        "tasks": {task: {} for task in tasks},
        "seed": seed,
        "updates": updates,
        "examples": updates * batch_size,
        "batches_by_source": dict(Counter(plan)),
        "microbatch": microbatch,
        "train_seconds": seconds,
        "device": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else torch_device.type,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "curve": curve,
    }
    _save(output / "report_probs.npz", probabilities["report"])
    _save(output / "select_probs.npz", probabilities["select"])
    torch.save(
        {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        output / "adapter.pt",
    )
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def merge(
    config: dict,
    specialists: list[Path],
    method: str,
    device: str,
    output: Path,
    root: Path = ROOT,
    batch_size: int = 128,
) -> dict:
    """Merge one seed's specialists; the scale is chosen on dev_select and its search counted."""
    from copy import deepcopy

    import torch

    from src.models.adapters import extract_effective_delta, materialize_merge, merge_deltas

    runs = [read_json(path / "run.json") for path in specialists]
    tasks = [task for run in runs for task in run["tasks"]]
    if (
        any(run["kind"] != "neural" or len(run["tasks"]) != 1 for run in runs)
        or sorted(tasks) != sorted(config["tasks"])
        or len({run["seed"] for run in runs}) != 1
    ):
        raise ValueError("Merging needs one single-task specialist per study task, one seed")
    seed = runs[0]["seed"]
    output.mkdir(parents=True, exist_ok=False)
    cfg, torch_device, snapshot, tokenizer = _runtime(device, root)
    vocab = vocabulary(config, root)
    base = build_base(config, cfg, tokenizer, snapshot, seed, vocab)
    deltas = []
    for path, task in zip(specialists, tasks, strict=True):
        model = deepcopy(base)
        binding = attach(config, cfg, model, seed, [task])
        state = torch.load(path / "adapter.pt", map_location="cpu")
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(state) != trainable:
            raise ValueError(f"{path} does not hold exactly the {task} specialist's parameters")
        model.load_state_dict(state, strict=False)
        deltas.append(extract_effective_delta(model, binding, task_id=task))
        del model
    settings = config["merge"][method]
    selection_rows = {
        source: load(config, source, config["evaluation"]["selection_split"], root)
        for source in {spec["source"] for spec in config["tasks"].values()}
    }
    best = None
    searched = []
    for scale in settings["scales"]:
        merged = merge_deltas(deltas, method=method, density=settings["density"], scale=scale)
        model = materialize_merge(base, merged, private_tasks=tasks).to(torch_device)
        probabilities = _evaluate(config, model, tokenizer, tasks, torch_device, batch_size, root)
        score = selection_score(config, probabilities["select"], selection_rows, vocab)
        searched.append({"scale": scale, "selection_score": score})
        print(f"  {method} scale {scale}: dev_select score {score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, scale, probabilities)
        del model
    assert best is not None
    _, scale, probabilities = best
    result = {
        "kind": "merged",
        "method": method,
        "density": settings["density"],
        "scale": scale,
        "seed": seed,
        "tasks": {task: {} for task in config["tasks"]},
        "specialists": [str(path) for path in specialists],
        "updates": sum(run["updates"] for run in runs),
        "examples": sum(run["examples"] for run in runs),
        "selection": searched,
    }
    _save(output / "report_probs.npz", probabilities["report"])
    _save(output / "select_probs.npz", probabilities["select"])
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------------- report


def _arm(run: dict) -> str:
    if run["kind"] == "merged":
        return f"{run['method']}@{run['updates']}"
    return "+".join(sorted(run["tasks"])) + f"@{run['updates']}"


def report(config: dict, runs: list[Path], root: Path = ROOT) -> dict:
    """Seed-averaged results per task and arm against the lexical baseline, on dev_report."""
    evaluation = config["evaluation"]
    vocab = vocabulary(config, root)
    support = evaluation["min_label_support"]
    loaded = [
        (
            read_json(path / "run.json"),
            np.load(path / "select_probs.npz"),
            np.load(path / "report_probs.npz"),
        )
        for path in runs
    ]
    summary: dict = {"study_id": config["study_id"], "split": evaluation["report_split"]}
    summary["tasks"] = {}
    for source in sorted({spec["source"] for spec in config["tasks"].values()}):
        rows = load(config, source, evaluation["report_split"], root)
        select_rows = load(config, source, evaluation["selection_split"], root)
        weights = group_weights(
            [row["group"] for row in rows],
            evaluation["bootstrap_resamples"],
            evaluation["bootstrap_seed"],
        )
        for task, spec in config["tasks"].items():
            if spec["source"] != source:
                continue
            truth = targets(rows, task, vocab[task])
            select_truth = targets(select_rows, task, vocab[task])

            def score(
                select_probs,
                report_probs,
                task=task,
                truth=truth,
                select_truth=select_truth,
                weights=weights,
            ):
                if config["tasks"][task]["type"] != "multi_label":
                    return None, distribution_metrics(report_probs[task], truth, weights)
                threshold = select_threshold(
                    select_probs[task], select_truth, evaluation["threshold_grid"], support
                )
                return threshold, metrics(report_probs[task], truth, threshold, weights, support)

            entry: dict = {"rows": len(rows)}
            if spec["type"] == "multi_label":
                entry["excluded_labels"] = [
                    label for label, n in zip(vocab[task], truth.sum(0), strict=True) if n < support
                ]
            baseline = None
            arms: dict[str, list] = {}
            for run, select_probs, report_probs in loaded:
                if task not in run["tasks"] or task not in report_probs:
                    continue
                threshold, values = score(select_probs, report_probs)
                if run["kind"] == "lexical":
                    baseline = values
                    entry["lexical"] = summarize([values]) | {
                        "C": run["tasks"][task]["C"],
                        "thresholds": [threshold],
                    }
                else:
                    arms.setdefault(_arm(run), []).append((run["seed"], threshold, values))
            for arm, values in arms.items():
                values.sort(key=lambda item: item[0])
                entry[arm] = summarize([v[2] for v in values], baseline) | {
                    "seeds_run": [v[0] for v in values],
                    "thresholds": [v[1] for v in values],
                }
            summary["tasks"][task] = entry
    return summary


# ---------------------------------------------------------------------- CLI


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CONFIG)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prepare", help="Build BGC and CR4 splits with closed-world labels")
    lexical_parser = actions.add_parser("lexical", help="Fit the tuned TF-IDF baselines")
    lexical_parser.add_argument("--output", type=Path, required=True)
    train_parser = actions.add_parser("train", help="Train one LoRA adapter for a fixed budget")
    train_parser.add_argument("--tasks", required=True, help="Comma-separated, e.g. genre or all")
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument(
        "--updates", type=int, help="Default: the specialist budget times the number of tasks"
    )
    train_parser.add_argument("--device", choices=["cuda", "mps"], required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument(
        "--microbatch", type=int, help="Gradient-accumulation chunk; must divide the batch"
    )
    merge_parser = actions.add_parser("merge", help="Merge one seed's specialists")
    merge_parser.add_argument("specialists", type=Path, nargs="+")
    merge_parser.add_argument("--method", choices=["task_arithmetic", "ties"], required=True)
    merge_parser.add_argument("--device", choices=["cuda", "mps"], required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--batch-size", type=int, default=128)
    report_parser = actions.add_parser("report", help="Summarize run directories")
    report_parser.add_argument("runs", type=Path, nargs="+")
    report_parser.add_argument("--output", type=Path, required=True)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config = read_json(args.config)
    if args.action == "prepare":
        summary = prepare(config)
        print(
            json.dumps(
                {
                    source: {split: value["rows"] for split, value in splits.items()}
                    for source, splits in summary["sources"].items()
                }
            )
        )
    elif args.action == "lexical":
        print(json.dumps(lexical(config, args.output), indent=2))
    elif args.action == "train":
        tasks = list(config["tasks"]) if args.tasks == "all" else args.tasks.split(",")
        updates = args.updates or config["budget"]["specialist_updates"] * len(tasks)
        result = train(
            config, tasks, args.seed, updates, args.device, args.output, microbatch=args.microbatch
        )
        print(json.dumps({key: result[key] for key in ("tasks", "examples", "train_seconds")}))
    elif args.action == "merge":
        result = merge(
            config,
            args.specialists,
            args.method,
            args.device,
            args.output,
            batch_size=args.batch_size,
        )
        print(json.dumps({key: result[key] for key in ("method", "scale", "selection")}))
    else:
        summary = report(config, args.runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        for task, arms in summary["tasks"].items():
            for arm, values in arms.items():
                if not isinstance(values, dict):
                    continue
                cells = "  ".join(
                    f"{name} {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
                    for name, v in values.items()
                    if isinstance(v, dict) and "ci95" in v
                )
                print(f"{task:7} {arm:24} {cells}")
    return 0
