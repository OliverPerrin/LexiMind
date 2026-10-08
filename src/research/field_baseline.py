"""Bounded weak source-label ranking diagnostics and research-only neural recovery.

Only observed positives define recovery metrics. Unlisted labels remain unknown,
and rank scores are neither probabilities nor judgments of semantic relevance.
"""

from __future__ import annotations

import argparse
import platform
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from src.research.book_fields import FACETS, input_hash, validate_states
from src.research.candidate_io import create_or_verify, json_bytes
from src.research.io import file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("frequency", "label_name", "positive_centroid")
TOP_K = (1, 3, 5)
TFIDF = {
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_features": 20_000,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": "english",
}


def validate_config(config: dict) -> None:
    """Freeze the small exploratory run; descriptive and source bindings may vary."""
    fixed = {
        "schema_version": 1,
        "kind": "book_field_retrieval_diagnostic",
        "methods": list(METHODS),
        "facets": list(FACETS),
        "top_k": list(TOP_K),
        "train_limit": 4096,
        "dev_limit": 1024,
        "selection_salt": "bgc-field-retrieval-v1",
        "tfidf": TFIDF,
        "max_total_seconds": 600,
        "promote": False,
        "paid_spend_authorized": False,
    }
    if not isinstance(config, dict) or "seeds" in config:
        raise ValueError("Expected a fixed deterministic field diagnostic, without seeds")
    for key, expected in fixed.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Field diagnostic requires the fixed {key} contract")
    # Python otherwise considers True == 1, including inside dictionary equality.
    if any(type(config["tfidf"][key]) is not type(value) for key, value in TFIDF.items()):
        raise ValueError("TF-IDF options require their exact declared types")
    if any(type(k) is not int for k in config["top_k"] + config["tfidf"]["ngram_range"]):
        raise ValueError("Ranking cutoffs and n-gram bounds must be integers")


def _labels(mapping: dict) -> dict[str, list[str]]:
    if set(mapping["facets"]) != set(FACETS):
        raise ValueError("Expected the four independent field vocabularies")
    result = {}
    for facet in FACETS:
        labels = mapping["facets"][facet]["labels"]
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Field vocabularies require unique nonempty labels")
        result[facet] = sorted(labels)
    return result


def _validate_rows(rows: list[dict], mapping: dict, split: str) -> None:
    seen_ids, seen_groups = set(), set()
    for row in rows:
        if (
            row["effective_split"] != split
            or row["record_id"] in seen_ids
            or row["group_id"] in seen_groups
        ):
            raise ValueError("Ranking rows require unique records/groups in their declared split")
        seen_ids.add(row["record_id"])
        seen_groups.add(row["group_id"])
        if input_hash(row["input"]) != row["input_sha256"]:
            raise ValueError("Ranking input differs from its pinned title/description")
        validate_states(row["fields"], mapping)


def _text(row: dict) -> str:
    payload: dict[str, str] = row["input"]
    return payload["title"] + "\n" + payload["description"]


def fit_rankers(
    train_rows: list[dict], mapping: dict, tfidf_config: dict, *, texts: list[str] | None = None
) -> dict:
    """Fit vocabulary/IDF and prototypes on training text and observed positives only."""
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if not train_rows:
        raise ValueError("Ranker fitting requires training rows")
    labels = _labels(mapping)
    _validate_rows(train_rows, mapping, "train")
    started = time.monotonic()
    options = dict(tfidf_config)
    options["ngram_range"] = tuple(options["ngram_range"])
    vectorizer = TfidfVectorizer(**options, norm="l2", dtype=np.float64)
    if texts is not None and (
        len(texts) != len(train_rows) or any(not isinstance(t, str) for t in texts)
    ):
        raise ValueError("Lexical text controls must match every training row")
    train = vectorizer.fit_transform(
        texts if texts is not None else [_text(row) for row in train_rows]
    ).tocsr()
    centroids, names, frequency, support = {}, {}, {}, {}
    for facet in FACETS:
        vocabulary = labels[facet]
        positions = {label: index for index, label in enumerate(vocabulary)}
        row_indices, column_indices = [], []
        for index, row in enumerate(train_rows):
            for label in row["fields"][facet]["positive"]:
                row_indices.append(positions[label])
                column_indices.append(index)
        indicators = sparse.csr_matrix(
            (np.ones(len(row_indices)), (row_indices, column_indices)),
            shape=(len(vocabulary), len(train_rows)),
            dtype=np.float64,
        )
        counts = np.asarray(indicators.sum(axis=1)).ravel()
        means = sparse.diags(1.0 / np.maximum(counts, 1.0)) @ indicators @ train
        centroids[facet] = normalize(means, norm="l2", axis=1).tocsr()
        names[facet] = vectorizer.transform(
            [label.replace("_", " ") for label in vocabulary]
        ).tocsr()
        frequency[facet] = counts
        support[facet] = {
            "positive_groups": {label: int(counts[i]) for i, label in enumerate(vocabulary)},
            "zero_positive_support_labels": [
                label for i, label in enumerate(vocabulary) if counts[i] == 0
            ],
            "zero_centroid_labels": [
                label
                for i, label in enumerate(vocabulary)
                if centroids[facet].indptr[i] == centroids[facet].indptr[i + 1]
            ],
            "zero_label_name_labels": [
                label
                for i, label in enumerate(vocabulary)
                if names[facet].indptr[i] == names[facet].indptr[i + 1]
            ],
        }
    return {
        "vectorizer": vectorizer,
        "labels": labels,
        "mapping": mapping,
        "centroids": centroids,
        "label_name_vectors": names,
        "frequency": frequency,
        "support": support,
        "train_record_ids": {row["record_id"] for row in train_rows},
        "train_group_ids": {row["group_id"] for row in train_rows},
        "diagnostics": {
            "train_groups": len(train_rows),
            "vocabulary_features": len(vectorizer.vocabulary_),
            "zero_training_tfidf_vectors": int(np.count_nonzero(np.diff(train.indptr) == 0)),
            "negative_training_labels": sum(
                len(row["fields"][facet]["negative"]) for row in train_rows for facet in FACETS
            ),
            "negative_labels_used_for_fitting": 0,
        },
        "fit_seconds": time.monotonic() - started,
    }


def _rank(scores, labels: list[str], positives: set[str], top_k: tuple[int, ...]) -> dict:
    ordered = sorted(zip(labels, scores, strict=True), key=lambda pair: (-float(pair[1]), pair[0]))
    ranking = [{"label": label, "score": float(score)} for label, score in ordered]
    ranks = [i + 1 for i, (label, _) in enumerate(ordered) if label in positives]
    return {
        "ranking": ranking,
        "effective_k": {str(k): min(k, len(labels)) for k in top_k},
        "recall_at_k": {
            str(k): sum(rank <= k for rank in ranks) / len(positives) if positives else None
            for k in top_k
        },
        "reciprocal_rank": 1.0 / min(ranks) if ranks else None,
        "all_scores_zero": all(float(score) == 0.0 for _, score in ordered),
        "distinct_score_count": len({float(score) for _, score in ordered}),
    }


def rank_rows(
    fitted: dict,
    rows: list[dict],
    top_k: tuple[int, ...] = TOP_K,
    *,
    texts: list[str] | None = None,
) -> list[dict]:
    """Rank every vocabulary label for development inputs, with fixed alphabetical ties."""
    import numpy as np

    if not rows:
        raise ValueError("Ranking evaluation requires development rows")
    _validate_rows(rows, fitted["mapping"], "dev")
    if any(
        row["group_id"] in fitted["train_group_ids"]
        or row["record_id"] in fitted["train_record_ids"]
        for row in rows
    ):
        raise ValueError("Development rows overlap fitted records or groups")
    if any(type(k) is not int or k < 1 for k in top_k) or not top_k:
        raise ValueError("Ranking cutoffs must be positive integers")
    if texts is not None and (
        len(texts) != len(rows) or any(not isinstance(t, str) for t in texts)
    ):
        raise ValueError("Lexical text controls must match every development row")
    matrix = (
        fitted["vectorizer"]
        .transform(texts if texts is not None else [_text(row) for row in rows])
        .tocsr()
    )
    scores = {
        method: {
            facet: (
                np.broadcast_to(fitted["frequency"][facet], (len(rows), len(labels)))
                if method == "frequency"
                else (
                    matrix
                    @ fitted["label_name_vectors" if method == "label_name" else "centroids"][
                        facet
                    ].T
                ).toarray()
            )
            for facet, labels in fitted["labels"].items()
        }
        for method in METHODS
    }
    if any(
        not np.isfinite(values).all() for facets in scores.values() for values in facets.values()
    ):
        raise ValueError("Ranker produced non-finite scores")
    return [
        {
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "source_split": row.get("source_split"),
            "effective_split": row["effective_split"],
            "input_sha256": row["input_sha256"],
            "observed_positive": {
                facet: sorted(row["fields"][facet]["positive"]) for facet in FACETS
            },
            "observed_negative": {
                facet: sorted(row["fields"][facet]["negative"]) for facet in FACETS
            },
            "zero_tfidf_vector": bool(matrix.indptr[index] == matrix.indptr[index + 1]),
            "methods": {
                method: {
                    facet: _rank(
                        scores[method][facet][index],
                        labels,
                        set(row["fields"][facet]["positive"]),
                        top_k,
                    )
                    for facet, labels in fitted["labels"].items()
                }
                for method in METHODS
            },
        }
        for index, row in enumerate(rows)
    ]


def ranking_metrics(
    ranked_rows: list[dict], mapping: dict, top_k: tuple[int, ...] = TOP_K, *, methods=METHODS
) -> dict:
    """Observed-positive recovery only; exclude unobserved facets, never score negatives."""
    labels = _labels(mapping)
    if len({row["group_id"] for row in ranked_rows}) != len(ranked_rows):
        raise ValueError("Group-macro metrics require one unique row per group")
    result: dict = {}
    for method in methods:
        result[method] = {}
        for facet, vocabulary in labels.items():
            eligible = [row for row in ranked_rows if row["observed_positive"][facet]]
            support: Counter[str] = Counter()
            hits: dict[int, Counter[str]] = {k: Counter() for k in top_k}
            recall_sums = {k: 0.0 for k in top_k}
            reciprocal_sum = 0.0
            for row in eligible:
                positives = set(row["observed_positive"][facet])
                if not positives <= set(vocabulary):
                    raise ValueError("Observed positives contain labels outside the vocabulary")
                support.update(positives)
                ordered = row["methods"][method][facet]["ranking"]
                ranking_labels = [entry["label"] for entry in ordered]
                if len(ranking_labels) != len(vocabulary) or set(ranking_labels) != set(vocabulary):
                    raise ValueError("Every ranker must retain every vocabulary label exactly once")
                ranks = [
                    index + 1 for index, label in enumerate(ranking_labels) if label in positives
                ]
                reciprocal_sum += 1.0 / min(ranks)
                for k in top_k:
                    recovered = positives & set(ranking_labels[:k])
                    hits[k].update(recovered)
                    recall_sums[k] += len(recovered) / len(positives)
            by_label: dict[str, dict] = {
                label: {
                    "observed_positive_groups": support[label],
                    "recall_at_k": {
                        str(k): hits[k][label] / support[label] if support[label] else None
                        for k in top_k
                    },
                }
                for label in vocabulary
            }
            present = [label for label in vocabulary if support[label]]
            result[method][facet] = {
                "effective_k": {str(k): min(k, len(vocabulary)) for k in top_k},
                "evaluated_groups": len(eligible),
                "excluded_groups_without_observed_positives": len(ranked_rows) - len(eligible),
                "observed_positive_count": sum(support.values()),
                "group_macro_recall_at_k": {
                    str(k): (recall_sums[k] / len(eligible) if eligible else None) for k in top_k
                },
                "group_macro_reciprocal_rank": (
                    reciprocal_sum / len(eligible) if eligible else None
                ),
                "label_macro_recall_at_k": {
                    str(k): (
                        sum(by_label[label]["recall_at_k"][str(k)] for label in present)
                        / len(present)
                        if present
                        else None
                    )
                    for k in top_k
                },
                "zero_evaluation_support_labels": [
                    label for label in vocabulary if not support[label]
                ],
                "all_scores_zero_groups": sum(
                    row["methods"][method][facet]["all_scores_zero"] for row in ranked_rows
                ),
                "by_label": by_label,
            }
    return result


def _reference(root: Path, path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def _write(path: Path, value: dict) -> None:
    create_or_verify(path, [json_bytes(value)])


def _save_fitted(root: Path, output: Path, fitted: dict, tfidf_config: dict) -> dict:
    """Persist inert vocabulary/IDF JSON and sparse NPZ arrays; never pickle or load code."""
    from scipy import sparse

    directory = output / "fitted"
    directory.mkdir()
    vectorizer = fitted["vectorizer"]
    state = {
        "schema_version": 1,
        "tfidf_config": tfidf_config,
        "norm": "l2",
        "dtype": "float64",
        "vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()},
        "idf": [float(value) for value in vectorizer.idf_],
        "labels": fitted["labels"],
        "frequency": {facet: counts.tolist() for facet, counts in fitted["frequency"].items()},
    }
    state_path = directory / "vectorizer.json"
    _write(state_path, state)
    references = {"vectorizer": _reference(root, state_path)}
    for kind in ("centroids", "label_name_vectors"):
        references[kind] = {}
        for facet, matrix in fitted[kind].items():
            path = directory / f"{kind}_{facet}.npz"
            sparse.save_npz(path, matrix, compressed=True)
            references[kind][facet] = _reference(root, path)
    return references


@contextmanager
def _time_limit(seconds: int):
    """Interrupt the bounded CLI run, including library work, without persistent workers."""
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("A preexisting process timer prevents an isolated diagnostic budget")

    def expired(_signum, _frame):
        raise TimeoutError("Field diagnostic exceeded its fixed wall-clock budget")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def execute(
    config_path: Path, output: Path, *, root: Path = ROOT, prepare_only: bool = False
) -> dict:
    """Prepare or execute one fixed local run, exclusively in a fresh outputs directory."""
    from src.research.field_baseline_data import prepare_field_baseline_data
    from src.research.field_baseline_review import build_review

    root, config_path, output = root.resolve(), config_path.resolve(), output.resolve()
    if not output.is_relative_to(root / "outputs") or output == root / "outputs":
        raise ValueError("Field baseline output must be a fresh directory beneath outputs")
    if output.exists():
        raise ValueError(
            "Field baseline output already exists; preserve it and use a new directory"
        )
    config = read_json(config_path)
    if config.get("kind") in {
        "book_source_assignment_recovery_pilot",
        "book_source_assignment_learning_curves",
        "book_source_assignment_data_scaling",
        "book_source_assignment_loss_weighting",
    }:
        return execute_source_recovery(config_path, output, root=root, prepare_only=prepare_only)
    validate_config(config)
    config_ref = _reference(root, config_path)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with _time_limit(config["max_total_seconds"]):
            bundle = prepare_field_baseline_data(root, config)
            train, dev, mapping = bundle["train"], bundle["dev"], bundle["mapping"]
            rows = [*train, *dev]
            if len(train) != config["train_limit"] or len(dev) != config["dev_limit"]:
                raise ValueError("Prepared rows do not meet the fixed train/development sizes")
            _validate_rows(train, mapping, "train")
            _validate_rows(dev, mapping, "dev")
            if len({row["group_id"] for row in rows}) != len(rows):
                raise ValueError("Diagnostic rows are not globally unique groups")
            examples_path = output / "examples.json"
            _write(
                examples_path,
                {
                    "schema_version": 1,
                    "kind": "book_field_retrieval_diagnostic_examples",
                    "config_reference": config_ref,
                    "provenance": bundle["provenance"],
                    "mapping": mapping,
                    "train": train,
                    "dev": dev,
                },
            )
            review = build_review(
                root,
                config_path,
                examples_path,
                root
                / "data/research_candidates/bgc/field-baseline-review"
                / output.relative_to(root / "outputs"),
            )
            report = {
                "schema_version": 1,
                "kind": "book_field_retrieval_diagnostic_result",
                "status": "prepared_not_fitted"
                if prepare_only
                else "completed_weak_label_diagnostic",
                "config": config,
                "config_reference": config_ref,
                "source_provenance": bundle["provenance"],
                "examples": _reference(root, examples_path),
                "review": review,
                "implementation_sha256": {
                    name: file_hash(root / name)
                    for name in (
                        "src/research/field_baseline.py",
                        "src/research/field_baseline_data.py",
                        "src/research/field_baseline_review.py",
                        "src/research/book_fields.py",
                        "src/research/candidate_io.py",
                        "src/research/io.py",
                    )
                },
                "selected": [
                    {
                        key: row[key]
                        for key in (
                            "record_id",
                            "group_id",
                            "source_split",
                            "effective_split",
                            "input_sha256",
                        )
                    }
                    for row in rows
                ],
                "selected_label_support": {
                    split: {
                        facet: {
                            "positive_groups": {
                                label: sum(
                                    label in row["fields"][facet]["positive"] for row in cohort
                                )
                                for label in labels
                            },
                            "negative_groups": {
                                label: sum(
                                    label in row["fields"][facet]["negative"] for row in cohort
                                )
                                for label in labels
                            },
                            "groups_without_observed_positives": sum(
                                not row["fields"][facet]["positive"] for row in cohort
                            ),
                        }
                        for facet, labels in _labels(mapping).items()
                    }
                    for split, cohort in (("train", train), ("dev", dev))
                },
                "mapping": mapping,
                "human_gold": False,
                "formal_dataset_admission": False,
                "neural_training_performed": False,
                "rl_training_performed": False,
                "model_promoted": False,
                "paid_spend": 0,
                "statistical_fitting_performed": not prepare_only,
                "semantics": {
                    "input": "Literal title + newline + description; no provider labels or identity metadata.",
                    "missing_labels": "Unknown, never semantic negatives or automatically completed labels.",
                    "metrics": "Observed-positive source-label recovery on selected development groups, not precision, classification accuracy, human gold or recommendation relevance.",
                    "frequency_score": "Number of selected positive training groups; not a probability.",
                    "lexical_score": "Cosine TF-IDF similarity; not a probability.",
                    "ties": "Descending score, then ascending canonical label ID; includes zero-score and unsupported labels.",
                    "group_weight": "One metadata-selected source record per selected singleton group.",
                    "fit_scope": "Vocabulary, IDF, frequency and prototypes use selected training rows only; label-name text is transformed without refitting.",
                },
                "timings": {"preparation_seconds": time.monotonic() - started},
            }
            if not prepare_only:
                fitted = fit_rankers(train, mapping, config["tfidf"])
                report["training_support"] = fitted["support"]
                report["fit_diagnostics"] = fitted["diagnostics"]
                report["timings"]["fit_seconds"] = fitted["fit_seconds"]
                eval_started = time.monotonic()
                ranked = rank_rows(fitted, dev, tuple(config["top_k"]))
                report["metrics"] = ranking_metrics(ranked, mapping, tuple(config["top_k"]))
                report["zero_development_tfidf_vectors"] = sum(
                    row["zero_tfidf_vector"] for row in ranked
                )
                report["timings"]["evaluation_seconds"] = time.monotonic() - eval_started
                rankings_path = output / "rankings.json"
                _write(rankings_path, {"schema_version": 1, "records": ranked})
                report["rankings"] = _reference(root, rankings_path)
                report["fitted_artifacts"] = _save_fitted(root, output, fitted, config["tfidf"])
                report["dependency_versions"] = {
                    "python": platform.python_version(),
                    **{name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                }
            if _reference(root, config_path) != config_ref:
                raise ValueError("Protocol changed during the diagnostic")
            report["timings"]["total_seconds"] = time.monotonic() - started
            _write(output / "report.json", report)
            return report
    except BaseException as error:
        _write(
            output / "failure.json",
            {
                "schema_version": 1,
                "status": "failed_incomplete_do_not_use",
                "config_reference": config_ref,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/research/book_field_baseline.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        report = execute(args.config, args.output, prepare_only=args.prepare_only)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json_bytes(
            {
                "status": report["status"],
                "output": str(args.output),
                "selected_records": len(report["selected"]),
                "statistical_fitting_performed": report["statistical_fitting_performed"],
                "total_seconds": report["timings"]["total_seconds"],
            }
        ).decode(),
        end="",
    )
    return 0


SOURCE_RECOVERY = {
    "schema_version": 1,
    "kind": "book_source_assignment_recovery_pilot",
    "seeds": [17, 29],
    "arms": ["head_only", "lora_head"],
    "facets": list(FACETS),
    "top_k": list(TOP_K),
    "train_limit": 4096,
    "dev_limit": 1024,
    "selection_salt": "bgc-field-retrieval-v1",
    "tfidf": TFIDF,
    "batch_size": 16,
    "epochs": 2,
    "updates": 512,
    "endpoints": [0, 256, 512],
    "primary_endpoint": 512,
    "max_length": 256,
    "max_total_seconds": 900,
    "device": "cuda",
    "parameter_dtype": "float32",
    "autocast_dtype": "bfloat16",
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,
    "optimizer": {"lr": 0.0005, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "scheduler": "constant",
    "lora": {"rank": 4, "alpha": 8, "dropout": 0.0},
    "dropout": 0.0,
    "base": {
        "repo": "google/flan-t5-base",
        "revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
        "weight_sha256": "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8",
    },
    "schedule": "Python random.Random(seed + epoch).shuffle over fixed cohort indices; epoch is 1 or 2; same schedule in both arms",
    "promote": False,
    "paid_spend_authorized": False,
    "formal_dataset_admission": False,
    "human_gold": False,
}


SOURCE_LEARNING_CURVES = {
    **SOURCE_RECOVERY,
    "kind": "book_source_assignment_learning_curves",
    "epochs": 16,
    "updates": 4096,
    "endpoints": [0, 512, 1024, 2048, 4096],
    "primary_endpoint": 4096,
    "max_total_seconds": 1800,
    "schedule": "Python random.Random(seed + epoch).shuffle over fixed cohort indices; epoch is 1 through 16; same schedule in both arms",
    "batch_evidence": "shared_schedule_token_hashes",
    "fit_diagnostics": "dev_every_endpoint_and_final_train",
    "initialization": "fresh_base_head_factors_and_optimizer_not_resume",
}


SOURCE_DATA_SCALING = {
    **SOURCE_LEARNING_CURVES,
    "kind": "book_source_assignment_data_scaling",
    "train_limit": 16384,
    "epochs": 4,
    "endpoints": [0, 1024, 2048, 4096],
    "schedule": "Python random.Random(seed + epoch).shuffle over fixed cohort indices; epoch is 1 through 4; same schedule in both arms",
    "scheduled_presentations_per_arm": 65536,
    "source_supervised_presentations_per_arm": 65532,
    "expected_whole_empty_train_rows": 1,
    "comparison": "audited_historical4096_cohort_descriptive_only_not_isolated_data_size_causal_effect",
}


SOURCE_LOSS_WEIGHTING = {
    **SOURCE_DATA_SCALING,
    "kind": "book_source_assignment_loss_weighting",
    "arms": ["unweighted", "weighted"],
    "max_total_seconds": 2400,
    "comparison": "fresh_contemporaneous_unweighted_vs_weighted_lora_controls_historical_scaling_descriptive_only",
    "weighting": {
        "rule": "train_global_coefficient_inverse_sqrt_per_facet",
        "lower": 0.25,
        "upper": 4.0,
        "bisection_iterations": 96,
        "float64_residual_tolerance": 1e-12,
        "float32_residual_tolerance": 1e-7,
        "renormalize_after_cast_or_clipping": False,
    },
    "postrun_audit_caps_seconds": {"cpu": 600, "gpu": 600},
}


def validate_source_recovery_config(config: dict) -> None:
    """Reject protocol drift rather than silently adapting this one bounded pilot."""
    import json

    contract = (
        SOURCE_LOSS_WEIGHTING
        if config.get("kind") == SOURCE_LOSS_WEIGHTING["kind"]
        else SOURCE_DATA_SCALING
        if config.get("kind") == SOURCE_DATA_SCALING["kind"]
        else SOURCE_LEARNING_CURVES
        if config.get("kind") == SOURCE_LEARNING_CURVES["kind"]
        else SOURCE_RECOVERY
    )
    for key, expected in contract.items():
        if json.dumps(config.get(key), sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f"Source recovery requires fixed {key}")
    legacy = read_json(ROOT / "configs/research/book_field_baseline.json")
    for key in ("field_manifest", "partition_manifest"):
        if config.get(key) != legacy[key]:
            raise ValueError(f"Source recovery must retain the existing {key} pin")


def _validate_source_weights(logits, weights) -> None:
    if weights is None:
        return
    import torch

    if (
        weights.shape != (logits.shape[1],)
        or weights.dtype != torch.float32
        or weights.device != logits.device
        or not torch.isfinite(weights).all()
        or (weights < 0.25).any()
        or (weights > 4).any()
    ):
        raise ValueError(
            "Applied source weights require finite FP32 C-vector on logits device in[0.25,4]"
        )


def source_recovery_weights(train, mapping, schedules) -> dict:
    """One deterministic train-only vector; preserve global facet mass within fixed tolerances."""
    import math

    import torch

    labels = _labels(mapping)
    active = [[f for f in FACETS if row["fields"][f]["positive"]] for row in train]
    eligible = sum(bool(facets) for facets in active)
    if eligible != 16383 or len(train) != 16384:
        raise ValueError("Weighting requires the fixed16383eligible/16384selected train rows")
    mass = {f: {label: 0.0 for label in labels[f]} for f in FACETS}
    support = {f: {label: 0 for label in labels[f]} for f in FACETS}
    for row, facets in zip(train, active, strict=True):
        for f in facets:
            share = 1 / (eligible * len(facets) * len(row["fields"][f]["positive"]))
            for label in row["fields"][f]["positive"]:
                mass[f][label] += share
                support[f][label] += 1
    vector, result = [], {}
    for f in FACETS:
        total = math.fsum(mass[f].values())
        probabilities = [mass[f][label] / total for label in labels[f]]
        if any(m <= 0 or not math.isfinite(m) for m in probabilities):
            raise ValueError("All48 labels require positive train support/coefficient mass")
        lower, upper = 0.0, 4 * max(math.sqrt(m) for m in probabilities)
        for _ in range(96):
            midpoint = (lower + upper) / 2
            mean = math.fsum(
                m * min(4.0, max(0.25, midpoint / math.sqrt(m))) for m in probabilities
            )
            if mean < 1:
                lower = midpoint
            else:
                upper = midpoint
        scalar = (lower + upper) / 2
        exact = [min(4.0, max(0.25, scalar / math.sqrt(m))) for m in probabilities]
        applied = torch.tensor(exact, dtype=torch.float32).tolist()
        residual64 = math.fsum(m * w for m, w in zip(probabilities, exact, strict=True)) - 1
        residual32 = math.fsum(m * w for m, w in zip(probabilities, applied, strict=True)) - 1
        if abs(residual64) > 1e-12 or abs(residual32) > 1e-7:
            raise ValueError("Fixed coefficient normalization gate failed")
        vector.extend(applied)
        result[f] = {
            "labels": labels[f],
            "positive_groups": support[f],
            "global_mass": mass[f],
            "facet_mass": total,
            "conditional_mass": probabilities,
            "scalar": scalar,
            "weights_float64": exact,
            "weights_applied_float32": applied,
            "normalization_residual_float64": residual64,
            "normalization_residual_float32": residual32,
            "lower_cap_count": sum(w == 0.25 for w in applied),
            "upper_cap_count": sum(w == 4 for w in applied),
            "applied_global_weighted_facet_mass": math.fsum(
                mass[f][label] * w for label, w in zip(labels[f], applied, strict=True)
            ),
        }
    tensor = torch.tensor(vector, dtype=torch.float32)
    scheduled = {}
    weights_by_label = {
        f: dict(zip(labels[f], result[f]["weights_applied_float32"], strict=True)) for f in FACETS
    }
    for seed in (17, 29):
        totals = {f: 0.0 for f in FACETS}
        unweighted = {f: 0.0 for f in FACETS}
        batches: Counter[int] = Counter()
        for epoch in schedules[str(seed)]:
            if sorted(epoch) != list(range(len(train))):
                raise ValueError("Weight schedule must retain every fixed train row")
            for start in range(0, len(epoch), 16):
                indices = [i for i in epoch[start : start + 16] if active[i]]
                if not indices:
                    raise ValueError("Weight schedule has no supervised rows")
                batches[len(indices)] += 1
                for i in indices:
                    for f in active[i]:
                        base = 1 / (
                            4096
                            * len(indices)
                            * len(active[i])
                            * len(train[i]["fields"][f]["positive"])
                        )
                        for label in train[i]["fields"][f]["positive"]:
                            unweighted[f] += base
                            totals[f] += base * weights_by_label[f][label]
        if dict(batches) != {15: 4, 16: 4092} and dict(batches) != {16: 4092, 15: 4}:
            raise ValueError("Weight schedule eligible batch counts changed")
        scheduled[str(seed)] = {
            "unweighted_facet_mass": unweighted,
            "weighted_facet_mass": totals,
            "weighted_minus_unweighted_facet_mass": {f: totals[f] - unweighted[f] for f in FACETS},
            "weighted_total": math.fsum(totals.values()),
            "eligible_batch_size_counts": dict(batches),
        }
    return {
        "schema_version": 1,
        "kind": "train_only_source_loss_weight_vector",
        "facets": result,
        "label_order": [label for f in FACETS for label in labels[f]],
        "applied_weights": vector,
        "tensor_sha256": _recovery_tensor_hash(tensor),
        "scheduled_mass": scheduled,
        "semantics": "Global train facet normalization only, not exact batch-schedule normalization. Weights scale rows/facets and redistribute co-positive targets; gradient norms, clipping and update magnitudes are not preserved. No within-row renormalization.",
    }


def source_recovery_loss(logits, positives, facet_bounds, *, label_weights=None):
    """Uniform source-presence CE; facet mean per row, then eligible-row mean.

    The softmax competes across every label, including unassigned and co-positive
    labels. This is source-assignment recovery, not semantic-negative supervision.
    """
    import torch

    if (
        logits.ndim != 2
        or positives.shape != logits.shape
        or positives.dtype != torch.bool
        or not torch.isfinite(logits).all()
        or not facet_bounds
    ):
        raise ValueError("Recovery logits/positive indicators must be finite matching BxC tensors")
    _validate_source_weights(logits, label_weights)
    cursor = 0
    losses, eligible = [], []
    for start, stop in facet_bounds.values():
        if start != cursor or stop <= start or stop > logits.shape[1]:
            raise ValueError("Recovery facets must partition every logit in order")
        cursor = stop
        observed = positives[:, start:stop]
        count = observed.sum(dim=1)
        log_probs = torch.log_softmax(logits[:, start:stop].float(), dim=1)
        terms = log_probs * observed
        if label_weights is not None:
            terms = terms * label_weights[start:stop]
        losses.append(-terms.sum(dim=1) / count.clamp_min(1))
        eligible.append(count > 0)
    if cursor != logits.shape[1]:
        raise ValueError("Recovery facets must cover every logit")
    available = torch.stack(eligible, dim=1)
    row_count = available.sum(dim=1)
    row_loss = (torch.stack(losses, dim=1) * available).sum(dim=1) / row_count.clamp_min(1)
    if not (row_count > 0).any():
        raise ValueError("A recovery batch must contain an eligible source-positive row")
    return row_loss[row_count > 0].mean()


def source_recovery_fit_totals(logits, positives, facet_bounds, *, label_weights=None) -> dict:
    """Additive eval-mode objective totals, correctly weighted by eligible rows."""
    import torch

    # Reuse objective validation even when no row is eligible in this batch.
    if logits.ndim != 2 or positives.shape != logits.shape or positives.dtype != torch.bool:
        raise ValueError("Fit diagnostic requires matching BxC logits and positive indicators")
    if not torch.isfinite(logits).all():
        raise ValueError("Fit diagnostic requires finite logits")
    _validate_source_weights(logits, label_weights)
    entropy, availability = [], []
    cursor = 0
    for start, stop in facet_bounds.values():
        if start != cursor or stop <= start or stop > logits.shape[1]:
            raise ValueError("Fit facets must partition every logit")
        cursor = stop
        counts = positives[:, start:stop].sum(dim=1)
        if label_weights is None:
            entropy.append(counts.clamp_min(1).float().log())
        else:
            weights = label_weights[start:stop]
            coefficients = positives[:, start:stop] * weights
            mass = coefficients.sum(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            entropy.append(
                -(coefficients * (weights.log() - mass.log()[:, None])).sum(dim=1)
                / counts.clamp_min(1)
            )
        availability.append(counts > 0)
    if cursor != logits.shape[1] or not availability:
        raise ValueError("Fit facets must cover every logit")
    mask = torch.stack(availability, dim=1)
    facet_count = mask.sum(dim=1)
    eligible = facet_count > 0
    count = int(eligible.sum())
    floors = (torch.stack(entropy, dim=1) * mask).sum(dim=1) / facet_count.clamp_min(1)
    return {
        "eligible_rows": count,
        "excluded_rows": len(logits) - count,
        "cross_entropy_sum": float(
            source_recovery_loss(logits, positives, facet_bounds, label_weights=label_weights)
        )
        * count
        if count
        else 0.0,
        "target_entropy_floor_sum": float(floors[eligible].sum()),
    }


def source_recovery_rankings(
    scores, rows: list[dict], mapping: dict, method: str, *, log_scores=None
) -> list[dict]:
    """Validate facet probability rankings and retain all labels and source outcomes."""
    import numpy as np

    labels = _labels(mapping)
    if set(scores) != set(FACETS):
        raise ValueError("Recovery evaluation requires all four facets")
    for facet, vocabulary in labels.items():
        values = np.asarray(scores[facet])
        if (
            values.shape != (len(rows), len(vocabulary))
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=1e-5)
        ):
            raise ValueError("Recovery scores must be finite per-facet probability vectors")
    ranked = [
        {
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "source_split": row["source_split"],
            "effective_split": row["effective_split"],
            "input_sha256": row["input_sha256"],
            "observed_positive": {f: sorted(row["fields"][f]["positive"]) for f in FACETS},
            "observed_negative": {f: sorted(row["fields"][f]["negative"]) for f in FACETS},
            "methods": {
                method: {
                    f: _rank(scores[f][i], labels[f], set(row["fields"][f]["positive"]), TOP_K)
                    for f in FACETS
                }
            },
        }
        for i, row in enumerate(rows)
    ]

    if log_scores is not None:
        if set(log_scores) != set(FACETS):
            raise ValueError("Recovery log probabilities require all facets")
        for facet, vocabulary in labels.items():
            log_values = np.asarray(log_scores[facet])
            if (
                log_values.shape != np.asarray(scores[facet]).shape
                or not np.isfinite(log_values).all()
                or not np.allclose(np.exp(log_values), scores[facet], atol=1e-7, rtol=1e-5)
            ):
                raise ValueError(
                    "Recovery log probabilities must be finite and exponentiate to saved scores"
                )
            positions = {label: i for i, label in enumerate(vocabulary)}
            for i, row in enumerate(ranked):
                for entry in row["methods"][method][facet]["ranking"]:
                    entry["log_probability"] = float(log_values[i, positions[entry["label"]]])
    return ranked


def _recovery_tensor_hash(tensor) -> str:
    import hashlib

    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        json_bytes({"shape": list(value.shape), "dtype": str(value.dtype)})
        + value.numpy().tobytes()
    ).hexdigest()


def _scaling_cohort_support(rows, mapping) -> dict:
    labels = _labels(mapping)
    empty = sum(not any(row["fields"][f]["positive"] for f in FACETS) for row in rows)
    return {
        "rows": len(rows),
        "unique_groups": len({row["group_id"] for row in rows}),
        "objective_eligible_rows": len(rows) - empty,
        "whole_empty_rows": empty,
        "facets": {
            f: {
                "eligible_rows": sum(bool(row["fields"][f]["positive"]) for row in rows),
                "groups_without_observed_positives": sum(
                    not row["fields"][f]["positive"] for row in rows
                ),
                "positive_groups": {
                    label: sum(label in row["fields"][f]["positive"] for row in rows)
                    for label in labels[f]
                },
            }
            for f in FACETS
        },
    }


def _verify_scaling_comparison(root, config, train, dev, mapping, *, encoded=None) -> dict:
    """Check full old-row equality and visible-token equality, never approximate identity."""
    refs = {
        name: config[name]
        for name in (
            "previous_source_learning_curves_observation",
            "historical_examples",
            "historical_tokenization",
        )
    }
    for reference in refs.values():
        if _reference(root, root / reference["path"]) != reference:
            raise ValueError("Data exposure comparison reference changed")
    old = read_json(root / refs["historical_examples"]["path"])
    if (
        len(old["train"]) != 4096
        or len(old["dev"]) != 1024
        or train[:4096] != old["train"]
        or dev != old["dev"]
        or mapping != old["mapping"]
    ):
        raise ValueError(
            "Expanded cohort must retain exact historical full train prefix and development rows"
        )
    supports = {
        name: _scaling_cohort_support(rows, mapping)
        for name, rows in (
            ("original_train", train[:4096]),
            ("additional_train", train[4096:]),
            ("expanded_train", train),
            ("dev", dev),
        )
    }
    if supports["expanded_train"]["whole_empty_rows"] != config["expected_whole_empty_train_rows"]:
        raise ValueError("Expanded cohort whole-empty source-positive row count changed")
    proof = {
        "references": refs,
        "historical_train_prefix_rows": 4096,
        "additional_train_rows": 12288,
        "development_rows": 1024,
        "full_train_prefix_rows_exact": True,
        "full_dev_rows_exact": True,
        "mapping_exact": True,
        "visible_tokens_and_decoded_text_verified": False,
        "support_strata": supports,
        "comparison": config["comparison"],
    }
    if encoded is not None:
        old_tokens = read_json(root / refs["historical_tokenization"]["path"])["records"]
        for split, limit in (("train", 4096), ("dev", 1024)):
            if len(old_tokens[split]) != limit:
                raise ValueError("Historical token cohort count differs")
            for current, historical in zip(encoded[split][:limit], old_tokens[split], strict=True):
                current_visible = [
                    token
                    for token, valid in zip(
                        current["input_ids"], current["attention_mask"], strict=True
                    )
                    if valid
                ]
                old_visible = [
                    token
                    for token, valid in zip(
                        historical["input_ids"], historical["attention_mask"], strict=True
                    )
                    if valid
                ]
                if (
                    current["record_id"] != historical["record_id"]
                    or current_visible != old_visible
                    or current["decoded_control_text"] != historical["decoded_control_text"]
                    or current["decoded_control_sha256"] != historical["decoded_control_sha256"]
                ):
                    raise ValueError(
                        "Historical visible tokens or decoded lexical control text changed"
                    )
        proof["visible_tokens_and_decoded_text_verified"] = True
    return proof


def _scaling_primary_comparison(root, config, report) -> dict:
    """Descriptive historical comparison, retaining every arm and both seeds."""
    reference = config["previous_source_learning_curves_observation"]
    if _reference(root, root / reference["path"]) != reference:
        raise ValueError("Historical learning-curve observation changed before comparison")
    previous = read_json(root / reference["path"])
    result = {
        "historical_reference": config["previous_source_learning_curves_observation"],
        "claim_boundary": config["comparison"],
        "endpoint": 4096,
        "arms": {},
    }
    for key, arm in report["arms"].items():
        current = arm["endpoints"]["4096"]["metrics"][arm["arm"]]
        old = previous["arms"][key]["endpoints"]["4096"]["metrics"][arm["arm"]]
        result["arms"][key] = {
            facet: {
                metric: {
                    "current": current[facet][metric]["3"],
                    "historical": old[facet][metric]["3"],
                    "difference": current[facet][metric]["3"] - old[facet][metric]["3"],
                }
                for metric in ("group_macro_recall_at_k", "label_macro_recall_at_k")
            }
            for facet in ("genre", "topic")
        }
    result["descriptive_two_seed_mean_differences"] = {
        arm: {
            facet: {
                metric: sum(
                    result["arms"][f"{seed}_{arm}"][facet][metric]["difference"]
                    for seed in (17, 29)
                )
                / 2
                for metric in ("group_macro_recall_at_k", "label_macro_recall_at_k")
            }
            for facet in ("genre", "topic")
        }
        for arm in ("head_only", "lora_head")
    }
    return result


def _verify_weighting_history(
    root, config, train, dev, mapping, schedules, *, encoded=None, lexical_ranked=None
) -> dict:
    names = (
        "historical_scaling_examples",
        "historical_scaling_tokenization",
        "historical_scaling_schedules",
        "historical_scaling_lexical_rankings",
        "historical_scaling_report",
    )
    refs = {name: config[name] for name in names}
    for reference in refs.values():
        if _reference(root, root / reference["path"]) != reference:
            raise ValueError("Weighting historical reference changed")
    old = read_json(root / refs["historical_scaling_examples"]["path"])
    if train != old["train"] or dev != old["dev"] or mapping != old["mapping"]:
        raise ValueError("Weight trial must retain every exact prior scaling row and mapping")
    old_schedule = read_json(root / refs["historical_scaling_schedules"]["path"])["indices"]
    if schedules != old_schedule:
        raise ValueError("Weight trial schedules differ from scaling")
    old_lex = read_json(root / refs["historical_scaling_lexical_rankings"]["path"])["records"]
    if [r["record_id"] for r in old_lex] != [r["record_id"] for r in dev]:
        raise ValueError("Historical lexical development identities differ")
    proof = {
        "references": refs,
        "full_rows_mapping_schedules_exact": True,
        "historical_lexical_rows_bound": True,
        "visible_tokens_decoded_text_exact": False,
        "fresh_lexical_refit_exact": False,
    }
    if encoded is not None:
        previous = read_json(root / refs["historical_scaling_tokenization"]["path"])["records"]
        for split in ("train", "dev"):
            if len(encoded[split]) != len(previous[split]):
                raise ValueError("Weight token counts differ")
            for current, prior in zip(encoded[split], previous[split], strict=True):

                def visible(record):
                    return [
                        t
                        for t, m in zip(record["input_ids"], record["attention_mask"], strict=True)
                        if m
                    ]

                if (
                    current["record_id"] != prior["record_id"]
                    or visible(current) != visible(prior)
                    or current["decoded_control_text"] != prior["decoded_control_text"]
                    or current["decoded_control_sha256"] != prior["decoded_control_sha256"]
                ):
                    raise ValueError("Weight trial prior visible tokens/decoded controls changed")
        proof["visible_tokens_decoded_text_exact"] = True
    if lexical_ranked is not None:
        if lexical_ranked != old_lex:
            raise ValueError(
                "Fresh matched lexical refit differs from audited prior scores/rankings"
            )
        proof["fresh_lexical_refit_exact"] = True
    return proof


def execute_source_recovery(
    config_path: Path, output: Path, *, root: Path = ROOT, prepare_only: bool = False
) -> dict:
    """One immutable research-only pilot; never retry, select a checkpoint or promote."""
    import hashlib
    import random

    from src.research.field_baseline_data import prepare_field_baseline_data

    root, config_path, output = root.resolve(), config_path.resolve(), output.resolve()
    if not output.is_relative_to(root / "outputs") or output == root / "outputs" or output.exists():
        raise ValueError("Recovery needs a fresh directory beneath outputs")
    config = read_json(config_path)
    validate_source_recovery_config(config)
    config_ref = _reference(root, config_path)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "preparation", "actual_updates": {}}
    runtime_references = {
        name: _reference(root, root / name)
        for name in (
            "configs/config.yaml",
            "configs/model/base.yaml",
            "configs/data/datasets.yaml",
            "configs/training/default.yaml",
            "configs/training/book_lora.yaml",
        )
    }

    def check(phase):
        progress["phase"] = phase
        if time.monotonic() - started > config["max_total_seconds"]:
            raise TimeoutError(
                f"Source recovery exceeded its fixed {config['max_total_seconds']}-second cooperative budget"
            )
        if _reference(root, config_path) != config_ref:
            raise ValueError("Source recovery protocol changed during execution")
        if any(_reference(root, root / name) != ref for name, ref in runtime_references.items()):
            raise ValueError("Source recovery runtime configuration changed during execution")

    try:
        bundle = prepare_field_baseline_data(root, config)
        check("source_verified")
        if config.get("kind") == SOURCE_LEARNING_CURVES["kind"]:
            prior = config["previous_source_recovery_observation"]
            if _reference(root, root / prior["path"]) != prior:
                raise ValueError(
                    "Previous source-recovery observation differs from its protocol pin"
                )
        historical = config["prior_full_text_reference"]
        if _reference(root, root / historical["path"]) != historical:
            raise ValueError("Historical full-text reference changed")
        train, dev, mapping = bundle["train"], bundle["dev"], bundle["mapping"]
        labels = _labels(mapping)
        if (
            len(train) != config["train_limit"]
            or len(dev) != config["dev_limit"]
            or sum(map(len, labels.values())) != 48
        ):
            raise ValueError("Recovery requires exactly its fixed train/dev cohort and 48 labels")
        for rows, split in ((train, "train"), (dev, "dev")):
            _validate_rows(rows, mapping, split)
        if len({r["group_id"] for r in train + dev}) != len(train) + len(dev):
            raise ValueError("Recovery train/development groups overlap")
        if any(r["fields"][f]["negative"] for r in train + dev for f in FACETS):
            raise ValueError("This source-presence pilot requires zero assigned source negatives")
        scaling_proof = None
        if config["kind"] == SOURCE_DATA_SCALING["kind"]:
            scaling_proof = _verify_scaling_comparison(root, config, train, dev, mapping)
            check("historical_source_cohort_verified")
        bounds, cursor = {}, 0
        for f in FACETS:
            bounds[f] = [cursor, cursor + len(labels[f])]
            cursor += len(labels[f])
        schedules: dict[str, list[list[int]]] = {}
        for seed in config["seeds"]:
            schedules[str(seed)] = []
            for epoch in range(1, config["epochs"] + 1):
                indices = list(range(len(train)))
                random.Random(seed + epoch).shuffle(indices)
                for begin in range(0, len(indices), 16):
                    if not any(
                        train[i]["fields"][f]["positive"]
                        for i in indices[begin : begin + 16]
                        for f in FACETS
                    ):
                        raise ValueError(
                            "Planned batch has no source-positive row; no replacement or resampling"
                        )
                schedules[str(seed)].append(indices)
        weighting_proof = None
        weight_receipt = None
        if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"]:
            weighting_proof = _verify_weighting_history(
                root, config, train, dev, mapping, schedules
            )
            weight_receipt = source_recovery_weights(train, mapping, schedules)
            _write(output / "loss_weights.json", weight_receipt)
            check("train_only_weights_and_history_verified")
        examples_path = output / "examples.json"
        _write(
            examples_path,
            {
                "schema_version": 1,
                "kind": "source_recovery_examples",
                "mapping": mapping,
                "provenance": bundle["provenance"],
                "train": train,
                "dev": dev,
            },
        )
        schedule_path = output / "schedules.json"
        _write(
            schedule_path,
            {
                "rule": config["schedule"],
                "indices": schedules,
                "record_ids": {
                    str(seed): [
                        [train[i]["record_id"] for i in epoch] for epoch in schedules[str(seed)]
                    ]
                    for seed in config["seeds"]
                },
            },
        )
        # Cached tokenizer only: prepare-only does not resolve/read base model weights.
        from huggingface_hub import snapshot_download

        from src.data.tokenization import Tokenizer, TokenizerConfig
        from src.utils.labels import format_book_input

        recipe = config["base"]
        snapshot = Path(
            snapshot_download(recipe["repo"], revision=recipe["revision"], local_files_only=True)
        )
        tokenizer = Tokenizer(
            TokenizerConfig(
                pretrained_model_name=str(snapshot),
                max_length=256,
                padding="longest",
                truncation=True,
            )
        )
        check("tokenizer_loaded")
        encoded, lexical_texts = {}, {}
        for split, rows in (("train", train), ("dev", dev)):
            records, texts = [], []
            for begin in range(0, len(rows), 16):
                batch = tokenizer.batch_encode(
                    [format_book_input(**r["input"]) for r in rows[begin : begin + 16]],
                    max_length=256,
                    padding="longest",
                    pad_to_multiple_of=8,
                )
                for row, ids, mask in zip(
                    rows[begin : begin + 16],
                    batch["input_ids"].tolist(),
                    batch["attention_mask"].tolist(),
                    strict=True,
                ):
                    visible = [token for token, valid in zip(ids, mask, strict=True) if valid]
                    text = tokenizer.decode(visible)
                    texts.append(text)
                    records.append(
                        {
                            "record_id": row["record_id"],
                            "input_ids": ids,
                            "attention_mask": mask,
                            "decoded_control_text": text,
                            "decoded_control_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
                check(f"tokenization_{split}")
            encoded[split], lexical_texts[split] = records, texts
        if scaling_proof is not None:
            scaling_proof = _verify_scaling_comparison(
                root, config, train, dev, mapping, encoded=encoded
            )
            check("historical_visible_tokens_verified")
        if weighting_proof is not None:
            weighting_proof = _verify_weighting_history(
                root, config, train, dev, mapping, schedules, encoded=encoded
            )
            check("weighting_tokens_verified")
        token_path = output / "tokenization.json"
        _write(
            token_path,
            {
                "schema_version": 1,
                "input_format": "book_title_description_v1",
                "max_length": 256,
                "control": "Decoded exact visible model token IDs; special tokens removed",
                "records": encoded,
            },
        )
        report = {
            "schema_version": 1,
            "kind": "book_source_assignment_recovery_result",
            "status": "prepared_not_fitted",
            "config": config,
            "config_reference": config_ref,
            "runtime_configuration_references": runtime_references,
            "source_provenance": bundle["provenance"],
            "mapping": mapping,
            "facet_bounds": bounds,
            "examples": _reference(root, examples_path),
            "schedules": _reference(root, schedule_path),
            "tokenization": _reference(root, token_path),
            "selected": [
                {k: r[k] for k in ("record_id", "group_id", "effective_split", "input_sha256")}
                for r in train + dev
            ],
            "selected_label_support": {
                split: {
                    f: {
                        "positive_groups": {
                            label: sum(label in r["fields"][f]["positive"] for r in rows)
                            for label in labels[f]
                        },
                        "groups_without_observed_positives": sum(
                            not r["fields"][f]["positive"] for r in rows
                        ),
                    }
                    for f in FACETS
                }
                for split, rows in (("train", train), ("dev", dev))
            },
            "rows_without_any_observed_positive": {
                split: sum(not any(r["fields"][f]["positive"] for f in FACETS) for r in rows)
                for split, rows in (("train", train), ("dev", dev))
            },
            "human_gold": False,
            "formal_dataset_admission": False,
            "model_promoted": False,
            "paid_spend": 0,
            "neural_training_performed": False,
            "rl_training_performed": False,
            "statistical_fitting_performed": False,
            "arms": {},
            "prior_full_text_reference": config["prior_full_text_reference"],
            "implementation_sha256": {
                name: file_hash(root / name)
                for name in (
                    "src/research/field_baseline.py",
                    "src/research/field_baseline_data.py",
                    "src/training/trainer.py",
                    "src/training/utils.py",
                    "src/models/adapters.py",
                )
            },
            "semantics": {
                "loss": "Uniform known-positive distribution per facet; mean eligible facets within each row, then mean eligible rows. Whole-empty rows excluded only from loss. Softmax pressures unassigned labels and co-positives compete; omissions remain semantically unknown.",
                "metrics": "Observed publisher-source assignment recovery only, not semantic negatives, human gold, precision, F1 or accuracy.",
                "comparison": "Matched lexical control decodes exact truncated formatted neural input. Prior full literal title/newline/description TF-IDF is an external reference, not an input-matched comparison.",
                "selection": f"Endpoint{config['primary_endpoint']} primary; all other declared endpoints diagnostic; all seeds/arms retained, no best checkpoint selection.",
            },
            "timings": {},
        }
        if scaling_proof is not None:
            supervised_presentations = (
                scaling_proof["support_strata"]["expanded_train"]["objective_eligible_rows"]
                * config["epochs"]
            )
            scheduled_presentations = len(train) * config["epochs"]
            if (
                supervised_presentations != config["source_supervised_presentations_per_arm"]
                or scheduled_presentations != config["scheduled_presentations_per_arm"]
            ):
                raise ValueError("Fixed source-supervised or scheduled presentation counts changed")
            report["comparison_cohort_proof"] = scaling_proof
            report["comparison_support_strata"] = scaling_proof["support_strata"]
            report["scheduled_presentations_per_arm"] = scheduled_presentations
            report["source_supervised_presentations_per_arm"] = supervised_presentations
            report["semantics"]["historical_comparison"] = config["comparison"]
        if weighting_proof is not None:
            report["weighting_history_proof"] = weighting_proof
            report["loss_weights"] = _reference(root, output / "loss_weights.json")
            report["weight_vector"] = weight_receipt
            report["semantics"]["loss"] = config["loss"]
            report["semantics"]["weighting"] = config["weighting_semantics"]
            report["scheduled_presentations_per_arm"] = config["scheduled_presentations_per_arm"]
            report["source_supervised_presentations_per_arm"] = config[
                "source_supervised_presentations_per_arm"
            ]
        if not prepare_only or weighting_proof is not None:
            check("matched_lexical_fitting")
            fitted = fit_rankers(train, mapping, config["tfidf"], texts=lexical_texts["train"])
            check("matched_lexical_evaluation")
            lexical_ranked = rank_rows(fitted, dev, texts=lexical_texts["dev"])
            report["matched_lexical_metrics"] = ranking_metrics(lexical_ranked, mapping)
            report["training_support"] = fitted["support"]
            path = output / "matched_lexical_rankings.json"
            _write(path, {"records": lexical_ranked})
            report["matched_lexical_rankings"] = _reference(root, path)
            report["matched_lexical_artifacts"] = _save_fitted(
                root, output, fitted, config["tfidf"]
            )
            if weighting_proof is not None:
                report["weighting_history_proof"] = _verify_weighting_history(
                    root,
                    config,
                    train,
                    dev,
                    mapping,
                    schedules,
                    encoded=encoded,
                    lexical_ranked=lexical_ranked,
                )
                report["statistical_fitting_performed"] = True
                if prepare_only:
                    report["status"] = "prepared_weights_and_lexical_verified_not_neural_trained"
                check("fresh_lexical_refit_verified")
        if not prepare_only:
            check("neural_initialization")
            _fit_source_recovery(
                root,
                output,
                config,
                report,
                train,
                dev,
                encoded,
                tokenizer,
                schedules,
                bounds,
                check,
                progress,
            )
            if scaling_proof is not None:
                report["historical_primary_comparison"] = _scaling_primary_comparison(
                    root, config, report
                )
                check("historical_primary_comparison_saved")
            report.update(
                status="completed_source_assignment_recovery_pilot",
                neural_training_performed=True,
                statistical_fitting_performed=True,
            )
        check("final_receipt")
        report["timings"]["total_seconds"] = time.monotonic() - started
        _write(output / "report.json", report)
        check("receipt_saved")
        return report
    except BaseException as error:
        # A terminal post-save failure must invalidate any earlier apparent success.
        if (output / "report.json").exists():
            from src.utils.atomic import atomic_write

            failed_report = read_json(output / "report.json")
            failed_report.update(status="failed_incomplete_do_not_use", terminal_error=str(error))
            atomic_write(
                output / "report.json", lambda stream: stream.write(json_bytes(failed_report))
            )
        _write(
            output / "failure.json",
            {
                "schema_version": 1,
                "status": "failed_incomplete_do_not_use",
                "config_reference": config_ref,
                "progress": progress,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
            },
        )
        raise


def _fit_source_recovery(
    root, output, config, report, train, dev, encoded, tokenizer, schedules, bounds, check, progress
):
    """Reuse native model/Trainer plumbing, with a research-local forward objective."""
    import gc
    from dataclasses import asdict

    import mlflow
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir

    from src.models.adapters import extract_effective_delta
    from src.training.trainer import Trainer, TrainerConfig
    from src.training.utils import (
        build_training_model,
        build_training_optimizer,
        prepare_training_runtime,
        set_seed,
    )
    from src.utils.atomic import atomic_write

    labels = _labels(report["mapping"])
    positions = {f: {label: bounds[f][0] + i for i, label in enumerate(labels[f])} for f in FACETS}
    indicators = {}
    for split, rows in (("train", train), ("dev", dev)):
        indicators[split] = torch.zeros((len(rows), 48), dtype=torch.bool)
        for i, row in enumerate(rows):
            for f in FACETS:
                for label in row["fields"][f]["positive"]:
                    indicators[split][i, positions[f][label]] = True

    def collate(indices, split):
        records = [encoded[split][i] for i in indices]
        width = max(sum(r["attention_mask"]) for r in records)
        width = ((width + 7) // 8) * 8
        ids = torch.full((len(indices), width), tokenizer.pad_token_id, dtype=torch.long)
        masks = torch.zeros_like(ids, dtype=torch.bool)
        for j, row in enumerate(records):
            visible = [
                token
                for token, valid in zip(row["input_ids"], row["attention_mask"], strict=True)
                if valid
            ]
            ids[j, : len(visible)] = torch.tensor(visible)
            masks[j, : len(visible)] = True
        batch = {"input_ids": ids, "attention_mask": masks}
        if split == "train":
            batch["labels"] = indicators[split][indices]
        return batch

    class SourceRecoveryTrainer(Trainer):
        source_weights: torch.Tensor | None = None

        def _forward_task(self, task, batch, *, summarize_metrics=True):
            if task != "source_recovery":
                return super()._forward_task(task, batch, summarize_metrics=summarize_metrics)
            check("source_recovery_forward")
            logits = self.model.forward(
                "topic", {k: batch[k] for k in ("input_ids", "attention_mask")}
            )
            return source_recovery_loss(
                logits, batch["labels"], bounds, label_weights=getattr(self, "source_weights", None)
            ), {}

    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["training=book_lora", "device=cuda"])
    cfg.training.optimizer = config["optimizer"]
    cfg.training.trainer.gradient_accumulation_steps = 1
    cfg.model.dropout = 0.0
    _validate_recovery_runtime(cfg, config)
    from omegaconf import OmegaConf

    report["resolved_runtime_config"] = OmegaConf.to_container(cfg, resolve=True)
    check("runtime_configuration_bound")
    device, snapshot = prepare_training_runtime(cfg)
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Authorized pilot requires CUDA BF16; no device fallback")
    check("runtime_verified")
    report["device"] = {"name": torch.cuda.get_device_name(device), "bf16_supported": True}
    report["dependency_versions"] = {
        "python": platform.python_version(),
        **{
            name: metadata.version(name)
            for name in ("torch", "transformers", "numpy", "scipy", "scikit-learn")
        },
    }
    learning_curves = config["kind"] in {
        SOURCE_LEARNING_CURVES["kind"],
        SOURCE_DATA_SCALING["kind"],
        SOURCE_LOSS_WEIGHTING["kind"],
    }
    applied_weights = (
        torch.tensor(report["weight_vector"]["applied_weights"], dtype=torch.float32, device=device)
        if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"]
        else None
    )

    def evaluate(model, split, method, key, update):
        rows = train if split == "train" else dev
        model.eval()
        scores: dict[str, list] = {f: [] for f in FACETS}
        log_scores: dict[str, list] = {f: [] for f in FACETS}
        totals = {
            "eligible_rows": 0,
            "excluded_rows": 0,
            "cross_entropy_sum": 0.0,
            "target_entropy_floor_sum": 0.0,
        }
        weighted_totals = dict(totals)
        with torch.no_grad():
            for begin in range(0, len(rows), 16):
                indices = list(range(begin, min(begin + 16, len(rows))))
                batch = {
                    k: v.to(device) for k, v in collate(indices, split).items() if k != "labels"
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.forward("topic", batch)
                if logits.shape != (len(indices), 48) or not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite or malformed neural evaluation logits")
                if learning_curves:
                    values = source_recovery_fit_totals(
                        logits, indicators[split][indices].to(device), bounds
                    )
                    for name, value in values.items():
                        totals[name] += value
                    if applied_weights is not None:
                        weighted_values = source_recovery_fit_totals(
                            logits,
                            indicators[split][indices].to(device),
                            bounds,
                            label_weights=applied_weights,
                        )
                        for name, value in weighted_values.items():
                            weighted_totals[name] += value
                for f, (start, stop) in bounds.items():
                    scores[f].append(logits[:, start:stop].float().softmax(dim=1).cpu().numpy())
                    if learning_curves:
                        log_scores[f].append(
                            logits[:, start:stop].float().log_softmax(dim=1).cpu().numpy()
                        )
                check(f"evaluate_{key}_{update}_{split}")
        ranked = source_recovery_rankings(
            {f: np.concatenate(v) for f, v in scores.items()},
            rows,
            report["mapping"],
            method,
            log_scores={f: np.concatenate(v) for f, v in log_scores.items()}
            if learning_curves
            else None,
        )
        fit: dict | None = None
        if learning_curves:
            denominator = totals["eligible_rows"]
            ce = totals["cross_entropy_sum"] / denominator if denominator else None
            floor = totals["target_entropy_floor_sum"] / denominator if denominator else None
            fit = {
                **totals,
                "cross_entropy": ce,
                "target_entropy_floor": floor,
                "excess_cross_entropy": ce - floor
                if ce is not None and floor is not None
                else None,
                "mode": "eval_bf16_autocast",
                "row_weighting": "eligible_row_count",
            }
            if applied_weights is not None:
                weight_ce = (
                    weighted_totals["cross_entropy_sum"] / denominator if denominator else None
                )
                weight_floor = (
                    weighted_totals["target_entropy_floor_sum"] / denominator
                    if denominator
                    else None
                )
                fit["weighted_objective"] = {
                    **weighted_totals,
                    "cross_entropy": weight_ce,
                    "target_entropy_floor": weight_floor,
                    "excess_cross_entropy": weight_ce - weight_floor
                    if weight_ce is not None and weight_floor is not None
                    else None,
                    "weight_tensor_sha256": report["weight_vector"]["tensor_sha256"],
                }
        return ranked, ranking_metrics(ranked, report["mapping"], methods=(method,)), fit

    for seed in config["seeds"]:
        expected_initial = None
        expected_endpoint_zero: dict[str, str] = {}
        for arm in config["arms"]:
            check(f"initialize_{seed}_{arm}")
            cfg.seed = seed
            set_seed(seed)
            model, binding = build_training_model(
                cfg,
                tokenizer,
                num_emotions=0,
                num_topics=48,
                topic_problem_type="multi_label",
                snapshot=snapshot,
                compile_modules=False,
            )
            check(f"initialized_{seed}_{arm}")
            private = set(binding.private_parameters)
            factors = {n for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))}
            state_names = private | factors
            initial_hashes = {
                n: _recovery_tensor_hash(model.get_parameter(n)) for n in sorted(state_names)
            }
            identity = {"binding": asdict(binding), "tensor_hashes": initial_hashes}
            if expected_initial is None:
                expected_initial = identity
            elif identity != expected_initial:
                raise ValueError(
                    "Paired arms did not share exact head/factor initialization and frozen base"
                )
            if arm == "head_only":
                for n in factors:
                    model.get_parameter(n).requires_grad_(False)
            actual_trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            if actual_trainable != (private if arm == "head_only" else state_names):
                raise ValueError("Recovery arm trainable parameters differ from exact allowlist")
            optimizer = build_training_optimizer(model, cfg, device)
            trainer = SourceRecoveryTrainer(
                model,
                optimizer,
                TrainerConfig(
                    max_epochs=config["epochs"],
                    gradient_accumulation_steps=1,
                    gradient_clip_norm=1.0,
                    scheduler_type="constant",
                    warmup_steps=0,
                    early_stopping_patience=None,
                    task_sampling="round_robin",
                    generation_metrics=False,
                    gradient_conflict_frequency=0,
                    use_pcgrad=False,
                    tracking_uri="sqlite:///" + str(output / "mlflow.db"),
                    experiment_name="SourceAssignmentRecovery",
                    run_name=f"{seed}_{arm}",
                ),
                device,
                tokenizer,
            )
            if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"]:
                trainer.source_weights = applied_weights if arm == "weighted" else None
                arm_report_objective = (
                    "weighted_positive_terms"
                    if arm == "weighted"
                    else "original_unweighted_positive_terms"
                )
            if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"] and optimizer.state:
                raise ValueError("Fresh weighting optimizer must have zero initial state entries")
            if not trainer.use_bfloat16:
                raise ValueError("Trainer did not enable BF16 autocast")
            key = f"{seed}_{arm}"
            progress["actual_updates"][key] = 0
            arm_report = {
                "seed": seed,
                "arm": arm,
                "initialization": identity,
                "trainable_names": sorted(actual_trainable),
                "endpoints": {},
                "training": {},
            }
            if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"]:
                arm_report["training_objective"] = arm_report_objective
                arm_report["weight_tensor_sha256"] = report["weight_vector"]["tensor_sha256"]
                arm_report["initial_optimizer_state_entries"] = len(optimizer.state)
                arm_report["fresh_optimizer_verified"] = True
            report["arms"][key] = arm_report
            arm_output = output / key
            arm_output.mkdir()

            study_started = time.monotonic()

            def endpoint(
                key=key,
                trainer=trainer,
                model=model,
                arm=arm,
                seed=seed,
                binding=binding,
                state_names=state_names,
                initial_hashes=initial_hashes,
                factors=factors,
                arm_output=arm_output,
                arm_report=arm_report,
                study_started=study_started,
                expected_endpoint_zero=expected_endpoint_zero,
            ):
                check(f"evaluate_{key}_{trainer.global_step}")
                evaluation_start = time.monotonic()
                ranked, metrics, dev_fit = evaluate(model, "dev", arm, key, trainer.global_step)
                if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"] and trainer.global_step == 0:
                    import hashlib

                    zero_hash = hashlib.sha256(
                        json_bytes([row["methods"][arm] for row in ranked])
                    ).hexdigest()
                    if expected_endpoint_zero and expected_endpoint_zero["sha256"] != zero_hash:
                        raise ValueError(
                            "Paired endpoint0 predictions differ ignoring method names"
                        )
                    is_second_arm = bool(expected_endpoint_zero)
                    expected_endpoint_zero["sha256"] = zero_hash
                    arm_report["endpoint_zero_prediction_sha256"] = zero_hash
                    arm_report["paired_endpoint_zero_predictions_verified"] = is_second_arm
                    if is_second_arm:
                        report["arms"][f"{seed}_unweighted"][
                            "paired_endpoint_zero_predictions_verified"
                        ] = True
                train_fit = None
                if learning_curves and trainer.global_step == config["primary_endpoint"]:
                    train_started = time.monotonic()
                    train_ranked, train_metrics, train_objective = evaluate(
                        model, "train", arm, key, trainer.global_step
                    )
                    train_path = arm_output / "final_train.rankings.json"
                    _write(train_path, {"schema_version": 1, "records": train_ranked})
                    train_fit = {
                        "metrics": train_metrics,
                        "objective": train_objective,
                        "rankings": _reference(root, train_path),
                        "evaluation_and_save_seconds": time.monotonic() - train_started,
                    }
                    arm_report["final_train_fit"] = train_fit
                    _write(arm_output / "final_train.receipt.json", train_fit)
                    check(f"final_train_saved_{key}")
                delta = extract_effective_delta(model, binding, task_id="source_recovery")
                adaptation_norm = float(
                    sum(v.double().square().sum().item() for v in delta.shared.values()) ** 0.5
                )
                hashes = {
                    n: _recovery_tensor_hash(model.get_parameter(n)) for n in sorted(state_names)
                }
                if arm == "head_only" and any(hashes[n] != initial_hashes[n] for n in factors):
                    raise ValueError("Head-only factors changed")
                if arm == "head_only" and adaptation_norm != 0:
                    raise ValueError("Head-only encoder acquired an effective adapter delta")
                if (
                    (arm == "lora_head" or config["kind"] == SOURCE_LOSS_WEIGHTING["kind"])
                    and trainer.global_step
                    and adaptation_norm == 0
                ):
                    raise ValueError("LoRA arm has no effective adaptation")
                check(f"save_{key}_{trainer.global_step}")
                checkpoint = arm_output / f"endpoint_{trainer.global_step}.research.pt"
                artifact = {
                    "schema_version": 1,
                    "kind": "book_source_assignment_recovery_artifact",
                    "production_inference_supported": False,
                    "resume_supported": False,
                    "arm": arm,
                    "seed": seed,
                    "actual_updates": trainer.global_step,
                    "binding": asdict(binding),
                    "config": config,
                    "config_reference": report["config_reference"],
                    "model_config": model._leximind_training_model_config,
                    "resolved_runtime_config": OmegaConf.to_container(cfg, resolve=True),
                    "runtime_configuration_references": report["runtime_configuration_references"],
                    "implementation_sha256": report["implementation_sha256"],
                    "vocabulary": labels,
                    "facet_bounds": bounds,
                    "tokenizer_contract": model._leximind_training_tokenizer_contract,
                    "source_provenance": report["source_provenance"],
                    "examples": report["examples"],
                    "schedules": report["schedules"],
                    "tokenization": report["tokenization"],
                    "tensor_hashes": hashes,
                    "research_state": {
                        n: model.get_parameter(n).detach().cpu().clone()
                        for n in sorted(state_names)
                    },
                }
                if config["kind"] == SOURCE_LOSS_WEIGHTING["kind"]:
                    artifact["loss_weight_vector"] = report["weight_vector"]
                    artifact["training_objective"] = arm_report["training_objective"]
                atomic_write(checkpoint, lambda stream: torch.save(artifact, stream))
                rank_path = arm_output / f"endpoint_{trainer.global_step}.rankings.json"
                _write(rank_path, {"schema_version": 1, "records": ranked})
                receipt = {
                    "actual_updates": trainer.global_step,
                    "metrics": metrics,
                    "artifact": _reference(root, checkpoint),
                    "rankings": _reference(root, rank_path),
                    "tensor_hashes": hashes,
                    "effective_adapter_l2": adaptation_norm,
                    "frozen_base_verified": True,
                    "evaluation_and_save_seconds": time.monotonic() - evaluation_start,
                    "elapsed_seconds": time.monotonic() - study_started,
                }
                if learning_curves:
                    receipt["dev_fit"] = dev_fit
                    if train_fit is not None:
                        receipt["final_train_fit"] = train_fit
                arm_report["endpoints"][str(trainer.global_step)] = receipt
                _write(arm_output / f"endpoint_{trainer.global_step}.receipt.json", receipt)
                check(f"saved_{key}_{trainer.global_step}")

            endpoint()
            with mlflow.start_run(run_name=key):
                for epoch in range(1, config["epochs"] + 1):
                    batches = [
                        collate(schedules[str(seed)][epoch - 1][i : i + 16], "train")
                        for i in range(0, len(train), config["batch_size"])
                    ]
                    if learning_curves:
                        batch_path = output / f"seed_{seed}_epoch_{epoch}.batch_hashes.json"
                        _write(
                            batch_path,
                            {
                                "schema_version": 1,
                                "kind": "reconstructable_actual_batch_tensor_proof",
                                "seed": seed,
                                "epoch": epoch,
                                "schedule": report["schedules"],
                                "tokenization": report["tokenization"],
                                "examples": report["examples"],
                                "padding": {
                                    "side": "right",
                                    "pad_token_id": tokenizer.pad_token_id,
                                    "rule": "longest visible row rounded up to multiple8",
                                },
                                "tensor_hash_format": "SHA256(json_bytes(shape,dtype) + contiguous CPU NumPy C-order tensor bytes)",
                                "batches": [
                                    {
                                        "batch_index": j,
                                        "schedule_offset": j * 16,
                                        "rows": len(b["input_ids"]),
                                        "tensor_shapes": {
                                            name: list(value.shape) for name, value in b.items()
                                        },
                                        "tensor_sha256": {
                                            name: _recovery_tensor_hash(value)
                                            for name, value in b.items()
                                        },
                                    }
                                    for j, b in enumerate(batches)
                                ],
                            },
                        )
                    else:
                        batch_path = arm_output / f"epoch_{epoch}.batches.json"
                        _write(
                            batch_path,
                            {
                                "schedule": report["schedules"],
                                "epoch": epoch,
                                "batches": [
                                    {
                                        "input_ids": b["input_ids"].tolist(),
                                        "attention_mask": b["attention_mask"].tolist(),
                                    }
                                    for b in batches
                                ],
                            },
                        )
                    arm_report.setdefault("actual_batch_inputs", {})[str(epoch)] = _reference(
                        root, batch_path
                    )
                    check(f"batches_saved_{key}_{epoch}")
                    loaders = {"source_recovery": batches}
                    trainer._setup_scheduler(loaders, epoch)
                    epoch_start = time.monotonic()

                    def step_checked(key=key, trainer=trainer):
                        progress["actual_updates"][key] = trainer.global_step
                        check(f"train_{key}")

                    arm_report["training"][str(epoch)] = trainer._run_epoch(
                        loaders, train=True, epoch=epoch, step_callback=step_checked
                    )
                    arm_report["training"][str(epoch)]["seconds"] = time.monotonic() - epoch_start
                    if config["kind"] in {
                        SOURCE_DATA_SCALING["kind"],
                        SOURCE_LOSS_WEIGHTING["kind"],
                    }:
                        arm_report["actual_scheduled_presentations"] = (
                            trainer.global_step * config["batch_size"]
                        )
                        arm_report["actual_source_supervised_presentations"] = epoch * int(
                            indicators["train"].any(dim=1).sum()
                        )
                    if trainer.global_step != epoch * (len(train) // config["batch_size"]):
                        raise ValueError(
                            "Recovery update count differs from its fixed record schedule"
                        )
                    if trainer.global_step in config["endpoints"]:
                        endpoint()
            del endpoint, step_checked, trainer, optimizer, model, batches
            gc.collect()
            torch.cuda.empty_cache()
            check(f"completed_{key}")


def _validate_recovery_runtime(cfg, config) -> None:
    """Bind the actual composed recipe to the same tokenizer/base and adapter scope."""
    import json

    recipe = cfg.training.book_lora
    expected = {
        "base_repo": config["base"]["repo"],
        "base_revision": config["base"]["revision"],
        "base_weight_sha256": config["base"]["weight_sha256"],
        **config["lora"],
        "dtype": config["parameter_dtype"],
    }
    for key, value in expected.items():
        if json.dumps(recipe.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Resolved recovery runtime differs from frozen {key}")
    if cfg.model.dropout != config["dropout"] or cfg.device != config["device"]:
        raise ValueError("Resolved recovery model dropout/device differs from frozen protocol")
