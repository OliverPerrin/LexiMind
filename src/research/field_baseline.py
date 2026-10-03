"""Bounded weak source-label ranking diagnostics; no neural training or label admission.

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


def fit_rankers(train_rows: list[dict], mapping: dict, tfidf_config: dict) -> dict:
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
    train = vectorizer.fit_transform([_text(row) for row in train_rows]).tocsr()
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


def rank_rows(fitted: dict, rows: list[dict], top_k: tuple[int, ...] = TOP_K) -> list[dict]:
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
    matrix = fitted["vectorizer"].transform([_text(row) for row in rows]).tocsr()
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


def ranking_metrics(ranked_rows: list[dict], mapping: dict, top_k: tuple[int, ...] = TOP_K) -> dict:
    """Observed-positive recovery only; exclude unobserved facets, never score negatives."""
    labels = _labels(mapping)
    if len({row["group_id"] for row in ranked_rows}) != len(ranked_rows):
        raise ValueError("Group-macro metrics require one unique row per group")
    result: dict = {}
    for method in METHODS:
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
