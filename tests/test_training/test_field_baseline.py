"""Synthetic field-ranking checks; no pretrained models or real book text."""

from __future__ import annotations

import json
from copy import deepcopy
from math import log
from threading import Event

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.research import field_baseline as baseline
from src.research.book_fields import FACETS, input_hash

TFIDF = {**baseline.TFIDF, "min_df": 1, "ngram_range": [1, 1], "stop_words": None}


@pytest.fixture
def mapping():
    return {
        "facets": {
            "genre": {"labels": ["alpha", "beta", "gamma"]},
            "topic": {"labels": ["history", "science"]},
            "form": {"labels": ["fiction"]},
            "audience": {"labels": ["adult", "children"]},
        }
    }


def fields(**positive):
    return {facet: {"positive": list(positive.get(facet, ())), "negative": []} for facet in FACETS}


def row(identifier, description, **positive):
    payload = {"title": "", "description": description}
    split = identifier.split(":", 1)[0]
    return {
        "record_id": identifier,
        "group_id": f"group:{identifier}",
        "source_split": split,
        "effective_split": split,
        "input": payload,
        "input_sha256": input_hash(payload),
        "fields": fields(**positive),
    }


@pytest.fixture
def train_rows():
    return [
        row("train:1", "azure river calm", genre=["alpha"], form=["fiction"]),
        row("train:2", "azure river water", genre=["alpha"], topic=["science"]),
        row("train:3", "bronze desert sand", genre=["beta"], audience=["adult"]),
    ]


def test_idf_and_centroids_use_only_train_documents_and_positive_labels(mapping, train_rows):
    fitted = baseline.fit_rankers(train_rows, mapping, TFIDF)
    vectorizer = fitted["vectorizer"]
    vocab = vectorizer.vocabulary_
    assert vectorizer.idf_[vocab["azure"]] == pytest.approx(1 + log(4 / 3))
    assert vectorizer.idf_[vocab["bronze"]] == pytest.approx(1 + log(4 / 2))
    # Label names are transformed only; they cannot create vocabulary entries.
    assert "alpha" not in vocab and "gamma" not in vocab
    documents = vectorizer.transform([row["input"]["description"] for row in train_rows]).toarray()
    expected_alpha = documents[:2].mean(axis=0)
    expected_alpha /= np.linalg.norm(expected_alpha)
    np.testing.assert_allclose(fitted["centroids"]["genre"].toarray()[0], expected_alpha)
    np.testing.assert_allclose(fitted["centroids"]["genre"].toarray()[1], documents[2])
    assert not fitted["centroids"]["genre"].toarray()[2].any()
    assert fitted["support"]["genre"]["positive_groups"] == {"alpha": 2, "beta": 1, "gamma": 0}


def test_development_only_tokens_and_labels_cannot_update_vocab_idf_or_prototypes(
    mapping, train_rows
):
    fitted = baseline.fit_rankers(train_rows, mapping, TFIDF)
    vocab = dict(fitted["vectorizer"].vocabulary_)
    idf = fitted["vectorizer"].idf_.copy()
    matrices = {facet: value.toarray().copy() for facet, value in fitted["centroids"].items()}
    frequency = {facet: value.copy() for facet, value in fitted["frequency"].items()}
    baseline.rank_rows(
        fitted,
        [row("dev:1", "quasaronly quasaronly", genre=["gamma"], topic=["history"])],
    )
    assert "quasaronly" not in fitted["vectorizer"].vocabulary_
    assert fitted["vectorizer"].vocabulary_ == vocab
    np.testing.assert_array_equal(fitted["vectorizer"].idf_, idf)
    for facet in FACETS:
        np.testing.assert_array_equal(fitted["centroids"][facet].toarray(), matrices[facet])
        np.testing.assert_array_equal(fitted["frequency"][facet], frequency[facet])


def test_explicit_negatives_do_not_turn_unknowns_into_training_negatives(mapping, train_rows):
    ordinary = baseline.fit_rankers(train_rows, mapping, TFIDF)
    amended = deepcopy(train_rows)
    amended[0]["fields"]["genre"]["negative"] = ["beta"]
    amended[2]["fields"]["genre"]["negative"] = ["alpha"]
    with_negatives = baseline.fit_rankers(amended, mapping, TFIDF)
    assert ordinary["diagnostics"]["negative_training_labels"] == 0
    assert with_negatives["diagnostics"]["negative_training_labels"] == 2
    assert with_negatives["diagnostics"]["negative_labels_used_for_fitting"] == 0
    assert with_negatives["support"] == ordinary["support"]
    for facet in FACETS:
        np.testing.assert_array_equal(
            with_negatives["centroids"][facet].toarray(), ordinary["centroids"][facet].toarray()
        )
        np.testing.assert_array_equal(
            with_negatives["frequency"][facet], ordinary["frequency"][facet]
        )


def test_ties_are_alphabetical_zero_support_labels_remain_and_large_k_is_bounded(
    mapping, train_rows
):
    mapping["facets"]["genre"]["labels"].reverse()
    fitted = baseline.fit_rankers(train_rows, mapping, TFIDF)
    (ranked,) = baseline.rank_rows(
        fitted, [row("dev:1", "outofvocabulary", genre=["gamma"], form=["fiction"])]
    )
    assert ranked["zero_tfidf_vector"] is True
    for method in baseline.METHODS:
        genre = ranked["methods"][method]["genre"]
        assert [entry["label"] for entry in genre["ranking"]] == ["alpha", "beta", "gamma"]
        assert genre["recall_at_k"] == {"1": 0, "3": 1, "5": 1}
        assert genre["effective_k"] == {"1": 1, "3": 3, "5": 3}
        assert len(ranked["methods"][method]["form"]["ranking"]) == 1
        assert ranked["methods"][method]["form"]["recall_at_k"] == {"1": 1, "3": 1, "5": 1}
        assert ranked["methods"][method]["form"]["effective_k"] == {"1": 1, "3": 1, "5": 1}
        if method != "frequency":
            assert genre["all_scores_zero"] is True
            assert genre["distinct_score_count"] == 1
    assert fitted["support"]["genre"]["zero_positive_support_labels"] == ["gamma"]
    metrics = baseline.ranking_metrics([ranked], mapping)
    for method in baseline.METHODS:
        assert metrics[method]["form"]["effective_k"] == {"1": 1, "3": 1, "5": 1}
        gamma = metrics[method]["genre"]["by_label"]["gamma"]
        assert gamma["observed_positive_groups"] == 1
        assert gamma["recall_at_k"] == {"1": 0, "3": 1, "5": 1}


def test_known_negative_and_unknown_predictions_neither_count_as_errors_nor_change_recovery(
    mapping, train_rows
):
    fitted = baseline.fit_rankers(train_rows, mapping, TFIDF)
    original = row("dev:1", "azure", genre=["gamma"])
    changed = deepcopy(original)
    changed["fields"]["genre"]["negative"] = ["alpha"]
    before = baseline.rank_rows(fitted, [original])
    after = baseline.rank_rows(fitted, [changed])
    assert before[0]["observed_negative"]["genre"] == []
    assert after[0]["observed_negative"]["genre"] == ["alpha"]
    assert before[0]["methods"] == after[0]["methods"]
    assert baseline.ranking_metrics(before, mapping) == baseline.ranking_metrics(after, mapping)


def hand_ranked_rows(mapping):
    positive_sets = [["alpha", "beta"], ["alpha"], ["alpha"], []]
    genre_orders = [
        ["alpha", "beta", "gamma"],
        ["alpha", "beta", "gamma"],
        ["beta", "alpha", "gamma"],
        ["alpha", "beta", "gamma"],
    ]
    results = []
    for index, (positives, genre_order) in enumerate(zip(positive_sets, genre_orders, strict=True)):
        results.append(
            {
                "record_id": f"dev:{index}",
                "group_id": f"group:{index}",
                "observed_positive": {
                    facet: positives if facet == "genre" else [] for facet in FACETS
                },
                "observed_negative": {facet: [] for facet in FACETS},
                "methods": {
                    method: {
                        facet: {
                            "ranking": [
                                {"label": label, "score": float(10 - rank)}
                                for rank, label in enumerate(
                                    genre_order
                                    if facet == "genre"
                                    else mapping["facets"][facet]["labels"]
                                )
                            ],
                            "all_scores_zero": False,
                            # Cached values are intentionally false: aggregates must use actual rankings.
                            "recall_at_k": {"1": 999, "3": 999, "5": 999},
                            "reciprocal_rank": 999,
                        }
                        for facet in FACETS
                    }
                    for method in baseline.METHODS
                },
            }
        )
    return results


def test_group_and_label_macro_recall_have_independent_hand_calculated_denominators(mapping):
    metrics = baseline.ranking_metrics(hand_ranked_rows(mapping), mapping)
    for method in baseline.METHODS:
        genre = metrics[method]["genre"]
        assert genre["evaluated_groups"] == 3
        assert genre["excluded_groups_without_observed_positives"] == 1
        assert genre["observed_positive_count"] == 4
        # Row recalls: 1/2, 1, 0. Label recalls: alpha 2/3, beta 0/1.
        assert genre["group_macro_recall_at_k"]["1"] == pytest.approx(0.5)
        assert genre["label_macro_recall_at_k"]["1"] == pytest.approx(1 / 3)
        assert genre["group_macro_reciprocal_rank"] == pytest.approx(5 / 6)
        assert genre["label_macro_recall_at_k"]["3"] == 1
        assert genre["zero_evaluation_support_labels"] == ["gamma"]
        assert genre["by_label"]["gamma"]["recall_at_k"]["1"] is None
        assert genre["by_label"]["alpha"]["observed_positive_groups"] == 3
        assert genre["by_label"]["beta"]["observed_positive_groups"] == 1


def test_facets_with_no_positive_observations_are_excluded_with_null_metrics(mapping):
    rows = hand_ranked_rows(mapping)
    # An explicit negative does not create an observed-positive evaluation opportunity.
    rows[0]["observed_negative"]["topic"] = ["history"]
    metrics = baseline.ranking_metrics(rows, mapping)
    for method in baseline.METHODS:
        topic = metrics[method]["topic"]
        assert topic["evaluated_groups"] == 0
        assert topic["excluded_groups_without_observed_positives"] == 4
        assert topic["group_macro_recall_at_k"] == {"1": None, "3": None, "5": None}
        assert topic["label_macro_recall_at_k"] == {"1": None, "3": None, "5": None}
        assert topic["group_macro_reciprocal_rank"] is None


@pytest.mark.parametrize("overlap", ["record_id", "group_id"])
def test_development_must_be_disjoint_from_fitted_ids_and_groups(mapping, train_rows, overlap):
    fitted = baseline.fit_rankers(train_rows, mapping, TFIDF)
    development = row("dev:1", "azure")
    development[overlap] = train_rows[0][overlap]
    with pytest.raises(ValueError, match="overlap"):
        baseline.rank_rows(fitted, [development])


@pytest.mark.parametrize("mutation", ["group", "role", "hash", "contradictory_state"])
def test_invalid_training_rows_cannot_reach_fitting(mapping, train_rows, mutation):
    if mutation == "group":
        train_rows[1]["group_id"] = train_rows[0]["group_id"]
    elif mutation == "role":
        train_rows[0]["effective_split"] = "dev"
    elif mutation == "hash":
        train_rows[0]["input"]["title"] = "Changed after receipt"
    else:
        train_rows[0]["fields"]["genre"]["negative"] = ["alpha"]
    with pytest.raises(ValueError):
        baseline.fit_rankers(train_rows, mapping, TFIDF)


def fixed_config():
    return {
        "schema_version": 1,
        "kind": "book_field_retrieval_diagnostic",
        "methods": list(baseline.METHODS),
        "facets": list(FACETS),
        "top_k": list(baseline.TOP_K),
        "train_limit": 4096,
        "dev_limit": 1024,
        "selection_salt": "bgc-field-retrieval-v1",
        "tfidf": deepcopy(baseline.TFIDF),
        "max_total_seconds": 600,
        "promote": False,
        "paid_spend_authorized": False,
    }


@pytest.mark.parametrize(
    "mutation", ["time_budget", "cohort", "features", "seeds", "boolean_cutoff", "promote"]
)
def test_execute_rejects_changed_budget_or_protocol_before_creating_output(
    tmp_path, monkeypatch, mutation
):
    config = fixed_config()
    if mutation == "time_budget":
        config["max_total_seconds"] += 1
    elif mutation == "cohort":
        config["train_limit"] += 1
    elif mutation == "features":
        config["tfidf"]["max_features"] += 1
    elif mutation == "seeds":
        config["seeds"] = [1, 2]
    elif mutation == "boolean_cutoff":
        config["top_k"][0] = True
    else:
        config["promote"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "outputs/run"
    with pytest.raises(ValueError):
        baseline.execute(path, output, root=tmp_path)
    assert not output.exists()


def test_existing_output_is_preserved_even_before_config_is_opened(tmp_path):
    output = tmp_path / "outputs/prior-run"
    output.mkdir(parents=True)
    sentinel = output / "report.json"
    sentinel.write_bytes(b"prior immutable diagnostic")
    with pytest.raises(ValueError, match="already exists"):
        baseline.execute(tmp_path / "nonexistent-config.json", output, root=tmp_path)
    assert sentinel.read_bytes() == b"prior immutable diagnostic"
    assert list(output.iterdir()) == [sentinel]


def test_failed_preparation_records_failure_without_fitting_or_success_report(
    tmp_path, monkeypatch
):
    from src.research import field_baseline_data

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fixed_config()))
    output = tmp_path / "outputs/failed-run"

    def expired(*args, **kwargs):
        raise TimeoutError("Synthetic fixed-budget expiration")

    monkeypatch.setattr(field_baseline_data, "prepare_field_baseline_data", expired)
    with pytest.raises(TimeoutError, match="fixed-budget"):
        baseline.execute(config_path, output, root=tmp_path)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed_incomplete_do_not_use"
    assert failure["error_type"] == "TimeoutError"
    assert not (output / "report.json").exists()
    assert not (output / "fitted").exists()
    with pytest.raises(ValueError, match="already exists"):
        baseline.execute(config_path, output, root=tmp_path)


def test_wall_clock_budget_interrupts_and_restores_handler_and_timer():
    previous_handler = baseline.signal.getsignal(baseline.signal.SIGALRM)
    assert baseline.signal.getitimer(baseline.signal.ITIMER_REAL) == (0.0, 0.0)
    with pytest.raises(TimeoutError, match="wall-clock budget"):
        with baseline._time_limit(0.02):
            Event().wait(1)
    assert baseline.signal.getsignal(baseline.signal.SIGALRM) == previous_handler
    assert baseline.signal.getitimer(baseline.signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("prepare_only", [True, False])
def test_execute_persists_auditable_synthetic_run_or_preparation_without_fitting(
    tmp_path, monkeypatch, mapping, train_rows, prepare_only
):
    from scipy import sparse

    from src.research import field_baseline_data, field_baseline_review

    development = [
        row("dev:1", "bronze desert", genre=["beta"]),
        row("dev:2", "azure river", genre=["alpha"]),
        row("dev:3", "unknownonly", genre=["gamma"]),
        row("dev:4", "azure"),
    ]
    config = fixed_config()
    config.update(train_limit=len(train_rows), dev_limit=len(development), tfidf=TFIDF)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # Dedicated protocol tests cover the real fixed sizes. This harness exercises
    # persistence and reporting with three invented training documents.
    monkeypatch.setattr(baseline, "validate_config", lambda value: None)
    monkeypatch.setattr(
        field_baseline_data,
        "prepare_field_baseline_data",
        lambda root, config: {
            "train": train_rows,
            "dev": development,
            "mapping": mapping,
            "provenance": {"source": "invented fixture"},
        },
    )
    monkeypatch.setattr(field_baseline_review, "build_review", lambda *args: {"synthetic": True})
    for name in (
        "field_baseline",
        "field_baseline_data",
        "field_baseline_review",
        "book_fields",
        "candidate_io",
        "io",
    ):
        path = tmp_path / f"src/research/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic provenance for {name}\n")
    if prepare_only:
        monkeypatch.setattr(
            baseline, "fit_rankers", lambda *args: pytest.fail("Preparation attempted fitting")
        )
    output = tmp_path / "outputs/synthetic"
    report = baseline.execute(config_path, output, root=tmp_path, prepare_only=prepare_only)
    assert json.loads((output / "report.json").read_text()) == report
    assert report["statistical_fitting_performed"] is (not prepare_only)
    assert report["human_gold"] is report["model_promoted"] is False
    assert report["neural_training_performed"] is report["rl_training_performed"] is False
    assert report["formal_dataset_admission"] is False
    assert not (output / "failure.json").exists()
    assert len(json.loads((output / "examples.json").read_text())["dev"]) == 4
    if prepare_only:
        assert report["status"] == "prepared_not_fitted"
        assert "metrics" not in report
        assert not (output / "fitted").exists()
        assert not (output / "rankings.json").exists()
    else:
        assert report["status"] == "completed_weak_label_diagnostic"
        assert report["training_support"]["genre"]["zero_positive_support_labels"] == ["gamma"]
        state = json.loads((output / "fitted/vectorizer.json").read_text())
        assert "unknownonly" not in state["vocabulary"]
        assert len(state["idf"]) == len(state["vocabulary"])
        assert state["frequency"]["genre"] == [2, 1, 0]
        centroids = sparse.load_npz(output / "fitted/centroids_genre.npz").toarray()
        assert centroids.shape == (3, len(state["vocabulary"]))
        assert not centroids[2].any()
        assert len(json.loads((output / "rankings.json").read_text())["records"]) == 4
        assert report["metrics"]["positive_centroid"]["genre"]["evaluated_groups"] == 3


def test_source_weighting_all_ones_preserves_loss_and_gradient():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[True, False, True], [False, True, False]])
    bounds = {"facet": [0, 3]}
    original = torch.tensor([[1.0, -0.5, 0.2], [0.1, 0.3, -1.0]], requires_grad=True)
    weighted = original.detach().clone().requires_grad_(True)
    loss = baseline.source_recovery_loss(original, labels, bounds)
    other = baseline.source_recovery_loss(weighted, labels, bounds, label_weights=torch.ones(3))
    loss.backward()
    other.backward()
    assert torch.equal(loss, other)
    assert torch.equal(original.grad, weighted.grad)
    a = baseline.source_recovery_fit_totals(original.detach(), labels, bounds)
    b = baseline.source_recovery_fit_totals(
        weighted.detach(), labels, bounds, label_weights=torch.ones(3)
    )
    assert a["cross_entropy_sum"] == b["cross_entropy_sum"]
    assert a["target_entropy_floor_sum"] == pytest.approx(b["target_entropy_floor_sum"], abs=1e-7)


def test_source_weighting_closed_form_gradient_and_correct_entropy_floor():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[0.4, -0.2, 0.1, 0.0, 1.0], [0.0, 0.5, -0.3, 0.2, -0.1], [0.0] * 5], requires_grad=True
    )
    positives = torch.tensor(
        [[True, True, False, False, False], [False, False, True, True, True], [False] * 5]
    )
    weights = torch.tensor([0.5, 2.0, 4.0, 1.0, 3.0])
    bounds = {"a": [0, 3], "b": [3, 5]}
    loss = baseline.source_recovery_loss(logits, positives, bounds, label_weights=weights)
    loss.backward()
    expected = torch.zeros_like(logits)
    for row, facet_count in [(0, 1), (1, 2)]:
        for start, stop in bounds.values():
            target = positives[row, start:stop]
            if not target.any():
                continue
            coefficients = target * weights[start:stop] / target.sum()
            probabilities = logits.detach()[row, start:stop].softmax(0)
            expected[row, start:stop] = (coefficients.sum() * probabilities - coefficients) / (
                2 * facet_count
            )
    assert torch.allclose(logits.grad, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(logits.grad[2], torch.zeros(5))
    assert torch.equal(logits.grad[0, 3:], torch.zeros(2))
    optimum = torch.tensor([[0.0, log(3.0), -1000.0]])
    known = torch.tensor([[True, True, False]])
    fit = baseline.source_recovery_fit_totals(
        optimum, known, {"facet": [0, 3]}, label_weights=torch.tensor([1.0, 3.0, 2.0])
    )
    assert fit["cross_entropy_sum"] == pytest.approx(fit["target_entropy_floor_sum"], abs=2e-7)
    assert fit["target_entropy_floor_sum"] != pytest.approx(log(2.0))
    singleton = baseline.source_recovery_fit_totals(
        torch.zeros((1, 2)),
        torch.tensor([[True, False]]),
        {"facet": [0, 2]},
        label_weights=torch.tensor([0.25, 4.0]),
    )
    assert singleton["target_entropy_floor_sum"] == 0.0
    assert singleton["cross_entropy_sum"] == pytest.approx(0.25 * log(2.0))


def test_source_weighting_fixed_protocol_and_invalid_weight_guards():
    torch = pytest.importorskip("torch")
    config = json.loads(
        (baseline.ROOT / "configs/research/book_source_loss_weighting.json").read_text()
    )
    baseline.validate_source_recovery_config(config)
    for mutate in [
        lambda c: c.update(arms=["head_only", "weighted"]),
        lambda c: c["weighting"].update(bisection_iterations=95),
        lambda c: c["weighting"].update(upper=5.0),
    ]:
        invalid = deepcopy(config)
        mutate(invalid)
        with pytest.raises(ValueError):
            baseline.validate_source_recovery_config(invalid)
    logits = torch.zeros((1, 2))
    known = torch.tensor([[True, False]])
    for weights in [
        torch.tensor([0.1, 1.0]),
        torch.tensor([1.0, 5.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.ones(2, dtype=torch.float64),
    ]:
        with pytest.raises(ValueError):
            baseline.source_recovery_loss(logits, known, {"facet": [0, 2]}, label_weights=weights)
