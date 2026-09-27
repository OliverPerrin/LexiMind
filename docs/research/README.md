# Research preparation, 26 September 2026

Training, model execution/evaluation, research scoring and paid teacher calls are
paused at the user's request. This packet contains source review, data preparation
and software checks. It contains **no new model results or human judgments**.

Start with [study decisions](study_decisions.md) and the machine-readable
[working design](../../configs/research/study_design.json). The research question
is now a controlled replication/application, not a new merging algorithm claim.
The model-recipe study (M1) and book-relevance study (B1) have separate admission
requirements. Neither one validates the other's scientific claims.

## Evidence prepared

| Area | Deliverable and what it establishes | What it does not establish |
| --- | --- | --- |
| Model methods | [Nine-paper methods/appendix review](model_recipe_review.md), with pinned versions, source locators and search record | Exhaustive novelty search or independently reproduced results |
| Book discovery | [Recommendation, narrative emotion and evaluation review](book_discovery_review.md), including provider/source-use notes | Whole-book mood gold or representative catalogue coverage |
| Current data | [Streaming audit](data_readiness.md) of 156,796 records in 12 files; retained identities are missing and 73 normalized-input groups span splits | Automatic deletion policy or reconstruction of historical training data |
| Fresh emotion candidate | [Pinned GoEmotions reconstruction](goemotions_reconstruction.md): 54,263 source IDs, with 53,963 unique legacy matches and 300 ambiguous matches | Admitted training data, literary mood labels or a new split |
| Source assignments | [Grouped partition preparation](partitions.md) keeps identical text together within development splits and preserves official tests | Admitted data or unseen-text generalization |
| Fresh topic candidate | [AG News reconstruction](ag_news_reconstruction.md): 127,600 source rows with pinned file/row identities | Original article identities or resolved source-use rights |
| Academic extension | [Author-source reconstruction](arxiv_reconstruction.md) retains article IDs and official source partitions | Approved article-text reuse or a trained summarizer |
| Backbone interface | [Three pinned metadata/source reviews](backbone_interface_review.md) for FLAN-T5, T5Gemma and T5Gemma 2 | Model loading, adapter integration, GPU fit or a selected runtime |
| Cost accounting | [Ledger schema](compute_accounting.md) that preserves failed spend, unknown values and reused expert costs | Measured compute or an agreed numeric budget |
| Annotation | [Empty source-linked packet](annotation_preparation.md) for 102 catalogue works and draft rubrics | Collected queries, ratings, adjudication or gold labels |
| Admission | [Model](admission_contracts.md) and [book](book_admission_contract.md) evidence contracts with software fixtures | Proof that submitted human reviews or measurements are true |

The 18,753 legacy literary summary pairs remain logically quarantined pending
source/work reconciliation. The overlap counts are diagnostic signals: repeated
text can reflect different source records or annotations. Preserve official IDs
and report a reviewed policy instead of deleting every identical text blindly.
The legacy processed files and small historical reports have not been rewritten.
The old all-in-one downloader, campaign launchers and paper/plot generators have
been removed; the custom transformer, core trainer/inference path and Gradio demo remain.

## Decisions still needed before training

For M1, admit fresh sources and label contracts, settle the final task suite, choose
a compatible pinned backbone/runtime, and enumerate shared/private/frozen parameters.
GoEmotions and AG News now have [deterministic candidate assignments](partitions.md);
their repeated-text/generalization and source-use policies still require review.
The optional arXiv extension needs the same source/identity/use review. Freeze task sampling, expert budget allocation, merge
coefficient selection, stopping/overshoot rules and the intended effect-size claim.
Hardware feasibility and numeric B remain deferred until experiments are resumed.

For B1, review the relevance rubric with raters, collect independent query families,
define eligible works and complete independent judgments/adjudication before scoring.
The first target is new query families over the **fixed catalogue**; candidate works
may recur across partitions, while query-family and seed-work groups may not.
Mood queries remain disabled. Rater judgments belong in separate collection receipts,
not in the immutable empty packet. No automatic method can manufacture human gold.

Several useful tasks remain possible without a GPU: dataset-use review, source reconciliation, precise preprocessing/split specifications,
query sampling and annotation instructions, and implementing software-only adapter
interface checks. Actual feasibility, data-driven power estimates and any scoring
must wait for the paused experimental work to resume.

## Reproducible preparation checks

These commands use Python and the lightweight development/test dependencies. They
load no model, calculate no research-system scores and make no network requests:

```sh
python scripts/audit_research_preparation.py --target model_study
python scripts/audit_research_preparation.py --target book_study
python scripts/prepare_annotation_packet.py --mode validate --packet research/preparation/annotation_packet.json
python scripts/validate_compute_ledger.py research/preparation/compute_ledger_template.json
python -m pytest tests/test_research -q
```

The preparation checker returns 0 for internally consistent preparation evidence,
1 for missing/stale/malformed evidence, and **2 with `--require-ready`** while the
requested stage remains blocked. The default successful check is not readiness.
The ledger's `--require-observed` similarly rejects the empty template. These are
inspection tools, not interlocks around the legacy training scripts.

[`manifest.json`](../../research/preparation/manifest.json) pins the reviewed files.
Checks are scoped to common evidence plus the requested study: changing the book
catalogue cannot invalidate M1. Nested bindings also connect audits to their source
inventory/scripts and the annotation packet to its catalogue/rubrics. The corpus
inventory is a snapshot, not an automatic rescan of current local data. Future
admitted dataset receipts must validate their own fresh files. Raw Reddit text is
kept under ignored `data/research_candidates/`, outside the committed packet.

After deliberately changing and reviewing evidence, refresh its dependent receipts
first, then run `python scripts/build_preparation_manifest.py --replace`. The
checker never refreshes hashes automatically. A new hash is a consistency record,
not scientific approval. The packet and source-linked reviews make
unfinished decisions explicit for the next research session.
