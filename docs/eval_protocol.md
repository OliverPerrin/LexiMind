# Evaluation protocol — DRAFT

**Prepared 2026-09-22. Not frozen, preregistered, or an authorization to run.** The
user has paused training and all research experiments. No evaluations, inference,
teacher generation, backbone pilots, or paid jobs may be launched under this draft.
The items below specify preparation and decisions for a later explicit resumption.

**September 26 refinement:** [study_decisions.md](research/study_decisions.md) and
[`study_design.json`](../configs/research/study_design.json) specify the working
first comparison. It matches a proposed student-training allowance and reports
additional development cost; it does not silently claim equal total cost. Shared
encoder attention deltas and retained private modules are explicit. Benchmark and
book studies have separate readiness checks. Legacy files remain unadmitted, and
numeric/empirical feasibility choices remain open. This is still a draft protocol.

## Questions and scope

1. Applied: do independently validated topic/genre/mood features learned jointly
   improve book recommendation relevance over plain text retrieval?
2. Model: for the selected task suite and one fixed backbone, how do joint SFT,
   budget-matched specialists, and merged task adapters compare in quality and cost?

These are separate estimands. Improvements on GoEmotions or arXiv do not answer the
book recommendation question. A suite with one dataset per output type also cannot
identify an output-type effect independently of domain, dataset size or label quality.
No planned outcome or novelty claim is a result.

## Data identity and splits

Every future run bundle must include:

- Dataset provider, immutable revision, license/source attribution and collection date;
  preprocessing code commit, resolved parameters, label map/order and SHA-256 hashes
  of final split files. Archive source-to-canonical identity decisions and exclusions.
- Stable example IDs plus a parent work/document ID. Editions, translations and
  excerpts from the same work stay in one split; duplicate descriptions and
  near-duplicate source documents are grouped before splitting. Author-held-out
  analysis may be a separate domain-shift probe, not silently mixed into test.
- Explicit train, model-selection validation, calibration, and untouched test IDs.
  Assert empty source-record/declared independent-group intersections between
  partitions. Report text duplicates separately and apply a frozen, source-aware
  policy; preserving official comment splits is different from claiming unseen-text
  generalization. For
  GoEmotions, pin the actual example IDs assigned by `split_emotion_val`; apply that
  same partition to every model, including BERT. Do not use full validation for one
  arm and half validation for another.
- Domain counts and exclusions before and after all filters. The old literary
  corpus is quarantined from new research until title/author identity and grouping
  are repaired. Present-day file hashes are not historical dataset revisions.

Use arXiv/GoEmotions only if relevant to the agreed model question; the old topic
set's 189 test examples do not provide adequate precision by assumption. Dataset,
label taxonomy, annotator agreement and detectable effect need assessment before
selecting a final size. No dataset count in the August plan is automatically frozen.

## Book-relevance judgments

No gold relevance set has been collected. Proposed schema per record:

`query_id`, `query_family_id`, `query_text`, `candidate_work_id`, `relevance_0_to_3`,
`constraint_violations`, `annotator_id`, `judgment_source`, `catalogue_revision`.

Write query needs independently of model outputs. Include topic, genre, atmosphere,
combined requests and “more like this book.” Pool and shuffle candidates from each
baseline with background candidates; hide system names/scores during judging.
Define relevance grades and distinguish a book's subject matter from its emotional
atmosphere. Obtain independent judgments on an overlap subset, report agreement and
adjudication. Keep query families and anchor works disjoint between development and
test; plan author-held-out cases separately. Do not turn model-generated mood tags
or synthetic clicks into gold human judgments.

Proposed primary metric: query-level nDCG@10 with 0–3 grades. Secondary measures:
constraint adherence, catalogue coverage and result diversity. Recall@10 is only
interpretable within a fully judged candidate pool or with its incompleteness made
explicit. Freeze candidate eligibility, deduplication, top-k and treatment of ties
and unjudged items before comparing systems. A small curated catalogue limits claims
to that catalogue; it is not a population sample of all books.

## Proposed arms and fair interfaces

| Arm | Role | Training budget | Deployment |
| --- | --- | --- | --- |
| Text retrieval | Applied baseline; lexical now, pinned embedding model later | No LexiMind training | Fixed index and ranking recipe |
| Metadata-assisted retrieval | Topic/genre and validated mood controls | Annotated/publisher labels; provenance recorded | Same candidate pool |
| Joint SFT | First model arm | Total B across all tasks | One backbone plus agreed heads |
| Specialists | Same-backbone control | Sum across tasks equals B | All N models counted |
| Merged adapters | Same specialist checkpoints as above | Same B plus merge/selection cost | One agreed merged backbone and task heads |

A generous B-per-specialist control may be useful later but must be labeled N×B.
It is not part of the matched comparison. Backbone feasibility is a separate pilot
before the final model choice; no such pilot has run. Distillation and RL are deferred.

For merging, pin the shared base revision, adapter targets/rank/scaling and label
order. Merge effective weight deltas, not arbitrarily paired LoRA factors. Distinct
classification heads are task-specific parameters: explicitly retain/train them by
the same policy in each arm and include their cost. Do not average unrelated heads
or present a routed ensemble as one static merged model. Freeze the policy for heads
and generation before the comparison; unknown compatibility blocks a run.

## Compute ledger and budget

The primary matched-budget unit is **not decided**. Equal token totals alone cannot
make encoder-only classification, encoder-decoder generation and different backbones
equal in compute. Choose one primary unit (e.g. measured device time on the same
hardware or validated FLOP accounting), its boundaries, stopping rule and tolerance
after authorized feasibility measurement; retain the following full ledger per task
and arm regardless:

- Non-padding source and target tokens separately; padded token/sequence totals,
  input/target length distributions and truncation rates; modules executed.
- Optimizer steps, gradient accumulation, examples, trainable/frozen parameters,
  dtype, checkpointing/compile settings and peak allocated/reserved memory.
- Synchronized GPU execution time where measurable; wall time and hardware/software
  versions, warm-up and data-loading time separately. FLOPs must state the estimator
  and omitted operations rather than masquerading as direct measurements.
- All task-specialist training, failed attempts, validation, calibration, merge
  coefficient search and merge computation. Report final-run and total-development
  costs separately; give each recipe comparable tuning opportunities.
- If teacher/RL arms are later approved, separately count teacher input/output tokens,
  dollars, policy/reference/reward inference and every rollout including rejected
  samples, in addition to student gradient work.

Do not label local electricity/time free compute. Choose B only with the user's
available resources; the old $15 estimate is historical. Freeze how partial final
batches and failed runs count. Record observed budget use; exceeding the declared
rule disqualifies an arm from a matched comparison until explained or rerun.

## Metrics, selection and statistics

For emotion, propose macro F1 at per-class thresholds tuned only on the pinned
calibration set. Also report common fixed-threshold, micro and sample F1, per-label
support, calibration behavior and rare-label failures. Threshold vectors and comparison
operators are artifacts; no test-time threshold tuning.

For generation, propose ROUGE-L with secondary ROUGE-1/2 and BERTScore, accompanied
by human checks of faithfulness/usefulness where the application needs them. Pin
metric package versions, BERTScore model revision, normalization, truncation,
decoding settings and generation limits. Do not substitute overlap with a marketing
blurb for a validated full-book summary target.

For topic/genre, define multi-class versus multi-label targets first, then pin macro
F1/accuracy or the appropriate multi-label metrics. For all arms, agree validation-only
selection criteria and search spaces before training. Store example-level outputs and
losses; aggregate JSON alone is insufficient for a future paired analysis.

Propose at least three matched training seeds for headline comparisons, chosen before
runs. Report each seed as well as mean and spread; test examples are not independent
training runs. Use paired resampling of differences over the correct independent unit
(work/document or query family), with a prespecified seed and resample count. Decide
how training-seed variation is represented, and define a primary comparison plus
multiple-comparison handling before the final test is opened. No blanket “6-point
noise floor,” no significance inference from overlapping model CIs, and no guaranteed
power from raw example count alone.

## Required future result bundle

Each bundle should include a run ID, code/environment revision, all data and label
hashes, backbone/tokenizer revisions, resolved config, random seeds, hardware, budget
ledger, checkpoint hash, validation-selection decision, frozen thresholds, decoding
settings, per-example predictions and aggregate metrics with metric-code revision.
A report must refer to its bundle, not a mutable `best.pt` path. Preserve failures and
missing tasks; do not fill missing metrics with zeros.

Current archive tools verify historical file integrity and render its stored metrics.
They do not validate statistical correctness, enforce this draft in legacy trainers,
or turn an incomplete historical report into a reproducible experiment.

## Resume gate

[Preparation manifest](../configs/research/preparation.json) intentionally records all
unresolved gates. To resume: obtain the user's instruction to restart experiments,
confirm available compute, complete the full-paper literature matrix, repair/pin data,
collect and review relevance judgments, agree the backbone pilot, then freeze the
headline protocol after its remaining choices are resolved. Commit the actual protocol
and artifact hashes before running its arms. Until then the status remains **draft**.
