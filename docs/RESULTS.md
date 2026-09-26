# Historical results and provenance

**Status, 2026-09-22:** historical report files have been preserved and checked by
SHA-256. They have not been reproduced. Training, inference, model evaluation and
research experiments are paused at the user's request; no paid compute is authorized.

The previous version called these results reproducible from the current checkpoint.
That was too strong: the reports do not pin checkpoint hashes, dataset revisions,
resolved evaluation settings or code revisions. This record distinguishes report
contents from evidence linking a report to a particular training run.

**September 26 code-only update:** [model-stack repairs](model_stack_audit_2026-09-26.md) correct future training, calibration and decoding behavior. In particular, future BERT runs now split model-selection and calibration data. Statements below about the “current” code describe the September 22 audit snapshot; the archived historical results have not been rerun or retroactively repaired.

## Preserved evidence

[Manifest](../research/results/manifest.json) records original paths, archived file
hashes, current checkpoint observations and missing run linkage. Small report/config
files live in `research/results/historical/`; model weights and datasets stay outside
Git. [Generated tables](../research/results/historical_tables.md) contain every
reported aggregate below with its exact source JSON path. Table generation checks
archive hashes and rejects absent, non-numeric or out-of-range metrics.

```sh
python3 scripts/audit_research_artifacts.py
python3 scripts/build_tables.py --output research/results/historical_tables.md
python3 scripts/build_tables.py --format latex --output research/results/historical_tables.tex
```

The LaTeX fragment requires `longtable`. These commands format and validate existing
files; none loads a model. Optional `--check-local` on the audit checks current local
data/checkpoint hashes against the September 22 inventory. `--require-ready` exits 2
while preparation remains incomplete. It does not disable legacy training commands.

## Root LexiMind test report: training-run identity unresolved

Source: [leximind_test.json](../research/results/historical/leximind_test.json), copied
byte-for-byte from `outputs/evaluation_report_test.json`. Its metadata identifies the
test split and the path `checkpoints/best.pt`, but no seed or immutable checkpoint ID.
Do **not** relabel these metrics as the completed seed-17 campaign.

- Summarization: **2,727** examples; ROUGE-1 **0.306**, ROUGE-2 **0.091**, ROUGE-L
  **0.184**, BLEU-4 **0.024**, BERTScore F1 **0.830**. Stored ROUGE-L interval:
  **[0.181, 0.186]**; this is an archived interval, not a new calculation.
- Academic subset: **2,506** examples, ROUGE-L **0.188**. Literary subset: **221**,
  ROUGE-L **0.131**. Literary matching/split integrity is unresolved; these numbers
  do not validate book summarization or transfer to book discovery.
- Emotion: **5,427** examples, **28** labels. Fixed-setting sample/macro/micro F1:
  **0.351 / 0.143 / 0.445**. Frozen-tuned fields: **0.496 / 0.290 / 0.483**.
  Twenty of 28 labels have zero fixed-setting F1 in the report. The fixed threshold
  is not recorded there; **0.5** is the current inference-code default, not a
  report-pinned setting. Frozen per-label thresholds are present in the JSON.
- Topic: **189** examples, accuracy **0.857**, macro F1 **0.861**; stored accuracy
  interval **[0.804, 0.905]**. Small sample size limits precision. There is no fixed
  universal delta below which a difference is uninterpretable: paired predictions,
  the estimand and an appropriate uncertainty analysis are needed.

Current LexiMind evaluation code tunes on the calibration half returned by
`split_emotion_val` (seed 20260416), disjoint from model selection. This is the
intended protocol; the historical report does not establish all exact input bytes
or the evaluation code revision used to produce it.

## Completed logged seed-17 training run: a separate artifact

The [logged resolved config](../research/results/historical/seed17_logged_config.yaml)
is extracted from the start of `outputs/multiseed_emnlp/seed_17/train.log`. It records
FLAN-T5-base initialization, seed 17, eight epochs, four frozen encoder layers,
label smoothing 0.1, temperature sampling alpha 0.5, and task weights
**summarization 1.0 / emotion 1.0 / topic 0.3**. PCGrad is false. The current
`configs/training/full.yaml` instead has emotion weight **1.2** and a changed
conflict-diagnostic frequency; it is not the historical run's resolved config.

The two currently available checkpoints have **different SHA-256 hashes**:

| Current local path | SHA-256 |
| --- | --- |
| `checkpoints/best.pt` | `3f9ac2cdd30f6a3e205b8ff28cb43acbc21573fd0f1bcf0d8d8b87e539e9cc2c` |
| `outputs/multiseed_emnlp/seed_17/checkpoints/best.pt` | `f833727b1117cbf2a109083c34dafdc061bed2482551a23d973a0b7e86ef07df` |

They are not byte-identical; tensor equivalence was not tested. Neither hash was
recorded in the historical test report. File names and equal sizes do not establish
equivalence. The log's final progress line records about **10 h 47 min** of training
loop elapsed time; the exact GPU model and precision are not pinned in the archived
resolved configuration. Historical hardware descriptions are not a measured compute
ledger.

Two distinct histories are also preserved. `root_training_history.json` records
validation loss **4.298 → 3.925** and emotion F1 **0.197 → 0.459**. The seed-17
history instead records **4.353 → 3.974** and **0.304 → 0.485**. These should never
be combined into one run narrative. No test report has been linked immutably to the
logged seed-17 run.

## BERT reports: descriptive references, not a controlled MTL effect

Source: [bert_combined_test.json](../research/results/historical/bert_combined_test.json),
byte-for-byte copy of `outputs/bert_baseline/combined_results.json`. Each mode records
`split: test`. These reports do not pin training configuration, checkpoints, or data
hashes. The current baseline script is useful context, not proof of historical settings.

| Setting | BERT single-emotion macro F1 | BERT multitask macro F1 |
| --- | --- | --- |
| Explicit fixed threshold **0.3** | 0.468 | 0.493 |
| Report's frozen-tuned fields | 0.496 | 0.508 |

BERT single-topic accuracy is **0.831**, multitask accuracy **0.794**. These values
are observations, not significance claims. Comparing LexiMind's fixed-setting emotion
F1 with BERT's conflates threshold policy and architecture. Even the frozen-tuned
fields do not establish a matched comparison: current BERT eval-only code tunes on
**the entire validation split**, while LexiMind uses a separate calibration half.
All arms need the same calibration/model-selection partition in a future campaign.

## Current data are not the historical evaluation snapshot

The [local data inventory](../research/results/historical/local_data_snapshot_2026-09-22.json)
records present-day file hashes and counts, with no claim of historical run identity.
The current summarization test contains **3,440** rows: **2,506 academic + 934 literary**.
The old report contains **2,506 academic + 221 literary**. Matching academic counts
alone does not establish matching examples. The August plan's **3,440 academic**
headline count was therefore incorrect for both this snapshot and the old report.

Training counts observed in the seed-17 log are 61,877 summarization, 43,410 emotion
and 3,402 topic rows. Counts alone do not pin a data recipe. Title-only literary
joins, author/edition identity and work-level splitting need repair and new frozen
manifests before any future research. Existing report bytes are preserved even if
new catalogue ingestion rules later change.

## Not measured / not established

- Same-backbone single-task versus joint-training effects and multi-seed variance.
- Any benefit of MTL features for book search or recommendation relevance.
- Validity of GoEmotions predictions as book atmosphere/mood labels.
- A T5Gemma integration, feasibility run or selected replacement backbone.
- Adapter-merging, distillation, RL, or a new PCGrad campaign.
- Reproduction of historical test metrics from immutable complete run bundles.

See [draft evaluation protocol](eval_protocol.md), [initial related-work gate](related_work.md),
[revised plan](research_plan_2026.md), and [older paper audit](archive/README.md).
