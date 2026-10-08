# Research: current work

**The local training runtime is ready for reviewed data.** Bounded MacBook work
and the user's RTX 4070/WSL profiling were authorized through 8 October. Bounded
synthetic execution and checkpoint portability are verified on both devices. Formal MTL and
book studies still need their data/protocol evidence; paid cloud compute remains
unapproved. The Gradio demo and custom FLAN/T5 implementation stay.

Read [dataset decisions](dataset_decisions.md) for the current direction and next
actions. The priority is book-aligned genre/topic supervision, with narrative
emotion as an auxiliary and reader mood kept separate. AG News, GoEmotions and
arXiv are optional controls, not the primary field-training plan.

The [study decisions](study_decisions.md) and
[machine-readable design](../../configs/research/study_design.json) retain joint
adaptation, specialists, task arithmetic and TIES under comparable training budgets.
Book recommendation relevance is evaluated separately from model-task accuracy.

## Interactive visual archive

[Download the standalone visual gallery](visuals.html) and open the HTML locally;
GitHub displays HTML source rather than rendering it. The single file works
offline and preserves the three interactive 8 October snapshots with stable
anchors, PR links, commit-pinned result/research-note links and source hashes.

Append future snapshots to the gallery's `archive-payload` entries, retaining
original complete chart documents, data/code hashes, date, PR and pinned evidence.
Preserve old entries; record corrections as new versions and embed licensed
runtime resources so the gallery remains portable. No website deployment is needed.

## Current runtime readiness

The opt-in `training=book_lora` recipe shares model construction, adapter attachment,
freezing and optimizer setup between `train.py` and the existing profiler. It uses
the cached, revision-pinned native FLAN-T5-base with `gated-gelu-tanh`, encoder Q/V
rank-four LoRA, a private topic head, and a frozen decoder. The 48-column synthetic
fixture has 184,368 trainable parameters that receive AdamW state; the real count
depends on the exported label vocabulary. MPS uses float32, disabled CPU operator
fallback and a 35% memory fraction. CUDA uses float32 weights with native BF16 autocast; unsupported
unscaled FP16 execution is rejected. Compilation remains disabled.

The [8 October engineering receipt](../../research/results/training_readiness_20261008.json)
records fabricated title/description inputs, 48 synthetic target columns and no
real book judgments. Each completed M5 condition ran three warmup and 17 timed
outer microbatches at length 256, with nominal effective batch eight:

| Microbatch / accumulation | Mean seconds per microbatch | Fabricated rows/sec | Maximum sampled MPS driver GiB |
| --- | ---: | ---: | ---: |
| 1 / 8 | 0.0617 | 16.21 | 2.09 |
| 2 / 4 | 0.1088 | 18.38 | 2.02 |
| 4 / 2 | 0.2017 | 19.83 | 2.02 |
| 8 / 1 | 0.4104 | 19.49 | 3.09 |

Batch four is the starting preset. Batch eight plateaued with greater observed
memory. These are diagnostic timings with profiling, metrics and per-step device
synchronization; they exclude initialization and do not estimate real book model
quality. Separate warmup/timed loops flush partial accumulation windows, so the
conditions are not gradient-equivalent comparisons. The initial batch-one trace
export failed after its loop completed; its failed receipt is preserved, the
duplicate export was fixed, and a fresh batch-one measurement completed.

The isolated WSL Ubuntu environment used Python 3.10.12, PyTorch 2.14.0+cu130,
CUDA 13.0, Windows driver 617.42, Transformers 5.17.0 and tokenizers 0.23.2. The
RTX 4070 reported capability 8.9 and native BF16 support. The original checkout
remained clean, with its original PyTorch 2.9.1+cu128 environment unchanged.

| CUDA microbatch / accumulation | Effective batch | Mean seconds per microbatch | Fabricated rows/sec | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| 4 / 2 | 8 | 0.0729 | 54.88 | 1.64 / 1.66 |
| 8 / 1 | 8 | 0.0804 | 99.48 | 2.04 / 2.11 |
| 16 / 1 | 16 | 0.1002 | 159.67 | 2.85 / 2.92 |
| 32 / 1 | 32 | 0.2067 | 154.81 | 4.44 / 4.57 |

The CUDA starting point is batch 16 with accumulation one. It gave higher observed
throughput and lower reserved memory than batch 32. Larger conditions change the
effective batch and are capacity diagnostics, not matched-learning comparisons.
A timing-only batch-16 run retained metrics and per-step synchronization, processing
272 timed examples at 172.51 rows/sec. Removing the heavy trace lowered mean step
time by 7.4%. All five CUDA conditions verified frozen-base hashes after updates;
initialization and that audit remain outside the reported timing/allocator boundary.
The CUDA allocator statistics and MPS driver samples measure different memory bases.
These observations select starting settings, not a universal optimum or book model quality.

Two fabricated rows then exercised one optimizer update and native merged-weight
checkpoint saving. Adapter continuation and ordinary inference reload were checked
with exact short/long token IDs and masks. Ordered labels, loss/input/mapping,
tokenizer length/padding, vocabulary, backend normalization and special-token IDs
are bound before restoration. The final v4 artifacts add backend/direction binding without
another optimizer update; earlier artifacts remain preserved.

The small Mac v4 adapter restored unchanged on the RTX with strict base/head,
label and tokenizer bindings. Two fabricated rows exercised one BF16 optimizer
update; frozen weights matched before and after. The CUDA-saved native merged
checkpoint reloaded through ordinary inference, with maximum logit difference
`4.10e-08` against the same adapter weights in CPU FP32. Materialization used FP32
with TF32 disabled. No full Mac model transfer or real book judgments were needed.

Real field training still waits on independently reviewed labels and source/split
admission: the current packet has **zero completed human reviews**. Supply explicit
positive and negative evidence for every enabled label and report both counts;
omissions remain unknown. Runtime feasibility does not fill these data gates,
calibrate thresholds or establish book model quality.

Use Python 3.11 with the existing pinned dependency files. Install the appropriate
PyTorch build for the target hardware; the observed Mac environment used PyTorch
2.14.0. The existing MLflow SQLite backend also needs SQL dependencies:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install torch -r requirements-test.txt SQLAlchemy alembic sqlparse
```

The verified WSL environment was created separately from the existing checkout,
using the [official CUDA wheel index](https://download.pytorch.org/whl/cu130):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements-test.txt SQLAlchemy alembic sqlparse tokenizers==0.23.2
```

After separate data/experiment approval, set an explicit reviewed split directory
and a fresh output directory. The pinned FLAN snapshot must already be cached:

```sh
LEXIMIND_SPLITS=/absolute/path/to/reviewed/book-field-splits
LEXIMIND_RUN=outputs/reviewed-field-run-001
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python scripts/train.py training=book_lora device=mps \
  "data.processed.topic=$LEXIMIND_SPLITS" \
  "checkpoint_out=$LEXIMIND_RUN/checkpoints/best.pt" \
  "labels_out=$LEXIMIND_RUN/labels.json" "history_out=$LEXIMIND_RUN/history.json" \
  "+training.trainer.tracking_uri=sqlite:///$LEXIMIND_RUN/mlruns.db"
```

For the verified user-owned RTX host, use the explicit measured overrides:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/train.py training=book_lora device=cuda \
  training.dataloader.batch_size=16 training.trainer.gradient_accumulation_steps=1 \
  "data.processed.topic=$LEXIMIND_SPLITS" \
  "checkpoint_out=$LEXIMIND_RUN/checkpoints/best.pt" \
  "labels_out=$LEXIMIND_RUN/labels.json" "history_out=$LEXIMIND_RUN/history.json" \
  "+training.trainer.tracking_uri=sqlite:///$LEXIMIND_RUN/mlruns.db"
```

Both commands require reviewed splits; candidate source files remain unadmitted.

Profile a fresh bounded run through the same construction and loss path:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PROFILE_STEPS=20 PROFILE_OUTPUT_DIR=outputs/reviewed-field-profile-001 \
  .venv/bin/python scripts/profile_training.py training=book_lora device=mps \
  "data.processed.topic=$LEXIMIND_SPLITS" training.scheduler.name=constant \
  "+training.trainer.tracking_uri=sqlite:///outputs/reviewed-field-profile-001.db"
```

For CUDA, select its measured starting batch explicitly. `PROFILE_TRACE=0` selects
timing-only measurement; the default `1` retains the heavy operator trace:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PROFILE_STEPS=20 PROFILE_TRACE=0 PROFILE_OUTPUT_DIR=outputs/reviewed-cuda-profile-001 \
  .venv/bin/python scripts/profile_training.py training=book_lora device=cuda \
  training.dataloader.batch_size=16 training.trainer.gradient_accumulation_steps=1 \
  "data.processed.topic=$LEXIMIND_SPLITS" training.scheduler.name=constant \
  "+training.trainer.tracking_uri=sqlite:///outputs/reviewed-cuda-profile-001.db"
```

Profiling reads training rows only; the summary records actual per-step batch sizes,
example counts, optimizer updates, timing boundaries and allocated/reserved memory.
The constant scheduler isolates this diagnostic from the separate warmup boundary.

Native `last.pt`/`best.pt` contain merged ordinary model weights; matching
`.adapter.pt` files contain small base-bound factors and the private head. Paired
`labels.json`, `model_config.yaml` and `tokenizer_config.json` are required contracts.
Files are individually atomic, not an atomic multi-file transaction. Inference
restores the paired architecture and encoding settings automatically:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/inference.py \
  --checkpoint "$LEXIMIND_RUN/checkpoints/last.pt" --device mps \
  --title "Book title" --threshold 0.5 "Book description"
```

Use `--device cuda` on the verified RTX host. The explicit threshold is an operating
choice, not a calibration result. Keep each title paired with its description.

Weights-only continuation uses the adapter artifact, exact reviewed labels and the
same base/seed/tokenization in a fresh output directory. It resets optimizer,
scheduler and RNG state. `max_epochs=2` below means continue through epoch two:

```sh
LEXIMIND_PREVIOUS=outputs/reviewed-field-run-001
LEXIMIND_RUN=outputs/reviewed-field-run-002
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python scripts/train.py training=book_lora device=mps \
  "data.processed.topic=$LEXIMIND_SPLITS" \
  "resume_from=$LEXIMIND_PREVIOUS/checkpoints/last.adapter.pt" \
  "resume_labels=$LEXIMIND_PREVIOUS/checkpoints/labels.json" training.trainer.max_epochs=2 \
  "checkpoint_out=$LEXIMIND_RUN/checkpoints/best.pt" \
  "labels_out=$LEXIMIND_RUN/labels.json" "history_out=$LEXIMIND_RUN/history.json" \
  "+training.trainer.tracking_uri=sqlite:///$LEXIMIND_RUN/mlruns.db"
```

For CUDA continuation, replace `device=mps` with `device=cuda` and add
`training.dataloader.batch_size=16 training.trainer.gradient_accumulation_steps=1`.
A merged full-model
checkpoint is an inference artifact, not an adapter-resume substitute.

## Working commands

The bounded local continuation pilot reuses the training entry point. It preserves
the global test set and writes to a fresh ignored directory:

```sh
python scripts/train.py --pilot configs/research/macbook_pilot.json --output outputs/macbook-pilot --prepare-only
# Use another fresh output directory and omit --prepare-only to execute locally.
```

The configuration limits steps, wall time and MPS memory. Its three-work sample is
a feasibility/overfit check; the diagnostic work remains a global training work.
The prefix reward adds a four-canonical-token minimum and is not original RPT scoring.

The [M5 observations](../../research/results/macbook_pilot_20261002.json) record
two 64-update LoRA continuation pilots: 12–16 seconds for supervised updates and
about 2.32 GB maximum observed MPS driver memory. The 32-token RL probe had no
reward signal; a separately declared eight-token probe made one RL update, with
no diagnostic improvement. These tiny runs establish local execution feasibility.

The [expanded Book Dash cohort](../../research/preparation/bookdash_manifest.json)
adds 35 pinned works: 23 train, five validation, five test, two quarantined. The
[paired infilling comparison](../../research/results/book_denoising_20261002.json)
uses 179 training/36 validation examples. Across two seeds, continued CE reduced
validation target loss more than RL; exact recovery stayed low and only eight of
128 RL groups produced updates. Test narratives remained excluded. Reproduce via
`train.py --pilot configs/research/book_denoising.json --output outputs/new-run`.

The [fixed supervised comparison](../../configs/research/book_supervision.json)
retains those exact examples and compares instruction/word-only output with native
T5 sentinel-marked span output, at 128 and 512 updates in two seeds. It measures
final training reward coverage without RL updates. This changes both input/output
format and relative content-loss weighting, with equal prompt/update budgets,
not equal compute. Development books were already inspected in the previous run;
the reserved test narratives remain excluded. Run through the same entry point:

```sh
python scripts/train.py --pilot configs/research/book_supervision.json --output outputs/new-supervision-run --prepare-only
```

Its [recorded M5 results](../../research/results/book_supervision_20261003.json)
cover four 512-update conditions in 336 seconds, with 2.28 GB maximum observed
Metal driver memory. Final work-macro exact recovery:

| Format | Development, seed 17 / 29 | Training, seed 17 / 29 | Mixed probe groups, seed 17 / 29 |
| --- | --- | --- | --- |
| Instruction + word | 2.5% / 0% | 60.4% / 60.4% | 17/46 / 18/46 |
| T5 sentinel span | 5% / 16.7% | 58.7% / 58.4% | 21/46 / 18/46 |

The span format's descriptive advantage needs broader evidence; the large
train–development gap remains. Both formats passed the declared training reward
coverage gate, but no RL update ran. All 1,884 outcomes were independently
recomputed, and eight adapters reloaded with 40 matching CPU/MPS greedy cases.
Both word-format 128-update adapters also reproduce the earlier warm-start hashes.
Prioritize reviewed book-field labels and broader evaluation. Any further RL
comparison needs a new fixed protocol; training reward coverage alone is not a
reason to scale it or promote a model.

The [RNN/PufferLib review](rnn_ppo_review.md) distinguishes supervised recurrent
passage/session aggregation from PPO. Current PufferLib 5.0 uses a CUDA trainer;
its CPU evaluation support does not provide an M5 training path. Keep the native
runtime while testing the supervised baseline and preparing proper book labels.

## Book-field source recovery

The [fixed field diagnostic](../../configs/research/book_field_baseline.json)
uses 4,096 training and 1,024 development BGC records. It selects singleton groups
by hash, excludes every cross-source overlay component, and requires original and
effective roles to agree. Selection precedes text resolution; the original test
ZIP member is never opened. Raw train/dev member bytes are streamed and checked,
while only selected record frames are decoded.

```sh
python scripts/research.py field-baseline --output outputs/new-field-run --prepare-only
# Omit --prepare-only in a different fresh output directory to fit and score.
```

Frequency, label-name matching and positive TF-IDF prototypes rank the 48 mapped
labels separately within genre/topic/form/audience. Vocabulary and IDF use
training text only. The fixed metrics are observed-positive recall at 1/3/5 and
first-positive reciprocal rank, with group and label support reported. Omissions
remain unknown: this does not measure false positives, calibrated field quality,
reader mood or recommendation relevance. The [provider's description](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)
explicitly notes missing specific categories. Small facets can achieve trivial
recall at large k; inspect the actual cutoff and rare-label support.

The [3 October results](../../research/results/book_field_baseline_20261003.json)
completed in 6.06 seconds on CPU, including preparation, fitting and scoring.
Observed-positive group-macro recall@3 on development:

| Method | Genre (353 groups) | Topic (379) | Form (805) | Audience (258) |
| --- | --- | --- | --- | --- |
| Training frequency | 52.4% | 26.3% | 93.3% | 100% |
| Label-name matching | 27.9% | 41.1% | 54.7% | 100% |
| Positive TF-IDF prototype | 85.3% | 89.0% | 97.5% | 100% |

Audience has only three labels, so its recall@3 is trivial for every method.
Prototype label-macro recall@3 was 72.6% for genres and 80.1% for topics.
The erotica label has one training positive and no development positives;
gothic/horror and games each have only two development positives. The fixed
sample was retained. Independent reconstruction verified all 20,000 training-only
IDFs, eight saved sparse arrays, 12,288 ranking lists and 147,456 scores.
This supplies a useful fixed lexical control for future field models, with false
positives and human label quality still unresolved.

Each preparation/run also creates a 32-record **training-only, blind review**
worksheet under `data/research_candidates/bgc/field-baseline-review/`, with its
path and hashes in `report.json`. All human and agent slots start blank. The
worksheet is hash-selected before scoring and binds the protocol, cohort,
assignments and partition overlay. Use its own `review_manifest.json` when
importing an exported draft:

```sh
python scripts/research.py review import --manifest <review-folder>/review_manifest.json --draft <exported-draft.json> --output data/research_candidates/new-human-candidate.json
```

Imports remain candidates. Source-conflicting decisions still require separate
adjudication and the existing importer rejects them; preserve the exported draft.
The older 32-row purposive packet includes effective test records and must not be
used as this diagnostic's training seed. No worksheet creates human evidence by
being generated or viewed.

### Bounded neural source-assignment recovery, 8 October

The [fixed protocol](../../configs/research/book_source_recovery.json) completed
once on the owned RTX4070: 4096 training / 1024 development groups, four facets,
48 labels, seeds 17 and 29, and 512 updates per arm in two passes. The
[recorded result](../../research/results/book_source_recovery_20261008.json)
retains every seed and endpoint 0/256/512; 512 was fixed as primary. The full run
took 212.03 seconds within the cooperative 900-second budget. Independent audit
recomputed all metrics and matched lexical scores, verified paired initialization
and batches, frozen bases and unchanged head-only adapter factors, and restored
all 12 checkpoints. BF16 replay matched exactly on the same 32 recorded rows per
checkpoint. Separate FP32 replay sensitivity reached 0.01214 in probability;
this was not a BF16 restore failure.

Primary development group-macro observed-positive recall at 3:

| Arm / control | Genre | Topic |
| --- | ---: | ---: |
| Frozen encoder, seed 17 | 53.96% | 48.26% |
| Frozen encoder, seed 29 | 53.96% | 46.77% |
| Q/V LoRA + head, seed 17 | 72.52% | 70.58% |
| Q/V LoRA + head, seed 29 | 77.97% | 78.74% |
| Q/V LoRA + head, descriptive two-seed mean | 75.24% | 74.66% |
| Matched positive-centroid TF-IDF | 83.27% | 88.90% |

LoRA improved recovery over each paired frozen-encoder head, but both seeds
remained below the strong input-matched lexical control for genre/topic recall
at 3. Full-precision values, all facets and other cutoffs remain in the result.
The prior full-text lexical baseline is a separate historical reference.

This objective learns publisher source assignments: uniform observed-positive
targets, a facet mean within each eligible row, then a row mean. Softmax pressures
unassigned labels and co-positives compete; omissions remain semantically unknown.
It is distinct from semantic partial-label BCE. These results establish neither
semantic correctness, precision/F1, significance nor recommendation relevance.
FLAN may have encountered public BGC records or blurbs during pretraining;
exposure is unknown. The shared base controls exposure within each neural pair,
but lexical versus neural recovery does not isolate pretraining.

Retain the strong lexical control. Investigate source data/objective and any
longer fixed neural budget only under a future protocol; this 512-update result
does not establish that longer training would win. Human field labels remain
empty, formal admission remains unresolved, and there is no automatic RL,
neural website promotion or paid compute. Research artifacts preserve all head
and adapter factors and are rejected by ordinary inference loading.

### Source learning curves, 8 October

The [fixed duration protocol](../../configs/research/book_source_learning_curves.json)
completed once: four fresh arms, 16 epochs / 4096 updates each, in 1191.70 seconds.
The [result](../../research/results/book_source_learning_curves_20261008.json)
retains all 20 development endpoints and four final training reports. The
predeclared 4096 endpoint remains primary; intermediate checkpoints are
exploratory diagnostics. Independent audit passed: all 20 development objectives
and four final training
aggregates were reconstructed, alongside 32 shared batch proofs / 24,576 tensor
hashes and the matched lexical scores. All 20 BF16 checkpoint restores matched
probabilities and log probabilities exactly on 32 fixed development rows each;
frozen bindings and ordinary inference-loader rejection were verified. An
initial CPU probe failed only on tuple/list serialization comparison; that
failure was preserved, the checker alone was normalized, and the corrected
CPU audit passed. No training rerun was performed.

Head only means a frozen encoder with a trained head. Primary observed-source
recall at 3, shown as **group macro / label macro**:

| Arm / seed | Genre | Topic |
| --- | ---: | ---: |
| Head only, 17 | 79.91% / 56.59% | 79.84% / 54.11% |
| Head only, 29 | 80.17% / 56.47% | 80.67% / 53.94% |
| LoRA + head, 17 | 80.87% / 69.47% | 89.64% / 80.43% |
| LoRA + head, 29 | 79.88% / 68.06% | 89.07% / 83.80% |
| LoRA descriptive two-seed mean | 80.38% / 68.76% | 89.36% / 82.11% |
| Matched positive-centroid TF-IDF | 83.27% / 70.31% | 88.90% / 79.67% |

Final LoRA means remained below lexical genre recovery and exceeded lexical
topic recovery on both primary aggregation measures. Seed-29 LoRA genre group
recall was slightly below its paired head-only arm; LoRA did not dominate every
outcome. All facet/cutoff aggregates, label supports and per-seed paired
differences remain available with hash-addressed per-label evidence.

LoRA development source cross-entropy worsened from 2048 to 4096 updates:
0.72474 to 0.80267 for seed 17 and 0.72686 to 0.82962 for seed 29. Final training
cross-entropy was only 0.31146 / 0.31316. This pattern is consistent with
source-objective overfitting; it does not justify selecting an earlier checkpoint
after inspecting development results. Eligible-row-weighted loss, target entropy
floor and excess loss are recorded at every endpoint, with full final training
rankings. Saved float32 log probabilities enable independent loss reconstruction
when probabilities underflow; shared tensor hashes reconstruct paired batches.

This repeatedly inspected development cohort makes the study exploratory and
descriptive, not confirmatory. Publisher-assignment competition is distinct from
semantic partial-label BCE: unknown omissions remain unknown even though softmax
pressures unassigned and co-positive labels. No semantic-quality, precision/F1,
significance or recommendation claim follows. Seven of eight original
endpoint-0/512 state-and-prediction comparisons were exact. Only seed-17 LoRA at 512 differed: maximum parameter difference
0.00107625 and probability difference 0.02084035 despite matching initialization,
cohort, inputs and schedules. The cause is unproven; CUDA nondeterminism is
possible, and the duration protocol omitted the earlier 256-update evaluation.
Exact replication across all earlier LoRA checkpoints is not claimed.
Retain the strong lexical control and investigate source/objective and label
support before proposing a new fixed experiment. Do not automatically add more
epochs or promote PPO/neural website inference. Human field labels remain empty.

### Broader source exposure, 8 October

The [fixed data-exposure protocol](../../configs/research/book_source_data_scaling.json)
completed once and [passed independent audit](../../research/results/book_source_data_scaling_20261008.json).
Four fresh arms trained on 16,384 original-training singleton groups for four
passes / 4,096 updates each, retaining seeds 17/29 and both head-only/LoRA arms.
The exact old 4,096-row training prefix and all 1,024 development rows/tokens
were preserved. Each arm had 65,536 scheduled presentations, of which 65,532
had observed positives: the one whole-empty row remained selected and contributed
no objective loss. The whole run took 1341.77 seconds within its 1800-second cap.

At the predeclared final 4,096-update endpoint, observed-source recall at 3 is
shown as **group macro / label macro**. Head only means frozen encoder, trained head.

| Arm / seed | Genre | Topic |
| --- | ---: | ---: |
| Head only, 17 | 81.98% / 59.34% | 80.80% / 54.89% |
| Head only, 29 | 80.87% / 58.29% | 81.51% / 56.49% |
| LoRA + head, 17 | 88.06% / 79.33% | 91.62% / 86.04% |
| LoRA + head, 29 | 86.90% / 75.43% | 93.21% / 89.00% |
| LoRA descriptive two-seed mean | 87.48% / 77.38% | 92.41% / 87.52% |
| Refitted positive-centroid TF-IDF | 84.97% / 77.72% | 91.09% / 88.54% |

Both LoRA seeds exceeded the refitted lexical **group-macro** recall at 3 for
both facets, while the LoRA **label-macro means remained slightly below** it.
The label results cross by seed: genre LoRA-17 is above lexical and LoRA-29
below; topic LoRA-17 is below and LoRA-29 above. No all-metric lexical-superiority
claim follows. Both head-only and LoRA historical differences, including negative
form differences, remain in the compact result and hash-addressed full evidence.

Against the earlier audited smaller-cohort LoRA finals, descriptive mean changes
were +7.10 / +8.62 percentage points for genre group/label recall at 3, and
+3.06 / +5.41 points for topic. Unique data exposure and repetition changed
jointly, and the smaller-cohort arms were historical rather than fresh
contemporaneous controls; this cannot isolate a causal data-size effect.

Expanded-cohort LoRA development cross-entropy decreased from 2048 to 4096
updates: 0.65906 to 0.61221 for seed 17 and 0.65317 to 0.61249 for seed 29.
Final training values were 0.52541 / 0.51826. Unlike the historical smaller-cohort
pattern, final development loss continued to improve and the train/dev gap was
smaller. This is consistent with less observed source-objective overfitting,
with the same historical-comparison limitation. All 16 development checkpoints,
four final training reports, target entropy floors/excess loss, all facet/cutoff
metrics and original/additional/expanded/development label supports are retained.

Independent CPU audit reconstructed every development/final-training objective,
refitted the lexical controls, recounted supports and verified eight shared batch
proofs / 16 paired arm-epoch references / 24,576 tensor hashes. All 16 BF16
checkpoint restores matched probabilities and log probabilities exactly on 32
fixed development rows each; frozen bindings and ordinary-loader rejection
passed. CPU/GPU audits both passed on their first runs, without a training rerun.
The 82 original raw files (1,134,267,269 bytes) were transferred and hash-verified.

Keep the lexical controls and inspect label/support/objective limits before a
new fixed experiment. Development has already been inspected and weak publisher
assignments remain distinct from semantic partial-label BCE: omissions stay
unknown despite softmax competition. Rare development supports still limit
interpretation. This supplies no semantic gold, precision/F1, significance,
recommendation or product-admission evidence. There is no automatic extra-epoch,
PPO or neural website promotion; human semantic field labels remain empty.

### Label support and source objective, 8 October

The [read-only analysis](../../research/results/book_source_objective_analysis_20261008.json)
reconstructed final recall at 3 from the audited saved rankings for all 48 labels,
four facets, both head-only and LoRA seeds, and refitted centroid TF-IDF. No model,
GPU, fitting or source acquisition was used. It retains exact neural-only and
lexical-only source-hit cancellations, original/additional/expanded/development
supports, fixed support bins and single/multiple-positive row strata. Separate
independent computations agree. The final independent checker passed 1,747
arithmetic/structural checks after two checker-only errors were corrected. The
receipt distinguishes retained successful checkers from reconstructed failed
versions; original analyses were unchanged.

Different denominators explain the arithmetic divergence: group macro averages
each book's recovered fraction, while label macro gives each supported label
equal weight. Signed per-label contributions reconstruct both gaps and their
difference exactly. Two-seed LoRA-minus-lexical gaps are +2.5142 / −0.3346
percentage points for genre group/label macro, and +1.3193 / −1.0249 for topic.
The negative label means are seed-dependent: genre is positive for seed 17 and
negative for 29; topic has the reverse signs.

Fixed **development-support** bins contribute the following to the full facet
gap, in percentage points. These are sums of per-label contributions, not
within-bin means; empty and unsupported bins remain visible.

| Facet | Dev positives per label | Labels | Group contribution | Label contribution |
| --- | --- | ---: | ---: | ---: |
| Genre | 0 | 1 | +0.0000 | +0.0000 |
| Genre | 1-4 | 1 | +0.0000 | +0.0000 |
| Genre | 5-19 | 3 | -0.2007 | -1.9558 |
| Genre | 20+ | 10 | +2.7148 | +1.6212 |
| Topic | 0 | 0 | +0.0000 | +0.0000 |
| Topic | 1-4 | 2 | -0.1319 | -1.0870 |
| Topic | 5-19 | 11 | +0.2639 | -0.5995 |
| Topic | 20+ | 10 | +1.1873 | +0.6616 |

Small development counts do **not** establish missing or rare training labels.
Spiritual fiction has 117 training / 8 dev positives, paranormal fiction 293 /
18, finance 50 / 5, and technology 114 / 4. Finance's saved LoRA hits are 3/5 in
both seeds versus lexical 5/5; technology is 3/4 versus 4/4. One technology hit
changes topic label macro by 1/(4×23) = 1.0870 percentage points, exceeding the
1.0249-point mean deficit. This is denominator sensitivity, not a confidence
interval or significance claim. The result separately retains training-support
bins, including mixed outcomes within bins. Erotica has 9 training and zero dev
positives: recall/delta remain null, contributions zero, and genre label macro
uses 14 supported labels rather than its 15-label vocabulary.

The uniform source objective assigns global target coefficient mass of 17.20%
/ 17.92% / 43.67% / 21.22% to genre/topic/form/audience. Actual schedule-weighted
mass is recorded separately: each seed has four 15-eligible-row batches and
4092 16-eligible-row batches. Scheduled versus globally uniform row-weight L1
is only 0.0001216233; it does not establish an explanation for the macro gap. Train/dev label target
coefficient L1 is 0.08415756. These coefficients are neither gradient mass nor
causal evidence. Structural group recall-at-3 ceilings are 98.8244% genre and
99.9340% topic; audience's three-label vocabulary makes recall at 3 mechanically
100%. Raw recall remains primary, with no ceiling normalization.

A future **training-only capped/tempered weighting hypothesis** is defensible
for controlled testing, not established as a remedy. Select one coherent variant
from training supports/target coefficients, freeze its normalization/cap before
execution, and compare it with fresh contemporaneous unweighted LoRA controls
under the same seeds, cohort, schedules and update budget. No formula is a
supported winner; do not choose weights from dev-label misses or seed flips.
Keep lexical controls, group/label outcomes, all supports and original unweighted
source-loss diagnostics. No new training is authorized by this analysis itself;
unknown omissions remain unknown, human semantic gold stays empty, and no PPO,
semantic-negative BCE or product promotion follows.

### Training-only source-loss weighting, 8 October

The [fixed trial](../../configs/research/book_source_loss_weighting.json)
completed once in 1727.84 seconds and [passed independent audit](../../research/results/book_source_loss_weighting_20261008.json).
Fresh unweighted/weighted LoRA controls used the same 16,384/1,024 rows, cached
tokens, schedules, initializations and empty optimizers, with seeds 17/29 and
4,096 updates each. The applied train-only weights were fixed before training;
all 16 development endpoints and four final training reports remain available.

At the predeclared final endpoint, **group macro / label macro** source recall
at 3 was:

| Arm / seed | Genre | Topic |
| --- | ---: | ---: |
| Unweighted, 17 | 88.30% / 79.69% | 91.62% / 86.39% |
| Weighted, 17 | 84.32% / 78.55% | 92.77% / 89.70% |
| Unweighted, 29 | 86.76% / 72.77% | 92.68% / 88.79% |
| Weighted, 29 | 84.79% / 78.84% | 92.81% / 89.15% |
| Unweighted descriptive two-seed mean | 87.53% / 76.23% | 92.15% / 87.59% |
| Weighted descriptive two-seed mean | 84.56% / 78.69% | 92.79% / 89.42% |
| Matched centroid TF-IDF | 84.97% / 77.72% | 91.09% / 88.54% |

Every weighted-minus-unweighted primary change is retained, in percentage points:

| Seed | Genre group | Genre label | Topic group | Topic label |
| --- | ---: | ---: | ---: | ---: |
| 17 | -3.9754 | -1.1397 | +1.1434 | +3.3087 |
| 29 | -1.9688 | +6.0662 | +0.1319 | +0.3544 |
| Descriptive mean | -2.9721 | +2.4633 | +0.6376 | +1.8315 |

Genre group recovery fell in both seeds, while genre label recovery had opposite
signs. Topic group and label recovery improved in both seeds. This is a mixed
four-primary-outcome tradeoff, not a general improvement or promotion result.
All four facets, cutoffs and per-label values—including unsupported nulls—remain
in the compact result and hash-addressed raw evidence.

Weighted-arm development CE improved **under the weighted objective** against
its paired controls (0.81363 to 0.76061 for seed 17; 0.86325 to 0.76893 for 29),
while original unweighted CE worsened (0.61203 to 0.67187; 0.61054 to 0.64831).
Both arms are evaluated under each common objective separately, with its correct
entropy floor/excess. These losses are not interchangeable. The weights also
change row/facet scale and co-positive target proportions; equal gradient norms,
clipping or update magnitudes were not claimed.

The initial Mac prepare failed the unchanged exact lexical gate: 510 exchanged
features were tied at total frequency nine. Its failure and diagnosis remain;
this did not uniquely identify an architecture or dependency cause. A separately
numbered WSL prepare using the original runtime passed exact lexical, row, token
and schedule checks before training. The first CPU audit then failed only on
integer-versus-JSON-string keys in a serialized weight vector; a separately frozen
checker normalized those comparisons and passed. No training/runtime/source
changes or training rerun followed. The first GPU audit restored all 16
checkpoints with exact probability/log-probability agreement on 32 fixed
development rows per checkpoint, frozen bindings
and production-loader rejection.

Retain unweighted and lexical controls. The joint primary tradeoff does not
support applying this rule across every facet; any follow-up requires its own
fixed hypothesis and fresh controls. Publisher assignments and unknown omissions
remain distinct from human semantic gold. No significance, generalization,
semantic-negative BCE, PPO or product promotion follows. The audited visual will
be appended to the [single portable gallery](visuals.html) with its actual PR and
commit-pinned evidence after publication.

Rebuild prepared book candidates offline from the preserved source cache:

```sh
python scripts/research.py bgc-groups
python scripts/research.py book-fields
python scripts/research.py licensed-books
python scripts/research.py bookdash
python scripts/research.py book-groups-review
python scripts/research.py field-review
python scripts/research.py book-partitions
python scripts/research.py cr4
python scripts/research.py review build --output data/research_candidates/human-review/index.html
```

The source builders are offline by default; licensed-books, bookdash and cr4 expose an
explicit `--fetch` for missing pinned public files. Open the generated HTML locally
to review labels, export a draft, then use `review import --help` to import it into
a new candidate packet. No human judgments are filled automatically.

Optional continuation preparation uses the existing tokenizer JSON and the
effective book partitions, without loading model weights:

```sh
python scripts/research.py licensed-books --rpt-tokenizer artifacts/hf_tokenizer/tokenizer.json
```

All research preparation shares this entry point. Most commands use the standard
library; this optional tokenizer step needs `tokenizers`. Check compact evidence:

```sh
python scripts/research.py status --target model_study
python scripts/research.py status --target book_study
python scripts/research.py annotation --mode validate --packet research/preparation/annotation_packet.json
python -m pytest tests/test_research -q
```

Successful inspection means consistent preparation evidence, not experiment
readiness. Optional control artifacts are checked only with `--check-archive`.
`--require-ready` returns 2 while that stage remains unresolved. These
commands do not load models or grant permission to run experiments.

## Reference only when needed

- [Source register](../../research/preparation/book_field_sources.json): dataset
  alternatives, methods, terms and CR4 metadata observations.
- [Protocol](../eval_protocol.md), [mood rubric](../mood_annotation_guide.md),
  [relevance rubric](../recommendation_judgments.md).
- [Model methods](model_recipe_review.md), [book-retrieval methods](book_discovery_review.md),
  [backbone compatibility](backbone_interface_review.md).
- [RL methods through October 2026](../../research/preparation/rl_methods.json):
  implemented objective contracts, recent evidence and deferred techniques.
- [Retained source archive](source_archive.md): old corpus audit, optional control
  reconstructions and their reproducible commands.
- [Model receipts](admission_contracts.md), [book receipts](book_admission_contract.md),
  [annotation tooling](annotation_preparation.md), [compute ledger](compute_accounting.md).

Keep one current decision page; source details belong in manifests and implementation
contracts in code/tests. Update dependent evidence deliberately, then run
`python scripts/research.py snapshot --replace`. Do not refresh hashes to
hide an unexplained change. New evidence should replace stale decisions, not add
another parallel plan document.
