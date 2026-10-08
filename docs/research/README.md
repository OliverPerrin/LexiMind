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
