# Research: current work

**Bounded local MacBook training is authorized.** The formal MTL and book studies
still need their data/protocol evidence; paid calls and remote compute remain
unapproved. The Gradio demo and custom FLAN/T5 implementation stay.

Read [dataset decisions](dataset_decisions.md) for the current direction and next
actions. The priority is book-aligned genre/topic supervision, with narrative
emotion as an auxiliary and reader mood kept separate. AG News, GoEmotions and
arXiv are optional controls, not the primary field-training plan.

The [study decisions](study_decisions.md) and
[machine-readable design](../../configs/research/study_design.json) retain joint
adaptation, specialists, task arithmetic and TIES under comparable training budgets.
Book recommendation relevance is evaluated separately from model-task accuracy.

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
