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

Rebuild prepared book candidates offline from the preserved source cache:

```sh
python scripts/research.py bgc-groups
python scripts/research.py book-fields
python scripts/research.py licensed-books
python scripts/research.py book-groups-review
python scripts/research.py field-review
python scripts/research.py book-partitions
python scripts/research.py cr4
python scripts/research.py review build --output data/research_candidates/human-review/index.html
```

The source builders are offline by default; licensed-books and cr4 expose an
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
