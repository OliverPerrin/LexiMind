# Research: current work

**Training and research experiments remain paused.** Source research, preparation
and software checks continue. The Gradio demo and custom FLAN/T5 implementation stay.

Read [dataset decisions](dataset_decisions.md) for the current direction and next
three actions. The priority is book-aligned genre/topic supervision, with narrative
emotion as an auxiliary and reader mood kept separate. AG News, GoEmotions and
arXiv are optional controls, not the primary field-training plan.

The [study decisions](study_decisions.md) and
[machine-readable design](../../configs/research/study_design.json) retain joint
adaptation, specialists, task arithmetic and TIES under comparable training budgets.
Book recommendation relevance is evaluated separately from model-task accuracy.

## Working commands

Rebuild prepared book candidates offline from the preserved source cache:

```sh
python scripts/prepare_bgc_groups.py
python scripts/prepare_book_fields.py
python scripts/prepare_licensed_books.py
python scripts/review_book_groups.py
python scripts/prepare_field_review.py
```

Only the licensed-book command has an explicit `--fetch` option for missing pinned
public files. These commands use the standard library and do not load models.
Then check the compact preparation evidence:

```sh
python scripts/audit_research_preparation.py --target model_study
python scripts/audit_research_preparation.py --target book_study
python scripts/prepare_annotation_packet.py --mode validate --packet research/preparation/annotation_packet.json
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
- [Retained source archive](source_archive.md): old corpus audit, optional control
  reconstructions and their reproducible commands.
- [Model receipts](admission_contracts.md), [book receipts](book_admission_contract.md),
  [annotation tooling](annotation_preparation.md), [compute ledger](compute_accounting.md).

Keep one current decision page; source details belong in manifests and implementation
contracts in code/tests. Update dependent evidence deliberately, then run
`python scripts/build_preparation_manifest.py --replace`. Do not refresh hashes to
hide an unexplained change. New evidence should replace stale decisions, not add
another parallel plan document.
