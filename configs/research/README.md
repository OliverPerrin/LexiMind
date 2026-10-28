# Research configuration

[`merge_study.json`](merge_study.json) is the current protocol: the budget-matched
joint, specialist and merged LoRA study, run through
`python scripts/research.py merge-study`. See the
[research index](../../docs/research/README.md) for its design and status.

The remaining files belong to earlier work. `study_design.json` and
`preparation.json` are the superseded admission-gated design, still read by
`python scripts/research.py status`. The `book_*` and `macbook_pilot.json` files fix
the September–October pilots recorded under `research/results/`. None of these is
a Hydra training config; executable defaults live under `configs/training/`.
