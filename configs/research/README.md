# Research configuration

`study_design.json` describes the proposed comparison; `preparation.json` records
its current evidence and unresolved decisions. Neither is a Hydra training config.
Formal study execution still needs admission evidence. Separately authorized,
bounded local pilots use `macbook_pilot.json`, `book_denoising.json` and
`book_supervision.json` through `scripts/train.py --pilot`; each records its own
limits and does not grant formal study readiness.

`book_field_baseline.json` fixes a separate weak source-label retrieval diagnostic.
Run it through `scripts/research.py field-baseline --output outputs/new-field-run`;
add `--prepare-only` to verify the cohort and build a blank training-review
worksheet without fitting. Actual execution fits training-only TF-IDF statistics
and positive-label prototypes; it does not train the neural model or admit gold.

Use `python scripts/research.py status --target model_study --require-ready`
or `--target book_study` to inspect the relevant blockers. Exit 2 means that stage
is not ready. The checker never launches a job or grants permission.

See [the research index](../../docs/research/README.md) for reconstruction commands,
source boundaries and next decisions. Executable software defaults live separately
under `configs/training/default.yaml` and require explicit dataset directories.
