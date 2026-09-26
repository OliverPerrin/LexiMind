# Research configuration

`study_design.json` describes the proposed comparison; `preparation.json` records
its current evidence and unresolved decisions. Neither is a Hydra training config.
Training and research experiments remain paused.

Use `python scripts/audit_research_preparation.py --target model_study --require-ready`
or `--target book_study` to inspect the relevant blockers. Exit 2 means that stage
is not ready. The checker never launches a job or grants permission.

See [the research index](../../docs/research/README.md) for reconstruction commands,
source boundaries and next decisions. Executable software defaults live separately
under `configs/training/default.yaml` and require explicit dataset directories.
