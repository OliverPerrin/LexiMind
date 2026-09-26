# Research preparation

`preparation.json` is a status/gate document, **not a Hydra training configuration**.
Research is paused by the user's September 22 instruction. No model, seed, budget,
protocol or literature novelty verdict has been approved for a new experiment.

Run `python3 scripts/audit_research_artifacts.py --require-ready` to inspect blockers.
It exits 2 until preparation is ready; this does not technically disable existing
training scripts. File integrity and unit-test success are not experiment readiness.

Historical resolved seed-17 configuration is separately preserved at
`research/results/historical/seed17_logged_config.yaml`. It records emotion weight
1.0 and must not be silently replaced by the current `training/full.yaml` (1.2).
