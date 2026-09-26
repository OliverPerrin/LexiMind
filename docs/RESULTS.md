# Historical research evidence

The small original reports and resolved seed-17 configuration remain under
[`research/results/historical/`](../research/results/historical/), with their
original byte hashes in the [manifest](../research/results/manifest.json).
They are historical observations, not reproduced results or evidence for the
current website or proposed model study.

The old test report names a checkpoint path but does not pin its hash, dataset
revision or evaluation code revision. Its relationship to the completed seed-17
training history is unresolved. The retained checkpoint observations differ;
calibration protocols and literary source matching also limit comparisons.
Do not relabel that report as a verified seed-17 result or use it as book-mood gold.

Old paper drafts, campaign launchers, plot/table generators and duplicate result
narratives have been removed from the active codebase. Git history preserves them;
local paper/figure edits were additionally backed up before cleanup. The underlying
historical report JSON/YAML files were not rewritten.

Their byte integrity is covered by `tests/test_research/test_evidence_io.py`.
Current source reconstruction and study decisions live in
[the research index](research/README.md). Training and research experiments remain
paused; software checks do not establish model quality.
