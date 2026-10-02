# Compute accounting reference

The [ledger template](../../research/preparation/compute_ledger_template.json) is
unobserved: runs, events and recipes are empty. The chosen budget boundary and
unresolved numeric allowance are in [study_decisions.md](study_decisions.md).

```sh
python3 scripts/research.py ledger
python3 scripts/research.py ledger --require-observed
python3 scripts/research.py ledger path/to/observed-ledger.json --report path/to/report.json --require-observed
python3 -m pytest tests/test_research/test_compute_ledger.py -q
```

Normal validation accepts a structurally valid empty template; `--require-observed`
rejects it with exit 2. Malformed ledgers also exit 2. A report pins the source
ledger hash and cannot overwrite that ledger. Exact schemas and examples are in
[ledger.py](../../src/research/ledger.py) and [synthetic fixtures](../../tests/test_research/test_compute_ledger.py).

## Records and units

| Record | Required identity |
| --- | --- |
| Run | Unique run ID, hardware ID, environment hash, immutable backbone revision, code commit and timing-boundary ID |
| Event | Unique event ID, known run, phase, completed/failed/interrupted outcome, metric object |
| Recipe | Unique recipe ID and distinct existing leaf run references; no recursive recipe references |

Phases separate training, validation, calibration, merge search, final merge,
final evaluation, teacher work and preprocessing. Retain failed/interrupted spend.
Count source/target and padded tokens separately; also retain examples, optimizer
steps, teacher input/output tokens, wall/device-window seconds, estimated FLOPs
and external USD costs. Counts must be exact nonnegative integers; other values
must be finite and nonnegative. Padding cannot be below the corresponding count.

Missing/null means unknown, never zero. Each aggregate supplies a total only if
all included events report the quantity; otherwise it supplies null, a known
subtotal and explicit missing-event IDs. Empty phases remain unobserved. The
ledger cannot establish that every relevant event was recorded.

## Reuse and timing

A merged recipe's equivalent cost includes its experts and its own search/merge
work. Reused experts appear in each recipe-equivalent comparison, but project-unique
cost counts every event once. Do not sum recipe totals as project expenditure.
Unreferenced runs still count in the project total and produce a warning.

Timing boundaries must declare initialization/compilation, transfers, accumulation,
optimizer steps, validation, saving and overlap handling. Events represent disjoint
accounting intervals; summed run time is not automatically project makespan.
Mixed hardware/boundaries are flagged, not converted into equivalent FLOPs.
Equal tokens, ranks or trial counts do not establish equal compute.

The ledger records observations supplied by a future measured process; it does not
verify their truth or grant execution authority. Reports retain measurements as
not independently verified. Keep preprocessing/selection/failure costs visible
alongside the primary student-training allowance rather than calling them free.
