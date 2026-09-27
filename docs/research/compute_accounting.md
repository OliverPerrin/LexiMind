# Compute-ledger preparation

**Status: unobserved preparation.** No profiling, training, evaluation, teacher
calls, device measurements, or paid workloads have been run for this ledger.
The [committed template](../../research/preparation/compute_ledger_template.json)
has `status: unobserved_template` and empty `runs`, `events`, and `recipes` arrays.
It contains no invented costs, timings, observations, or approved budget.

The primary matched-budget criterion remains **proposed and requires feasibility
and protocol decisions**. Equal token counts do not imply equal compute across
encoder-only classification, encoder-decoder generation, specialist training,
merging, teacher generation, or reinforcement learning. This ledger preserves
separate quantities; it does not select a budget or convert them into equivalents.

## Validation and reporting

```sh
python3 scripts/validate_compute_ledger.py
python3 scripts/validate_compute_ledger.py --require-observed
python3 scripts/validate_compute_ledger.py path/to/observed-ledger.json --report path/to/report.json --require-observed
python3 -m pytest tests/test_research/test_compute_ledger.py -q
```

Ordinary validation/reporting exits 0 for a structurally valid template or observed
ledger. `--require-observed` exits **2 for the template**, and 0 only for a valid
ledger explicitly marked `observed`. Malformed ledgers exit 2. A zero exit code
means schema validation succeeded; it does **not** independently verify any
measurement, establish experimental fairness, or authorize training. Reports
always retain `measurements_independently_verified: false` and
`training_authorized: false`.

The report pins the input ledger's SHA-256 and never overwrites the source ledger.
Generation requires only the Python standard library and local source files.
Synthetic tests exercise accounting rules; their numbers are not research
observations and are not copied into the committed template.

## Observed-ledger contract

An observed ledger uses `schema_version: 1` and contains these records:

| Record | Required fields and constraints |
| --- | --- |
| Run | Unique `run_id`, `hardware_id`, full `environment_sha256`, full lowercase 40-character HF commit `backbone_revision`, full Git `code_commit`, `timing_boundary_id` |
| Event | Unique `event_id`, existing `run_id`, supported `phase`, `outcome`, and a `metrics` object |
| Recipe | Unique `recipe_id` and a nonempty `run_ids` list containing unique leaf run references |

Every observed run must have at least one event. Recipes cannot reference other
recipes, missing runs, or the same run twice. The template cannot carry any of
these records while calling itself unobserved. Revision and environment fields
identify the intended provenance; the corresponding manifests and measurement
receipts still need external review. The current HF backbone protocol requires the
full commit SHA: branch paths, tags, shortened hashes and bare mutable aliases are
rejected. Other revision authorities need an explicit future schema/protocol change.

Allowed phases are `training`, `validation`, `calibration`, `merge_search`,
`merge`, `final_evaluation`, `teacher`, and `preprocessing`. Outcomes are
`completed`, `failed`, and `interrupted`. A failed or interrupted event is retained
and contributes every known quantity; a later successful attempt must not erase
its cost. Use a distinct event ID for each disjoint accounting interval.

Keep training, validation/model selection, calibration, merge search, final
merging, final evaluation, teacher production, and preprocessing in their own
phases. This makes both the primary training allocation and the additional tuning
or preparation cost visible. Do not hide tuning by reporting only the successful
final training run.

## Quantities and unknown values

| Metrics | Type and unit |
| --- | --- |
| `source_tokens`, `target_tokens` | Nonnegative exact integer counts under the reviewed exposure policy |
| `padded_source_tokens`, `padded_target_tokens` | Nonnegative exact integer counts; at least the corresponding actual count when both are known |
| `examples`, `optimizer_steps` | Nonnegative exact integer counts |
| `teacher_input_tokens`, `teacher_output_tokens` | Nonnegative exact integer teacher counters |
| `wall_seconds`, `device_window_seconds` | Finite nonnegative numeric seconds, with timing semantics pinned by `timing_boundary_id` |
| `estimated_flops` | Finite nonnegative numeric estimate; estimator assumptions must be retained externally |
| `external_cost_usd` | Finite nonnegative recorded USD cost; no inferred rate or currency conversion |

Booleans, fractional counts, negative quantities, NaN, infinities, unsupported
metric names, and contradictory padding counts are rejected. Missing fields and
explicit `null` both mean **unknown**, never zero. Record zero only when the
measurement/policy establishes zero; do not replace an unavailable counter with
zero merely to obtain a complete total.

Each aggregate metric reports:

- `value`: the sum only when every included event supplies that quantity; otherwise
  `null`. An aggregate containing no events also has `value: null`.
- `known_subtotal`: the sum of supplied values, or `null` if none are supplied.
- `known_event_count` and `unknown_event_ids`: explicit coverage of that subtotal.

A known subtotal is not a full cost when any event is unknown. Absence of an event
for a phase is not evidence that the phase cost zero; missing phases are listed
as unobserved. The project total covers recorded events only, so independently
checking that all relevant spend was recorded remains necessary.

## Reused experts and unique project cost

A recipe references all leaf runs needed to reproduce that recipe. For a merged
model, that normally includes its expert preparation/training runs, merge-search
runs, final merge, and relevant validation/calibration/evaluation runs under the
reviewed protocol. Merely referencing the cheap final merge would omit the costs
of producing its experts.

For a recipe using experts A and B, recipe-equivalent cost includes A + B + its
other required runs. If A and B are also reused by another recipe, their recorded
costs appear in that recipe's equivalent cost as well. **Project-unique cost counts
each event once**, regardless of how many recipes reuse it. Do not add the
recipe-equivalent totals together and label the result project expenditure.

Reports provide project-unique totals, project totals by phase, each recipe's
equivalent total, and each recipe's phase breakdown. Runs that no recipe references
still count toward project-unique expenditure and produce a warning. The validator
checks references and duplicate accounting entries; the protocol must determine
whether a recipe's dependency list is complete.

## Boundaries still requiring decisions

A `timing_boundary_id` must link to a definition of what was timed: initialization,
compilation, data/tokenization work, synchronization, accumulation, optimizer
steps, validation, saving, and parallel or overlapping device windows must be
accounted for explicitly. Events must represent disjoint intervals for each
quantity, or their overlap must be reconciled before summing. The schema does not
infer overlap from timings or turn a sum of run wall times into project makespan.

Mixed hardware and mixed timing boundaries produce warnings. Their raw totals
remain visible, but device-window seconds are not normalized across hardware and
are not automatically FLOPs. Environment hashes, hardware identifiers, and
backbone revisions need verifiable supporting records.

Before adopting a matched-budget comparison, the separate protocol must settle:

- The primary budget measure and the feasibility evidence needed to measure it.
- Whether and how specialist preparation, merging/search, teacher generation,
  calibration, preprocessing, failed trials, and final evaluation fit within that
  budget versus separately reported additional costs.
- Task/head exposure, padding, batching, token definitions, data partitions, and
  comparable treatment of classification and generation.
- Measurement receipts, device/timing boundaries, cost reconciliation, and approval
  to execute the eventual study.

This ledger prepares those decisions and exposes missing evidence. It resolves
none of them merely by adding up fields.
