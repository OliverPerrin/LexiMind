# Evaluation protocol: working draft

Data/field choices: [dataset decisions](research/dataset_decisions.md).
Recipe choices: [study decisions](research/study_decisions.md).
No model experiment is authorized by this document.

## Model comparison

- Compare joint adaptation, total-budget specialists, task arithmetic and TIES
  with the same per-task input, supervision, initialization and selection access.
  Retain the custom transformer and verify the actual adapter/module inventory.
- Group editions, translations, passages, series and cross-source copies under
  reviewed work identities before partitioning. Report unresolved identities and
  text overlap; do not silently relabel a changed benchmark split as the original.
- Separate training, model selection, calibration and final test. Learn mappings,
  negative-sampling rules and label thresholds from permitted development data only.
  Book target metadata must not leak into the classifier input.
- Genre/topic targets are multi-label and may be incomplete. Preserve unknowns;
  reviewed negatives/complete-label examples are needed for the proposed masked
  supervision. Report the observed-label fraction. Narrative character emotion,
  passage tone and reader experience are different evaluation targets.
- On a complete reviewed field test set, report macro/micro F1 and per-label support,
  per-field precision/coverage, hierarchy violations and abstentions. With incomplete
  truth, restrict metrics to the judged scope and do not claim exhaustive recall.
- Keep primary final-budget checkpoints and matched training seeds. Any selected
  checkpoint, merge coefficient search, calibration or post-merge adjustment must
  use the predeclared development allowance and appear in the compute ledger.
- Match synchronized student-training wall time on one declared hardware/timing
  boundary. Report total expert creation, failures, preprocessing, validation and
  search separately. Tokens are supporting measurements, not compute equivalence.
- Report task regressions and source/domain slices before any aggregate. Freeze the
  effect-size claim and uncertainty analysis before viewing model comparisons.

## Book relevance

- B1 targets new query families over the fixed catalogue, not unseen candidate works.
  Split query families and seed-work groups; exclude the seed from its own results.
- Collect requests independently of target-book reviews and model suggestions.
  Use the same evidence availability for all compared systems. Compare text-only,
  metadata-rich lexical and later learned representations explicitly.
- Primary metric: query-level nDCG@10, gains `[0,1,3,7]`, log2 discount, over all
  eligible works with complete reviewed judgments. Preserve deterministic ties.
  Decide zero-relevance query handling before scoring.
- A frozen pooled/partially judged comparison is secondary and pool-conditional.
  Unjudged or abstained pairs are not nonrelevant. Report coverage, hard-constraint
  violations and diversity with their denominators.
- Keep individual ratings and adjudication separate. Use multiple independent
  raters; preserve disagreement, evidence scope and edition/translation information.
  See the [relevance rubric](recommendation_judgments.md).
- Mood evaluation uses the [reader-mood rubric](mood_annotation_guide.md), not a
  generic emotion benchmark. Description-only impressions cannot certify full-book
  experience. Model quality and recommendation usefulness remain separate outcomes.

## Execution record

A run needs exact source/split/label hashes, code and backbone/runtime revisions,
resolved configuration, seeds, parameter partitions, checkpoints, calibration and
selection records, predictions and resource measurements. Unknown quantities remain
unknown; missing/failed runs stay visible. Existing receipt validators check
consistency, not the truth of a human review or measurement.

The [research index](research/README.md) lists current work and commands. Old source
reconstructions remain available as optional controls; they are not the default
book-field task suite. Hardware feasibility and all model runs remain paused.
