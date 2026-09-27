# Study decisions

Current data/field choices: [dataset_decisions.md](dataset_decisions.md).
Configuration: [study_design.json](../../configs/research/study_design.json).

## M1: model recipes

A controlled replication/application study of book-domain task retention. It is
not a claim of a new merging algorithm or the first heterogeneous-task study.
The first proposed tasks are partial multi-label book metadata and character-scoped
narrative emotion. Reader mood needs separate evidence; generation is optional.

| Arm | Adaptation / deployment | Accounting |
| --- | --- | --- |
| Joint | Shared encoder adapter and task-private modules | One total training allowance B |
| Specialists | Separate task encoder adapters and private modules | B total across tasks, not per task |
| Task arithmetic | Merge the exact specialist encoder deltas | Same expert creation plus merge/selection |
| TIES | Merge the same specialist encoder deltas | Same expert creation plus merge/selection |

Preserve the from-scratch transformer. A selected runtime still needs verified
adapter paths and tied-parameter behavior; upstream metadata alone is insufficient.
Use the same base, data views, label/loss contracts and per-seed head initialization
across arms. Freeze pretrained base/embeddings/norms during LoRA adaptation.
Merge effective encoder weight deltas into the same base without rank recompression;
retain corresponding specialist private modules unchanged. Any post-merge tuning
is a separate budgeted arm. Random untrained heads are not an informative baseline.

The proposed primary budget is synchronized training-window wall time on one device.
Record expert reuse, failed runs, validation, calibration and search separately;
equal training allowance does not imply equal total development cost. Use final
budget checkpoints primarily. Per-task results and precision/coverage precede an
aggregate. Backbone, B, selection allowances and effect-size criteria remain open.
Optional domain adaptation must produce one shared initialization with recorded cost.

## B1: book relevance

Test new query families over the fixed, identified catalogue. Query families and
seed-work groups are partitioned; candidate works may recur across query partitions.
This makes no unseen-work claim. Rank all eligible works, exclude the seed book,
and use independently collected graded relevance judgments. Metadata-rich lexical
retrieval and text-only retrieval are distinct baselines.

M1 scores do not prove recommendation gains. B1 judgments do not gate an otherwise
valid model-only study. Evaluation queries cannot be generated from target-book
reviews and then described as independent prospective reader requests.

## Preparation scope

Fix book-domain labels and work grouping first, then integrate the new output/loss
contract. Optional source controls and later mood/generation work are not requirements
for making progress on the genre/topic track. GPU feasibility, numeric budgets and
model runs remain deferred under the current pause.
