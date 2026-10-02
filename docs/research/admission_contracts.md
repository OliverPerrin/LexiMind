# Model-study evidence contract

`validate_model_admission(root, plan)` in [admission.py](../../src/research/admission.py)
returns structural/provenance blockers. Use it with the study-design validator;
book relevance is a separate admission path. Current choices are in
[study_decisions.md](study_decisions.md), not this schema reference.

```sh
python scripts/research.py status --target model_study --require-ready
python -m pytest tests/test_research/test_model_admission.py -q
```

The preflight returns 1 for invalid evidence and 2 for a blocked requested stage.
Complete temporary schemas and adversarial cases live in
[test_model_admission.py](../../tests/test_research/test_model_admission.py); do not
copy its synthetic values into real observations.

## Evidence chain

| Plan reference | Kind and required binding |
| --- | --- |
| `selected_dataset_manifest` | `dataset_admission`: ordered tasks, label/tokenizer contracts, four split files, grouping/overlap policy and review receipt |
| `adapter_module_manifest` | `adapter_module_manifest`: exact dataset/label hashes, complete parameter partition, initialization and structure receipt |
| `budget.feasibility_evidence` | `feasibility_timing_evidence`: exact dataset/module/label/budget hashes, per-task measurements and raw/review receipts |

References contain exactly `path`, `bytes`, `sha256`: positive byte length,
lowercase SHA-256 and a repository-relative file. Escapes, changed bytes, duplicate
JSON keys and non-finite values fail; structured evidence is bounded at 64 MiB.
Each record has schema version 1, exact kind, reviewed status and a binding to
`study_id`, backbone repository/revision, and the complete selected runtime.
Runtime pins include torch/Transformers/PEFT versions, environment hash and
implementation commit. An unrelated metric report cannot replace this evidence.

Major records have a payload and separate typed receipt. The receipt binds the
payload's SHA-256 (sorted compact UTF-8 JSON, `ensure_ascii=False`, no NaN), passed
outcome, issuer, exact required checks and nonempty supporting-file references.
The plan separately hashes the whole major-record file, avoiding circular hashes.

## Requirements preserved by the validator

- Data: ordered label contracts; distinct `train`, `model_selection`, `calibration`,
  `test` files; disjoint declared group IDs; source-use and grouping review.
  Preserved official text overlap is compatible with a **new-source-ID** claim;
  unseen-text/works claims require their corresponding stronger grouping evidence.
- Modules: standard LoRA only, explicit rank/alpha/dropout, no bias/RSLoRA/DoRA;
  sorted nonoverlapping shared/private allowlists and complete parameter inventory.
  All base-origin weights and aliases stay frozen. Shared parameters are encoder
  LoRA; private parameters match their declared task roles. Shapes and initial
  hashes must agree with tied groups and matched joint/specialist head manifests.
- Budget: positive B on named hardware/timing boundary; task allocations sum to 1;
  explicit finish-current-window overshoot tolerance; both time and trial limits
  for shared-development selection. Additional development costs stay separate.
- Feasibility: every task has measured optimizer-window timing, lengths, batch
  size and raw evidence under exactly that budget/module/data/runtime binding.
  Label, grouping, structure, save/reload and timing checks each have typed receipts.

Full field names, allowed policies and receipt check sets are enforced in
[the validator](../../src/research/admission.py) and its fixtures. Missing references
remain missing; no helper manufactures a review or repairs a manifest. Passing
schema/hash checks cannot authenticate an issuer, verify actual tensor contents,
prove truthful timing/source rights, or grant experiment authorization.
