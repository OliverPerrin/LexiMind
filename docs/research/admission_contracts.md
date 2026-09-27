# Model-study admission contracts

**Preparation tooling only. No admission records or observations have been created.**
`validate_model_admission(root: Path, plan: dict) -> list[str]` in
[`src/research/admission.py`](../../src/research/admission.py) returns consistency
blockers. An empty list means the supplied files satisfy this contract; it does not
prove their human assertions, source rights, measurements, model quality or permission
to run anything. The caller must keep authorization and protocol-freeze gates separate.
The current study keeps all evidence references, runtime selection and B unset.

All file references below have exactly `path`, `bytes`, and `sha256`: a repository-relative
regular file, positive exact byte count and lowercase SHA-256. Absolute paths, parent
traversal, escaping symlinks, changed bytes and unknown hashes fail. Structured evidence
records must be JSON, at most 64 MiB; duplicate JSON keys and non-finite JSON constants
fail. This is not an evidence-authoring or repair utility, and it performs no network,
model loading, scoring, training or source-data modification.

## Plan fields

The helper complements the broader study-design validator; both should be used.
It requires the existing encoder-only standard-LoRA/effective-delta/private-retention
policies, `post_merge_training=false`, `pretrained_base_updates=false`, and
`embedding_and_layernorm_updates=false`. `head_initialization` must be
`identical task-specific initialization within each matched training seed`.

| Location under `model_study` | Contract |
| --- | --- |
| `backbone.selected_repo`, `selected_revision` | Explicit owner/repository and complete 40-character lowercase revision |
| `backbone.selected_runtime` | Exact fields `transformers_version`, `torch_version`, `peft_version`, `environment_sha256`, `implementation_commit`; versions are nonblank, hashes are full SHA-256/Git SHA-1 |
| `selected_dataset_manifest` | Hash reference to a `dataset_admission` record |
| `adapter_module_manifest` | Hash reference to an `adapter_module_manifest` record |
| `budget.feasibility_evidence` | Hash reference to a `feasibility_timing_evidence` record |
| `budget.B`, `hardware_id`, `timing_boundary_id` | Finite positive training seconds, named hardware and a named timing boundary |
| `budget.specialist_allocation` | Exactly the admitted task IDs, with finite fractions in `(0,1]` summing to 1 within 1e-9 |
| `budget.overshoot_rule` | Exactly `{name: "finish_current_optimizer_window", max_fraction: positive <= 1}`; the declared tolerance is reviewed evidence, not a recommended numerical value |
| `budget.selection_allowance` | Exactly `{unit: "synchronized_selection_wall_seconds", per_recipe_seconds: positive, max_trials_per_recipe: positive integer, data_access: "shared_development_partitions"}` |

Budget status must be `frozen`. The four safeguards `equal_selection_access`,
`count_all_expert_training`, `report_total_development_cost`, and
`tokens_are_auxiliary_not_compute_equivalence` must be true. The global selection
allowance applies to each recipe; both time and trial caps are explicit. It does not
pretend total development costs automatically equal the primary training allowance.

## Common evidence envelope and binding

Every structured record has `schema_version: 1`, an exact `kind`, `status: "reviewed"`,
and this exact `binding` object, matching the selected study:

```text
binding = {
  study_id,
  backbone_repo,
  backbone_revision,
  runtime: {transformers_version, torch_version, peft_version,
            environment_sha256, implementation_commit}
}
```

A historical metric JSON, unrelated reviewed record or different runtime therefore
cannot stand in for an admission artifact merely by receiving a current hash.
`reviewed` is an assertion made by its issuer; the checker cannot authenticate it.

Major records have a `payload` and a `receipt` file reference. Their receipt has its
own kind, the same binding, `outcome: "passed"`, a nonblank `recorded_by`, an exact
`checks` object with all required values true, and nonempty `evidence_files` hash
references. It binds `payload_sha256` computed as:

```text
SHA256(UTF8(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))))
```

This hashes the payload without its receipt reference, avoiding a circular hash.
The plan pins each completed major record's **full file bytes**, including its
receipt reference. Supporting files and receipts are also checked. A schema-valid
receipt is not independent confirmation that an observation really occurred.

## Dataset admission

`kind: "dataset_admission"`; `payload.tasks` is an array in the exact order of
`model_study.tasks`. Every task entry has:

- `task_id` and `label_contract` reference. The referenced `task_label_contract`
  has the common envelope, that task ID, `output_type` (`single_label`, `multi_label`,
  or `sequence`), ordered unique `label_order`, and pinned `tokenizer: {repo, revision}`.
  Classification labels are nonempty; sequence label order is empty.
- `splits` with exactly `train`, `model_selection`, `calibration`, `test` file references.
  Separate partitions may not be identical file content. This checks bytes, not row
  schemas or human correctness; those remain the source auditor/review's responsibility.
- `group_assignment` referencing a `dataset_group_assignment` record. It binds the
  task and exact `split_sha256` mapping, and declares `group_ids_by_split`. Group IDs
  are nonempty, unique within each partition and disjoint across partitions.

The grouping record also contains `grouping_policy` and an `overlap_report` reference:

| Policy field | Allowed values |
| --- | --- |
| `unit` | `provider_document_id`, `canonical_work_id`, `reviewed_group_id` |
| `generalization_claim` | `new_source_ids`, `unseen_normalized_text`, `unseen_canonical_works` |
| `text_overlap_policy` | `preserve_official_splits_report_overlap`, `deduplicated_variant` |

A `dataset_overlap_report` binds the same task and split hashes and records a
nonnegative integer `cross_split_normalized_text_groups`. Repeated text is permitted
under a reviewed preserve/report policy when the claim is new source IDs. It is not
permitted alongside an unseen-text/deduplicated claim. An unseen-canonical-work claim
requires canonical-work grouping; distinct provider IDs alone do not establish it.
The helper never imposes a universal zero-text-overlap rule.

The `dataset_admission_receipt` checks are exactly: `source_use_reviewed`,
`group_disjointness_verified`, `labels_verified`, `split_files_verified`,
`no_quarantined_inputs`, and `grouping_policy_reviewed`.
A legacy data audit is historical context, not an unconditional veto of a different,
freshly admitted dataset. This helper uses the fresh admission and its receipts.

## Adapter/module admission

`kind: "adapter_module_manifest"`. Its payload includes:

- `dataset_admission_sha256` and `label_contract_sha256` (exact task-to-hash map),
  binding the module design to the selected data and label interfaces.
- `private_scope`, identical to the study plan; `inspection_seed`, one of its unique
  integer `training_seeds.values`.
- `lora_config` with exact fields `r`, `alpha`, `dropout`, `bias`, `use_rslora`,
  `use_dora`: positive integer rank, positive alpha, dropout in `[0,1)`, bias `none`
  and both variant flags false for this first standard-LoRA contract.
- `module_allowlist.shared` and `module_allowlist.private[task_id]`: nonempty,
  unique, lexicographically ordered full module names. Shared entries must explicitly
  name encoder attention; private lists cannot include the encoder. Exact and nested
  parent/child allowlist overlap is rejected.
- `shared_parameters`, `private_parameters[task_id]`, `frozen_parameters`: ordered,
  nonempty name lists, disjoint and together covering the complete `parameter_inventory`.
  Every allowed trainable module must contain declared parameters.
- `parameter_inventory[name]` with positive integer `shape`, `initial_sha256` and
  `origin` (`base`, `lora`, `private_head`). Every base-origin parameter must remain
  frozen. Shared parameters must be LoRA; private origins must match their task's
  pooler/classifier or decoder-adapter policy.
- `tied_parameter_groups`: explicitly recorded, possibly empty. Each group contains
  at least two frozen names with the same initial digest and shape. Alias groups
  cannot overlap or cross into trainable partitions.

`private_initialization_files[seed][arm_id][task_id]` covers every declared seed,
exactly one joint and one specialist arm, and all tasks. Each file is a
`private_initialization_manifest` with the common envelope, integer seed, task ID
and `parameter_initial_sha256` mapping covering that task's exact private parameter
set. Hashes must agree across matched arms. At the inspection seed they must also
match the inventory. Merge arms reuse these private components; they do not invent
an independent initialization.

The `adapter_structure_receipt` checks are exactly: `complete_parameter_inventory`,
`encoder_only_shared_scope`, `private_task_scope`, `frozen_base_and_aliases`,
`matched_head_initialization`, and `save_reload_verified`. Static names and receipts
cannot themselves prove correspondence to a live model; the originating inspected
implementation and its supporting evidence must be reviewed separately.

## Feasibility and timing admission

`kind: "feasibility_timing_evidence"`. Its payload pins the full dataset and module
artifact hashes, the label-contract hash map, and `budget_contract` containing the
seven plan fields: B, hardware ID, timing-boundary ID, primary unit, specialist
allocation, overshoot rule, selection allowance.

`measurements` must cover every admitted task. Each has `task_id`, the same hardware
and timing-boundary IDs, positive `training_window_seconds`, positive integer
`optimizer_windows`, `source_length`, `batch_size`, nonnegative integer `target_length`,
and a pinned `raw_receipt`. These are observations supplied by a future authorized
feasibility process, not estimates generated by the checker.

The `feasibility_review_receipt` checks are exactly: `runtime_and_tokenizer_verified`,
`timing_boundary_verified`, `hardware_feasible`, `allocation_and_overshoot_reviewed`,
`selection_allowance_reviewed`.

## Validation and limits

Tests in [`test_model_admission.py`](../../tests/test_research/test_model_admission.py)
use temporary, explicitly synthetic metadata and tiny file fixtures. They exercise
placeholder rejection, cross-runtime/data/label binding, partial budgets, tampered
receipts, parameter intersections, frozen aliases, matched head initialization,
per-task grouping policies and incomplete timing coverage. They neither collect
research observations nor establish dataset or model quality.

The helper does not authorize experiments, rewrite the plan, manufacture reviews,
certify copyright/permission decisions, authenticate issuers, verify claimed tensor
contents against loaded weights, or prove that raw timing logs are truthful. Its
purpose is to reject missing and inconsistent evidence before those external reviews.
