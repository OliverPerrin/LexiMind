# First-study decisions — working design, 26 September 2026

This is a concrete preparation proposal, not a frozen protocol or permission to
run. The machine-readable counterpart is
[`study_design.json`](../../configs/research/study_design.json). Training, model
execution, research scoring and paid teacher calls remain paused.

## Research position

Treat the first model study as a **controlled replication/application**. Existing
work already merges classification and generation tasks and compares to joint
learning. The [methods review](model_recipe_review.md) identifies differences in
data access, output interfaces, expert creation and selection cost that make an
unqualified leaderboard comparison unsuitable.

The question we can defend is: *given the same student-training allowance and
task interfaces, what task utility do joint adaptation and merged specialists
retain, and what additional selection/deployment costs do they incur?*

This does not claim a novel merge algorithm, a universal recipe ranking, or a
causal effect of output type. A single dataset for each output type confounds
domain, size, supervision and interface. The study can be useful as a transparent
case study without a conference target or a promised positive result.

## Two independent studies

**M1 — model recipes.** Start by resolving a comment-emotion and news-topic
classification core. Academic summarization is an extension candidate after its
identity, source-use and length audit. The exact suite remains open. It need not
wait for book-relevance labels, and its benchmark scores do not establish book
recommendation utility.

**B1 — book relevance.** Prepare graded query-to-work judgments for a declared
catalogue and evidence scope. Rank all eligible works in the small catalogue.
The current site is a *lexical plus metadata* baseline: its index already includes
subjects and genres. A genuinely text-only control must restrict fields explicitly.
No evaluation has run and no query or judgment collection has started.

M1 can inform a later book-domain feature model, but the current GoEmotions head
has no admission as literary mood evidence. B1 mood queries remain disabled.

B1's first proposed generalization target is **new query families over the fixed
catalogue**. Candidate works may appear for queries in multiple partitions; query
families and seed-work groups must remain separated. This is not an unseen-work
claim. A later work-held-out model-feature study needs a separately stated split
and candidate-eligibility policy.

## Common interface and four initial model arms

The working interface retains separate task heads. Merge only encoder attention
updates. Emotion/topic pooling and classifiers are private; the summarization
decoder's allowed adapter updates are private to that task. Embeddings and layer
norms stay frozen in this proposal to avoid hidden tied-weight updates. Exact
module allowlists still require a selected implementation and feasibility checks.
All pretrained base weights stay frozen during adaptation. Later materializing
merged effective deltas into base matrices is a distinct merge operation, not an
exception to the training policy.

| Arm | What is adapted | Resource accounting | Deployed representation |
| --- | --- | --- | --- |
| Joint | One shared encoder adapter plus the same task-private modules | One total student-training allowance B | One encoder plus task-private modules |
| Specialists | Independently adapted task encoders and private modules | B total across tasks; allocation must be declared | One encoder per task, all required private modules counted |
| Task arithmetic | The exact specialist encoder deltas above | Same expert creation plus arithmetic and any selection cost | One materialized encoder plus retained specialist private modules |
| TIES | The same specialist encoder deltas | Same expert creation plus trimming/sign election/selection | One materialized encoder plus retained specialist private modules |

Use the same base/tokenizer revisions, task interfaces, label order, adapter family,
head initializations and allowed training data. Do not compare full joint fine-tuning
to LoRA specialists as if only merging changed. Do not retrain experts separately
for each merge method. Preserve their original run IDs and hashes.

The first four arms do not include an “unadapted” random-classifier-head reference.
That would not measure the pretrained encoder's usable classification ability. A
frozen-encoder, trained-head control is a possible additional arm with its own
training budget; a generative zero-shot reference has different prerequisites.

Private-module retention is a recipe decision: a merged encoder may present a
changed representation to a specialist head or decoder. A post-merge head-adaptation
diagnostic needs its own declared budget and development data; it is not hidden
inside the primary merge arm. An all-text-output variant is a separate study axis.

The first merge representation is an effective weight delta, materialized into the
same base matrices. Separately averaging LoRA factors introduces cross terms and
is not the same operation. Recompression to a fixed adapter rank is deferred and
must report approximation error and cost if added. The official
[PEFT discussion](https://huggingface.co/blog/peft_merging#methods-for-combiningmerging-lora-adapters)
distinguishes factor-space and effective-product variants.

## Budget and selection choices

The working primary unit is **synchronized training-window wall seconds on one
declared device**, enclosing forward, backward and optimizer work. B, timing
implementation, warm-up/compile boundaries, task allocation and overshoot tolerance
remain unset until authorized feasibility work. Tokens, padding, examples, trainable
parameters and device-window/FLOP estimates remain separate ledger quantities.

This first proposal matches student training, not automatically total development
cost. Validation, calibration, merge search, failures, final evaluation, preprocessing
and any later teacher work are visible in the full ledger. Selection access and
allowances must be agreed before comparing methods. Equal trial counts need not
mean equal compute; report both. No local hardware cost is declared “free.”

Primary checkpoints are the declared final-budget checkpoints. Validation-selected
checkpoints may be a separate prespecified secondary analysis. Failed/partial runs
and missing tasks remain visible. Proposed seeds 17, 42 and 123 are a design choice,
not a completed or approved campaign. No average may hide a large task regression;
report per-task outcomes and the deployment/cost tradeoff before choosing a winner.

Compatible *existing experts* answer a second resource question. Keep their
incremental reuse costs and acquisition/history distinct from the all-data-available
regime. The first four-arm comparison does not silently switch between the regimes.

## What is decided versus open

Decided for preparation: a replication/application framing; independent M1/B1
questions; common-data access; common adaptation family; explicit private modules;
effective-delta merging; reused expert identities; no fabricated zero costs or labels.

Still open: eligible/pinned fresh data; source-use review; canonical work/edition
reconciliation where claimed; actual adapter integration; chosen backbone/runtime;
B and stopping policy; hyperparameter/merge-selection allowance; effect-size and
uncertainty design; human judgments; protocol freeze; explicit experiment resumption.
None can be resolved by changing a status flag or hashing an old result report.
