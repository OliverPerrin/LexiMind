# LexiMind Phase 2 — Research Plan

## September 26 research preparation — current decision record

The current proposal is in [research/study_decisions.md](research/study_decisions.md)
and [the machine-readable design](../configs/research/study_design.json). This update
takes precedence where the older sections differ. Training, model execution and
research evaluations remain paused; literature/source preparation and software
contract checks have progressed.

- A nine-paper methods review establishes that heterogeneous-output merging and
  joint comparisons already exist. The first LexiMind study is framed as a controlled
  replication/application, with no originality claim or promised result.
- Keep benchmark-task retention (M1) separate from book relevance (B1). Book judgments
  do not gate an otherwise valid benchmark-only study, and benchmark emotion scores
  do not establish book mood. Both still need their own data/protocol admission.
- Working arms are joint LoRA, total-budget task specialists, and reuse of those
  specialist encoder deltas for task arithmetic and TIES. Private modules, selection
  access, materialization and full cost accounting are explicit. Numeric budget,
  backbone, dataset admission and adapter integration remain unresolved.
- [The data audit](research/data_readiness.md) covers 156,796 current rows. Missing
  parent/source identities and repeated inputs/targets require reconstruction and
  policy review before new research use; existing files are retained.
- [The backbone review](research/backbone_interface_review.md) uses pinned configs
  and source only. Nominal context does not establish usable context, consumer-GPU
  fit, or full-book understanding. A config read is not an integration.
- Annotation packets and compute ledgers have validators and explicit empty
  templates. No book labels, study measurements or trained results were manufactured.

Start at [research/README.md](research/README.md) for evidence and reproducible
preparation commands. The September 22 archive remains a historical result record.

## September 22, 2026 update — current scope and pause

**This section supersedes the August proposal below.** The earlier text is retained
as a historical design record, including assumptions that are now corrected here.
The user has asked to hold off on **training or any research experiments** because
the RTX 4070 is unavailable. No model inference/evaluation, backbone A/B, teacher
API generation, paid compute, or research campaign is authorized now. Ordinary
software unit tests, file-hash checks, catalogue engineering and website work can
continue. The old $15/December budget and hardware assumptions are not current
spending authorization or deadlines.

The product returns to books: a trustworthy catalogue, topic/genre discovery,
content-based recommendations and a website. The research track remains useful,
but shipping a catalogue does not establish model quality. Proposed applied question:
**Do jointly learned topic, genre and validated mood features improve book discovery
over a text-retrieval baseline?** Human book-level relevance judgments and a validated
mood taxonomy are prerequisites; GoEmotions scores are not book-mood ground truth.

The first prospective model comparison is deliberately narrower: joint SFT versus
same-backbone task specialists with a matched total budget, plus adapters merged
from those same specialists. Uniform task arithmetic and TIES are candidate merge
baselines; the final set follows the literature review. Distillation and RL remain
later possibilities. A backbone feasibility comparison is deferred until experiments
resume; T5Gemma remains a candidate requiring integration, not an implemented upgrade.

Corrections to the historical proposal:

- [RESULTS.md](RESULTS.md) now separates the root test report from the logged seed-17
  run. Their current checkpoint files are different bytes. Seed-17's logged emotion
  weight is **1.0**, while the present training YAML says **1.2**. Historical run/data
  identity cannot be recovered by reading current filenames.
- The current 3,440-row summarization test is **2,506 academic + 934 literary**, not
  3,440 academic. The historical report used only 221 literary examples. Fix identity
  matching and audit split groups before choosing any literary benchmark.
- Equal source-token counts are **not equal compute** across encoder-only heads,
  encoder-decoder generation, backbones or RL. Track source/target and padded tokens,
  active modules, optimizer steps, GPU/device time, validation/search overhead,
  memory, and measured or explicitly estimated FLOPs. Pin the primary budget and
  tolerance before runs; account for all specialist training and merge selection.
- One task per output type confounds output type with dataset and domain. An outcome
  on this suite cannot establish a general causal rule about output types.
- Long advertised context does not demonstrate useful full-book understanding,
  feasibility on 12 GB, or make long-document evaluation obsolete.
- The [official T5Gemma 2 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/t5gemma-2/)
  is dated **December 18, 2025**, not October 2025 as the historical table says. It
  describes architectural changes and pretrained releases. Recheck model cards,
  revisions and practical memory needs when selecting a backbone.
- The literature gate is **open**, and no novelty/publication claim is established.
  [Initial review](related_work.md) identifies prior work relating merging and joint
  learning; budget fairness and heterogeneous outputs are questions to investigate,
  not a verified unclaimed contribution.

Resumable preparation artifacts now available:

1. Historical report copies, SHA-256 manifest and generated tables in
   `research/results/`; `scripts/audit_research_artifacts.py` validates bytes without
   importing ML libraries.
2. [Evaluation protocol](eval_protocol.md), explicitly **draft**. It is neither frozen,
   preregistered, tagged nor agreed experimental authorization.
3. `configs/research/preparation.json`, with the user pause and unresolved gates.
   `python3 scripts/audit_research_artifacts.py --require-ready` intentionally exits
   nonzero. This checker is advisory preparation tooling; old training entry points
   are not technically locked by it.

On resumption, in order: confirm experiments may resume and available resources;
finish the full-paper literature matrix; repair/pin data and group-split audits;
collect blinded book-level relevance judgments; agree the backbone feasibility
protocol and budget; implement/test the selected model interface and budget ledger;
freeze the headline protocol with revisions and thresholds; then run the authorized
comparison. No dates or compute amounts are assumed. The website proceeds while
these research gates remain open.

---

## Historical proposal — August 4, 2026 (superseded where noted above)

**Working title:** Merge, Distill, or Jointly Train? A Fixed-Budget Comparison of
Multi-Task Post-Training Recipes for Small Language Models

**Author:** Oliver Perrin · Drafted 2026-08-04 · Target: arXiv preprint, no conference deadline

**Constraints this plan is built around:** worked intermittently alongside a full-time
MLE role and a side SaaS; ~$15 of paid compute until December 2026; an RTX 4070 (12GB)
and an M5 MacBook. Every design choice below that looks conservative is downstream of
one of those three.

---

## 1. Why the project is being redirected

Phase 1 asked: *does joint multi-task training beat single-task training for
summarization + emotion + topic on FLAN-T5-base?* Three problems make that question
a dead end regardless of how carefully it is re-run:

1. **It is answered.** Standley et al. (2020), Aribandi et al. (ExT5, 2022), and the
   whole MTL-in-NLP line already establish that transfer is heterogeneous and
   grouping-dependent. A three-task instance adds a data point, not a finding.
2. **The instrument can't measure it.** The topic test split is 189 samples with an
   11-point CI. The literary summarization split is 221 samples. Any MTL delta on
   those is unfalsifiable.
3. **The framing is dated.** In 2026 nobody chooses between "one FLAN-T5 fine-tuned on
   three tasks" and "three FLAN-T5s." The live question is how to *post-train* a small
   model into a multi-task specialist, and the candidate recipes are SFT mixtures,
   per-task adapters plus merging, distillation from a frontier teacher, and RL on
   verifiable rewards.

Phase 2 keeps the assets that are genuinely good — the evaluation discipline, the
calibration-split machinery, the reproducible harness — and replaces the question.

## 2. The question

> Given a fixed post-training compute budget **B** and **N** heterogeneous tasks,
> which recipe produces the best single deployable model — and does the answer depend
> on the *output type* of the task?

Practitioners make this call constantly and choose by folklore. The specific gap in
the literature is not "does merging work" (well studied) but the combination of:

- **budget-matched** comparison — merging papers compare merging methods to each
  other, rarely to joint SFT at equal total compute;
- **heterogeneous output types** in one suite — free-form generation, multi-label
  classification, and schema-constrained structured output have different
  interference profiles, and studies almost always use homogeneous benchmark suites;
- **distillation and RLVR as arms in the same currency** — normally studied in
  separate papers with incomparable budgets;
- **statistics that survive scrutiny** — multiple seeds, paired bootstrap on deltas,
  pre-registered eval.

### Phase 0 gate (do this before spending any compute)

Two evenings of literature review. If a paper already runs a budget-matched
joint-SFT vs. merge vs. distill comparison across heterogeneous output types, we
pivot the framing to the sharpest surviving sub-question — most likely *"which task
output types tolerate merging and which require joint training?"*, which is narrower
and, as far as I can tell, unclaimed. **Do not skip this gate.** Search: model
merging + multi-task, task arithmetic, TIES/DARE, model soups, data mixing laws for
post-training, RLVR multi-task.

## 3. Task suite

Chosen so that (a) every test split is large enough for a 3-point difference to be
detectable, (b) the output types are genuinely different, (c) at least one task is
*verifiable* so an RL arm is honest rather than a reward-model guess.

| # | Task | Output type | Source | Test n | Primary metric |
| - | ---- | ----------- | ------ | ------ | -------------- |
| T1 | Abstractive summarization | Free-form generation | arXiv body → abstract (existing, `data/processed/summarization`) | 3,440 | ROUGE-L + BERTScore F1 |
| T2 | Emotion detection | Multi-label, 28 classes | GoEmotions (existing) | 5,427 | Macro F1 @ frozen tuned thresholds |
| T3 | Attribute extraction | Schema-constrained JSON (**verifiable**) | Product-attribute corpus (MAVE or Amazon ESCI-derived) | ≥5,000 | Field-level exact-match F1 + schema-validity rate |
| T4 | Topic classification | Single-label, multi-class | 20 Newsgroups or AG News | 7,532 / 7,600 | Accuracy + macro F1 |

**Changes from Phase 1 and why:**

- **T3 is new and is the centre of gravity.** Structured extraction is verifiable
  (exact match against a schema), which makes the GRPO arm principled. It is also the
  task type that matters most for the product-data-harmonization work — the paper
  becomes directly useful to that audience.
- **T4 replaces the 189-sample topic set.** Same output type, 40× the test data, a
  standard benchmark with published numbers. The old topic set can stay as a small
  domain-shift probe, never as a headline.
- **T1 keeps only the academic half as headline.** The 221-sample literary split
  becomes an explicit out-of-distribution generalization probe, reported separately.
  That is more informative than dropping it and more honest than averaging it in.
- **T2 is unchanged.** It is the hardest arm, has published baselines to anchor
  against, and its long tail is exactly where interference shows up. The existing
  calibration-split threshold machinery carries over unchanged.

## 4. Base model

> **Status: PROVISIONAL — not yet measured.** The choice below is a reasoned default,
> not a result. It is settled by the A/B in chunk 1a (§11), and this section gets
> rewritten with numbers once that runs. Nothing downstream may cite it as decided.

**Stay encoder-decoder. Upgrade the backbone from FLAN-T5-base (2022) to T5Gemma.**

### Why encoder-decoder at all

1. **It is the setting that needs the paper.** Teams that cannot afford to pretrain —
   the overwhelming majority — post-train a small pretrained encoder-decoder on one
   GPU. The post-training literature is written at frontier scale and does not
   transfer down cleanly. Scoping explicitly to *small encoder-decoder, one consumer
   GPU, fixed budget* is a real and underserved audience, not a compromise.
2. **The pipeline already works.** Tokenization, data, trainer, eval, and the
   calibration-split machinery are debugged and shape-compatible. On intermittent
   evenings that is worth more than a fashionable backbone.
3. **The research question is architecture-agnostic.** "Does the best post-training
   recipe depend on the task's output type?" is the same question on any backbone, and
   an encoder-decoder is arguably the *cleaner* testbed — task heads are explicit
   rather than all funnelled through next-token prediction.

### Why not FLAN-T5-base

FLAN-T5 is a 2022 checkpoint with a 512-token context. Google has since shipped
**T5Gemma** — Gemma adapted back into an encoder-decoder via UL2 on ~2T tokens — which
keeps the architecture and replaces the pretraining. Google reports T5Gemma 2 beating
its Gemma 3 decoder-only counterparts on several benchmarks, crediting the separate
encoder for long-context handling.

| Candidate | Params | Context | Modality | License |
| --------- | ------ | ------- | -------- | ------- |
| `google/t5gemma-b-b-ul2` (v1, Jun 2025) | 591M | — | text-only | gemma (gated) |
| `google/t5gemma-2-270m-270m` (v2, Oct 2025) | 786M (~370M text) | 128K | multimodal | gemma (gated) |
| `google/flan-t5-base` (Phase 1) | 250M | 512 | text-only | apache-2.0 |

**Leaning `t5gemma-b-b-ul2`** for the grid: text-only, so no vision tower is carried
through every step of a 15-run campaign. `t5gemma-2-270m-270m` is the stronger model
but roughly a third of its checkpoint is a vision encoder irrelevant to these tasks.
The A/B decides.

### Consequence worth recording

**128K context retires the original Phase 1 research question.** "Train short, infer
long" existed to work around a 512-token limit that modern encoder-decoders no longer
have — chunking-and-aggregation was a workaround, not a phenomenon. RQ2 wasn't merely
under-powered on 221 samples; it was about to stop being a question at all. That is
independent confirmation the pivot was correct.

### Watch items

- **License.** T5Gemma is `license:gemma`, not Apache-2.0, and the checkpoints are
  gated. Fine for a research preprint. If any of this feeds commercial work, read the
  Gemma terms — [`Qwen/Qwen3.5-0.8B`](https://hf.co/Qwen/Qwen3.5-0.8B) is the
  Apache-2.0 fallback.
- **Not instruction-tuned.** T5Gemma 2 ships base checkpoints only. Irrelevant for
  full fine-tuning — arguably cleaner — but the A0 zero-shot reference arm needs an
  instruction-tuned checkpoint to be meaningful. Use a v1 `-it` variant for A0, and
  say so in the paper.
- **Throughput.** 591M is ~2.4× FLAN-T5-base. §8's grid estimate assumes this; the
  timed run in chunk 3 confirms or forces B down.

**Deferred to December, when there is budget: one Qwen3.5-0.8B arm** as a
decoder-only transfer spot-check on the top-2 recipes — labelled in the paper as a
single spot-check, not a second grid. Note the prior evidence is that Qwen3-0.6B
matched FLAN-T5-base on a constrained production task while training slower, so the
expected result is "recipe ranking transfers," not "decoder-only wins."

The from-scratch transformer keeps its role as a transparent reference implementation
and documented engineering artifact.

## 5. Arms

All student budgets equalized at **B** (definition in §6).

| Arm | Recipe | Budget | Seeds |
| --- | ------ | ------ | ----- |
| **A0** | Base model, zero-shot + 5-shot | 0 | 1 |
| **A1a** | Single-task specialists, B each (N models deployed) | N·B | 1 |
| **A1b** | Single-task specialists, B/N each | B total | 1 |
| **A2** | **Joint SFT**, mixed batches, proportional + temperature mixing | B | 3 |
| **A3** | Sequential SFT, B/N per stage, two task orders | B | 1 per order |
| **A4** | Per-task LoRA experts (B/N each) merged: uniform task arithmetic, TIES, DARE-TIES | B | 3 |
| **A5** | Distillation from a Claude teacher, Batch API (hard targets vs. rationale-augmented) — see §8 | B student | 3 |
| **A6** | A2 checkpoint + GRPO on the verifiable subset (T3, T4) | B split SFT/RL | 3 |

A1a is deliberately *not* budget-matched — it is the "you were willing to deploy N
models and spend N×B" reference. Label it as such in every table. A1b is the
budget-matched specialist control, and the pair of them separates "more compute" from
"more models."

A4's experts are the same runs as A1b's specialists. Train once, use for both.

## 6. Fixed-budget accounting (the methodological contribution)

Budget is measured in **student optimizer tokens**: the total number of tokens
processed through the student in forward+backward, summed over all optimizer steps.
Not wall-clock, not steps, not epochs — those are hardware- and batch-dependent and
are exactly why cross-paper comparisons in this area are incoherent.

Rules, applied uniformly:

- **SFT arms:** `B = Σ_steps (batch_size × seq_len)`. Straightforward.
- **Merging arms:** N experts × B/N. Merging itself is CPU-side and costs ~0; report
  it as such rather than pretending it is free of *engineering* cost.
- **Distillation arms:** the student's budget is B. **Teacher inference is reported
  separately**, in both tokens and dollars, and every table carries both a
  student-only column (fair to a practitioner who buys or already has the data) and a
  total-cost column (fair scientifically). This double accounting is the honest thing
  and is routinely fudged.
- **RL arms:** count *all* tokens through the policy, including rollouts and the
  discarded samples — not just the tokens in the gradient. RLVR looks far cheaper than
  it is when only gradient tokens are counted. Report the SFT/RL split explicitly.

Log the running token count in the trainer and assert it against B at run end. Make
budget violation a hard failure, not a footnote.

## 7. Evaluation protocol — frozen before the campaign starts

Written down and committed *before* any arm runs. This is the direct fix for what went
wrong in Phase 1.

- **Seeds:** 3 for headline arms (A2, A4, A5, A6), 1 for the rest. Report mean ± std.
- **Deltas:** paired bootstrap (10,000 resamples) on the same test items; report the
  CI of the *difference*, not two overlapping CIs.
- **Thresholds (T2):** tuned on the calibration half of validation, frozen, applied to
  test. Machinery already exists in `src/data/dataset.py::split_emotion_val`.
- **Decoding (T1, T3):** identical parameters across every arm. Fixed, documented,
  ablated once at the end — never tuned per arm.
- **Model selection:** identical criterion across arms, on validation only.
- **No metric may be tuned on test.** No exceptions, no "we checked."
- **Every table is generated from result JSON by a script.** No number is ever typed
  into a `.tex` file by hand. Add `scripts/build_tables.py`; make it the only path
  from results to paper.
- **`docs/RESULTS.md` is updated as each arm lands**, with the source path.

### Expected outcomes, written down in advance

Committing to predictions makes the negative results publishable rather than
embarrassing:

- Joint SFT beats budget-matched merging on **T1** (generative tasks need shared
  capacity throughout training).
- Merging is competitive or better on **T2/T4** (classification heads specialize
  cleanly; merging avoids interference).
- Distillation with **rationales** beats hard-target distillation on **T3** and
  matters little on T2/T4.
- GRPO helps **T3** substantially and **T4** marginally, and mildly degrades T1
  (alignment-tax style regression on the unrewarded task).

If these all hold, the paper is a clean "output type determines the right recipe"
result. If they don't, the surprises are the paper. Either way it is publishable —
which is the point of pre-registering.

## 8. Compute and cost

**Hard constraint: ~$15 total until December 2026, and no rented GPU at all.**
Everything trains on the 4070. The entire paid budget goes to one thing.

| Work | Where | Cost |
| ---- | ----- | ---- |
| Every training arm, all seeds | **RTX 4070**, overnight, sequential | $0 |
| GRPO rollouts | **RTX 4070** (250M policy, short JSON outputs) | $0 |
| Development, analysis, figures, writing | **M5 MacBook** | $0 |
| **Teacher data for the distillation arm** | **Claude API, Batch** | **~$15** |
| **Total** | | **~$15** |

### Why the whole budget goes to teacher data

It is the only spend that compounds. Teacher outputs are generated **once** and reused
by every distillation seed, every ablation, and every re-run for the life of the
project — where an hour of rented GPU is consumed once and gone. It is also the one
thing the 4070 genuinely cannot substitute for: a locally-quantized 8B teacher fits in
12GB, but its structured-output quality is exactly the variable the distillation arm is
supposed to isolate, so a weak teacher makes the arm uninterpretable rather than cheap.

**Use the Batch API** (`client.messages.batches.create`) — 50% off all token usage,
up to 100K requests per batch, typically finishing within an hour. This is a bulk
offline job with no latency requirement, so there is no reason to pay interactive
rates. Results come back in arbitrary order: key them by `custom_id`, never by
position.

**Model: Claude Haiku 4.5** (`claude-haiku-4-5`) — $1.00 / $5.00 per MTok, halved
under Batch to **$0.50 / $2.50**. Structured extraction with a fixed schema is well
within its range, and it is 3× cheaper than Sonnet on input and output alike.

Worked estimate at ~20K distillation examples, ~400 input and ~250 output tokens each:

```text
input:   20,000 × 400 = 8.0M tokens × $0.50/MTok  = $4.00
output:  20,000 × 250 = 5.0M tokens × $2.50/MTok  = $12.50
                                            total ≈ $16.50
```

Levers if that runs over: drop to 15K examples (≈$12.40), or shorten the rationale
field — output tokens are ~75% of the bill, so rationale length is the dominant term.
**Generate a 200-example pilot batch first**, inspect the outputs by hand, and only
then commit to the full run — a schema or prompt bug discovered after the full batch
is the one way to actually waste this money.

Don't bother with prompt caching here: Haiku 4.5's minimum cacheable prefix is 4096
tokens and the shared schema/instruction prefix will be far shorter, so `cache_control`
would pay the write premium for zero reads.

### 4070 throughput

T5Gemma `b-b` (591M) under LoRA, ~4 tasks: roughly **4–8 h per full-budget arm** —
about 2.4× the FLAN-T5-base estimate, scaling with parameter count. That is roughly
one arm per overnight run, so the ~15-run grid is **12–15 nights**. They need not be
consecutive; each run is independent and checkpoints to disk.

**These numbers are extrapolated, not measured.** Chunk 3 (§11) runs one timed arm and
sets B from observed tokens/sec. If the 4070 is slower than projected, **shrink B
uniformly rather than cutting arms** — a smaller equal budget is still a completely
valid experiment, while dropping an arm breaks the comparison that is the whole point.

If 591M proves too slow to finish the grid in reasonable calendar time, the fallback
order is: (1) shrink B, (2) drop T1's literary OOD probe from the training mix,
(3) fall back to FLAN-T5-base and state the backbone limitation in the paper. Do not
cut seeds — single-seed results are what Phase 1 already produced.

### December and after

When there is budget again, in priority order: (1) the Qwen3-0.6B architecture
spot-check on the top-2 recipes (~$40 on an H100), (2) a third seed on any arm whose
variance turned out wide, (3) a larger teacher for a distillation-quality ablation.
None of these are load-bearing — the paper stands without them, and each one is a
clearly-labelled extension rather than a gap.

## 9. Machine workflow (Mac ↔ 4070)

Skip rsync. It leaves you reconciling divergent copies, which is how Phase 1 ended up
with checkpoints on one machine and result files on another.

- **Code:** git. Push from the Mac, `git pull` on the 4070 box. One source of truth,
  full history, no directional sync to remember, and no "which machine has the newer
  version" question after a six-week gap.
- **Data:** HuggingFace Hub datasets (you already publish there). `download_data.py`
  becomes a pull-from-Hub, so every machine materializes identical data from a pinned
  revision. Data stays out of git.
- **Checkpoints:** push to HF Hub under a per-run ID; pull to the Mac for analysis.
- **Result JSON goes in git.** It is small, and it is the paper's evidence — it should
  be versioned alongside the code that produced it.
- **Environments:** `uv` with a committed lockfile. One command per machine.
- **Mac (M5):** development, analysis, plotting, paper writing. MPS runs forward
  passes for smoke tests; don't train there.
- **The 4070 box only needs to be on when a run is going.** Queue two arms, start
  them, come back the next evening. Nothing requires it to be reachable otherwise.

Net effect: both machines are symmetric and stateless, and nothing exists in exactly
one place. Adding Brev back in December is then a `git clone` plus a Hub pull.

## 10. What carries over from Phase 1

**Keep and reuse:**
- `src/training/metrics.py` — bootstrap CIs, F1 variants, threshold tuning
- `src/data/dataset.py::split_emotion_val` — the calibration/model-selection split
- Hydra config layout, MLflow tracking, the pytest suite, CI
- GoEmotions and arXiv summarization data pipelines
- The from-scratch transformer, as a documented reference implementation

**Rewrite:**
- `src/training/trainer.py` — needs token-budget accounting, LoRA, and a mixing-strategy
  abstraction. Probably cleaner as a new trainer alongside the old one than as a retrofit.

**New:**
- `src/merging/` — task arithmetic, TIES, DARE-TIES
- `src/distill/` — teacher data generation, rationale formatting
- `src/rl/` — GRPO loop over the verifiable tasks
- `scripts/build_tables.py` — results JSON → LaTeX, the only path to paper tables
- T3 data pipeline

## 11. Sequencing — designed for on-and-off work

This is a side project worked in gaps between a startup job and a SaaS. It is
sequenced as **independent, resumable chunks**, each leaving the project in a state
that survives a six-week gap. No chunk depends on remembering what you were mid-way
through in the last one.

| # | Chunk | Deliverable that persists | Effort |
| - | ----- | ------------------------- | ------ |
| 0 | Literature gate (§2) — confirm or pivot the framing | `docs/related_work.md` with the verdict written down | ~2 evenings |
| 1 | T3 corpus built, 200 samples hand-validated; T4 wired in | Dataset on HF Hub, pinned revision | ~1 weekend |
| **1a** | **Backbone A/B (§4): FLAN-T5-base vs `t5gemma-b-b-ul2` vs `t5gemma-2-270m-270m`** | **Measured table in `RESULTS.md`; §4 rewritten with numbers** | **~1 evening** |
| 2 | **Eval protocol frozen and committed** | `docs/eval_protocol.md`, git-tagged before any arm runs | ~1 evening |
| 3 | LoRA trainer with token-budget accounting; one timed arm | Calibrated value of B, committed to config | ~2 weekends |
| 4 | Teacher data generated (Batch API, §8) | JSONL on HF Hub — generated once, reused forever | ~1 evening + wait |
| 5 | Grid: A0–A4 on the 4070 | One result JSON per arm, appended to `RESULTS.md` | ~8 unattended nights |
| 6 | A5 distillation + A6 GRPO | Same | ~2 weekends |
| 7 | Analysis, figures, `build_tables.py` | Auto-generated paper tables | ~1 weekend |
| 8 | Writing | Preprint | ~4–6 evenings |

**Chunk 1a is small but load-bearing.** One short single-seed run per backbone on the
same data and hyperparameters, comparing task metric *and* wall-clock tokens/sec.
Cheap to run, and it converts §4 from a reasoned guess into a measured decision before
anything expensive depends on it. It also confirms the throughput assumption in §8.

Rules that make the gaps survivable:

- **Chunk 2 is the gate.** Nothing in chunks 5–7 may start until the eval protocol is
  frozen and tagged. This is the direct structural fix for what went wrong in Phase 1.
- **Provisional sections carry a status banner** until the run that settles them lands
  (§4 has one now). A plan may hold open questions; it may not state them as answers.
- **Every chunk ends with something committed.** A dataset revision, a config value, a
  result JSON — never a half-finished branch and a mental note.
- **`RESULTS.md` grows one arm at a time.** After any gap, it tells you exactly what is
  done and what isn't, without re-reading code.
- **Chunk 4 can run any time after chunk 1** — it's independent of the trainer work, so
  fire it early and let the teacher data sit ready.

No target date. Chunks 0–2 are ~4 evenings and unlock everything else; the rest can
take as long as it takes. The failure mode is starting the campaign before the
protocol is frozen — not running out of calendar.

## 12. Risks

| Risk | Mitigation |
| ---- | ---------- |
| The question turns out to be claimed | Phase 0 gate before any compute or money is spent |
| T3 corpus is noisier than expected | Hand-validate 200 samples before committing; arXiv-metadata extraction is the fallback |
| 4070 too slow for the grid | Shrink B uniformly. A smaller equal budget is still a valid experiment — never drop an arm, that breaks the comparison |
| Teacher-data spend wasted on a bad prompt or schema | 200-example pilot batch, inspected by hand, before the full run |
| Long gaps between sessions lose context | Chunked sequencing above; `RESULTS.md` as the resume point |
| Reviewers ask "why not a decoder-only model" | Scope stated explicitly up front (§4); Qwen3.5-0.8B spot-check deferred to December and labelled as such |
| T5Gemma too slow to finish the grid | Chunk 1a measures it early. Fallback order in §8: shrink B → drop the literary OOD probe → revert to FLAN-T5-base and state the limitation. Never cut seeds |
| Gemma license blocks a downstream use | Research preprint is unaffected. Qwen3.5-0.8B (Apache-2.0) is the escape hatch; read the terms before any commercial dependency |
| Merging results are boringly negative | Pre-registered predictions (§7) make a negative result a finding, not a failure |
| Scope creep back toward building a system | The demo and API are frozen. Phase 2 ships a paper and a reproducible harness, nothing else |
