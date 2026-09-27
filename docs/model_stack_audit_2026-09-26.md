# Python/model stack audit — September 26, 2026

This is a software review and synthetic unit-test repair. Training, real-checkpoint
inference, benchmark evaluation, research experiments and paid APIs remain paused.
No historical metrics were recomputed and no historical model/data artifacts were
changed. The website still uses its independent content-based recommender.

## Corrections and efficiency improvements

| Area | Previous behavior | Corrected behavior |
| --- | --- | --- |
| Joint inference | Tokenized and encoded the same input three times | `MultiTaskModel.classify_encoded` lets `batch_predict` tokenize/encode once and reuse memory for both heads and generation. Models without that API retain the old fallback. |
| Inference arguments | A multi-element model parameter was used as a boolean while choosing the device; threshold 0 fell back to 0.5; emotion labels could silently truncate | Explicit parameter/buffer presence checks, explicit `None` handling, threshold bounds and head/label-width validation. |
| Batched generation | A row that reached EOS could emit extra words while another row continued; unigram blocking used an incorrect prefix | Finished rows continue with EOS only; unigram blocking forbids previously generated tokens. |
| Attention and pooling | Finite masking allowed extreme masked scores through, and fully masked queries could average invalid values or produce NaNs | Boolean/negative-infinity masks with stable empty-row behavior, finite zero pooling for empty masks, and direct boolean SDPA masks when no position bias is required. |
| Gradient diagnostics | Zeroed all accumulated gradients, computed head gradients unnecessarily, and consumed extra training batches | `autograd.grad` over shared parameters using already-consumed batches; accumulated `.grad` values and CPU/CUDA RNG states remain intact. |
| Accumulation | Partial windows never stepped; joint-trainer remainder gradients could leak into the next epoch | Both joint and BERT trainers step/normalize the final partial window; scheduler update counts include it. |
| PCGrad | Repeated sampled tasks overwrote earlier losses; later projections used already-projected comparison gradients; unused private heads received zero grads | Sum all task draws, project against original other-task gradients, accumulate private grads directly, and preserve `None` for entirely unused parameters. |
| Validation | Joint trainer recycled shorter task loaders and weighted partial batches equally | Visit each validation batch once; classifier losses/metrics weight examples, summarization loss weights nonignored target tokens. BERT validation loss also weights examples. |
| Checkpoint callback | The early-stopping epoch exited before its checkpoint callback | The callback receives the stopping epoch and its final history. |
| BERT sampling/calibration | Alpha 0.5 used `n ** (1/alpha)`; early selection and threshold calibration shared all validation examples | Uses `n ** alpha` and the same deterministic selection/calibration split as LexiMind for future runs. |
| Reporting | BERT printed stale hard-coded LexiMind scores and missing scores as zero; multi-seed tables labeled requested seeds even after failures | Report only supplied BERT values. Multi-seed output tracks actual metric-specific seed IDs/counts, sample spread when estimable, missing seeds and failed training. Failed new runs cannot fall through to evaluation of old checkpoints. |
| Evaluation label order | Raw emotion logits could be compared against alphabetically resorted label metadata | Respect the checkpoint's supplied label order; empty/empty bootstrap F1 follows the same zero-division convention as its point estimate. |
| Data/setup | Training loaded unused test files; shifting ignored labels could retain -100 token IDs; odd sinusoidal dimensions failed | Test data is opt-in to the split helper; ignored labels shift to padding; odd hidden dimensions have valid sinusoidal tables. |
| API errors | Returned arbitrary internal exception strings to clients | Detailed exceptions stay in server logs and the client receives a stable generic error. |

The shared-encoding unit fixture checks exact result parity, matching padding masks,
unchanged checkpoint key names, and **one encoder/tokenizer call instead of three**.
This is a structural call-count check, not a measured GPU speedup. Attention fixtures
check extreme masked scores, fully masked rows, finite gradients and manual/SDPA
agreement. Mask semantics follow the [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Historical and future-run boundary

These changes intentionally correct future optimizer, sampler, validation and decoding
behavior. Future results must record this code revision and the resolved configuration;
they are a changed recipe, not a reproduction of Phase 1. In particular, splitting
validation now does not undo the old BERT checkpoint's historical model-selection
exposure. The September 22 archive and its hash manifest remain immutable evidence of
what was found then. See [RESULTS.md](RESULTS.md) for report provenance and
[eval_protocol.md](eval_protocol.md) for the still-draft prospective protocol.

Checkpoint module/parameter names and normal save/load interfaces are preserved.
`resume_from` is explicitly a **weights-only continuation**, not exact optimizer,
scheduler or RNG recovery. Epoch metadata from `last.pt` no longer identifies
`best.pt`; unknown best-checkpoint epochs start a new weights-only schedule. A true
resumable run bundle remains a prerequisite for the future research campaign.

## Verification boundary and remaining work

Tests use hand-constructed arrays, scalar gradient arithmetic, tiny randomly initialized
components, temporary JSON fixtures and mocked API results. They do not use a trained
checkpoint or report book/model quality. Visualization tests now write only to temporary directories; the final full-suite
run includes them without replacing pre-existing local figures.

CUDA kernel choice, AMP behavior, `torch.compile`, quantized modules and full-checkpoint
prediction parity have not been tested on the unavailable GPU. In particular, the
joint trainer's older fp16 fallback lacks gradient scaling; use of non-bf16 CUDA hardware
needs a separately validated mixed-precision design before experiments resume. CPU/CUDA
RNG preservation in diagnostics does not assert identical RNG behavior on other devices.

The following remain prospective work, not completed optimizations: exact resume bundles;
frozen data/group-split and gold-relevance manifests; a measured backbone integration;
GPU memory/throughput profiling; and optional dynamic-padding or generation-kernel
changes. No T5Gemma compatibility, recommendation gain, completed literature gate,
preregistered protocol, or new result is implied by these software fixes.
