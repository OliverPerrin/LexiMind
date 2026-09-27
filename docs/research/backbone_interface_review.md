# Backbone and task-interface review

**Status: source/configuration review only, 2026-09-26. `selected_model = null`.**
No model/config class was instantiated, no weights or tokenizer files were loaded,
and no inference, training or feasibility experiment was run. This document uses
[pinned repository metadata](../../research/preparation/repository_metadata.json),
installed Transformers **5.12.1** source and official documentation. The inspected source files match upstream commit `ddb849abe009d1089e6c691bfc897f27211c663c` byte-for-byte. The repository's
software-test dependency pin is **5.17.0**; this source review does not establish
runtime compatibility with either version. The machine-readable counterpart is
[`backbone_candidates.json`](../../research/preparation/backbone_candidates.json).

## Candidate facts, not measured capabilities

| Candidate and immutable config | Structural facts in that config | Position/token observations | Access status |
| --- | --- | --- | --- |
| [FLAN-T5-base, `7bcac57…`](https://huggingface.co/google/flan-t5-base/blob/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/config.json) | `T5ForConditionalGeneration`; encoder/decoder 12/12 layers, width 768, FFN 2048, 12 heads, head width 64; vocabulary 32,128. | `n_positions=512`; relative-position buckets, not an absolute learned position table. Saved decoder start/pad 0, EOS 1. This is not a demonstrated hard architectural 512-token boundary or evidence of quality beyond the chosen truncation. | Metadata says ungated, Apache-2.0. No weights acquired. |
| [T5Gemma b-b-ul2, `97ea9b7…`](https://huggingface.co/google/t5gemma-b-b-ul2/blob/97ea9b7e92738bb57437867277ae38e65345b8d7/config.json) | `T5GemmaForConditionalGeneration`; 12/12 layers, width 768, FFN 2048, 12 query/12 KV heads, head width 64; vocabulary 256,000. Similar widths do not make it T5-compatible. | Encoder/decoder position fields 8,192; alternating sliding/full attention, sliding window 4,096. Saved EOS is **[1, 107]**, pad 0; nested BOS is absent from raw JSON and defaults to 2 in inspected config source. | Metadata says manually gated, Gemma terms. Readable metadata/config does not establish account access, terms acceptance or permission for a proposed use. |
| [T5Gemma2 270m-270m, `7c38f16…`](https://huggingface.co/google/t5gemma-2-270m-270m/blob/7c38f16641f455ef0685b18431faf1b17722d5a1/config.json) | `T5Gemma2ForConditionalGeneration`; text encoder/decoder 18/18 layers, width 640, FFN 2048, 4 query/1 KV heads, head width 256; vocabulary 262,144. Encoder also contains a SigLIP vision configuration. | Actual pinned text encoder/decoder `max_position_embeddings` is **32,768**, with full-attention RoPE factor 8 and sliding window 512. Saved BOS 2, pad 0, EOS 1. Do not replace these fields with a family-wide “128K” claim or multiply them into a proven context length. | Same gated/terms distinction as T5Gemma. Text-only inputs do not prove the unused vision parameters disappear from a loaded model. |

The [T5 source](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5/modeling_t5.py#L939),
[T5Gemma source](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma/modeling_t5gemma.py#L947)
and [T5Gemma2 source](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma2/modeling_t5gemma2.py#L1089)
are separate implementations. Configuration metadata is not parameter-count measurement,
VRAM evidence, throughput evidence, tokenization parity or a quality ranking.

## Upstream model boundary

Use the upstream conditional-generation implementation as the first integration target,
with a small explicit task wrapper. Keep the custom LexiMind model as the historical
reference. Its [factory](../../src/models/factory.py) recognizes `"t5"` in a name and
routes to a T5-specific weight mapper; changing that string to T5Gemma is not an
integration. RoPE, masks, normalization, vocabulary, grouped KV projections and the
second-generation merged decoder attention need their actual implementations.

The task wrapper should obtain encoder hidden states and the matching source mask,
then apply private task heads or pass `encoder_outputs` into the upstream generator.
Do not swap among each family's built-in sequence-classification classes: their
encoder/decoder paths and pooling policies are not the same experimental interface.

| Conditional-generation model | Source-level encoder route | Decoder/private path | Candidate encoder LoRA names before wrapping |
| --- | --- | --- | --- |
| T5 | `get_encoder()` resolves `.encoder` | `.decoder`; encoder and decoder blocks have separate self/cross-attention modules | `encoder.block.<i>.layer.0.SelfAttention.q` and `.v` |
| T5Gemma | inherited `get_encoder()` resolves `.model.encoder` | `.model.decoder`; private self and cross attention are distinct | `model.encoder.layers.<i>.self_attn.q_proj` and `.v_proj` |
| T5Gemma2 | explicit `get_encoder()` resolves `.model.encoder`, a multimodal wrapper; text stack is `.text_model` | `.model.decoder.layers.<i>.self_attn` is **merged** attention: its projections serve decoder and encoder-memory inputs | `model.encoder.text_model.layers.<i>.self_attn.q_proj` and `.v_proj` |

These are inspected names, not a successful PEFT injection. T5Gemma2's text-only encoder
path bypasses image-feature computation when `pixel_values` is absent; construction
still creates a vision tower. Its text encoder explicitly builds bidirectional masks,
so the saved generic `use_bidirectional_attention=false` field alone is not the forward
semantics. Inspect the [wrapper and text forward](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma2/modeling_t5gemma2.py#L754)
and [merged attention](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma2/modeling_t5gemma2.py#L332).

## Shared versus private adaptation contract

The proposed comparison retains one task interface within each backbone:

- **Frozen common base:** all original model weights, embeddings, language-model output
  projection, norms and any vision/projector parameters. This avoids accidental encoder
  changes through a tied decoder embedding. Save the resolved alias/tying map later.
- **Shared trainable component:** encoder-only Q/V LoRA deltas, with the same target
  modules, rank, scaling, dropout and initial tensors across recipes for a given seed.
- **Private trainable components:** emotion and topic heads with the same masked pooling,
  label order, losses and initial tensors; summarization decoder adapters with an
  explicit per-family allowlist. Each specialist updates only its own private components.
- **Joint arm:** the encoder adapter receives all task losses; each private component
  receives its task's loss. **Specialists:** independent copies start from the same
  base and the same corresponding private initialization. **Merge:** combine only the
  effective encoder updates and retain each specialist's private head/decoder adapter.
  Any head recalibration is a separate budgeted condition, not implicit postprocessing.

This is a design proposal for the root study plan, not a frozen protocol. Across
backbones the hidden widths differ, so “same initialization” means matched copies
within a backbone/seed, not identical incompatible tensors across model families.

Use explicit full-name allowlists, not bare suffixes such as `q_proj`/`v_proj` or
`all-linear`, which can reach decoder and vision modules. The JSON records candidate
regular expressions; expected match counts are derived from the config and must later
be checked against actual modules. A nested research wrapper or PEFT wrapper changes
prefixes, so matching must happen against the actual pre-injection module tree.
[PEFT documents these targeting and save controls](https://huggingface.co/docs/peft/package_reference/lora).

The merge artifact should contain `delta_W = (alpha/r) * B @ A` for each allowed
encoder projection, plus the common base revision and private-component references.
For weighted merging, combine those effective deltas; independently averaging A and B
introduces cross terms. Decide whether the result is materialized into frozen-base
weights or recompressed to a fixed rank. Preserve and account for every retained
private head/adapter. Neither equal rank nor the candidate's name implies equal
trainable counts: for example, the T5Gemma2 Q and V projection widths differ under
its grouped-KV configuration.

## Decoder start, EOS and tokenizer contract

Use `prepare_decoder_input_ids_from_labels` from the selected upstream implementation,
and mask target padding with -100. T5 reads `decoder_start_token_id` (saved as 0 here).
The inspected Gemma implementations shift with `config.decoder.bos_token_id` (default
2) and replace ignored labels with the decoder pad ID. A universal “start with pad”
rule would therefore be wrong. Source:
[T5 shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5/modeling_t5.py#L582),
[T5Gemma shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma/modeling_t5gemma.py#L595),
[T5Gemma2 shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma2/modeling_t5gemma2.py#L707).

The tokenizer/processor files and `generation_config.json` have **not** been inspected
or resolved in this task. Before authorized feasibility, pin them alongside the model
revision and reconcile the complete EOS list, BOS/start IDs, padding side, added tokens,
prompt/chat formatting and decoding settings. Do not take an instruction-tuned chat
example from family documentation as the agreed format for a different checkpoint.
Use the matched upstream tokenizer/processor; do not reuse the legacy FLAN tokenizer
for a Gemma vocabulary. Classification should consume encoder states directly rather
than silently using a generator-backed classification wrapper.

There is a version-sensitive tying detail: the saved FLAN config has
`tie_word_embeddings=false`, while the inspected [5.12.1 T5 configuration code](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5/configuration_t5.py#L77)
normalizes that field and separately tracks decoder-output scaling. Its model declares
ties among shared, encoder, decoder and LM-head weights. Record the resolved config
and actual parameter aliases under the ultimately selected library; this source
observation is not a loaded-checkpoint equivalence result. T5Gemma ties its decoder
embedding/output projection; T5Gemma2 declares additional encoder/decoder embedding
ties. Frozen pretrained embeddings and output weights keep that ambiguity outside
the proposed trainable partition, but do not remove the loading-verification requirement.

## Context and feasibility remain open

The earlier suggestion that a 128K context window retires long-document inference
research is incorrect. A position configuration or family capability claim does not
establish effective retrieval/summary fidelity throughout that window, fit on the
available hardware, throughput, or coverage of a full book. Different tokenizers also
cover different amounts of text at the same token count. A future study must record
source coverage, truncation/chunking and target lengths independently of configured
position limits; long-document analysis remains a legitimate separate question.

Before choosing a backbone, authorized feasibility must confirm runtime/version and
weight-loading compatibility; tokenizer/generation semantics; the exact shared/private
trainable and tied-parameter partition; adapter injection/save/reload; task-head and
cached-generation behavior; and actual resource use at declared lengths/batch sizes.
None of these runtime checks has been performed here. Public config access is not
Gemma terms approval, and no account acceptance or gated weight request was made.
