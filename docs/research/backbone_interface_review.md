# Backbone compatibility reference

The custom FLAN/T5 implementation in [src/models](../../src/models/) remains an
active part of LexiMind. This upstream compatibility review does not replace it.
Runtime/backbone selection is still open; [study_decisions.md](study_decisions.md)
contains the working adaptation interface.

[backbone_candidates.json](../../research/preparation/backbone_candidates.json)
retains config revisions, source hashes, module patterns and unresolved checks.
The inspected Transformers 5.12.1 files match upstream commit `ddb849abe009d1089e6c691bfc897f27211c663c`;
the software-test pin is 5.17.0. Neither is a selected research runtime. This review
loaded no model weights and measured no quality, memory or throughput.

## Pinned upstream observations

| Candidate/config | Important compatibility facts |
| --- | --- |
| [FLAN-T5-base](https://huggingface.co/google/flan-t5-base/blob/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/config.json) | 12/12 layers, width 768, vocabulary 32,128; relative-position buckets; saved start/pad 0, EOS 1. `n_positions=512` is not a demonstrated quality or hard-length bound. |
| [T5Gemma b-b-ul2](https://huggingface.co/google/t5gemma-b-b-ul2/blob/97ea9b7e92738bb57437867277ae38e65345b8d7/config.json) | Vocabulary 256,000; sliding/full attention; saved EOS [1,107]; nested BOS defaults to 2 in inspected code. Similar widths do not imply T5 weight compatibility. |
| [T5Gemma2 270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m/blob/7c38f16641f455ef0685b18431faf1b17722d5a1/config.json) | 18/18 text layers, width 640, grouped KV, vocabulary 262,144 and a vision configuration. Pinned position fields are 32,768, not proof of effective “128K” context. |

Gemma metadata is readable but does not establish weight access or acceptance of
its terms. Text-only execution also does not imply vision parameters vanish.
Do not route another family through the custom T5 mapper by changing a model name.

## Shared/private contract to verify in the actual implementation

- Freeze original base weights, embeddings, output projection, norms and aliases.
- Shared updates are encoder-attention LoRA; private components are task-specific
  pooling/classifiers; decoder adapters only if a generation task is admitted.
- Match task interfaces and private initialization within each backbone/seed.
  Merge effective deltas and retain the corresponding specialist private modules.
- Derive complete module/parameter allowlists from the **actual custom/wrapped
  module tree**. Upstream name patterns in the JSON are references, not verified
  custom-model targets. Broad suffix matching can accidentally include decoders.
- Verify tying, save/reload and the `(alpha/r) B @ A` convention before merging;
  separately averaged factors are not equivalent to an effective-delta merge.

## Tokenization, decoding and feasibility

Pin tokenizer/processor and generation settings separately; they were not resolved
by the config review. T5 uses its decoder-start setting; inspected Gemma shifts
use decoder BOS. Preserve family-specific EOS lists, padding and target masking.
[Primary source: T5 shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5/modeling_t5.py#L582),
[T5Gemma shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma/modeling_t5gemma.py#L595),
[T5Gemma2 shift](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/t5gemma2/modeling_t5gemma2.py#L707).

Saved T5 tying settings and inspected library normalization differ; record resolved
config and actual aliases rather than assuming checkpoint equivalence. Context
fields likewise establish neither book coverage nor device fit. Loading parity,
masking, head/generation behavior and declared-length resource use remain actual
integration checks; the [admission contract](admission_contracts.md) records their evidence requirements.
