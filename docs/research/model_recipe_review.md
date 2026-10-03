# Model recipes: reference findings

The working experiment is in [study_decisions.md](study_decisions.md); current
training-data choices belong in [dataset_decisions.md](dataset_decisions.md).
[model_literature.json](../../research/preparation/model_literature.json) preserves
nine primary sources, versions, reading scope, claim locators and available hashes.
This is a bounded methods review, not an exhaustive novelty search or a replication.

## What prior work establishes

Merging across different NLP outputs and comparisons with joint learning already
exist. LexiMind's useful question is the controlled recipe comparison, not priority
for combining classification and generation.

| Primary source | Relevant distinction |
| --- | --- |
| [Task Arithmetic, v3](https://arxiv.org/pdf/2212.04089v3) | T5 classification, QA and generation coexist; public expert reuse and validation selection are not a matched development-cost campaign. |
| [TIES, v2](https://arxiv.org/pdf/2306.01708v2) | Sign conflicts and trimming; prompted/ranked label interfaces differ from private classifier heads. |
| [DARE, v3](https://arxiv.org/pdf/2311.03099v3) | Dropped/rescaled deltas; inspected GLUE code retains target classifiers. Cheap merging does not include expert creation. |
| [Fisher merging, v2](https://arxiv.org/pdf/2111.09832v2) | Shared-body merging with retained heads; statistics and coefficient selection consume data/compute. |
| [ATM, v4](https://arxiv.org/pdf/2411.03055v4) | Repeated tuning/merging uses training or validation data; inspected experiments are vision, not NLP. |
| [DF-Merge](https://aclanthology.org/2025.naacl-long.254.pdf) | T5 label ranking; Bayesian search and Fisher estimation are additional work, and joint/expert step budgets differ. |
| [Realistic Evaluation, v1](https://arxiv.org/pdf/2409.18314v1) | Task/language transfer and heterogeneous outputs; open-vocabulary interfaces avoid new private NLP heads. |
| [FeatCal, v1](https://arxiv.org/pdf/2605.13030v1) | Calibration can repair merged representations; calibration/search access belongs in accounting. |
| [NSC, v1](https://arxiv.org/html/2603.26317v1) | Heterogeneous outputs and learned merge coefficients; inspected evidence does not establish a matched joint arm. |

## Consequences for the accepted comparison

- Compare joint adaptation, specialists, task arithmetic and TIES from the same
  base, task interfaces, label order and matched private initialization.
- Match the declared **student-training window B**; specialists share B across
  tasks. Report selection, merging and other development costs separately.
- Reuse exactly the same specialists for both merges. Recipe-equivalent cost
  includes their creation; project-unique cost counts shared runs only once.
- Merge effective encoder deltas, not separately averaged LoRA factors. Retain
  specialist private heads/decoder adapters and count their deployment costs.
- A changed encoder can disrupt its retained head. Head adaptation or rank
  recompression needs a separate declared condition, not hidden postprocessing.
- Random private heads are not a meaningful unadapted baseline. A frozen-encoder,
  trained-head control would have its own training allowance.
- Report per-task changes before aggregates. One dataset per output type cannot
  isolate output type from domain, label quality, size or interface.

Keep book relevance separate: benchmark retention does not validate book metadata
or reader atmosphere, and book judgments are not required to run the independent
model comparison. Existing-expert reuse is a different cost regime from creating
all experts for this study. Detailed source-specific qualifications remain in the
[JSON evidence register](../../research/preparation/model_literature.json).

## RL extension: reviewed through 2 October 2026

The [18-source register](../../research/preparation/rl_methods.json) records exact
versions, reading scope and implementation limits. RL is a separate, disabled
extension; the four supervised/merge controls remain the first comparison.

| Direction | Decision |
| --- | --- |
| Post-training | Start with [Dr. GRPO](https://arxiv.org/html/2503.20783v2) and independently checked rewards. DAPO-style token reduction and GSPO are optional loss variants. [September BPO](https://arxiv.org/html/2609.15987v1) is experimental, with explicit clipping settings. |
| Offline preferences | DPO is implemented for genuine chosen/rejected pairs. CR4 disagreement and missing book labels are not preference pairs. |
| Continued pretraining | Prepare [RPT](https://arxiv.org/html/2506.08007v1) token-boundary prefix verification and [RLP](https://arxiv.org/html/2510.01265v2) information-gain/EMA primitives. Neither establishes that RL should replace initial cross-entropy training for this small encoder-decoder. |
| Recent evidence | [June RL excursions](https://arxiv.org/html/2606.04272v1), [July/August pretraining-to-RL analysis](https://arxiv.org/html/2607.16097v2), and [September initialization work](https://arxiv.org/html/2609.28145v2) make pretrained capability, initialization and total compute explicit controls. Their domains and scales differ from book fields. |

The native rollout/scoring adapter uses the existing trainer; no parallel training
framework or launch scripts were added. Reward/tokenizer/behavior-policy revisions,
EOS masks and complete-label requirements are checked. Uniform-reward or fully
clipped batches skip optimizer updates. Replay, asynchronous critics and learned
semantic graders remain deferred. The [bounded M5 pilot](../../research/results/macbook_pilot_20261002.json)
measures local execution with a sparse continuation reward. CUDA feasibility and
representative model quality remain unmeasured; recent papers do not establish a
LexiMind improvement.

The [two-seed local comparison](../../research/results/book_denoising_20261002.json)
tests exact word reconstruction, inspired by [T5 denoising](https://arxiv.org/abs/1910.10683).
Both branches share warm-start adapters and 64 prompts, but RL generates four
responses per prompt, so compute is not matched. Continued CE reduced validation
content NLL more in both seeds; sparse rewards and poor exact recovery support
improving the supervised task/baseline before increasing RL compute.
