# Model recipes: primary-source review and a narrower LexiMind study

Checked **2026-09-26**. This review reads methods and relevant experimental appendices
from nine primary papers, with code inspection where noted. It is not an exhaustive
survey, a replication, or evidence of a vacant novelty claim. Training, model inference,
benchmark evaluation and paid calls remain paused. Source locators, reading scope,
versions and available PDF hashes are in
[`model_literature.json`](../../research/preparation/model_literature.json).

## What changes the proposed direction

**A comparison of merging and joint training across different kinds of NLP output is
already established research territory.** [Task Arithmetic](https://arxiv.org/pdf/2212.04089v3#page=27) includes a T5 sentiment/QA/
summarization/question-generation/CommonGen setting; [Realistic Evaluation](https://arxiv.org/pdf/2409.18314v1#page=5) includes
cross-lingual classification, QA and summarization. See their rows below. LexiMind
should position its first study as a carefully controlled replication/application,
with any later contribution determined by what the evidence supports.

My proposed question is:

> With a fixed backbone, defined task interfaces and the same total student-training
> allowance, what task utility do joint adaptation and merged specialists retain,
> and what additional selection and deployment costs do they incur?

"Enough" for M1 needs prespecified task-specific tolerances or a stated superiority
criterion. The working plan matches training-window wall time, not total development
cost; additional costs remain separately visible. Book recommendation relevance is
the independent B1 study, not a prerequisite for this benchmark comparison. Neither
question can be answered by a convenient average after seeing results. The
most useful initial distinction is **what resources are available**: training all
experts from scratch versus composing experts that already exist. These answer
different practical questions and need separate cost accounts.

## Evidence matrix

“Not established” below means the inspected source does not establish the particular
LexiMind comparison; it is not a claim that the paper is invalid. Section/page references
refer to the pinned PDF version, counting its first PDF page as page 1.

| Source / version | Backbone, tasks and output interface | Task-specific head policy | Data, tuning and cost boundary / joint reference |
| --- | --- | --- | --- |
| **Task Arithmetic**, [v3, 2023-03-31](https://arxiv.org/pdf/2212.04089v3#page=27) | CLIP vision; T5-base includes IMDB, RACE/QASC, MultiNews, SQuAD question generation and CommonGen. Classification and generated text already coexist. | Open-vocabulary models avoid newly learned private heads; CLIP text-derived classifiers are frozen. New task-head merging is deferred (§2, B.1). | Validation selects scale; the external-GLUE study searches 427 checkpoints. Vision experts use 2,000 steps; D.2 reports joint training with the same hyperparameters. D.6 reuses public NLP checkpoints. Neither is a complete matched-total-cost comparison. **Read:** §2, §4; B.1, D.2–D.3, D.6–D.7, pp2–6, 19, 25, 27–28. |
| **TIES**, [v2, 2023-10-27](https://arxiv.org/pdf/2306.01708v2#page=23) | T5-base/large on seven prompted tasks; T0-3B with (IA)³ on eleven classification/choice datasets; CLIP vision. NLP evaluates candidate label strings, not literary long-form generation. | Shared text-output interface, inferred from rank classification (§C.6); (IA)³ is not a LoRA-factor experiment. | Joint concatenated-data reference exists. T5 uses up to 75,000 steps with early stopping; compute varies by run. Fixed top-20%/scale-1 recipe was selected on PEFT tasks before transfer. Distinguish this from within-suite tuning. **Read:** §3–6, Algorithm 1, C.1/C.4/C.6, pp4–7, 20–23. |
| **DARE**, [v3, 2024-06-13](https://arxiv.org/pdf/2311.03099v3#page=5) | BERT/RoBERTa-base GLUE including regression; separate Llama-2-13B instruction/math/code merging. These are different experimental settings. | [Official GLUE code](https://github.com/yule-BUAA/MergeLM/blob/6d49ad96fd69c92013654b837041b868aa806564/merge_plms_glue.py#L72) excludes classifiers, restores the target classifier and evaluates that target. | Encoder experts: ten epochs and two learning rates; 90/10 train/validation, original GLUE validation used as test. A.4 sweeps drop rates and merge settings. Encoder joint references exist; cheap parameter arithmetic does not include expert creation or selection costs. **Read:** §3–4, A.2–A.4, pp3–7, 14–15; pinned code L72–105. |
| **Fisher merging**, [v2, 2022-08-26](https://arxiv.org/pdf/2111.09832v2#page=5) | BERT/RoBERTa transfer, GLUE with STS-B discretized into 25 classes; domain adaptation and vision robustness. | Explicitly merge shared body and retain task heads (§2, p5); input distribution shift at those heads is acknowledged. | Up to 4,096 training examples estimate Fisher; 50-point coefficient grid uses up to 2,048 validation examples. §3.3 estimates merge/statistic/validation FLOPs against additional RTE fine-tuning. This is primarily reuse/intermediate-task transfer, not a matched joint recipe campaign. **Read:** §2–3.4; C/D, pp3–9, 14. |
| **ATM**, [v4, 2025-08-08](https://arxiv.org/pdf/2411.03055v4#page=6) | ViT-B/16 vision classification; Table 1 has eight tasks. This version supplies no NLP result. | Head implementation is not specified sufficiently in the inspected text. Do not assume the CLIP frozen-head protocol. | PA-ATM revisits training data each round; PH-ATM trains on validation data after merging. §4.1 reallocates ten epochs/task across rounds; Table 1 otherwise compares original baseline settings. It is not a uniform joint-training cost ledger. Exact gradient equivalence assumes one full-batch GD step. Prose omits Cars although Table 1 includes it. **Read:** §2–4, Table 1, §7 tables, pp3–6, 10–11. |
| **DF-Merge**, [NAACL 2025 proceedings](https://aclanthology.org/2025.naacl-long.254.pdf#page=12) | T5-base/large; six PromptSource-formatted QA/paraphrase/completion/coreference tasks, scored by ranking label strings. | Shared generative label interface (§A); not LexiMind's private classification heads. | Six experts use 2,500 steps each; joint training uses up to 25,000, batch 64. Search adds ten random evaluations plus 50 Bayesian iterations, recomputing Fisher using 30 unlabeled validation examples; objective uses labeled validation accuracy. Shared data/backbone does not make these total budgets equal. **Read:** §3–4, limitations/compute, A, PDF pp4–9, 12 (proceedings p4934). |
| **Realistic Evaluation**, [v1, 2024-09-26](https://arxiv.org/pdf/2409.18314v1#page=5) | mT5-xl-lm-adapt across five task/language pairs: QA, NLI, summarization, word sense and answerability; CLIP/Stable Diffusion in separate settings. | Explicit open-vocabulary policy; no private NLP heads. Vision uses a unified frozen text-derived classifier. LoRA image experiments merge effective products, not factors separately (B.3). | Includes pretrained, specialist and joint references; measures held-in and compositional transfer separately. NLP training: up to 5,000 steps, batch 1,024. Table 2 separates merge/statistic FLOPs; validation selects settings. It does not establish equal total expert-plus-search versus joint compute. **Read:** §2–4; B–G, pp2–10, 16–20. |
| **FeatCal**, [v1, 2026-05-13](https://arxiv.org/pdf/2605.13030v1#page=26) | CLIP; prompted GLUE on FLAN-T5-base (full fine-tuning) and large (LoRA); Llama MergeBench extension. FLAN-T5 outputs text, including numeric STS-B. | Existing merged weights are updated layerwise; no new inference modules. Do not treat large-LoRA versus base-full comparisons as a controlled PEFT ablation. | Typically 256 calibration examples/task; three hyperparameters need development/validation selection. Appendix K explicitly excludes final evaluation and offline model search from runtime, and uses different native calibration streams. Joint references are included where available, not matched end-to-end budgets. **Read:** §3–5, A/G/H/K, pp3–10, 15, 20–22, 25–26. |
| **NSC**, [v1, 2026-03-27; CVPR 2026](https://arxiv.org/html/2603.26317v1#S4) | Rank-16 ViT dense classification/regression; Llama-3-8B NLI; LLaVA-1.5-7B QA/captioning. Heterogeneous outputs are an explicit target. | Multimodal projector is merged; vision encoder frozen in LLaVA. Private dense/NLI-head retention is unresolved in inspected paper/README. | Unlabeled validation inputs drive learned coefficients: 100 vision or 500 language/VLM iterations. Some baselines use labeled validation-loss searches. Experts cost 40,000 vision iterations; VLM QA/captioning use five/one epochs. No matched joint arm is established. **Read:** §3–4; A.1–A.4, Algorithm 1; pinned official README. |

## Design decisions for LexiMind

These are recommendations from the comparison above, not conclusions of a LexiMind
experiment.

**Define two resource regimes.** In the first, LexiMind owns all task data and must
pay for every specialist and joint-model update. Count expert creation, validation,
merge search, calibration, failed attempts and final reporting separately, then report
their total. In the second, compatible experts are available already: report incremental
composition cost alongside their known acquisition/history, without pretending the
experts were free to create. Do not place these regimes in one undifferentiated ranking.

**Decide the head contract before the backbone contest.** The existing custom model
has a shared encoder, private emotion/topic heads, and a summarization decoder.
For an initial application study, a coherent option is to merge only common encoder
parameters or adapter deltas, retaining each specialist's private head/decoder. Compare
that with joint training using the identical task interfaces, initialization, labels and
allowed modules. This is a proposal, not the only valid design. If head recalibration
is allowed, give it a separately declared data/compute allowance in every relevant arm.
An all-text-output variant is a separate interface ablation, not a silent replacement.

**Avoid changing representation and training method simultaneously.** Use one verified
base revision first. Neither T5Gemma integration nor the current custom T5 implementation
should be assumed equivalent to the vanilla T5 checkpoints in these papers. A replication
may use an upstream implementation; a book application may retain the custom multi-head
system. Record that distinction rather than pooling their results.

**Make the LoRA storage constraint explicit.** A weighted sum of effective deltas
`sum_i lambda_i B_i A_i` is generally not equal to multiplying separately averaged
factors. Its rank can grow with the number of experts. Pin whether the deployed result
is materialized into base weights or recompressed to a fixed rank, and count that cost.
Task-head and decoder storage belong in the deployment total too.

**Treat selection as part of the method.** A fixed published TIES configuration and
one selected on LexiMind validation data are distinct arms. Give methods a declared
validation access policy and selection allowance; archive every tried setting and its
cost. DARE also needs merge-randomness seeds independent of training seeds. Freeze
which inputs may be used for unlabeled calibration: unlabeled does not mean absent
or free data, and using the final test inputs creates a transductive setting.

**Keep task retention separate from book utility.** Report each task and its worst
regression against an agreed reference before considering an aggregate. Add independent
book-query relevance judgments only after catalogue/work identity and annotation quality
are adequate. Neither Reddit emotion F1 nor prompted GLUE accuracy validates literary
atmosphere labels. One dataset per output family cannot identify a causal output-type
effect; domain, label quality, task size and output interface remain confounded.

## A first sequence, after experiments are explicitly resumed

1. Reproduce one small published merge setting or a sharply specified simplification,
   checking data/head/selection contracts before trying new methods. This is a software
   and protocol feasibility gate, not a new headline result.
2. Compare a joint arm and independently trained task specialists under the declared
   total student-training allowance. Reuse those same specialist checkpoints for
   task-arithmetic and TIES merges; count additional selection and merge costs
   separately. An unadapted backbone with random private classifiers is not a useful
   reference. A frozen-encoder, trained-head control would need its own specified
   training allowance. Do not charge reused experts twice within a recipe or omit
   their creation from its equivalent cost.
3. Examine per-task retention and failures. If the merged encoder disrupts private-head
   inputs, test a budgeted head-calibration diagnostic on development data; keep its
   downstream relevance evaluation separate.
4. Only then add one motivated extension: DARE for sparsification, or a calibration
   method for a demonstrated feature/head mismatch. ATM requires repeated data access;
   it is a different regime. Distillation and RL remain deferred until this comparison
   is reliable and their full costs/data permissions are specified.

There is no promised result, publication outcome, universal recipe ranking or training
date. This review supports a narrower study; it does not close the separate annotation,
licensing, data-split, compute, protocol-freeze or experiment-authorization gates.

## Reading limits and reproducibility notes

Eight pinned PDFs were downloaded to a temporary reading directory; the NSC methods
and appendix were read in its versioned arXiv HTML/PDF viewer after a local download
was truncated. Relevant DF-Merge, TIES, ATM, Realistic Evaluation and FeatCal pages were rendered for layout checks.
No downloaded experiment code was executed, and no model weights or benchmark datasets were loaded. Official DARE code
and the NSC README were inspected at pinned Git commits, which are distinct from the
papers' publication revisions. Bibliographic dates use version dates, not search-engine
crawl dates. ATM's earlier abstract must not be substituted for its v4 experiments.

The nine-paper set was selected for direct method/interface/cost relevance from the
initial review and targeted primary-source searches. This bounded set does not establish
absence of closer work. Remaining source-specific uncertainties are retained in the JSON
rather than filled with assumptions; private-head and full development-budget details
must be resolved before choosing a reproduction target.
