# Related work — review history

**26 September update:** the initial abstract-level review below is preserved as
history. Continue with the [model methods review](research/model_recipe_review.md),
[book-discovery review](research/book_discovery_review.md), and
[current study decisions](research/study_decisions.md). Those reviews resolve the
framing as controlled replication/application; they do not establish novelty or
close the remaining data, integration, budget and human-evidence requirements.

# Initial literature gate, 22 September

**Checked 2026-09-22. Status: OPEN; no novelty claim.** This is a bounded initial
review of primary abstracts, publication records and official model documentation.
It is not a systematic review or a full-paper experimental matrix. No experiments
were run. The narrow scope below is a proposed useful study, not an assertion that
others have not studied it.

## Sources and implications

| Primary source | What the source supports | Consequence for LexiMind / unresolved detail |
| --- | --- | --- |
| Aribandi et al., [ExT5](https://arxiv.org/abs/2111.10952) (initial 2021; ICLR 2022) | Multitask pretraining with ExMix's 107 NLP tasks; transfer and task-family composition are studied. | Broad “MTL helps across heterogeneous tasks” is not a sufficient contribution. Full training-budget/ablation comparison remains to be read. |
| Ilharco et al., [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) (initial 2022) | Task vectors are fine-tuned-minus-base weights, and addition can combine capabilities. | Use a simple task-vector baseline and common base identity. Does not establish that our tasks or budgets will merge successfully. |
| Yadav et al., [TIES-Merging](https://arxiv.org/abs/2306.01708) (2023) | Addresses redundant changes and sign disagreement; evaluates varied merging settings. | Candidate merge baseline; inspect exact backbone, task-head and validation-selection conventions before porting. |
| Yu et al., [Language Models are Super Mario / DARE](https://arxiv.org/abs/2311.03099) (initial 2023) | Drops and rescales parameter deltas and combines homologous fine-tuned models. | Optional later baseline. Authors' low-cost merge claims do not eliminate the cost of training or selecting experts. |
| Zhou et al., [ATM, v4](https://arxiv.org/abs/2411.03055v4) (2025 revision) | Connects task vectors to multitask gradients under single-epoch full-batch assumptions; alternates tuning and merging, with vision experiments in this revision. | Directly relevant to framing merging versus joint learning. Earlier indexed abstracts describe a broader NLP/computational comparison; do not conflate versions. Full v4 methodology and budget definitions need reading. |
| Lee et al., [Dynamic Fisher-weighted Model Merging via Bayesian Optimization](https://aclanthology.org/2025.naacl-long.254/) (NAACL 2025) | Combines model-level scaling with parameter-level importance and discusses the gap from multitask fine-tuning. | Comparisons to joint learning already exist. Account for coefficient-search cost; inspect matched-compute details before a gap claim. |
| Gu et al., [FeatCal](https://arxiv.org/abs/2605.13030v1) (May 2026 preprint) | Proposes calibration of merged models using a small calibration set; includes FLAN-T5-base GLUE results. | Recent relevant alternative to retraining. Calibration data and cost belong in the comparison; results are authors' claims, not independently replicated here. |
| Du and Lin, [Dynamic Model Merging Made Slim](https://arxiv.org/abs/2605.18904v1) (May 2026 preprint) | Studies parameter allocation and storage/accuracy trade-offs for dynamic merging. | Static merged models and dynamic expert systems have different deployment costs. State the intended single-model constraint precisely. |
| Demszky et al., [GoEmotions](https://aclanthology.org/2020.acl-main.372/) (ACL 2020) | English Reddit comments annotated for 27 emotions plus neutral. | The source establishes the annotation task, not validity as book-atmosphere labels. An applied book-mood evaluation requires separate human judgments. |
| Google, [T5Gemma 2 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/t5gemma-2/) (December 18, 2025) and [implementation README](https://github.com/google-deepmind/gemma/blob/main/gemma/research/t5gemma/README.md) | Encoder-decoder models with architecture changes; the T5Gemma 2 announcement describes pretrained releases and extended context. | A candidate backbone needs its own compatible implementation, pinned revision and measured feasibility. The current custom T5 weight mapping is not an established integration. |

## Gate assessment

**The broad novelty claim is not established.** Prior work already studies multitask
transfer, merging versus joint-learning relationships, heterogeneous capability
combination, and cost/parameter trade-offs. The August plan's suggestion that this
combination is unclaimed is a hypothesis to investigate, not an accepted verdict.
A negative or sparse search result is not evidence of absence.

A useful near-term question is an **applied, reproducible comparison on book discovery**:
under a stated resource budget and shared catalogue, do validated topic/genre/mood
features improve recommendations over a plain-text baseline? The model sub-study can
start with joint training, task specialists, and adapters merged from those specialists.
Its value does not depend on claiming a new merging algorithm. We have not yet reviewed
book-recommendation, emotion-aware retrieval or multi-task recommender literature
sufficiently to claim novelty for that application either.

No paper here licenses a blanket inference that generation “needs joint training” or
classification “tolerates merging.” Test-suite outcomes may depend on initialization,
capacity, label space, domain, data size, calibration and merge selection. One dataset
per output type would leave those causes confounded.

## Search record and remaining work

Initial queries included exact titles for ExT5, task arithmetic, TIES and DARE, plus
`model merging multi-task compute budget`, `model merging joint training 2025 2026`,
and the official T5Gemma documentation. Primary records used above were checked
through arXiv, ACL Anthology, Google and the authors' implementation repository.
Search engines sometimes returned obsolete ATM abstracts; the current v4 record was
opened to distinguish its narrower claims.

Before the gate can close:

1. Read full methods/appendices and code for the closest papers; build a matrix of
   tasks/output representations, datasets, backbone/revisions, expert training cost,
   total tuning budget, calibration data, merge selection and deployment size.
2. Follow citations and recent work in both model merging and book/emotion-aware
   recommendation. Include negative transfer, domain shift and benchmark contamination.
3. Write down the closest existing comparison and a precise remaining question—or
   explicitly choose a replication/application study. A study can be useful without
   a novelty claim or a submission target.
4. Update the draft protocol and resolve its data/compute decisions before any new
   experiment; keep the user's pause in effect until they resume research.
