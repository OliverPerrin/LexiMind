# Research: is model merging worth it once you pay for the experts?

LexiMind's study compares four ways of getting one model that handles several book
classification tasks, under one matched training budget:

| Arm | What is trained | Deployed encoders |
| --- | --- | --- |
| Joint | One LoRA adapter on all tasks, with a head per task | 1 |
| Specialists | One LoRA adapter per task; the budget is split across them | one per task |
| Task arithmetic | The same specialists, merged by summing their weight deltas | 1 |
| TIES | The same specialists, merged with trimming and sign election | 1 |

Merging papers usually start from experts that already exist. Here the cost of
training the specialists is charged to the merged arms, so the comparison asks
whether merging beats simply training jointly with the same compute. This is a
controlled replication on book data, not a new merging method.

The protocol is fixed in [`merge_study.json`](../../configs/research/merge_study.json)
and implemented in [`src/research/merge_study.py`](../../src/research/merge_study.py).

## Design

- **Tasks.** Book genre (15 labels) and topic (23 labels) from the
  [Blurb Genre Collection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)
  (BGC), predicted from title and blurb. The third task is narrative emotion from
  [CR4-NarrEmote](https://aclanthology.org/2025.emnlp-main.493/): given a passage
  and a highlighted character, predict the distribution of annotators' emotions.
  The provider maps each annotator's free-text answer to one of eight NRC
  categories (`NRC_TopEmotion`). The mapping is automatic, so the labels are only
  partly human; a strict majority of annotators agree in 69% of multi-annotator
  pairs. This task differs from BGC in both input domain and output type: one
  softmax distribution instead of independent labels.
- **Duplicate-aware splits.** 849 groups of matching BGC records (shared blurbs,
  ISBNs, or title and author) crossed the official splits, including 274 identical
  blurbs. Groups are kept together: 58,889 train, 14,630 dev and 18,326 test
  records, with 49 ambiguous records quarantined. CR4's 38,209 passage-character
  pairs are grouped by linked source documents and split 24,455 / 6,111 / 7,643.
  Dev is halved by group:
  `dev_select` picks thresholds, regularisation and merge coefficients, while
  `dev_report` is used for development reporting. The test split stays sealed
  until the final run.
- **Labels and metrics.** BGC publisher labels are treated as complete, so an
  absent label is a negative. We report micro-F1, macro-F1 and macro average
  precision. Emotion reports accuracy and macro-F1 against the unique majority
  label, plus annotator agreement: the share of annotators who chose the predicted
  category.
  Labels with fewer than 5 positives in the evaluated split are left out of the
  averages and listed. Each task uses one decision threshold, chosen on
  `dev_select` by micro-F1. The mapped 15/23-label vocabularies differ from BGC's
  146 raw categories, so numbers are not directly comparable with published BGC
  results.
- **Model.** The from-scratch FLAN-T5-base encoder (`src/models/`) loaded with
  pinned weights. LoRA rank 16, alpha 32, sits on every attention and feed-forward
  projection; each task has a mean-pooled linear head trained with sigmoid
  cross-entropy. Every arm of a seed starts from an identical base and heads,
  which merging requires.
- **Budget.** Examples processed (updates × batch 32). Each specialist gets 3,680
  updates (two BGC epochs). The joint adapter gets their total, 11,040 updates,
  drawing two BGC batches for every CR4 batch; each BGC batch supervises genre and
  topic in one pass. Merges reuse the specialists and choose a scale on
  `dev_select`, by mean macro-AP (genre, topic) and macro-F1 (emotion). Task
  arithmetic searches 0.4–1.0; TIES keeps the top 20% of each delta and searches
  0.8–1.5. Each search is reported as selection cost.
- **Uncertainty.** Five seeds (17, 29, 41, 53, 67) and 1,000 bootstrap resamples
  over book groups. Intervals cover the seed-averaged metric; seed spread is
  reported separately.
- **Baseline.** TF-IDF unigrams and bigrams with one-vs-rest logistic regression.
  C is tuned on `dev_select` macro-AP over 1–4,096; it peaks inside the grid at 256.

## Status

| Step | State |
| --- | --- |
| 1. Duplicate-aware data and closed-world labels | Done |
| 2. Fair single-task baseline: does LoRA beat tuned TF-IDF? | TF-IDF done; a seed-17 neural validation is running on the M5 |
| 3. CR4 emotion task and the four arms | Implemented and smoke-tested end to end; five seeds wait for the RTX 4070 |
| 4. Final test-split run and write-up | After the five-seed runs |

The five-seed specialists also answer step 2, since each is a single-task LoRA
model at the study budget. Roughly 1.5 hours per seed on the RTX 4070, extrapolated
from the earlier pilots' throughput; this recipe hasn't been timed there yet.

## Results so far (`dev_report`)

| Task | System | Micro-F1 | Macro-F1 | Macro-AP |
| --- | --- | --- | --- | --- |
| Genre (7,369 books) | Tuned TF-IDF | 0.685 [0.671, 0.696] | 0.634 [0.610, 0.652] | 0.688 [0.667, 0.709] |
| Topic (7,369 books) | Tuned TF-IDF | 0.709 [0.698, 0.722] | 0.688 [0.667, 0.706] | 0.749 [0.733, 0.769] |

| Task | System | Accuracy | Macro-F1 | Annotator agreement |
| --- | --- | --- | --- | --- |
| Emotion (2,904 passages) | Tuned TF-IDF | 0.533 [0.514, 0.554] | 0.484 [0.461, 0.507] | 0.467 [0.452, 0.483] |

Brackets are 95% group bootstrap intervals. Erotica has 2 `dev_report`
positives and is excluded from genre averages. For emotion, always predicting the
most common majority label would score about 0.26 accuracy. Lexical C peaks
inside its grid: 256 for genre and topic, 4 for emotion.

## Commands

```sh
python scripts/research.py merge-study prepare          # rebuild splits from the pinned BGC archive
python scripts/research.py merge-study lexical --output outputs/merge-study/lexical
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for seed in 17 29 41 53 67; do
  for task in genre topic emotion all; do
    python scripts/research.py merge-study train --tasks $task --seed $seed --device cuda \
      --output outputs/merge-study/$task-$seed
  done
  for method in task_arithmetic ties; do
    python scripts/research.py merge-study merge --method $method --device cuda \
      --output outputs/merge-study/$method-$seed outputs/merge-study/{genre,topic,emotion}-$seed
  done
done
python scripts/research.py merge-study report outputs/merge-study/*/ \
  --output research/results/merge_study.json
```

`--tasks all` trains the joint adapter. On the M5, add `--microbatch 4`.
It accumulates gradients within the protocol batch of 32 and fits the repo's 35%
MPS memory cap; one 3,680-update run takes about 2.6 hours there. The BGC archive
and partition overlay come from `python scripts/research.py bgc-groups` and
`book-partitions`. The CR4 file is downloaded from the URL in the protocol, which
also pins its MD5.

## Earlier pilots, September–October 2026

These used a different design that treated a missing label as unknown rather
than negative, so they could only report recall at k. They remain in
[`research/results/`](../../research/results/) with their configs and audits; the
detailed write-ups are in git history up to commit `acf13cb`.

- **Native runtime.** The from-scratch FLAN-T5 port matches the reference model,
  LoRA checkpoints move between the M5 and the RTX 4070, and a merged checkpoint
  reloads through ordinary inference.
- **Genre/topic recall, 4,096 → 16,384 training books.** Rank-4 LoRA on the
  attention queries and values, trained with a per-facet softmax, beat a frozen
  encoder. At 4,096 books it fell short of a TF-IDF centroid baseline on genre and
  roughly matched it on topic. At 16,384 books it edged past that baseline on
  group recall but not on label recall. Those choices handicapped
  the neural model, which is why the current protocol uses full-projection LoRA,
  sigmoid cross-entropy and a tuned logistic-regression baseline.
- **Loss weighting.** Inconclusive: the two-seed difference between identical runs
  (up to 6.9 points) exceeded the measured effects (−3.0 to +2.5 points).
- **Reinforcement learning.** In matched pilots, supervised cross-entropy beat RL
  on book-text reconstruction in both seeds, and only 8 of 128 RL groups produced
  updates. RL is out of scope: every task here has direct labels.

Undergraduate results remain [historical](../RESULTS.md) and have not been reproduced.
