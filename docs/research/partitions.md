# Source partition preparation

These are deterministic **candidate assignments**, not an admitted dataset or a
frozen protocol. No model is loaded and no quality score is calculated. Original
source files, rows, labels and official test membership remain unchanged.

| Source | Train | Model selection | Calibration | Test |
| --- | ---: | ---: | ---: | ---: |
| GoEmotions | 43,410 | 2,745 | 2,681 | 5,427 |
| AG News | 107,876 | 6,151 | 5,973 | 7,600 |

GoEmotions splits only its original validation set into selection/calibration at
50/50 hash thresholds. AG News has no provider validation set, so its original
training set uses 90/5/5 thresholds. These are group probabilities, not exact row
quotas or label stratification. The salt is `leximind-source-partitions-v1`.

Groups are the SHA-256 of NFC-normalized text with collapsed whitespace, retaining
case and punctuation. The assignment hashes the salt and group digest. Within an
original source split, the same normalized text always gets the same role,
regardless of row order, record ID, labels or model outputs. Label support is
reported after assignment; the policy does not search seeds to improve coverage.

Official cross-split overlap remains visible: GoEmotions has **71** normalized
text groups spanning assigned roles; AG News has **4**. No repeated group within
one original source split crosses the new development/calibration boundaries.
These assignments make no unseen-text claim. AG News IDs identify pinned source
rows, not original articles or a deduplicated news-event population.

The outputs under ignored `data/research_candidates/*/partitions-v1/` are compact
`assignments.jsonl` indices: record ID, source split/row, assigned role and text
group hash. They contain no copied source text. The committed
[GoEmotions report](../../research/preparation/goemotions_partition_manifest.json)
and [AG News report](../../research/preparation/ag_news_partition_manifest.json)
pin the source candidate manifest, exact source files, implementation, assignments,
counts and overlap. A temporary SQLite index bounds memory while checking identities
and repeated text. Candidate corruption or conflicting existing assignments fails
before publishing a new assignment file.

```sh
python scripts/prepare_research_partitions.py research/preparation/goemotions_candidate_manifest.json --output-dir data/research_candidates/go_emotions/add492243ff905527e67aeb8b80c082af02207c3/partitions-v1 --report research/preparation/goemotions_partition_manifest.json
python scripts/prepare_research_partitions.py research/preparation/ag_news_candidate_manifest.json --output-dir data/research_candidates/ag_news/eb185aade064a813bc0b7f42de02595523103ca4/partitions-v1 --report research/preparation/ag_news_partition_manifest.json
```

The sources must already be prepared locally; there is no network fallback.
Rights/source-use review, an explicit overlap/generalization decision, label and
threshold policy, reviewed task selection and protocol freeze still precede any
admission. Assignment indices do not silently replace the core trainer's existing
JSONL inputs or its legacy validation helper. A future admitted export must use
these recorded assignments explicitly; the current default data paths are unset.
