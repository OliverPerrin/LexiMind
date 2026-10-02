# Retained source reconstructions

These are **optional controls and historical evidence**, not the primary book-field
training sources. Current choices are in [dataset decisions](dataset_decisions.md).
Original processed data, source files and model inputs were not rewritten.

| Source | Verified snapshot | Remaining limits |
| --- | --- | --- |
| [GoEmotions manifest](../../research/preparation/goemotions_candidate_manifest.json) | 54,263 provider comment IDs; 53,963 unique legacy matches, 300 ambiguous. V2 JSONL is 14,745,654 bytes, 67.4% smaller than V1. | Comment emotion, not book mood. Original duplicates and annotation differences retained. V1/raw bytes preserved. |
| [AG News manifest](../../research/preparation/ag_news_candidate_manifest.json) | 127,600 original rows; 4 labels; 83 normalized duplicate groups, 4 crossing official splits. | Row IDs are pinned file/row references, not article IDs. Provider card license is unknown; no book-genre or product-use admission. |
| [arXiv manifest](../../research/preparation/arxiv_source_manifest.json) | 215,913 original IDs; train 203,037 / val 6,436 / test 6,440. One verified 3.624 GB ZIP plus 171.5 MB metadata index; no second 15.1 GB text extraction. | 79,273 old-style ID forms unresolved; versions unstated; 120 empty articles, 3 empty abstracts, 47 section mismatches. Two cross-split article and seven abstract duplicate groups. Article rights/versions remain unverified. |
| [Legacy inventory](../../research/preparation/data_inventory.json) / [audit](../../research/preparation/data_audit.json) | 156,796 rows in 12 JSONL files; 73 normalized-input groups span splits. | Missing parent/source identity; 18,753 literary pairs remain quarantined. Hashes do not repair the old joins. |

## Rebuild only when needed

Candidate preparers use verified local caches; `--fetch` explicitly enables source
acquisition. The auditor separately reads `data/processed`. Raw candidate text stays
in ignored `data/research_candidates/` and never enters
the preparation manifest. arXiv re-indexing is substantial disk/decompression work.

```sh
python scripts/research.py status --check-archive
python scripts/research.py goemotions
python scripts/research.py ag-news
python scripts/research.py arxiv
python scripts/research.py data-audit
```

Preparers pin provider files, shared helper/script hashes and stable source
identities. Source/data/index files are created once or byte-verified. Some manifests,
audit reports and partition reports are deliberately regenerated; review their diffs
before refreshing preparation hashes. The manifests carry file sizes, hashes, terms,
normalization rules, overlap counts and commands' input boundaries.

## Candidate role assignments

| Dataset | Train | Model selection | Calibration | Test |
| --- | ---: | ---: | ---: | ---: |
| GoEmotions | 43,410 | 2,745 | 2,681 | 5,427 |
| AG News | 107,876 | 6,151 | 5,973 | 7,600 |

The salt `leximind-source-partitions-v1` hashes NFC/whitespace-normalized text.
GoEmotions splits original validation 50/50; AG News splits original train 90/5/5.
These are group thresholds, not exact row quotas. Official tests stay intact;
71 GoEmotions and 4 AG News text groups still cross assigned roles. No within-source
split duplicate group crosses the newly assigned boundaries. No unseen-text claim.

[GoEmotions assignments](../../research/preparation/goemotions_partition_manifest.json)
and [AG News assignments](../../research/preparation/ag_news_partition_manifest.json)
pin compact record/row indices rather than copying text. Reproduce with:

```sh
python scripts/research.py source-partitions research/preparation/goemotions_candidate_manifest.json --output-dir data/research_candidates/go_emotions/add492243ff905527e67aeb8b80c082af02207c3/partitions-v1 --report research/preparation/goemotions_partition_manifest.json
python scripts/research.py source-partitions research/preparation/ag_news_candidate_manifest.json --output-dir data/research_candidates/ag_news/eb185aade064a813bc0b7f42de02595523103ca4/partitions-v1 --report research/preparation/ag_news_partition_manifest.json
```

Assignments are proposals, not admitted training inputs. The trainer does not
silently substitute these indices for explicit reviewed JSONL split directories.
