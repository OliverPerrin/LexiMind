# AG News source candidate

**Status: prepared locally, not admitted for research execution.** This reconstructs
one immutable public release without model calls, training, scoring, deduplication,
or changes to legacy processed data. Source text stays in ignored local storage.
The [manifest](../../research/preparation/ag_news_candidate_manifest.json) contains
hashes, counts, schema information and bounded source-row references only.

## Pinned source

The [Hub release](https://huggingface.co/datasets/fancyzhx/ag_news/tree/eb185aade064a813bc0b7f42de02595523103ca4)
is fixed to commit `eb185aade064a813bc0b7f42de02595523103ca4`, configuration
`default`. It exposes `text` and integer `label` fields, without article IDs,
article URLs, timestamps, or an official validation split. The labels are
**World, Sports, Business, Sci/Tech**, with IDs 0–3 in that order. They are not
mapped onto the seven-class legacy topic mix or presented as book genres.

| Provider split | Rows | Download bytes |
| --- | ---: | ---: |
| Train | 120,000 | 18,585,438 |
| Test | 7,600 | 1,234,829 |
| Total | 127,600 | 19,820,267 |

The preparer checks the provider's Parquet SHA-256 values and byte sizes before
using the files. It also captures the pinned Hub card and checks its Git blob hash.
Only these two data files and the small card were acquired. The raw download bound
is 50 MB. PyArrow is reused from the existing isolated preparation environment;
no additional ML stack or model weights are needed.

## Row identity without invented article identity

The lean JSONL records contain only `record_id`, `source_row`, `provider_split`,
`text`, `topic`, and `label_id`. A record ID combines the provider namespace,
immutable source-file SHA-256, and one-based row number. The manifest maps that
reference to the exact file URL and revision.

These are **provider-source-row identities**, not article/document IDs. Different
source rows remain different rows even if their texts match. The absence of
upstream article identity also prevents a claim that news stories, syndication,
or source documents are disjoint merely because row IDs differ.

Shared provider/config/revision/status metadata stays in the manifest instead of
being copied into every row. The prepared JSONL files total 56,071,688 bytes. The
original splits, text and class assignments are preserved exactly; no validation,
model-selection or calibration partition is assigned by reconstruction.

## Observed integrity and overlap

The snapshot has no byte-exact decoded-text duplicate groups. A separate,
conservative check applies Unicode NFC and whitespace collapse while preserving
case, punctuation and diacritics. That check finds:

- **83 duplicate groups**, involving 166 source rows.
- **4 groups crossing the original train/test split**.
- **15 groups containing different class labels**.

All are retained as review evidence, not automatically merged, deleted or relabeled.
Exact text comparison finds no matches to the 3,780 legacy topic rows; this does
not establish equivalence between the two tasks or prove absence of related
stories. Legacy file hashes are recorded and those files remain unchanged.

A separate preparation plan may assign training, selection and calibration roles
or quarantine overlap groups. Such assignments must be versioned separately,
leave the source candidate and official test records intact, and remain subject
to policy review. Near-duplicates and related/syndicated stories are not detected
by the exact or whitespace-normalized checks.

## Source-rights gap

The pinned Hub card marks the license **unknown**. Its dataset description states
research/non-commercial availability, but that statement is not general reuse
clearance. The [original provider page](https://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
could not be independently retrieved during this preparation. The manifest retains
that gap rather than treating the mirror as proof of unrestricted rights.

Research/source-rights review therefore remains unresolved, as do duplicate policy,
article-level identity, and validation/calibration choices. This candidate is not
approved for product redistribution or training. The record status remains
`candidate_prepared_not_admitted`, with `training_authorized: false`.

## Reproduce and verify

```sh
python3 scripts/prepare_ag_news_candidate.py --fetch
python3 scripts/prepare_ag_news_candidate.py
python3 -m pytest tests/test_research/test_ag_news_candidate.py -q
```

The default command requires a verified local cache; only `--fetch` permits the
bounded acquisition. Outputs live under
`data/research_candidates/ag_news/<revision>/raw/` and `prepared/`. Existing
candidate files must match exactly and are never overwritten with different bytes.
The script rejects output locations overlapping legacy data or tracked source text.

Hashing, immutable publication and batched Parquet decoding share the small
`src/research/candidate_io.py` helpers with GoEmotions. Both preparer and helper
hashes are pinned in the manifest. Synthetic tests exercise schema changes,
missing source partitions, duplicate retention, unknown rights, deterministic
reconstruction and safe write boundaries without downloading records or running
models.
