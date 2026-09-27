# GoEmotions provenance reconstruction

**Status: a new local candidate is prepared, but is not admitted for research.**
No training, model execution, evaluation, annotation scoring, automatic
deduplication, or split reassignment was performed. Existing emotion files in
`data/processed/emotion/` were read for comparison and remain unchanged.

The [candidate manifest](../../research/preparation/goemotions_candidate_manifest.json)
contains source URLs, exact revision and file hashes, label order, aggregate
counts, and bounded hashed references. Reddit text and opaque source comment IDs
are retained only in ignored local candidate files, not in committed reports.
No Reddit endpoints, user profiles, or account information were queried.

## Source and acquisition evidence

The source is the public, ungated
[Google Research Datasets Hub release](https://huggingface.co/datasets/google-research-datasets/go_emotions/tree/add492243ff905527e67aeb8b80c082af02207c3),
fixed at revision `add492243ff905527e67aeb8b80c082af02207c3`, configuration
`simplified`. Its pinned card declares `text`, integer-list `labels`, and string
`id` fields. The [official Google README](https://github.com/google-research/google-research/blob/master/goemotions/README.md)
describes the filtered release's comment identifier and original partitions.

Before acquisition, the provider metadata established the following bounded files:

| Split | Rows | Download bytes |
| --- | ---: | ---: |
| Train | 43,410 | 2,767,678 |
| Validation | 5,426 | 350,063 |
| Test | 5,427 | 346,630 |
| Total | 54,263 | 3,464,371 |

Every downloaded Parquet file was checked against its provider-published SHA-256
and byte size. Only these three data files were acquired: no full/raw annotation
release, user metadata, or model weights. Small copies of the pinned Hub card,
Google README, and upstream license are retained alongside their URLs and hashes.
The Google documentation URLs use `master`; their captured hashes identify what
was inspected, without pretending those URLs are immutable revisions.

The Hub card declares Apache-2.0 and the
[upstream repository carries Apache-2.0](https://github.com/google-research/google-research/blob/master/LICENSE).
Those are source declarations, not a blanket decision about every possible use of
Reddit text. The official README also records biases, limited representativeness,
and potentially problematic content. Rights, bias, and intended-use review remain
an admission gate.

## New local artifact

The candidate lives under:

```text
data/research_candidates/go_emotions/add492243ff905527e67aeb8b80c082af02207c3/
  raw/acquisition.json
  raw/simplified/{train,validation,test}-00000-of-00001.parquet
  raw/documents/
  prepared/{train,validation,test}.jsonl  # preserved v1 files
  prepared/labels.json                   # preserved v1 label map
  prepared_v2/{train,validation,test}.jsonl
  prepared_v2/labels.json
```

The current manifest selects **record version 2**, under `prepared_v2/`. Its
JSONL files total **14,745,654 bytes**, down from the preserved v1 files' 45,230,603
bytes: **67.4% less serialized data**, with no text, label, comment-ID, or partition
loss. Version 1 remains unchanged locally and its hashes are retained in the
manifest.

Version 2 rows contain exactly `text`, `emotions`, `label_ids`, `document_id`,
`provider_comment_id`, `provider_split`, and `source_row`. Common provider/config/
revision/status information stays in the manifest. The original split selects
`acquisition.files[provider_split]`; the one-based source row resolves that exact
Parquet file and its pinned hash. Thus provenance is preserved without copying
long source URLs and invariant metadata into every row.

The document ID still denotes the same provider comment as v1; it is not a book
identity or recovered thread identifier. Both representations retain exact text
and original label ordering.

The release has **54,263 distinct provider comment IDs**. There are no repeated
comment IDs or conflicting records under one comment ID in this snapshot. This
restores source identity for the new candidate directly from the provider; it does
not retroactively assign IDs to legacy rows or establish historical model-run
provenance.

## Comparison with the legacy emotion corpus

The join compares **exact text plus labels in their recorded order**. It does not
trim text, collapse whitespace, reorder labels, use fuzzy similarity, or infer
identity from row position. Every legacy row has an exact candidate in its original
split; none matches only a different split.

| Legacy split | Rows | Unique exact candidate across the pinned release | Ambiguous across the pinned release | Unmatched |
| --- | ---: | ---: | ---: | ---: |
| Train | 43,410 | 43,180 | 230 | 0 |
| Validation | 5,426 | 5,387 | 39 | 0 |
| Test | 5,427 | 5,396 | 31 | 0 |
| Total | 54,263 | 53,963 | 300 | 0 |

Uniqueness here means uniqueness among source records in this pinned release.
The **300 ambiguous rows remain ambiguous**. Within-split uniqueness can be higher
than global uniqueness; the manifest records both without using the narrower
comparison to manufacture certainty. No legacy record was rewritten with a guessed
comment ID.

## Duplicate and annotation review remains open

All source records and original partitions were retained. The source contains:

- 166 exact-text duplicate groups involving 435 rows; 71 groups cross splits.
- 121 exact-text-and-ordered-label duplicate groups involving 300 rows; 55 groups
  cross splits.
- 74 exact-text groups with different label sets across their records.

Different comments can have identical text and different annotations. These
counts are review signals, not proof of erroneous labels. Comment-ID disjointness
therefore does not eliminate exact-text overlap. They explain why restoring IDs
alone does not make the candidate a leakage-controlled research dataset.

Before admission, decide whether the study retains the official benchmark
partitions, uses a separately versioned grouping policy, or reports both under
clearly distinct conditions. A grouping or removal policy must state how repeated
text, distinct comment IDs, and differing annotations are handled; it must not
silently rewrite the official benchmark. Model-selection and calibration groups
also remain unassigned and must be frozen separately. The labels remain comment
emotion labels; they are not gold labels for book atmosphere.

## Reproduce without model dependencies

```sh
python3 scripts/prepare_goemotions_candidate.py --fetch
python3 scripts/prepare_goemotions_candidate.py
python3 -m pytest tests/test_research/test_goemotions_candidate.py -q
```

`--fetch` permits the bounded public-source acquisition. Without it, missing source
receipts fail rather than triggering a download. Once cached, source bytes and
receipts are checked again and existing prepared files must match exactly; the
script refuses to replace a differing candidate at the same location. The code
never writes to legacy data. Shared hashing, immutable publication and batched
Parquet reading live in `src/research/candidate_io.py`; both helper and preparer
hashes are pinned in the manifest.

Acquisition uses the existing Hugging Face Hub client; Parquet decoding needs only
PyArrow, with no `datasets` library or ML stack. This preparation used an isolated
Python 3.14.5 environment with PyArrow 25.0.1, installed from a cached 35.9 MB binary
wheel. The manifest records decoder and Python versions. Tests use synthetic
records and mocked decoders and do not fetch source data.

Successful preparation is not research admission. The manifest remains
`candidate_prepared_not_admitted` with `training_authorized: false` until the
source-policy, duplicate-handling, calibration, and explicit execution gates are
resolved elsewhere.
