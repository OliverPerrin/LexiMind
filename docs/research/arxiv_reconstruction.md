# arXiv source reconstruction

Status: **candidate prepared, not admitted; checked 2026-09-26**. The author-linked
archive was acquired, verified and indexed with its **215,913 original article IDs**
and source partitions intact. No model, training, inference, scoring, article
scraping, or paid service was involved. Legacy processed data remain untouched.

## Source selection and access

The [author README at commit `6ef082e22b8f49e7195f10c1cdeb5abcf428ff5e`](https://github.com/armancohan/long-summarization/blob/6ef082e22b8f49e7195f10c1cdeb5abcf428ff5e/README.md)
links both Google Drive and an [Internet Archive mirror](https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip).
It documents sentence-array records retaining `article_id`, article/abstract text,
section names and section text, with original `train.txt`, `val.txt`, and `test.txt`
partitions. This route preserves the author release's IDs rather than inventing
identities for a text-only repackaging.

The [mirror metadata](https://archive.org/metadata/armancohan-long-summarization-paper-code)
and HTTP headers report **3,624,420,843 bytes**. The initial 3 GB review boundary was
explicitly extended to 4 GB after checking that approximately 264 GiB was free.
The mirror declares SHA-1 `26f95b9e0f37e9d2bcef31dd1dcd3d25d9367b4d`
and MD5 `6242aaf5cfcc7814473eee8b779c1b9f`. These provider declarations are
distinct from the SHA-256 computed locally after acquisition. The Drive archive
has not been downloaded or asserted byte-identical.

The acquired archive matches both provider checksums. Its observed SHA-256 is
`82ed30dd7c66a6497eeb3d7c3090c274e9e32c012438f8e0bb3cce3e6c1fcada`.

The first transfer retains a serial prefix, then uses four bounded HTTP ranges
into the same staged file. Each response must have the requested range, exact
length and unchanged ETag; the complete archive must match both provider checksums
before publication. Partial files are never candidates. The reconstruction script
also supports a bounded serial download/resume for a fresh acquisition, with at
most three attempts and explicit `--fetch` permission. It refuses a concurrently
staged ranged acquisition rather than downloading another full copy.
Aliased/nonregular source files and escaping directories are rejected before
requests or writes. Download/resume uses directory-relative, no-follow file
operations; platforms lacking these operations fail closed for acquisition.

## Storage and provenance

Only one complete article-text archive is retained under ignored
`data/research_candidates/arxiv/author-release-26f95b9e0f37/raw/`.
Small cached provenance documents include the pinned author README, mirror metadata,
and arXiv's license-information page, each with its URL, retrieval time and SHA-256.
The acquisition receipt distinguishes provider checksums from observed checksums.

[Source manifest](../../research/preparation/arxiv_source_manifest.json) is the
compact, source-linked preparation report. No source article text or abstracts are copied into Git. The
original ZIP is not extracted into a second full corpus. Instead, `index/*.jsonl`
stores a metadata record for every original source row:

- Original `article_id`, source split and source line number.
- ZIP member reference, uncompressed byte offset/length, and exact source-line hash.
- Recognized arXiv base-ID/version shape, without inventing a missing version.
  Unrecognized provider IDs remain intact and explicitly unresolved.
- Character/sentence/section counts and field-consistency flags.
- Hashes of newline-joined article/abstract sentence arrays and separately
  normalized article text. Normalization collapses whitespace and applies Unicode
  NFC while retaining case and punctuation. The raw source remains unchanged.

Offsets refer to the **uncompressed member**, not seek offsets into compressed ZIP
bytes. Retrieve evidence by opening that member and streaming to the recorded
location. The ZIP checksum and member/source-line hashes bind the locator to an
exact source. Recognition of ID syntax does not verify article existence, version
history, or identity equivalence through a network lookup.

The indexer checks ZIP members without extracting them, rejects unsafe/duplicate
names and encrypted/symlink entries, and enforces expanded-size, row and per-line
limits. It processes one bounded line at a time. Duplicate grouping uses a temporary
SQLite database with a bounded cache; that database contains IDs/hashes only and
is removed afterward. The final metadata reports within/across-split duplicates
without deleting records or changing the author's partitions.

## Observed source audit

| Original split | Rows / unique original IDs | IDs with unresolved arXiv syntax | Empty articles | Empty abstracts | Section/body sentence-count mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 203,037 | 74,279 | 120 | 3 | 46 |
| `val` | 6,436 | 2,493 | 0 | 0 | 1 |
| `test` | 6,440 | 2,501 | 0 | 0 | 0 |
| **Total** | **215,913** | **79,273** | **120** | **3** | **47** |

All original IDs are globally unique. The conservative syntax recognizer accepts
136,640 arXiv-shaped base IDs; the remaining 79,273 are slashless older forms such
as `hep-ph0701277`. A read-only pass over the finished indexes confirmed that all
unresolved forms match a letters/category prefix followed by seven digits. Their
original strings are preserved; no slash, official URL, or asserted canonical ID
was manufactured. Resolving that provider naming convention remains a review task.
None of the original IDs states an article-version suffix, so versions remain unknown.

| Equality/grouping criterion | Duplicate groups | Rows in groups | Groups spanning source splits |
| --- | ---: | ---: | ---: |
| Original `article_id` | 0 | 0 | 0 |
| Recognized base ID, **recognized subset only** | 0 | 0 | 0 |
| Exact newline-joined article text | 18 | 155 | 2 |
| Whitespace/NFC-normalized article text | 19 | 157 | 2 |
| Exact newline-joined abstract text | 52 | 118 | 7 |

These counts include empty-text groups; they are audit findings, not automatic
deduplication or exclusion instructions. Original-ID uniqueness does not prove
publication-level independence. The maximum observed training article/abstract
lengths are 743,782 and 160,939 characters respectively; the large abstract outlier
also needs source/length review before any model input contract is selected.

The three original JSONL members total **15,147,709,231 uncompressed bytes**, all
streamed and hashed without extraction. Their metadata indexes total **171,543,023
bytes**. No second full-text corpus or persistent duplicate-audit database was created.

## Reproduction

The script uses Python's standard library and shared repository helpers:

```sh
# Initial acquisition only; explicit network permission.
python3 scripts/prepare_arxiv_candidate.py --fetch

# Recheck cached source bytes and regenerate/verify the metadata-only index offline.
python3 scripts/prepare_arxiv_candidate.py

# Synthetic fixtures only; no network or model execution.
python3 -m unittest discover -s tests/test_research -p test_arxiv_candidate.py
```

Offline reconstruction hashes the archive and streams its original JSONL members;
it is a source audit, but still substantial disk/decompression work. Index and
manifest publication is create-only or byte-identical verification. Changed existing
outputs require a separately reviewed revision; they are not silently overwritten.
Legacy `data/processed`, model checkpoints, the website and Gradio inputs are untouched.

The provider name `val` remains unchanged in the source index. The manifest declares
`validation -> val` as an alias; no new selection/calibration split is created.
Field anomalies and duplicate text are review signals, not automatic exclusions.

## Admission remains open

The inspected author README provides no specific uniform article-text license.
arXiv explains that submitters choose licenses and versions can have different
terms; public availability is distinct from a blanket reuse grant.
[arXiv license information](https://info.arxiv.org/help/license/index.html).

The preparation does not recover per-article rights or unstated version identifiers,
resolve duplicates, select length/truncation rules, assign calibration data, or
approve a training task. Those source-use and generalization decisions remain
required before admission. Model training and research evaluation stay paused.
