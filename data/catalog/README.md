# Book catalogue

`web/data/books.json` is the source-grounded work catalogue used by the website.
`manifest.json` records its coverage, selection queries, field provenance, rejected
records, and content hash. `raw/` preserves the exact Open Library API response
data, request URL, retrieval timestamp, and SHA-256 digest for each source.
`web/data/catalog-manifest.json` is the deployment receipt: it hashes the exact
catalogue bytes and the full source manifest. Both web and Gradio loaders verify
the receipt before using the catalogue.

Rebuild without network access or model execution:

```sh
python3 scripts/build_book_catalog.py --offline
python3 -m pytest tests/test_catalog -q
```

The initial selection used ten subject searches, at most ten results per subject.
The September 26 expansion added three six-result searches for popular science,
literary fiction, and essays: 19 API requests produced 16 new candidates, of which
13 were admitted after source review. The current catalogue contains 102 books,
96 source descriptions, and 16 mapped genres. The original catalogue, manifest,
and review are preserved in `snapshots/2026-09-22/`.

Queries require an English edition, but a work's original title or description
may be in another language. Source text is retained without generated translations.
The search chooses a work; its description comes only from
that same work ID. Search author IDs must agree with the work record. This is a
small discovery selection, not a balanced evaluation set or a comprehensive
catalogue. Community records may still contain errors; source links remain
visible so users can inspect them.

`selection_review.json` records source-hash-pinned exclusions: one duplicate work,
five works with unresolved author/contributor attribution, and two uninformative descriptions
withheld from display. The raw responses remain intact. A changed source hash
aborts the build pending fresh review. Corrupt cache hashes, invalid source
timestamps, duplicate title/author identities, and malformed search responses
also abort before publication; ordinary missing work metadata is recorded as an
explicit rejection.

Descriptions are preserved, not model-generated. Some Open Library records use
Markdown or contain short descriptions; absent descriptions remain empty.
Genre filters are deterministic mappings from explicit Open Library subjects;
the original subjects remain available. Mood labels are empty because no
validated editorial mood source is available. The old social-media emotion
model is not treated as a literary atmosphere classifier.

Work IDs identify works, not editions. `firstPublished` comes from the matching
search record. Covers represent an available edition, not necessarily the first
edition. ISBNs remain empty until edition records can be represented separately.

The importer caches all responses, identifies itself, limits calls to no more
than one per second, and has bounded input sizes. There are no API calls in the
deployed website's recommendation flow. Repeated builds use the cache. Use
[Open Library data dumps](https://openlibrary.org/developers/dumps) for future
large-scale ingestion rather than expanding this small lookup script.

Sources: [API guidance](https://openlibrary.org/developers/api),
[Search API](https://openlibrary.org/dev/docs/api/search), and
[metadata licensing](https://openlibrary.org/developers/licensing).
Open Library does not assert additional rights over its database and notes that
individual contributions may retain existing rights. Catalogue metadata and
cover artwork are not re-licensed by this repository's software MIT licence.

## Historical data repair boundary

Existing processed datasets and the published Hugging Face dataset are historical
artifacts and are not rewritten by this work. Future Gutenberg/Goodreads joins
require full title (including subtitle) and author evidence; missing or ambiguous
identity fails closed. Matched records carry a stable `work_id` and description
source. Sources without author metadata will produce fewer or zero pairs until
their identity can be established. A title match alone is never sufficient.

The legacy discovery builder now excludes unverifiable literary pairs, writes to
`data/discovery_dataset_verified.jsonl` by default, and emits `Unknown` tone labels
with explicit abstention status. Raw model emotion scores are retained only for
audit. Run neither builder on historical training data as an implicit migration;
a future rebuild needs a separately reviewed dataset version. Its splitter now
keeps every verified literary `work_id` together, preserves even small supplied
validation/test partitions, and rejects conflicting partition assignments. Hash
assignment is deterministic and unaffected by input order. No historical datasets
were rebuilt to test this behavior; tests use synthetic fixtures only.

Future Gutenberg language-model paragraphs are grouped by provider document, not
randomly divided paragraph by paragraph. The converter preserves
[`METADATA.text_id`](https://huggingface.co/datasets/sedthh/gutenberg_english/blob/main/README.md)
or the ebook ID in the [PG-19 provider URL](https://huggingface.co/datasets/deepmind/pg19/blob/main/README.md).
If neither is available, a hash of the complete source document groups its
paragraphs. These `document_id` values are namespaced by provider; all carry
`identity_scope: provider_document` and `work_identity_status: unresolved`.

The optional [BookSum converter](https://huggingface.co/datasets/kmfoda/booksum)
retains the parent `bid`, original chapter-bearing `book_id`, chapter path,
summary ID/source, and supplied split. It can recover the same provider parent
from a recognized chapter path when `bid` is absent; conflicting or missing parent
identity fails with an actionable error. Provider documents can use the shared
splitter without being mislabeled as canonical works. No cross-source edition or
work equivalence is inferred. **Work-level generalization remains unverified**
until those provider documents and editions have been reconciled across sources.

The Gradio demo and its Docker image use the same canonical book catalogue as the
website. Legacy JSONL contributes historical paper examples only; its literary
pairs are not displayed. Unvalidated tone labels are removed at load time, so
rebuilding the old JSONL is not required to make the interface honest. A missing
canonical catalogue shows an explicit unavailable notice, never a legacy fallback.

## Publication and recovery

Cache and JSONL writes use temporary files and atomic replacement, so interrupted
serialization does not truncate the existing file. Catalogue publication stages
all three artifacts before replacing them, holds an advisory writer lock, and
publishes the catalogue last. An ordinary write failure rolls previous files back;
if rollback itself fails, the error names retained recovery files.
The writer lock uses `fcntl.flock` on POSIX and `msvcrt.locking` on Windows; missing
backends or acquisition failures abort publication. Windows contention that
outlasts the CRT retry window raises an error, allowing a later retry. The Windows
branch has controlled backend tests; native Windows execution has not been verified.

Multiple independent paths cannot be replaced atomically as a group. After an
abrupt process or power interruption between replacements, loaders reject a
catalogue whose byte hash disagrees with its receipt. Re-running the offline
builder restores a coherent generation. Repeated builds from unchanged cached
sources are byte-identical and perform no network requests. Custom `--output`
paths default to sibling manifests/receipts, avoiding accidental writes to the
canonical catalogue's metadata during isolated validation.
