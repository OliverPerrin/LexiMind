# Book catalogue

`web/data/books.json` is the source-grounded work catalogue used by the website.
`manifest.json` records its coverage, selection queries, field provenance, rejected
records, and content hash. `raw/` preserves the exact Open Library API response
data, request URL, retrieval timestamp, and SHA-256 digest for each source.

Rebuild without network access or model execution:

```sh
python3 scripts/build_book_catalog.py --offline
python3 -m pytest tests/test_catalog -q
```

The initial selection uses ten English-language subject searches, at most ten
results per subject. The search chooses a work; its description comes only from
that same work ID. Search author IDs must agree with the work record. This is a
small discovery selection, not a balanced evaluation set or a comprehensive
catalogue. Community records may still contain errors; source links remain
visible so users can inspect them.

`selection_review.json` records source-hash-pinned exclusions: one duplicate work,
two works with questionable author attribution, and two uninformative descriptions
withheld from display. The raw responses remain intact. A changed source hash
requires a fresh review instead of silently reusing the old decision.

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
a future rebuild needs a separately reviewed dataset version and work-level split.

The Gradio demo and its Docker image use the same canonical book catalogue as the
website. Legacy JSONL contributes historical paper examples only; its literary
pairs are not displayed. Unvalidated tone labels are removed at load time, so
rebuilding the old JSONL is not required to make the interface honest. A missing
canonical catalogue shows an explicit unavailable notice, never a legacy fallback.
