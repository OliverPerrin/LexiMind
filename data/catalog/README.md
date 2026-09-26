# Book catalogue

`web/data/books.json` is the source-grounded work catalogue used by the website.
`manifest.json` records its coverage, selection queries, field provenance, rejected
records, and content hash. `raw/` preserves the exact Open Library API response
data, request URL, retrieval timestamp, and SHA-256 digest for each source.
`web/data/catalog-manifest.json` is the deployment receipt: it hashes the exact
catalogue bytes and the full source manifest. The website verifies
the receipt before using the catalogue.

Rebuild without network access or model execution:

```sh
python3 scripts/build_book_catalog.py --offline
python3 -m pytest tests/test_catalog -q
```

The catalogue uses bounded subject searches and contains 102 works, 96 source
descriptions, and 16 mapped genres. It is a small discovery selection, not a
balanced research dataset or a comprehensive catalogue.

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

## Publication and recovery

Cache and catalogue writes use temporary files and atomic replacement, so interrupted
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
