# Annotation preparation, without collection

Status: **unlabelled preparation; collection and research evaluation have not
started**. This tooling references the current 102-work metadata catalogue and the
two existing draft guides. It does not select held-out works, generate queries,
sample candidates, contact readers, assign ratings/moods, or score systems. It does
not read the website's browser-local shelves or any private history.

The checked-in [packet](../../research/preparation/annotation_packet.json) contains
every current work. Its `queries` and `judgments` are empty. There is no annotator
roster, production label, train/test assignment, or duplicate copy of descriptions.
See [book_discovery_review.md](book_discovery_review.md) for the research rationale,
[recommendation_judgments.md](../recommendation_judgments.md) for the proposed
relevance process, and [mood_annotation_guide.md](../mood_annotation_guide.md) for
the distinction between passage emotion and whole-work reading atmosphere.

The initial empty packet is retained in
[`snapshots/annotation_packet_initial_2026-09-26.json`](../../research/preparation/snapshots/annotation_packet_initial_2026-09-26.json).
The current empty packet was regenerated after clarifying the relevance rubric's
fixed-catalogue query-generalization policy. Neither revision contains judgments.

## Commands and write boundaries

All commands use the standard library. Run from the repository root:

```sh
# Default is a deterministic, read-only plan printed to stdout.
python3 scripts/prepare_annotation_packet.py

# Create a new packet only at an explicit, previously nonexistent destination.
python3 scripts/prepare_annotation_packet.py --mode create --output /path/to/new-packet.json

# Verify the checked-in empty packet against current catalogue and rubric bytes.
python3 scripts/prepare_annotation_packet.py --mode validate --packet research/preparation/annotation_packet.json

# Synthetic software fixtures only; no model or research evaluation.
python3 -m unittest discover -s tests/test_research -p test_annotation_packet.py
```

`--catalog`, `--recommendation-rubric`, and `--mood-rubric` can point to explicit
alternative inputs. Creation uses exclusive file creation and refuses an existing
file or symlink; it has no overwrite flag and does not create missing parent
directories. A stale packet must be preserved or consciously removed/moved by its
owner before any replacement; the tool will not silently replace it. Plan and
validate modes reject output-file arguments. Validation does not update old hashes.

## What is pinned

| Field | Meaning |
| --- | --- |
| `catalog.sha256` / `bytes` | SHA-256 and length of the exact original catalogue bytes, including formatting and final newline |
| `catalog.path` | Repository-relative reference when possible; otherwise the resolved explicit input path |
| `catalog.record_count` | All source records, without a sample or split |
| `items[].work_id` | Unique, canonical Open Library work ID, checked against that record's work identifier |
| `items[].record_sha256` | Hash of the complete parsed metadata record serialized with sorted keys, UTF-8, `ensure_ascii=False`, compact separators, and non-finite numbers forbidden |
| `items[].source_urls` | Deduplicated source-record and available description-source URLs; not copied source text |
| `items[].source_revision` / `source_content_sha256` | Existing catalogue provenance, preserved as references rather than recomputed claims about the remote provider |
| `rubrics` | Paths, exact-byte SHA-256 values, and `draft` status for both guides |

The item-record hash is explicitly a **canonical JSON hash**, not an exact-byte
slice of a pretty-printed array. The separate catalogue hash pins those exact
bytes. No network requests are made to confirm source URLs or provider metadata;
the packet states what the local snapshot references. Duplicate JSON keys,
non-finite values, duplicate works and malformed identity/provenance fail closed.

No timestamps are invented at generation time. Identical input files and paths
produce identical packet bytes. Changed catalogue/rubric bytes, item omissions,
unexpected fields, substituted booleans/numbers, or populated answers invalidate
an existing preparation packet. A changed catalogue means a new preparation
revision, not a silently equivalent research population.

## Future record schema validators

`src/research/annotations.py` exposes `validate_query`, `validate_judgment`, and
`validate_future_records`. They accept future records supplied by a caller; they
do not create, collect, save, or adjudicate them. First call `validate_packet`
against the actual source files. The record validators check the packet's internal
structure and references, but only byte-level packet validation compares it with
the underlying catalogue/rubrics. Strict fields mean schema changes need an explicit
code/rubric revision rather than being quietly ignored.

Every future query requires:

- `query_id`, `query_family_id`, `author_id`: trimmed IDs using the respective
  `query:`, `family:`, `author:` prefix and a bounded lowercase identifier suffix.
- `text`, `intent`: nonempty text supplied through the separately authorized
  collection process. The current tooling generates neither.
- `constraints`: objects containing a unique `constraint_id` and its description.
- `seed_works`: zero or more known work/hash reference objects, without duplicates.
- `catalog_sha256`, `rubric_sha256`: the pinned catalogue and recommendation rubric.
- `source_kind`, `fixture_only`: explicit origin. Production validation accepts
  `human` with `fixture_only: false`; these assertions remain subject to review.

Every future judgment requires `judgment_id`, `kind`, known `work_id`, matching
`record_sha256`, `catalog_sha256`, matching kind-specific `rubric_sha256`,
`rater_id`, `source_kind`, `fixture_only`, `rationale`, and an `evidence` object.
Evidence records source URLs, locations, language, scope, reading coverage and
edition/translation identifier. Excerpt and whole-work reading require a stable
edition identifier and evidence locations; description and metadata scopes cannot
claim complete reading. HTTPS formatting is checked, not the truth of the evidence.

For `kind: recommendation`, require a known `query_id`, `relevance`,
`abstention_reason`, and separate `constraint_violations` IDs from that query.
`relevance` is an integer 0–3 or `null`; booleans, floats and numeric strings are
rejected. **Zero is judged nonrelevance; null is abstention** and requires a
nonempty reason. A scored judgment cannot also claim abstention. Constraint
violations remain separate even when an assessor abstains on overall relevance.
Under the current rubric, a recorded hard-constraint violation permits zero or
explicit abstention, never a positive relevance grade. Primary adjudication cannot
silently discard a retained violation by omitting that judgment from its cited
basis; any override needs an explicit reviewed constraint-resolution schema.

For `kind: mood`, require `mood_observations`, `evidence_status`, and
`eligible_for_whole_work_gold`. Observations must use the eight candidate terms in
the current draft guide; this software check does not validate that taxonomy.
Metadata/description-only evidence is rejected for book-mood observations. An
excerpt may support an explicitly scoped, nongold observation. Any assertion of
whole-work eligibility requires complete whole-work evidence, identified edition,
locations, supported status, at least one observation, and nonfixture origin.
Uncertain or incomplete observations must not be promoted by changing a flag.

Batch validation rejects duplicate query/judgment IDs and repeated assignments of
the same rater to the same kind/query/work. Different raters' judgments remain
separate, including disagreement. Later adjudication needs a distinct schema and
history; this module does not overwrite either person's observation.
Book admission additionally rejects canonical query-content duplicates crossing
partitions under different query/family IDs. This fingerprint includes literal
text, clarified intent, constraint descriptions and seed-work IDs, with NFC and
whitespace normalization and order-independent constraints/seeds. It preserves
case and punctuation and does not infer semantic equivalence between paraphrases.
Distinct seed contexts remain distinct; human family-group review is still required.

## Checks are not certification

**A schema-valid row is not verified human gold.** A malicious or mistaken author
can claim human origin, complete reading or supported evidence. This tool cannot
prove consent, actual reading, rater competence, rights, impartiality, source
accuracy, or the truth of a label. It does not freeze the draft rubrics. Every
packet therefore retains `collection_authorized: false`, `gold_verified: false`,
and `human_review_required: true`. No combination of record flags changes those
facts or authorizes training/evaluation.

Illustrative and system/model-generated records are rejected by the production
validators. Tests use explicitly marked `test_fixture` records and require the
deliberate `allow_test_fixtures=True` argument; they are never accepted by default
and cannot assert whole-work gold eligibility. They are synthetic software inputs,
not fake participants or production labels. The CLI has no fixture bypass.

Remaining work is a human-approved collection design, source/edition access review,
rubric pilot and freeze, consent/handling decisions, qualified judgments and review,
split/analysis preregistration, and explicit authorization for the relevant research
steps. None has been completed by preparing this packet.
