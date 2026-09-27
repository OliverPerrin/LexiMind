# Admission receipts for future book relevance evidence

Status: **preparation only**. No collection, query text, annotation, held-out
assignment, human review, or scoring has been performed by this contract. The
current [annotation packet](../../research/preparation/annotation_packet.json)
remains immutable and empty of answers. The four receipt references below remain
unpopulated in the prospective plan until a separately authorized collection
actually produces them.

`validate_book_admission(root: Path, plan: dict, packet: dict) -> list[str]` in
`src/research/book_admission.py` performs standard-library, read-only consistency
checks. It returns blocker strings, or an empty list when those checks pass.
It never creates files, starts collection, evaluates a recommender, grants
execution permission, or certifies human truth. Paused state, protocol freeze,
authorization, and model-study admission are separate preflight responsibilities.
Book evidence is not a prerequisite for the independent benchmark model study.

## The first generalization claim

This contract supports **new query families over a fixed catalogue**, declared as
`new_query_families_fixed_catalogue` in both `plan.applied_study.generalization`
and the partition receipt. Query families and seed-work groups cannot cross
pilot/development/test partitions. Candidate works can legitimately be retrieved
in several query partitions: they are the same fixed catalogue. A development
seed can therefore be a candidate for a different test query, provided it is not
that query's own seed and it passes the frozen eligibility policy.

All catalogue work IDs are tracked. This is not an unseen-candidate-work or
unseen-to-pretraining claim. Work-held-out feature supervision would require a
distinct, reviewed contract; the validator rejects an undeclared switch to that
claim instead of silently imposing it on the first relevance study.

## Typed artifact references

Each of these `plan.applied_study` keys is either `null` (blocked) or an exact
reference object with `kind`, repository-relative `path`, lowercase SHA-256
`sha256`, and positive integer `bytes`:

| Plan key | Required artifact kind | Role |
| --- | --- | --- |
| `collection_manifest` | `book_collection` | Separately collected queries and individual judgments |
| `partition_manifest` | `book_partitions` | Frozen query-family and seed-work assignment policy |
| `eligibility_manifest` | `book_eligibility` | Complete candidate/exclusion accounting per query |
| `rubric_review` | `book_rubric_review` | Frozen reference to a separate human-review source artifact |

Files are read once for byte/hash verification and strict JSON parsing. Missing
files, changed bytes, duplicate keys, non-finite values, wrong kinds, absolute
paths, `..` paths, and symlinks escaping the repository block admission. Existing
files are never overwritten. The `judgments_collected` boolean is ignored as
evidence: toggling it cannot replace any receipt.

Every receipt, including the review source, has this common header:

| Field | Required value or relation |
| --- | --- |
| `schema_version` | Integer `1`, not a boolean or float |
| `kind` | Its exact typed kind from this document |
| `purpose` | `book_relevance_primary` |
| `study_id` | Exact nonempty `plan.study_id` |
| `catalog_sha256` | Exact catalogue-byte hash from the immutable packet |
| `rubric_sha256` | Object mapping `recommendation` and `mood` to their exact packet rubric hashes |

Only the additional fields specified below are allowed. Hashes bind a particular
artifact, not the honesty of the person who created it. The packet is first
revalidated against its actual catalogue and rubric files; populating its answer
arrays is rejected rather than treated as a future collection.

## Collection receipt

Additional fields are `queries`, `judgments`, and `primary_query_ids`.
`validate_future_records` checks the query/judgment schemas documented in
[annotation_preparation.md](annotation_preparation.md), including known work IDs,
record hashes, catalogue/rubric hashes, rater IDs, human-origin declarations,
separate constraint violations, and null abstentions. Illustrative and
system-generated records cannot pass as human collection. There is no production
fixture bypass in book admission.

Queries and judgments must be nonempty. This first contract accepts recommendation
judgments only; it does not admit mood gold. `primary_query_ids` must be exactly
the set assigned to the test partition, not a convenient subset selected after
seeing outcomes. Every judgment must refer to a candidate eligible for its query.
Individual disagreements and abstentions remain in the collection.

## Partition receipt

Additional fields are `generalization`, `policy`, `catalogue_work_ids`,
`seed_work_assignments`, `query_assignments`, and `query_family_assignments`.
The exact policy object is:

```json
{
  "status": "frozen",
  "unit": "query_family_and_seed_work",
  "seed_work_overlap": "disjoint",
  "query_family_overlap": "disjoint",
  "seed_policy": "same_partition_as_query",
  "candidate_population": "fixed_catalogue"
}
```

`catalogue_work_ids` contains every packet work exactly once. Assignment arrays
contain objects with the respective identifier (`work_id`, `query_id`, or
`query_family_id`) and `partition`, one of `pilot`, `development`, `test`.
Seed assignments cover exactly the works used as seeds; they do not partition
the entire candidate catalogue. Query and family assignments cover exactly the
collected queries and their families. Duplicates, unknown IDs, unmapped IDs,
query/family disagreement, or a seed used across query partitions block admission.
No such assignments are generated by this tooling or stored in the current packet.

## Eligibility receipt

Additional fields are `policy` and `queries`. The exact policy object is:

```json
{
  "status": "frozen",
  "population": "all_packet_works",
  "seed_policy": "exclude",
  "candidate_partition_policy": "shared_catalogue",
  "other_exclusions": "explicit_reason"
}
```

There is one row per collected query, containing `query_id`, `eligible_work_ids`,
and `excluded_works`. Each exclusion contains `work_id`, `reason`, and nonempty
`detail`. The two sets must be disjoint and together account for the whole packet
catalogue, without duplicates. Every own seed must be excluded with reason
`seed_work`. Other permitted reasons are `insufficient_evidence`,
`explicit_query_constraint`, and `rights_unresolved`; the later human review must
assess their appropriateness. Candidate membership is not restricted by other
queries' partitions. Each primary query needs at least one eligible candidate.

This is a full *eligible-catalogue* primary design. A sparse pooled study needs its
own explicitly secondary contract; omitting unjudged eligible candidates does not
make a primary denominator complete. Frozen exclusions require their own evidence
and review and cannot be chosen retrospectively to improve a system's score.

## Human review and adjudication source

The `book_rubric_review` receipt adds `status: frozen` and an `attestation`
reference, using the same four-field reference contract with kind
`book_human_review`. This requires a separate source artifact rather than approval
flags inside a plan. The source has the common header plus:

| Field | Contract |
| --- | --- |
| `source_kind` | `human_review` |
| `reviewer_id` | Pseudonymous `reviewer:` ID with bounded lowercase suffix |
| `completed_at` | ISO timestamp with timezone |
| `decision` | `approved_for_primary_relevance` |
| `scope` | `metadata_supported_relevance_without_whole_work_mood` |
| `review_notes` | Nonempty human-authored account of rubric, eligibility and partition-policy review |
| `reviewed_artifacts` | Exact references to the collection, partition and eligibility receipts used by this plan |
| `reviewed_judgment_ids` | Every individual judgment ID, including disagreements and abstentions |
| `adjudications` | One final reviewed record for every primary-query/eligible-work pair |

An adjudication has `query_id`, `work_id`, integer `relevance` from 0 to 3,
`basis_judgment_ids`, and a nonempty `rationale`. Its basis must cite at least two
distinct-rater, non-abstaining judgments of that same pair. Null, boolean and
floating-point final grades are rejected. Unknown, duplicate, missing or surplus
primary-pair adjudications block admission. Two rater identifiers assert separate
raters; software cannot prove they represent independent people.

An abstention remains null. It cannot serve as a scored basis, but can remain
preserved and reviewed when other adequate judgments later resolve the pair.
If no adequate judgments cover an eligible pair, the primary design remains
blocked; an adjudication flag or zero placeholder cannot fill the gap.

## Mechanical checks and evidence limits

These checks establish typed, byte-bound, internally consistent references. They
cannot establish consent, reader competence, actual reading, independent identity,
rights, truthful origin, correct labels, or whether review happened before model
selection. Human review of actual provenance and authorization remains required.
In particular, an approval-shaped JSON file is not itself proof of human truth or
a cryptographic signature from an authorized reviewer.

Tests use temporary synthetic records with `test_fixture` origin. The unchanged
production path rejects them. Successful mechanical-path tests explicitly patch
only the existing record validator to exercise its test-fixture mode; they retain
real byte/hash, linkage, partition, eligibility and coverage checks. The review
fixture states that no human review occurred. These tests are software evidence,
not a completed collection or an admission of research gold.
