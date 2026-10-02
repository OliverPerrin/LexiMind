# Book-relevance evidence contract

`validate_book_admission(root, plan, packet)` in
[book_admission.py](../../src/research/book_admission.py) checks separate future
receipts. The [preparation packet](../../research/preparation/annotation_packet.json)
stays immutable with empty answers; collected records never replace it.

```sh
python scripts/research.py status --target book_study --require-ready
python -m pytest tests/test_research/test_book_admission.py -q
```

## Receipt graph

| `applied_study` reference | Required kind |
| --- | --- |
| `collection_manifest` | `book_collection` |
| `partition_manifest` | `book_partitions` |
| `eligibility_manifest` | `book_eligibility` |
| `rubric_review` | `book_rubric_review`, referencing a separate `book_human_review` source |

Each reference contains exactly `kind`, relative `path`, positive integer `bytes`
and lowercase `sha256`. Every receipt binds schema version 1, its exact kind,
`purpose: book_relevance_primary`, study ID, catalogue hash and both rubric hashes.
Changed bytes, malformed JSON, path escapes and mismatched bindings fail. A flipped
`judgments_collected` flag supplies no evidence. Exact payloads are demonstrated
in [the schema fixtures](../../tests/test_research/test_book_admission.py).

## Scope, partitions and eligibility

The supported claim is `new_query_families_fixed_catalogue`, matching the plan.
Track every catalogue work, but partition **queries, query families and seed groups**
into pilot/development/test. Candidate books may recur across query partitions;
this is not unseen-work evaluation. Identical normalized query content cannot
cross partitions under different IDs, and seed/family assignments must agree.
Primary query IDs are exactly the test-partition queries.

Every query declares eligible work IDs plus reasoned exclusions covering the
entire catalogue without overlap. Exclude its own seeds. Other permitted reasons
are insufficient evidence, explicit query constraints and unresolved rights.
The frozen policy shares the candidate catalogue across query partitions; it does
not assign every candidate book to a query split. Primary queries need candidates.

## Judgments and review

Collection records pass [annotation validators](annotation_preparation.md), with
known work/record/rubric hashes, human-origin declarations and separate constraints.
The current contract admits recommendation relevance, not mood gold. Each primary
eligible pair needs a reviewed integer grade 0–3 backed by two distinct-rater,
non-abstaining judgments. Null is not zero; incomplete eligible coverage blocks
primary scoring. Pooled coverage belongs to a separately specified secondary study.

The human-review source pins the exact collection, partition and eligibility
artifacts; records reviewer, timezone-bearing time, scope, decision and rationale;
and accounts for **all** judgments, including disagreement/abstention. Adjudications
must cover exactly the primary pairs and cite their supporting judgment IDs.
Retained hard-constraint violations cannot be discarded to admit a positive grade.
The exact frozen policies and attestation fields remain in
[the validator](../../src/research/book_admission.py) and fixture contracts.

These checks establish consistency, not actual independence, consent, source rights
or truthful human review. There is no production fixture bypass; software tests
explicitly enable synthetic record validation only inside their test boundary.
