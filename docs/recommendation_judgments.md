# Book relevance rubric: draft

Target: how well a work satisfies a reader's stated request under the declared
available evidence. It is not a prediction of rating or eventual enjoyment.
No production queries or judgments have been collected.

| Grade | Meaning |
| --- | --- |
| 3 | Strong fit for the intent and all required constraints |
| 2 | Useful fit with a meaningful limitation, no known hard-constraint violation |
| 1 | Weak or tangential fit, no known hard-constraint violation |
| 0 | Judged nonrelevance or a known violation of a required constraint |
| null | Insufficient evidence to judge; never converted to zero |

## Query and evidence

Record literal request, clarified intent, required constraints, source/author,
query-family ID, seed work IDs and the exact catalogue/rubric revisions. Generate
queries independently of candidate results and target-book reviews. Use consented
collection; private shelf history is not an automatic research dataset.

For each query/work judgment, retain a rater ID, grade or abstention reason,
constraint violations, rationale, source URLs/locations, evidence scope, language
and edition/translation if relevant. Description-based judgments are limited to
that evidence; requests for unavailable whole-book mood require abstention.
Do not show scores, system names or generated explanations to raters.

## Split and candidate policy

B1 tests new query families over the fixed catalogue. Query families and seed-work
groups cannot cross partitions; candidate works may occur in several query pools.
Exclude the seed work and record every other eligibility exclusion. Canonically
repeated query content cannot evade splitting by changing IDs. This is not an
unseen-work evaluation; that would need a separate candidate/split policy.

## Review and scoring

Primary evaluation requires every eligible query/work pair to have two independent,
non-abstaining judgments and a separate adjudication record. Preserve disagreements
and all known hard violations; adjudication cannot discard an inconvenient rating
to create a positive grade. Freeze rubric/pooling decisions before comparison.

Use nDCG@10 with gains `[0,1,3,7]`, log2 discount and all eligible works. A separately
frozen pool-conditional analysis may be secondary; unjudged pairs are not negatives
and cannot establish full-catalogue recall. Report evidence/label coverage and hard
violations. The [protocol](eval_protocol.md) fixes the comparison boundary.

The empty preparation packet remains separate from future collection, partition,
eligibility and review receipts. Exact fields and synthetic fixtures live in
`src/research/annotations.py`, `src/research/book_admission.py` and their tests.
