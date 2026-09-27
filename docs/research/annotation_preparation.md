# Annotation packet and record schemas

The [packet](../../research/preparation/annotation_packet.json) references all 102
catalogue works and both draft rubrics. Queries and judgments are empty. It contains
no train/test assignment, copied descriptions, rater roster or generated labels.
Future collection uses the [relevance](../recommendation_judgments.md) and
[mood](../mood_annotation_guide.md) rubrics, then [separate admission receipts](book_admission_contract.md).

## Commands

```sh
# Deterministic read-only plan; no collection or scoring.
python3 scripts/prepare_annotation_packet.py
# Explicit, previously nonexistent destination; parent must already exist.
python3 scripts/prepare_annotation_packet.py --mode create --output /path/to/new-packet.json
# Read-only verification against the actual catalogue and rubric files.
python3 scripts/prepare_annotation_packet.py --mode validate --packet research/preparation/annotation_packet.json
python3 -m unittest discover -s tests/test_research -p test_annotation_packet.py
```

`--catalog`, `--recommendation-rubric` and `--mood-rubric` select explicit inputs.
Creation is exclusive, including refusal of existing symlinks; no overwrite flag
exists. Validation never refreshes stale hashes. Preserve old evidence before
choosing a separately reviewed revision.

## Pins and future records

The packet hashes exact catalogue/rubric bytes. Each item also hashes its complete
metadata record as sorted compact UTF-8 JSON (`ensure_ascii=False`, no NaN), retaining
work ID, source URLs/revision and source-content hash. This canonical record hash
is not a byte slice of the pretty-printed catalogue; the catalogue hash covers that.

[annotations.py](../../src/research/annotations.py) exposes `validate_packet`,
`validate_query`, `validate_judgment` and `validate_future_records`. First verify the
packet against real files. Record validators then enforce these strict schemas:

| Record | Required content |
| --- | --- |
| Query | Query/family/author IDs; text and clarified intent; unique constraints and seed work/hash references; catalogue/rubric hashes; source kind and fixture flag |
| Judgment | Judgment/kind/work/rater IDs; record/catalogue/rubric hashes; source kind; rationale; evidence scope, coverage, URLs, locations, edition and language |
| Relevance extension | Known query ID, integer 0–3 or null, abstention reason, separate known constraint-violation IDs |
| Mood extension | Draft-vocabulary observations, evidence status, whole-work eligibility assertion |

Zero means judged mismatch; null requires an abstention reason. Floats, booleans
and numeric strings are not grades. A known hard-constraint violation allows zero
or abstention, never a positive grade. Reading evidence needs edition/location
references; metadata or description evidence cannot establish book mood. Whole-work
eligibility requires complete supported evidence and nonfixture origin.

Batch checks reject duplicate records/rater assignments. Canonical query-content
hashes normalize whitespace/NFC and constraint/seed order without equating arbitrary
paraphrases. [Fixtures](../../tests/test_research/test_annotation_packet.py) show exact
fields and rejection cases; they are not collected annotations.

Schema-valid claims are not verified human gold. The packet retains collection
unauthorized, gold unverified and human-review-required flags. Generated/illustrative
origins are rejected. Fixture mode is opt-in for software testing; the production
admission path does not enable it.
