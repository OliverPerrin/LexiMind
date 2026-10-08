# Annotation packet and record schemas

The [packet](../../research/preparation/annotation_packet.json) references all 102
catalogue works and both draft rubrics. Queries and judgments are empty. It contains
no train/test assignment, copied descriptions, rater roster or generated labels.
Future collection uses the [relevance](../recommendation_judgments.md) and
[mood](../mood_annotation_guide.md) rubrics, then [separate admission receipts](book_admission_contract.md).

## Commands

```sh
# Deterministic read-only plan; no collection or scoring.
python3 scripts/research.py annotation
# Explicit, previously nonexistent destination; parent must already exist.
python3 scripts/research.py annotation --mode create --output /path/to/new-packet.json
# Read-only verification against the actual catalogue and rubric files.
python3 scripts/research.py annotation --mode validate --packet research/preparation/annotation_packet.json
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

## Book-field review to partial classifier data

The existing offline field worksheet supports a separate, explicit export path.
It does not grant formal study admission or establish human gold. The fixed
32-record worksheet contains original and effective **training** records only;
its existence does not mean anyone has completed human decisions.

Preserve an exported human draft and import it into a new candidate:

```sh
python3 scripts/research.py review import --preserve-conflicts \
  --manifest data/research_candidates/bgc/field-baseline-review/book_field_baseline_20261003/prepare_001/review_manifest.json \
  --draft /path/to/leximind-human-review.json \
  --output data/research_candidates/human-review/candidate-v2.json
```

Without `--preserve-conflicts`, legacy import still rejects opposing source and
human assertions. With it, version 2 preserves the original source bindings and
all explicit human decisions, and lists conflicts separately. Agent assertions
never become human decisions. Missing decisions and source omissions stay unknown.

For each conflict, create an adjudication draft with this exact structure:

```json
{
  "schema_version": 1,
  "kind": "leximind_field_adjudication_draft",
  "candidate": {"path": "data/research_candidates/human-review/candidate-v2.json", "sha256": "EXACT_FILE_SHA256", "bytes": 123},
  "adjudications": [{
    "assertion_sha256": "EXACT_HASH_FROM_CANDIDATE_CONFLICT",
    "choice": "human",
    "reviewer": {"kind": "human", "id": "YOUR_REVIEWER_ID", "method": "direct_source_review", "reviewed_at": "2026-10-08"},
    "evidence": [{"input_field": "description", "start": 0, "end": 10, "sha256": "EXACT_UTF8_SPAN_SHA256"}],
    "rationale": "Explain the explicit adjudication from the pinned source passage."
  }]
}
```

The values above are schema placeholders, not annotations. `choice` is `human`
(retain the explicit human state), `source` (explicitly endorse the opposing
source state), or `unknown` (mask this label). Every choice requires human
provenance, a rationale, and literal Unicode character offsets/digest. The
assertion hash binds the precise human decision, reviewer, source state and input.
A source choice applies only to that conflict; it never completes other labels.

```sh
python3 scripts/research.py review adjudicate --manifest /path/to/review_manifest.json \
  --candidate data/research_candidates/human-review/candidate-v2.json \
  --draft /path/to/adjudication.json \
  --output data/research_candidates/human-review/adjudicated-v2.json
python3 scripts/research.py review export --manifest /path/to/review_manifest.json \
  --candidate data/research_candidates/human-review/adjudicated-v2.json \
  --role train --labels genre:adventure \
  --output data/research_candidates/human-review/train-export-v1
```

Choose `--labels` deliberately from the pinned mapping, in the desired head order.
Omitting it requests the entire mapping. Export refuses before creating files
unless **each requested training column** has at least one explicit positive and
one explicit negative. One of each is a feasibility check, not enough evidence
for representative precision, calibration, or study quality. Export reports
per-column counts and masked unknowns, plus the number of decisions excluded by
an explicit subset. Unresolved conflicts block export. Reviews containing no
remaining supervised labels contribute no training row.

Export verifies the original source role and effective partition role from the
pinned assignments and components, requiring unambiguous singleton groups. The
historical purposive packet lacks these split bindings and contains development
and test examples; it is rejected as classifier training input. A train export
creates `train.jsonl`, `labels.json`, and `manifest.json` in a new ignored directory.
The input is exactly the literal source title and description; targets include
only explicit human decisions and explicit adjudications. Unknown labels remain
masked by the existing partial-label collator. The manifest pins inputs/output
bytes and records provenance without claiming admission or gold.

Use an independently reviewed original/effective development packet for
`--role dev --training-labels /path/to/train-export-v1/labels.json`. This creates
`val.jsonl` with the exact training label order, mapping hash and input format;
it verifies the companion train export manifest and data hashes. Train and
development must share exact archive, mapping, assignments, components and
partition bindings, and have disjoint record and effective group identities. Development
does not require both signs per column. No development labels or split are
manufactured from the 32 training records. Training alone can support a separately
authorized fixed-update feasibility run with validation-based selection disabled;
it cannot support a generalization claim.

The existing trainer accepts these partial targets via explicit overrides:
`data.topic_problem_type=multi_label`, `data.processed.topic=/path/to/splits`, and
`training.trainer.tasks=[topic]`. Supply independently reviewed validation files
only when available. Historical preparation manifests pin helper code hashes;
changing review implementation does not silently refresh that historical evidence.
The legacy validator and historical preparation hashes remain unchanged. Version
2 validates the source and agent assertions against the real candidate, then
validates the human assertions separately against an isolated candidate copy with
unknown comparison states. That copy supplies no targets and never replaces
source evidence. Conflicts are derived from the unchanged real source assertions,
and export still requires explicit bound adjudication. This preserves the legacy
contract and keeps offline preparation checks independent of local corpus files.
