# Book recommendation judgments — draft protocol

Status: **preparation only; no relevance collection or research evaluation has
started**. This guide is subordinate to the research pause in
[the current study decisions](research/study_decisions.md) and to the unresolved gates in
[eval_protocol.md](eval_protocol.md). Software regression tests and synthetic runtime
benchmarks do not constitute recommendation-quality evidence.

The proposed applied question is whether jointly learned topic, genre, and
validated mood features improve book discovery over a text-retrieval baseline.
This guide defines how a later human evaluation could distinguish useful matches
from plausible-looking metadata overlap. It does not claim that the present
recommender answers that question.

## Units and query intent

Judge a **query–candidate work** pair. Preserve an explicit seed-work ID when a
query asks for a similar book. A title match, a bibliographic work, a particular
edition, and a chapter must not be treated as interchangeable identities.

Proposed query families are:

- A stated subject or genre with explicit inclusions/exclusions.
- A natural-language reading request with a short clarification of what matters.
- “More like this” with the relevant aspect of the seed named when available.
- A request involving mood, only once the requested mood and candidate evidence
  have passed the separate [mood annotation guide](mood_annotation_guide.md).

Keep a query’s literal text, clarified intent, hard constraints, acceptable
interpretations, and evidence needed to judge it. An ambiguous request may need an
`unjudgeable` result; raters should not silently invent a different request.

A lexical hit is not automatically a useful recommendation. Conversely, an
unfamiliar or unavailable book is not automatically irrelevant. Separate failures
of identity, evidence availability, and recommendation relevance.

## Proposed relevance rubric

This ordinal scale is a draft to calibrate and freeze later:

| Value | Proposed meaning |
| --- | --- |
| `3` — strong fit | Directly addresses the important intent, satisfies hard constraints, and has adequate supporting evidence. |
| `2` — useful fit | A credible recommendation addressing the main intent, with a limited or clearly stated tradeoff. |
| `1` — weak fit | Some relevant connection, but it misses a material preference or relies on a peripheral aspect. |
| `0` — not a fit | Evidence supports a mismatch or a violated hard constraint. |
| `null` — unjudgeable | Identity, available material, language, or query ambiguity prevents a defensible judgment. This is not a zero. |

Record hard-constraint violations separately from the relevance score. Do not make
claims about themes or whole-book mood that the available evidence cannot support.
A label justified only by bibliographic metadata should explicitly say so; stronger
claims may require a reader familiar with the work.

For a recommendation list, also record distinct author/work coverage, redundancy,
and exposure to different relevant subtopics. Diversity is useful only among
credible matches; filling a list with unrelated books is not a quality improvement.
No proposed diversity preference overrides a user’s explicit constraints.

## Data and provenance

A proposed judgment record would contain:

- `query_id`, original text, frozen clarified intent, constraint IDs, and seed-work
  IDs; keep these in a separately versioned query record.
- Candidate work ID and edition/translation where relevant; the catalogue record
  hash or immutable snapshot identifier; source URLs and access/reading coverage.
- Pseudonymous rater ID, rubric version, language competence, prior familiarity,
  relevance value or abstention reason, constraint-violation IDs, concise rationale,
  and evidence locations.
- Candidate-pool membership and exposure/order seed in a separate evaluation
  manifest. Hide ranking scores, system names, model explanations, and other
  raters’ judgments from the annotation view.
- Original individual records plus separate adjudication records and a change log.
  Do not overwrite disagreements or drop difficult queries after seeing scores.

Do not include private reading histories, account identifiers, or personal query
text without informed permission and an agreed handling policy. Synthetic examples
below are not consented user records and are not real judgments.

## Query generalization and optional work holdout

The current B1 proposal is **new query families over a fixed catalogue**. Group
queries by family and seed work before assigning development/test roles. Candidate
works may legitimately be retrieved for queries in multiple partitions. The primary
eligible pool covers the declared catalogue with explicit exclusions, including the
seed work itself. This setting makes no claim that candidate works are unseen by
the representation model.

A separate future model-feature study may claim unseen-work generalization. The
work-group procedure below applies to that explicit claim; it must not silently
shrink the current B1 catalogue to match each query's partition.

The held-out work set is currently **empty**. After research resumes:

1. Repair and pin work identity across metadata, editions, excerpts, summaries,
   training examples, and any teacher outputs. Titles alone are insufficient group
   keys. An unresolved join is excluded from a claimed clean split.
2. Assign split groups at the work level: all known editions, chapters, excerpts,
   source duplicates, and derivative examples stay together. Review series/author
   overlap as an additional source of dependence and declare the intended scope.
3. Separate rubric/pilot material, recommendation tuning queries, and the final
   held-out collection. A work used to choose model features or thresholds is not
   subsequently described as unseen test material.
4. Freeze a manifest of work IDs, source snapshot hashes, query groups, duplicate
   decisions, selection procedure, and exclusions before the main comparison.
   The initial 89-book website snapshot is a product seed, not an automatically
   representative or approved research test set.
5. Define the intended generalization claim. A catalogue can legitimately contain
   held-out works for retrieval, but trained-feature models and tuning choices must
   not use their held-out supervision. Declare whether seed works and candidate
   works are unseen, rather than using the word “held-out” without qualification.

No split, sample size, minimum effect, or stopping rule has been approved here.
Choose them before collection based on the intended claim and available human
judgments, not an arbitrary number of convenient catalogue entries.

## Collection and comparison after explicit resumption

- Create and freeze the query sample, annotation interface, access conditions,
  candidate-pool procedure, and reporting rules. Do not use model-generated labels
  as human relevance ground truth.
- Pool candidates from the frozen comparison systems and a declared coverage
  sample. Deduplicate by work. Present candidates in independently randomized
  order with system identities and ranks hidden. Record how the pool was built;
  pooling only one system’s outputs would leave its competitors underjudged.
- Obtain independent judgments, preserve abstentions, and adjudicate according to
  an agreed policy. A pilot may improve the rubric but cannot be reused as an
  untouched final test after those decisions.
- Predeclare metrics such as nDCG@k for graded relevance and a constraint-violation
  rate, with the actual k, relevance mapping, aggregation unit, missing-judgment
  policy, and uncertainty method fixed before scoring. Report judgment coverage
  alongside scores. Unjudged candidates must not silently become relevance zero.
- Compare systems on the same query/candidate evidence conditions. Account for
  dependencies between queries sharing a seed or work when planning uncertainty
  estimates. Keep accuracy, catalogue coverage, diversity, and runtime separate.
- Report negative findings and incomplete coverage. Better synthetic runtime or
  passing software tests do not establish better recommendations, book-mood
  understanding, or an advantage for MTL.

## Illustrative record — not evaluation data

This example is invented; the candidate does not refer to a catalogue work. It
shows why incomplete evidence must not be converted into a negative relevance label.

```json
{
  "illustrative_only": true,
  "query_id": "example:quiet-exploration",
  "candidate_work_id": "example:invented-garden-story",
  "rubric_version": "draft-unapproved",
  "rater_id": "example:reader-b",
  "relevance": null,
  "abstention_reason": "Only the blurb was available; the requested sustained reading atmosphere cannot be judged.",
  "constraint_violations": [],
  "evidence_scope": "description_only",
  "eligible_for_scored_evaluation": false
}
```

## Unstarted manifest

This is an explicit empty template, not an evaluation receipt:

```json
{
  "status": "not_started",
  "research_resumption_authorized": false,
  "rubric_frozen": false,
  "catalogue_snapshot": null,
  "held_out_work_ids": [],
  "queries": [],
  "judgments": [],
  "metrics": null
}
```
