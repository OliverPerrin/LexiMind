# Book mood annotation — draft guide

Status: **preparation only; collection has not started**. This document does not
resume research, establish a taxonomy, or authorize model inference, teacher APIs,
paid annotation, or training. The research pause in
[research_plan_2026.md](research_plan_2026.md) and the draft
[eval_protocol.md](eval_protocol.md) still applies. No book has received a gold mood
label through this guide, and the website’s mood arrays remain empty.

## What a mood label would mean

The proposed target is the **emotional atmosphere a reader experiences while
reading the work**, at a stated scope. It is not the emotion expressed by one
character, the sentiment of a review, or the tone of a publisher’s description.
Readers may experience a work differently; disagreement is evidence to retain.

Keep these observations separate:

| Observation | Unit and evidence | Example using invented material |
| --- | --- | --- |
| Book mood | Sustained reading experience, with coverage of the work recorded | A reader finds a fictional story persistently unsettling despite its cheerful narrator. |
| Character emotion | A particular character in a particular scene | A character feels joy in one scene of that same story. |
| Description tone | The marketing or catalogue text itself | A playful blurb advertises that story. |
| Subject or genre | Explicit bibliographic categories or story content | The story concerns an expedition; this does not by itself establish wonder, fear, or excitement. |

A model score from the existing Reddit-comment emotion task is **not** evidence of
a book’s mood. A subject such as death does not automatically mean “melancholic,”
and a genre such as romance does not automatically mean “hopeful.”

## Proposed annotation fields

Before collection resumes, review the vocabulary with readers and freeze a version.
The following is a starting proposal, not validated ground truth:

| Candidate label | Working definition | Do not infer it solely from |
| --- | --- | --- |
| Reflective | Sustained invitation to contemplation or inward attention | A philosophical subject heading |
| Playful | Sustained lightness, wit, or mischievous energy in the reading experience | One joke or a comic cover |
| Tense | Sustained anticipation, pressure, or apprehension | A crime or action genre |
| Unsettling | A sustained sense of unease, disorientation, or disturbed expectations | Violence appearing somewhere |
| Melancholic | Sustained wistfulness, sorrow, or reflective loss | An isolated sad scene |
| Hopeful | Sustained sense that renewal or positive possibility remains available | A happy final paragraph |
| Bleak | Sustained desolation or constrained prospects | A difficult topic or an unhappy character |
| Wonder-filled | Sustained sense of discovery, awe, or imaginative possibility | A fantasy or science-fiction category |

These labels are not mutually exclusive. Do not require a fixed number of labels.
Use **insufficient evidence** when the material does not support a work-level
judgment. Do not force a neutral label to fill a missing answer. If the mood changes
substantially, record the change and scope rather than flattening it into a single
confident tag. The decision about which dimensions and terms survive a pilot remains
open until research is authorized again.

A proposed row would contain:

- Stable work identity, edition/translation actually read, and the catalogue/source
  snapshot revision. Edition-sensitive evidence must remain edition-sensitive.
- Pseudonymous annotator ID, rubric version, language, annotation date, and declared
  prior familiarity. Do not store annotator contact details in the research rows.
- `scope`: `whole_work` or `excerpt`; `reading_coverage`: `complete`, `partial`, or
  `description_only`. Excerpt and description-only impressions are ineligible for
  whole-work gold labels.
- Zero or more candidate mood labels; `evidence_status`: `supported`, `uncertain`,
  or `insufficient_evidence`; and a concise rationale written by the annotator.
- Evidence locations (chapter/page/section and edition), relevant transitions, and
  counterevidence. Store a paraphrase by default; do not copy long copyrighted
  passages into the dataset.
- Any separately observed character emotion or description tone, explicitly marked
  as a different target. Neither is converted automatically into book mood.
- Individual judgments and any later adjudication as separate records. An
  adjudicated value must not overwrite disagreement.

“Confidence” would describe the annotator’s evidence and uncertainty. It would not
be presented as a calibrated probability or a model quality score.

## Proposed procedure after explicit resumption

1. Verify the work/edition identity and lawful access to the material. Metadata or
   a blurb alone cannot establish a whole-work gold judgment.
2. Agree reader instructions, languages, permitted evidence, consent/privacy
   handling, annotation compensation if any, and a versioned vocabulary before
   collecting labels. No annotation procurement is authorized by this document.
3. Have at least two readers annotate independently without model predictions,
   existing inferred mood tags, or each other’s answers. Record coverage first.
4. Retain label-level agreement, abstentions, disagreements, and rationales. Review
   recurring confusion, including mood changes and cultural/translation differences.
   Do not invent an agreement threshold after seeing the results.
5. Freeze the final rubric, agreement/adjudication policy, and minimum evidence
   requirements before producing a held-out evaluation set. If a pilot changes the
   rubric, keep the pilot apart from the final held-out collection.
6. Only admit labels to the public catalogue after provenance, scope, coverage,
   label-schema, and disagreement handling have been explicitly reviewed. A future
   schema change and validation tests are required; this guide does not relax the
   current rejection of unvalidated moods.

## Illustrative record — not a book label

The following is invented solely to show the shape of a possible record. Its ID is
intentionally not a valid catalogue work ID; it must never enter production data or
an evaluation dataset.

```json
{
  "illustrative_only": true,
  "work_id": "example:invented-expedition",
  "edition_id": "example:invented-edition",
  "rubric_version": "draft-unapproved",
  "annotator_id": "example:reader-a",
  "scope": "excerpt",
  "reading_coverage": "partial",
  "mood_observations": ["unsettling"],
  "evidence_status": "uncertain",
  "rationale": "The invented opening creates unease, but the rest of the work has not been read.",
  "evidence_locations": ["invented opening scene"],
  "counterevidence": "The cheerful narrator might alter the reading experience later.",
  "eligible_for_whole_work_gold": false
}
```

## Unstarted collection

No annotators have been recruited, no agreement values have been measured, and no
held-out works have been chosen under this guide.

```json
{
  "status": "not_started",
  "rubric_frozen": false,
  "held_out_work_ids": [],
  "annotations": []
}
```
