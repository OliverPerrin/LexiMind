# Reader-mood annotation rubric: draft

Target: the reader experience of a specified work/edition, not a character's emotion,
a sentence's sentiment, blurb tone or review-star rating. The active data choices
are in [dataset decisions](research/dataset_decisions.md). No judgments exist yet.

| Candidate label | Working meaning |
| --- | --- |
| Reflective | Sustained contemplation or inward attention |
| Playful | Sustained lightness, wit or mischievous energy |
| Tense | Sustained anticipation, pressure or apprehension |
| Unsettling | Sustained unease or disturbed expectations |
| Melancholic | Sustained wistfulness, sorrow or reflective loss |
| Hopeful | Sustained renewal or positive possibility |
| Bleak | Sustained desolation or constrained prospects |
| Wonder-filled | Sustained discovery, awe or imaginative possibility |

Labels may coexist and change over a book. This vocabulary needs a reader pilot;
it is not a validated taxonomy. A genre, topic or isolated scene is insufficient
evidence. Unknown is not neutral, and disagreement is not automatically error.

## Required record

Keep work ID, edition/translation, source snapshot, language, pseudonymous rater,
rubric version, reading coverage and scope (`whole_work` or `excerpt`). Record
labels, evidence status, short rationale, locations and counterevidence. Do not
store rater contact details or long copyrighted quotations in judgment rows.

Complete reading with an identified edition and evidence locations is required
for whole-work gold eligibility. Excerpt observations stay passage-scoped;
metadata/description impressions cannot become whole-work gold. A human-origin
flag does not prove actual reading or the truth of a label.

## Collection and review

1. Pilot the vocabulary, source access and handling with readers before freezing it.
2. Obtain independent judgments without model outputs or other raters' labels.
3. Preserve raw responses, abstentions and disagreement; adjudication is a separate
   record. Define agreement/coverage rules before inspecting comparative results.
4. Group works/editions before model splits. Keep rubric-development examples
   separate from final evaluation.
5. Enable product mood fields only after the source scope and independent field
   accuracy are supported. Calibrate model confidence separately from rater certainty.

CR4-NarrEmote character labels and EmoBank reader sentence scores may help transfer;
neither substitutes for this target. ACRec-style review-derived requests are useful
weak/design evidence, not independent prospective test queries. Schemas and explicit
fixture rules live in `src/research/annotations.py`; production and synthetic data
must remain distinguishable.
