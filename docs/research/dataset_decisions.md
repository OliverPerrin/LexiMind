# Book-field supervision: decision, 2 October 2026

**Use book-domain labels as the main path.** AG News, GoEmotions and arXiv remain
optional controls; their reconstruction does not make them the right product targets.
Better alignment is a reason to test this direction, not a measured accuracy gain.

| Target | First choice | Why / boundary |
| --- | --- | --- |
| Genres | [BlurbGenreCollection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html): 91,894 audited records, 146 hierarchical categories | Matches description inputs. CC BY-NC 4.0; audit editions/work groups and map publisher categories into field facets. Product reuse is a separate decision. |
| Topics | Sourced catalogue headings, distinguishing [topical subjects](https://www.loc.gov/marc/bibliographic/bd650.html) from [genre/form](https://www.loc.gov/marc/bibliographic/bd655.html) | Strong immediate source evidence and possible weak labels. Missing headings are unknown, not negatives. Raw Open Library subjects are not a clean topic vocabulary. |
| Narrative affect | [CR4-NarrEmote](https://aclanthology.org/2025.emnlp-main.493/), preferred literary auxiliary | Human character-emotion responses with passage/document IDs. Provider V1 metadata declares CC0 and open files. This is not reader or whole-book mood. |
| Reader mood | Independently reviewed reading-experience labels; [EmoBank reader annotations](https://github.com/JULIELab/EmoBank) and [IDEST](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274480) as bridges | Sentence/short-story scope, not full-book gold. EmoBank's blended file mixes reader/writer perspectives; use `reader.csv` for that specific target. |

Secondary options: [CMU summaries](https://www.cs.cmu.edu/~dbamman/booksummaries.html)
for a separate plot-summary genre domain; [DOAB](https://www.doabooks.org/en/publishers/guidance)
for scholarly nonfiction topics; DENS as a separate passage-affect diagnostic,
not pooled highlighted-character labels.
ACRec is a useful reader-preference design, but its GPT-derived requests come from
reviews of the target book. Do not use those as independent prospective evaluation
queries. Newer Goodreads/Gemini-derived genre corpora and unverified tone challenges
are watchlist items, not automatic upgrades. Details and source terms are in the
[compact source register](../../research/preparation/book_field_sources.json).

## Training approach

1. **Separate fields.** Genres describe content conventions; topics describe
   aboutness; moods describe a scoped reader experience. Form and audience remain
   separate metadata. The site's current `genres` mix these facets and must not be
   copied directly into a training vocabulary.
2. **Match available input.** Start from title/description for metadata fields.
   Copy trusted source values when present. Exclude target genre/subject fields
   from classifier inputs; richer passages are a separate evidence condition.
3. **Use partial multi-label supervision.** Preserve positive/negative/unknown
   states and valid source-specific hierarchy links. Add reviewed negatives or a
   complete-label seed set: masking unknowns with positives alone can learn an
   all-positive predictor. Keep weak/model-derived labels out of human gold.
4. **Preserve evidence and disagreement.** CR4's raw `t1` responses are human;
   NRC, NRCBERT and EMO mappings are derived. Keep highlighted-character context,
   passage/work IDs and individual responses. Audit `t0` validity; blank/uncertain
   `t1` stays unknown, not neutral or all-negative. Supporting text spans are references,
   not proof that a prediction or its explanation is correct.
5. **Test on held-out works.** Reconcile editions/series and cross-source copies;
   calibrate per-field thresholds separately. Report accuracy with coverage and
   abstentions. Optional later book-text denoising uses the same adapted base and
   recorded cost for every comparison arm.

## Prepared; next work

BGC now has **89,910 connected leakage groups** across 91,894 rows. The proposed
64/16/20 allocation keeps all five detected identity/text match types together;
849 groups crossed the original splits. These are not adjudicated literary works:
shared blurbs connect 61 Shakespeare records. The [group audit](../../research/preparation/bgc_group_manifest.json)
retains conflicts and review references. Two rare raw categories are absent from
dev; all 48 mapped field targets occur in every proposed split.
The [field mapping](../../research/preparation/book_field_mapping.json) separates
genre/topic/form/audience, preserves composite genre umbrellas, and supplies only
weak positives; omissions stay unknown. Text remains in the original ZIP.

The [bounded group review](../../research/preparation/book_group_review.json) covers
7 components / 113 rows and screens the 120-book catalogue plus four licensed texts.
It finds 24 identity-candidate pairs, including Little Brother in BGC. The
[partition overlay](../../research/preparation/book_partition_manifest.json)
now keeps strong links together and quarantines unresolved components: effective
BGC train/dev/test counts are 58,889 / 14,630 / 18,326, with 49 rows quarantined.
All 48 mapped targets retain positive support in each split. Title-only matches
remain unresolved; the overlay is a candidate, not a frozen work-identity truth.
The [32-record field review](../../research/preparation/book_field_review.json) contains
29 agent positive suggestions, one explicit negative candidate and 17 abstentions.
Human review is empty; these suggestions do not enter training labels or gold.
The local review page records explicit judgments and cited spans, hides agent
suggestions initially, and imports drafts into a separate unadmitted packet.

The [CR4 audit](../../research/preparation/cr4_candidate_manifest.json) covers
207,721 raw annotations and 43,142 passage/character conditions. Raw `t1` is free
text, not a categorical emotion target; `t1_corrected`/`t1_unified` include LLM
cleaning. The audit preserves duplicates, ambiguous highlights and disagreement.
Document IDs are not verified literary works; ontology, rights and leakage review
remain open before any training export.

Partial-label loss, data masks, observed-only diagnostics and sigmoid field output
are implemented and tested on synthetic inputs; see [runtime contracts](../architecture.md).
Legacy topic CE remains the default. Next: human/source conflict adjudication,
representative negative or complete-label evidence, and work-level split freeze.
CR4 needs a character-conditioned target decision before label conversion.
The custom transformer and four recipe arms stay;
formal field training still needs reviewed data. The separately recorded local
continuation pilot establishes execution feasibility only. Optional corpora need
not block this track.

A [bounded weak source-label diagnostic](../../research/results/book_field_baseline_20261003.json)
now fits training-only TF-IDF prototypes on 4,096 singleton groups and evaluates
1,024 development groups. Known-label recall@3 exceeds frequency ranking for
genre/topic, but unknown labels supply no precision or false-positive evidence.
The [research index](README.md#book-field-source-recovery) records all controls,
rare-label limits and the separate 32-record training-only review worksheet.
The old purposive review packet contains effective test records and is not used
as a training seed. No human labels or formal field-study admission were created.

The [8 October neural source-recovery pilot](../../research/results/book_source_recovery_20261008.json)
completed and passed independent audit on the same fixed 4096/1024 cohort. At the
predeclared 512-update endpoint, Q/V LoRA improved genre/topic observed-positive
recall at 3 over each paired frozen-encoder head. Both seeds remained below the
matched positive-centroid TF-IDF control: descriptive LoRA means were 75.24% /
74.66%, versus 83.27% / 88.90% for genre/topic. All four facets, seeds and
endpoints are retained; no best-seed or earlier-checkpoint selection was made.

This is source-assignment learning, distinct from semantic partial-label BCE.
Per-facet softmax pressures unassigned labels and co-positives compete; source
omissions stay unknown. Publisher-label recovery supplies no precision/F1,
human-gold, semantic-quality, significance or product-admission evidence. Unknown
FLAN exposure to public blurbs also limits lexical-versus-neural interpretation.
Keep the strong lexical control. Data/objective investigation and any longer
fixed neural budget require a future protocol; this bounded result does not
predict that more updates would win. Human field labels remain empty. No
automatic RL or neural website promotion follows from this pilot.

The [subsequent broader-exposure diagnostic](../../research/results/book_source_data_scaling_20261008.json)
retained the old training prefix and development cohort while using 16,384
training groups in four passes at the same 4,096-update budget. It completed and
passed CPU/GPU audit. LoRA mean final genre/topic group recall at 3 was 87.48% /
92.41%, above the refitted lexical 84.97% / 91.09%; label macro was 77.38% /
87.52%, slightly below lexical 77.72% / 88.54%, with opposing seed crossings.
Source development loss kept improving at the final endpoint, with a smaller
train/dev gap than the historical smaller-cohort run. These are descriptive
historical comparisons: unique exposure and repetition changed together, with
no fresh small-cohort controls. Preserve lexical controls and inspect support
and source-objective limits; no automatic more-epoch, PPO or product promotion
follows. Human semantic field labels and formal admission remain unresolved.

The [saved-output support/objective analysis](../../research/results/book_source_objective_analysis_20261008.json)
now decomposes all 48 labels without new fitting or model execution. Genre/topic
LoRA two-seed group-macro gains coexist with small label-macro deficits because
book-positive and equal-label denominators weight the same source-hit changes
differently. Those deficits reverse by seed. Low development support is not
proof of training rarity: spiritual fiction/paranormal fiction have 117/293
training positives, and finance/technology 50/114. A single technology source hit
would move topic label macro more than its observed mean deficit; this is
arithmetic sensitivity, not statistical uncertainty or semantic error evidence.

Keep the training-support view separate. A future capped/tempered training-only
source-target weighting trial may test alignment with label-macro recovery using
fresh contemporaneous unweighted controls, but no formula is justified as a
winner. Freeze one variant, budget, normalization and all group/label outcomes
before execution; never derive weights from dev misses. Coefficient mass is not
gradient mass, and no causal claim or new training/promotion follows from these
saved-output correlations. Unknown omissions and human-label gaps remain.

## Modern books beyond Gutenberg

Open Library supplies modern **catalogue** coverage; publisher blurbs match field
inputs. A [four-work full-text pilot](../../research/preparation/licensed_books_manifest.json)
now preserves the official [Little Brother](https://craphound.com/littlebrother/download/)
text and three [Book Dash](https://bookdash.org/re-using-the-book-dash-content/)
English books, with source hashes, per-work licences, attribution and section boundaries.
Little Brother's NC/SA conditions remain separate from Book Dash's CC BY terms.
One YA novel dominates the text; picture books omit visual context. No field model
has been trained from these texts.
[OAPEN](https://www.oapen.org/article/metadata) nonfiction and
[Green Comet](https://greencomet.org/welcome/) remain expansion candidates. Reconcile
work identities before admitting full-text splits. Free access is not blanket
training permission; see the source register for provider-specific limits.

A [separate Book Dash expansion](../../research/preparation/bookdash_manifest.json)
now preserves 35 additional English works and 12,541 narrative words. Two BGC
title ambiguities are quarantined; the remaining 33 have fixed work-level
train/validation/test roles. The original four-work pilot stays unchanged. These
are children's texts with omitted illustrations, not broad fiction coverage.

The optional [RPT preparation](../../research/preparation/rpt_candidate_manifest.json)
contains 32 token-boundary continuation examples from those four works (24 train,
8 test, no dev). Targets use standalone tokenizer-normalized text with verified
round trips, not original source-byte identity. This tiny preparation pilot is
not an evaluation set or a reproduction of the 14B RPT experiment.
