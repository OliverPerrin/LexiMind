# Book-field supervision: decision, 27 September 2026

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

Next: review ambiguous groups and cross-source/series identities; review the mapping
and collect negative or complete-label seed evidence. Then implement partial-label
loss/output handling and audit CR4's character-conditioned input. The existing
`topic` runtime is still single-label. The custom transformer, four recipe arms and
independent book-relevance study remain; training and model evaluation stay paused.
Optional corpora and later mood work need not block the genre/topic track.

## Modern books beyond Gutenberg

Open Library supplies modern **catalogue** coverage; publisher blurbs match field
inputs. A [four-work full-text pilot](../../research/preparation/licensed_books_manifest.json)
now preserves the official [Little Brother](https://craphound.com/littlebrother/download/)
text and three [Book Dash](https://bookdash.org/re-using-the-book-dash-content/)
English books, with source hashes, per-work licences, attribution and section boundaries.
Little Brother's NC/SA conditions remain separate from Book Dash's CC BY terms.
One YA novel dominates the text; picture books omit visual context. No model ran.
[OAPEN](https://www.oapen.org/article/metadata) nonfiction and
[Green Comet](https://greencomet.org/welcome/) remain expansion candidates. Reconcile
work identities before assigning any full-text splits. Free access is not blanket
training permission; see the source register for provider-specific limits.
