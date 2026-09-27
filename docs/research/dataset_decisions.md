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

## Next work, in order

- Reconcile BGC work/edition groups and CR4 identities; define one small field
  vocabulary and a positive/negative/unknown mapping. BGC's published splits contain
  repeated identities and blurbs; see its [audit](../../research/preparation/bgc_candidate_manifest.json).
- Prepare a reviewed field-label seed set and work-group split; then implement
  masked multi-label loss and the character-conditioned input contract.
- Resume model feasibility/training only when the existing pause is lifted.

The custom transformer, four recipe arms and independent book-relevance study
remain. BGC's original archive is now preserved and audited; dates are edition dates and
must not be used as original-work ages. Model-quality evaluation remains paused.
The existing `topic` runtime is single-label, so this proposal needs an explicit
loss/output integration before training. Source checks are scoped to the data being
used; optional datasets and later mood work need not block the genre/topic track.

## Modern books beyond Gutenberg

Use Open Library for broad modern **catalogue** coverage. For field training,
publisher blurbs remain the closest input match. For later full-text work, shortlist
[OAPEN](https://www.oapen.org/article/metadata) nonfiction, author-released fiction
such as [Green Comet](https://greencomet.org/welcome/) and
[Little Brother](https://craphound.com/littlebrother/download/), and
[Book Dash](https://bookdash.org/re-using-the-book-dash-content/) children's stories.
Preserve exact edition/licence/source evidence; these collections do not supply
mood gold or representative adult-fiction coverage. Free reading/borrowing or a
preview is not a training licence. The source register records the per-provider
limits, including Google Books caching terms and OpenStax's stated AI restrictions.
