# Book discovery: literature and dataset suitability

Checked **2026-09-26**. This is research preparation, not a completed experiment,
dataset acquisition, rights clearance, or a novelty claim. No model was run, no
reader was contacted, and no relevance or mood labels were created. The current
product is the 102-work catalogue described in [product.md](../product.md);
its 96 sourced descriptions and empty mood arrays are product metadata, not gold
recommendation judgments. No browser-local reading history was accessed.

The evidence register is [book_literature.json](../../research/preparation/book_literature.json).
It records nine selected research source groups, provider documentation, versions,
reading scope, and claim locators. Methods and relevant appendices were inspected;
the scope is deliberately recorded rather than claiming every cited version was
read in full. In particular, the inaccessible KDD 2020 sampled-metrics PDF is not
marked read: its full 2019 precursor and the authors' substantive 2021 account were
read separately. This is a focused review, not an exhaustive recent-work survey.

## What question the evidence supports

The most defensible applied study is: **for a fixed, identified book catalogue and
stated reading requests, do learned features improve query–work relevance beyond
lexical retrieval, a frozen text-embedding baseline, and explicit metadata?**
This is a proposed comparison. It does not require launching a collaborative
recommender or inventing whole-book mood labels.

The current [B1 working decision](study_decisions.md) specifies **new query families
over the fixed catalogue**. Query families and seed-work groups are partitioned;
candidate works may be reused across those query partitions. It makes no unseen-work
claim. Primary scoring requires complete eligible-catalogue judgments; a pooled
comparison is secondary only. The [admission contract](book_admission_contract.md)
implements these boundaries without creating queries, partitions, or labels.

Keep three targets separate:

| Target | Observation needed | What does not establish it |
| --- | --- | --- |
| Query–work fit | Independently judged reading request and candidate, with evidence scope | Matching genre strings, a save event, or a model's explanation |
| Personal preference / future interaction | A consented user's preferences or properly scoped interaction history | General topic relevance or absence of an interaction |
| Model task quality | Held-out targets for the stated classification/generation task | A ranking improvement, or the same task's metric in another domain |

The model comparison of joint training, specialists, and merged adapters belongs
in [eval_protocol.md](../eval_protocol.md). A book-ranking benefit cannot isolate
an effect of output type; a generation/classification result cannot establish that
readers discover better books. Neither implication is established by the studies
below.

## Methods evidence and limits

| ID / primary reading | Method and experimental design actually inspected | Consequence for LexiMind |
| --- | --- | --- |
| **LIBRA** — Mooney & Roy, [DL 2000 paper](https://www.cs.utexas.edu/users/ml/papers/libra-dl-00.pdf), system description and experimental methodology | Slot-specific bag-of-words naïve Bayes learns a user's positive/negative profile from ratings. Amazon descriptions, subjects, authors, and related-item fields are features. Five genre-specific rating sets use ten-fold cross-validation and top-3/top-10 precision, average rating, and rank correlation. Unfamiliar books could be rated from their pages. | Direct precedent for content-based book discovery and feature explanations. Page-based anticipated interest is not post-reading satisfaction. Small reader coverage and genre-selected candidates limit generalization; related-item fields also carry collaborative information. Preserve a plain-text baseline and report metadata ablations. |
| **GOODREADS** — Wan & McAuley, [RecSys 2018](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf), §§2–5 | chainRec shares user/item factors across ordered interaction stages, enforces monotonic scores, and optimizes adjacent-stage evidence. Goodreads uses shelve/read/rate/recommend chains. Filtering removes inactive users and items below five interactions; validation and test sample interaction chains. AUC and nDCG assess ranking. The authors report sliceTF slightly outperforming chainRec on Goodreads. | Multi-signal joint learning is established prior work, not automatic proof for LexiMind's tasks. Shelf, read, and high-rating signals have distinct semantics. This setup is neither unseen-work retrieval nor a causal estimate of reading satisfaction. |
| **NRT** — Li et al., [SIGIR 2017 / arXiv v1](https://arxiv.org/pdf/1708.00154v1), §§3.2–4.5 | Shared user/item factors feed rating regression, review-text prediction, and GRU tip generation; a weighted joint loss couples them. Amazon Books is included; review-summary fields serve as tips. The paper uses an 80/10/10 split, MAE/RMSE for ratings, and ROUGE for tips. Its LexRank comparison uses ground-truth ratings for review filtering. | Relevant precedent for joint numeric/text outputs, but its metrics do not demonstrate top-k discovery or book-mood understanding. User/item embeddings do not directly solve cold-start text queries. Do not provide test ratings or test reviews to a LexiMind baseline. |
| **CMU** — Bamman & Smith, [2013 v1](https://arxiv.org/pdf/1305.1319v1), §§3–5 and appendix example | Passage- and token-HMM alignments produce sentence-inclusion supervision for logistic-regression extractive summarization. The paper evaluates 439 Gutenberg/Wikipedia pairs with ten-fold cross-validation and ROUGE against plot summaries; the public dataset is a different, larger release. | Useful alignment/summary research, not relevance supervision. A plot recap is a different target from a spoiler-light discovery description. Group all derived sentences under the verified parent work. |
| **BOOKSUM** — Kryściński et al., [Findings EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.488.pdf), §§3–4, appendices A–C | Book/chapter pairing checks title and author; paragraph alignments use semantic similarity and stable matching. Splits group all granularities by book title. The alignment pilot uses three judges on 100 pairs. Long summaries use generate-and-rank baselines; human relevance/factuality checks cover paragraphs only, not chapters/books. | Strong precedent for identity checks and hierarchical splits. Algorithmically aligned paragraphs are not independently authored paragraph gold. Neither summary overlap nor paragraph factuality proves work-level recommendation value. |
| **DENS** — Liu et al., [EMNLP 2019](https://aclanthology.org/D19-1656.pdf), §§3–4; [arXiv appendix A](https://arxiv.org/pdf/1910.11769) | Annotators select dominant emotion for 40–200-token narrative passages. Three raters, majority decisions, expert fallback, popularity-based modern-story sampling, and partial lexicon filtering shape the data. The benchmark excludes Surprise and Disgust, masks entities, and uses five-fold micro-F1. | Closer to literary language than Reddit comments, but still passage emotion, not sustained reader atmosphere. Work-group isolation is not established by the stated folds. Keep this a possible separate transfer diagnostic, not mood ground truth. |
| **IMPLICIT** — Hu, Koren & Volinsky, [ICDM 2008](https://yifanhu.net/PUB/cf.pdf), §§4–6 | Weighted matrix factorization separates binary consumption preference from confidence, using alternating least squares over observed and unobserved pairs. Television data uses four training weeks and the next week for testing; repeated programs are removed for discovery evaluation. | Non-consumption is uncertain evidence, and more consumption need not mean greater liking. Future saves, dismissals, exposure, and reading completion require distinct definitions; current local shelves are not an authorized interaction dataset. |
| **SAMPLED** — Rendle, [2019 v1](https://arxiv.org/pdf/1912.02263v1), §§2–4.5; Krichene & Rendle, [IJCAI 2021 account](https://www.ijcai.org/proceedings/2021/0651.pdf), §§2–5 | Analysis of sampled ranks demonstrates that ranking among sampled negatives can reverse system comparisons. The single-relevant-item derivation uses a binomial rank distribution. The later account describes bias/variance corrections and MovieLens experiments. | Rank the full eligible 102-work catalogue. Candidate sampling and incomplete human judgments are different problems; this result does not make an unjudged item irrelevant or prove a particular pooling estimator valid. |
| **INCOMPLETE** — Buckley & Voorhees, [SIGIR 2004](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469), §§2–6 | Progressively reduced TREC relevance sets test system-order stability. bpref compares judged relevant/nonrelevant pairs and is more resilient than rank-dependent measures under their conditions. The discussion still requires adequate, fair pools and cautions about novel noncontributing systems. | Keep unjudged and explicitly irrelevant separate. Pooling is a documented evidence-collection design, not a license to claim exhaustive recall. Binary bpref can be a sensitivity analysis; it does not replace graded relevance or resolve biased pools. |

## Dataset suitability and terms

Access statements describe provider documentation checked today. No listed dataset
was downloaded, no access request was sent, and no release was admitted to training
or deployment. A paper's copyright, a repository's code license, and rights in the
underlying text are separate. The final admissible snapshot and intended-use review
remain open; a mirror's license label must not override the originating provider.

| Resource | Actual labels, unit, and available signals | Access / stated terms / provenance | Proposed use and principal mismatch |
| --- | --- | --- | --- |
| [UCSD Goodreads Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) | User–book interactions, ratings/reviews, book metadata, separate work records. Genre tags are keyword-derived from popular shelves; books may belong to several genre subsets. No query-relevance or whole-work mood gold. | Late-2017 collection; provider lists public downloads and says academic use only, no redistribution or commercial use. Exact files, hashes, and edition→work mapping would need pinning. | Optional offline preference benchmark after an intended-use review, not a public catalogue feed. Public-shelf selection, active-user filtering, popularity, exposure, and multiple editions affect results. |
| [BookSum author release](https://github.com/salesforce/booksum#legal-note) | Literary source/summary pairs at paragraph, chapter, book level; multiple summaries per work. No user preference, genre gold, or reading-atmosphere labels. | Repository supplies alignments, chapterized Gutenberg access, and source-collection scripts. Legal Note limits scripts to research and leaves source terms/rights to users. BSD-3 applies to code, not blanket summary-text rights. Repository is archived. | Possible bounded summarization task after source review. Classics/study-guide selection, spoilers, long context, and source-dependent style differ from public discovery blurbs. |
| [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html) | 16,559 Wikipedia plot summaries with Freebase-linked author/title/genre metadata; public release is not the paper's 439 paired-book experiment. Wikipedia/Freebase identity is work-like, not verified edition identity. | Provider links a 17 MB archive and explicitly links [CC BY-SA 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/legalcode). Snapshot hashes, source attribution, changes, and intended reuse obligations still need documenting. | Candidate genre/summary-text resource; not a book-text corpus or recommendation test set. Incomplete/overlapping genres and encyclopedic plot style require auditing; do not reproduce the legacy title-only join. |
| [DENS](https://aclanthology.org/D19-1656.pdf) | 9,710 narrative passages with a dominant-emotion label and agreement information; nine annotation categories, seven in the reported benchmark. | Paper offers access by request for noncommercial research only. No request was sent; present delivery, exact version, work IDs, and detailed redistribution terms are unverified. | Conditional narrative-emotion diagnostic. Cannot supply whole-work mood labels or unrestricted website features. Keep source works together if acquired. |
| [Open Library provider](https://openlibrary.org/developers/licensing) and existing catalogue | Work/edition/author identities, community subjects, descriptions, covers and bibliographic metadata. Current site has no independent relevance or mood judgments. | Licensing page, last edited 2021-06-27, says Internet Archive asserts no new rights but existing rights may remain. This is not a universal clearance for descriptions/covers. Record the per-record source and any reuse uncertainty. [API documentation](https://openlibrary.org/developers/api) is the access authority. | Current product-aligned retrieval corpus. Preserve work/edition distinctions and source hashes. Community subjects are evidence-backed metadata, not automatically adjudicated genre truth; the curated 102 works do not represent all books/readers. |

LIBRA's historical Amazon collection, NRT's Amazon/Yelp resources, and Hu et al.'s
television data are methodological evidence here, **not acquired candidate datasets**.
Their paper descriptions do not establish current access or reuse permission. We
should not add a new dataset merely to reproduce a paper whose task differs from the
book-discovery question.

## Proposed collection and evaluation decisions

These are recommendations for the later frozen protocol, not completed gates.
They refine [recommendation_judgments.md](../recommendation_judgments.md) and
[mood_annotation_guide.md](../mood_annotation_guide.md), without authorizing collection.

1. **Fix the retrieval unit and evidence conditions first.** Rank a work once;
   retain edition/translation-specific restrictions separately. Use the same
   description/subject snapshot and availability conditions for every arm. Do not
   give a learned model full text or user reviews while calling a blurb-only lexical
   comparison an isolated model-method effect. If richer evidence is studied, make
   it a separate, explicit factor.
2. **Separate ordinary relevance from reader experience.** Start with topic/genre
   and aspect-specific seed-book requests that the permitted evidence can support.
   Annotators must record whether a judgment uses metadata, a description, an
   excerpt, or complete reading. A description-only relevance judgment is eligible
   for that limited task; it cannot establish sustained whole-work mood. A request
   needing unavailable evidence receives an abstention, not zero relevance.
3. **Prepare queries independently of the systems.** Freeze literal request,
   clarified intent, exclusions, seed IDs, and query-family IDs before showing
   results. Keep pilot/rubric development, tuning, and final queries separate.
   Multi-constraint requests should preserve what the reader actually prioritized.
   The source of queries and the consent/privacy arrangements remain undecided.
4. **Prefer complete judgment within this small catalogue.** Later, assess whether
   qualified readers can judge all eligible works per chosen query. If not, pool
   top candidates from every frozen baseline plus a prespecified coverage sample,
   deduplicate by work, and independently randomize presentation without scores,
   system names, or generated explanations. Retain contributor IDs and pool depth
   in a separate manifest. Recheck coverage if another system is added. No pool,
   annotator roster, or actual query set has been created.
5. **Use a graded relevance target with explicit missingness.** The draft 0–3
   rubric is usable for piloting. Proposed primary endpoint: query-level nDCG@10,
   gain `2^grade - 1`, log-base-2 discount, averaged across the frozen query set.
   Rank all eligible works; freeze tie-breaking and handling of lists below ten.
   Primary scoring requires every eligible query–work pair to have an adequate
   reviewed judgment; compute its ideal ranking over that same catalogue. Any
   unresolved eligible pair blocks the primary comparison, not just an unjudged
   top-ten result. A separately specified secondary *pool-conditional nDCG* may
   disclose coverage and sensitivity analyses, but cannot replace the primary
   complete-catalogue endpoint. Preserve abstentions; never impute zero. Freeze
   treatment of queries with no relevant candidates before observing differences.
6. **Keep secondary metrics interpretable.** Report hard-constraint violations,
   judged precision@10 (if grade ≥2 is frozen as the binary threshold), judgment
   and catalogue coverage, duplicate-work rate, and relevant-author/subject diversity.
   Recall@10 needs a complete denominator or an explicit judged-pool qualifier.
   Report latency/storage separately. Human ratings of explanation support are a
   separate outcome; fluent explanations are not evidence of ranking correctness.
7. **Preserve uncertainty.** Collect judgments only after authorization. The current
   primary admission contract requires two distinct-rater, non-abstaining judgments
   per eligible primary pair and a separately pinned human adjudication source.
   Retain other disagreements and abstentions for review. Predefine agreement
   reporting and paired uncertainty over query
   families/seed-work clusters. Training seeds are another source of variation, not
   extra independent readers. Do not choose sample size or a success threshold from
   convenient catalogue size or a desired positive result.

## Identity, leakage, and feedback controls

These controls are protocol proposals informed by the studies, not evidence that
the current research data has passed them.

- Build a versioned mapping from dataset record → edition/translation → canonical
  work, preserving source IDs and unresolved joins. For a work-held-out supervised
  study, group its chapters, excerpts, summaries and duplicate editions before
  splitting, including cross-dataset overlap. Title normalization alone cannot
  resolve identity. This supervised-data rule does not partition B1's fixed
  candidate catalogue: its held-out units are query families and seed-work groups.
- Declare the intended cold-start claim: unseen supervised work, unseen user,
  unseen seed/query family, or future interaction. These are different tests. Text
  can be indexed at retrieval time without using its held-out labels for training,
  calibration, or feature selection. A fixed pretrained encoder may already know
  famous books; work-level splits do not prove absence from pretraining.
- For any later interaction study, order training/validation/test by the declared
  time boundary, include only information available then, and distinguish new-book
  discovery from repeats. User reviews written after consumption must not become
  features for predicting that earlier consumption. Author/series holdouts are
  separate generalization analyses; they must not silently change B1's stated scope.
- Do not reinterpret browser-local favourites as ratings, dismissals as universal
  dislike, or unobserved works as negatives. There is currently no consented central
  interaction dataset. If collection is ever separately authorized, document exposure,
  event meaning, provenance and missingness before choosing a feedback model.
- Separate input-derived metadata targets from discovery judgments. Predicting a
  supplied subject or a deterministic genre mapping can test label recovery; using
  the identical overlap rule to judge recommendations would be circular.
- Retain the legacy random tones and title-matched Gutenberg/Goodreads pairs only
  as quarantined historical artifacts. They cannot be training targets, evaluation
  labels, or new catalogue features.

## Remaining decisions

The preparation supports a **bounded application/replication study**, not a claim
that content-based books, multi-task recommendation, or affective text modelling is
new. No reviewed resource directly supplies the proposed sustained whole-work mood
judgments. This is a limitation of the reviewed sources, not a claim that none exist.

Before execution: choose the catalogue snapshot and evidence scope; settle admissible
training resources and source rights; freeze work/edition mappings and splits; decide
query sampling and judgment coverage; approve the collection process; freeze baselines,
metrics, uncertainty and success criteria; then obtain explicit authorization for the
relevant experiments. Dataset access, modern recommendation/reader-advisory follow-up,
and an operational review of source terms remain open. Training and evaluation stay
paused throughout this preparation.
