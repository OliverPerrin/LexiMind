# Dataset decisions before training

Status: source candidates and assignment plans, still unadmitted. The original
processed files remain unchanged. [GoEmotions](goemotions_reconstruction.md) and
[AG News](ag_news_reconstruction.md) have pinned local reconstructions and
[deterministic development/calibration assignments](partitions.md). The arXiv
extension has a separate [author-source reconstruction](arxiv_reconstruction.md).

| Resource | Proposed role | Admission work still needed |
| --- | --- | --- |
| GoEmotions simplified | M1 multilabel comment-emotion task | Provider comment IDs, original label order and official partitions are restored. Review repeated inputs/annotation disagreements and the proposed selection/calibration assignments. Repository card declares Apache-2.0, which is a recorded declaration rather than a blanket source-rights opinion. |
| AG News | M1 single-label topic candidate replacing the small seven-class legacy mix | Review original provider/source terms; the inspected Hub card says license unknown. Pinned source-row identities are reconstructed; original article IDs are not supplied. Review the proposed partitions and duplicate/generalization policy. |
| arXiv summarization | M1 generation extension candidate | Recover paper IDs/revisions, source-use basis, article/abstract pairing and lengths. The inspected Hub card has no license value; no blanket permission is inferred. Academic abstract generation does not validate literary blurbs. |
| Current 102-work catalogue | B1 product population and unlabelled annotation preparation | Choose query/eligible-work scope and evidence languages; verify identities; collect independent judgments later. It is curated, small and not population-representative. |
| BookSum | Possible later document/chapter generation study | Preserve provider parent bid and official partitions, reconcile cross-source works, separate paragraphs/chapters/full books, inspect original summary rights. A code license does not license every text. |
| CMU book summaries | Possible genre/text research candidate | Review CC BY-SA 3.0 US source terms and work identity; it contains plots/genre metadata, not reader-preference or whole-work mood gold. |
| Goodreads UCSD | Conditional offline interaction benchmark | Provider statements restrict use to academic work and prohibit redistribution/commercial use. Keep eligibility review and data access separate from the public website. Ratings/interactions are not query relevance or atmosphere labels. |
| DENS | Conditional passage-emotion research | Provider access/usage conditions apply. Short-passage dominant emotion is a different target from sustained book atmosphere. |

Provider URLs, precise scope and unresolved rights statements are in the
[book-data review](book_discovery_review.md) and its structured evidence register.
Missing license metadata does not by itself establish either permission or prohibition.

## Current processed files

The [local audit](data_readiness.md) makes all four current corpora ineligible for a
fresh controlled study until reconstruction/review. Files remain unchanged. In
particular, the 18,753 literary source/summary pairs must not be revived by accepting
title-only joins or by assigning the same blurb to arbitrary excerpts.

Repeated strings are review units, not automatic deletion instructions. For comment
classification, identical text may occur in distinct source comments. For source
descriptions repeated across excerpts, the parent work may be the same, but the
current files cannot establish that. Rehydrate provider IDs before deciding whether
to preserve official benchmark splits, group duplicates in a new version, or report
a separate contamination-sensitive analysis. Changing published splits must be named
as a new dataset/protocol rather than silently retaining the old benchmark label.

Source provenance may live in a pinned dataset-level manifest plus stable row IDs;
it need not duplicate a full URL in every row. The current audit records absent
row-level provenance fields and absent parent IDs, while the historical source
revision also remains unresolved. New manifests must explicitly connect the two.

## Admission boundary

Before any download/rebuild is admitted, write a manifest containing provider URL,
immutable revision, source terms reviewed, config/subset, original record and parent
IDs, label order, preprocessing revision, source-to-canonical mappings and unresolved
cases. Output files need hashes, counts, exclusion reasons and an audit of train,
selection, calibration and final-test boundaries. Preserve raw versus normalized
duplicate decisions separately. Parent-document grouping alone does not prove
canonical work/edition separation across datasets.

Keep the reconstructed source candidates and their reports separate from admitted training data. Do not overwrite
`data/processed`, publish restricted datasets, manufacture human labels, or interpret
the existing website's software tests as a relevance evaluation.
