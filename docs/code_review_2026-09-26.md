# Continuation and code review — 26 September 2026

This review covered the website, ranking, catalogue/source pipeline, model and task
heads, training/evaluation scripts, inference/API, persistence, plotting, provenance
tools and CI. It is a software review, not a new model or recommendation study.
Training, real-checkpoint inference, model evaluations and paid teacher calls remain
paused. The historical source data, checkpoints, figures and archived result JSON
were not replaced by new model outputs.

## Product continuation

- The catalogue now contains 102 work records and 96 source descriptions. Three
  additional subject batches required 19 API requests, all cached. Three new records
  were excluded because contributor roles were not sufficiently clear. Original
  response bytes and the first release's snapshot remain available.
- Source subjects link into exact subject browsing. Genres and subjects combine
  with text queries and seed-book recommendations.
- Shelf exports/imports preserve saved, favourite and hidden IDs, including IDs not
  in the current catalogue. Imports merge, have explicit size/schema bounds, and
  never silently replace the existing shelf. Corrupt stored data is not overwritten.
- Shelf operations re-read current storage, preserve queued failed-write intentions,
  and use Web Locks for cross-tab serialization where available. Older browsers
  without Web Locks have a read-latest fallback, not an atomic cross-tab guarantee.
- Cards, cover fallbacks, facets and shelf indexes are memoized; inactive views skip
  unnecessary ranking. A Strict Mode dialog race and storage-clear handling were fixed.
- Mood and recommendation-judgment guides are drafts with empty collections. No
  mood tags, gold judgments or inferred human preferences were invented.

## Ranking and measured software performance

The former interface fully ranked the catalogue, then displayed 16 results. The new
`searchBooks` API returns the exact matching total while ranking only the requested
prefix. Increasing the page size continues the same greedy diversity ordering;
bounded in-memory query caches reuse scores and progress. Plain browsing does not
construct the TF-IDF index until content similarity is actually needed.

The synthetic engineering microbenchmark uses deterministic catalogues, five cold
index samples per case, the same Apple M5/Node 26.7.0 process environment, and compares
the previous UI's full-ranking work with the visible-prefix API. It measures code
execution only—neither recommendation quality nor browser/network latency.

| Synthetic catalogue | Action | Before median | After median |
| --- | --- | ---: | ---: |
| 89 books | Browse first 16 | 1.367 ms | 0.479 ms |
| 89 books | Search first 16 | 1.005 ms | 1.099 ms |
| 1,000 books | Browse first 16 | 27.873 ms | 2.934 ms |
| 5,000 books | Browse first 16 | 574.111 ms | 13.023 ms |
| 5,000 books | Search first 16 | 172.822 ms | 43.691 ms |
| 5,000 books | Similar books first 16 | 506.487 ms | 46.186 ms |

Every measured case returned the same first 16 IDs. A separate comparison checked
2,880 prefix/total cases against the original implementation, including ties,
personalization and changing limits, without a mismatch. The small-catalogue search
case is slightly slower in this sample; the improvement is principally avoiding work
on unseen results as the catalogue grows. Full ranking still has quadratic worst-case
diversity selection when explicitly requested; the UI no longer requests all results.

Raw timings, source hash and machine details: [performance record](performance/recommendations_2026-09-26.json).
The executable timing harness is `web/scripts/benchmark-recommendations.ts`.
Search also now handles prototype-like words such as `constructor` safely, and
longest-genre exclusions cannot confuse “without science fiction” with “without science.”

## Source and artifact integrity

Cache timestamps, hashes, URLs, authors and scalar/list types are validated. Corrupt
caches and stale review decisions abort a catalogue build instead of silently
removing books. Catalogue, manifest and public receipt writes are staged and rolled
back on ordinary failure. A hash mismatch blocks Next.js/Gradio consumption after an
interrupted multi-file publication. POSIX and Windows advisory locks use their
documented standard-library backends; native Windows execution was not available.

Checkpoint and label writers use sibling temporary files, flush, and atomic replace.
Compile-wrapper normalization strips only whole `_orig_mod` path components, retains
module version metadata and rejects collisions. Legacy helper APIs share the same
implementation, including label-size property aliases and absent-task empty lists.

## Model, training and reporting

See [the model-stack audit](model_stack_audit_2026-09-26.md) for the exact future-recipe
changes. The main repairs are accumulated-gradient preservation during diagnostics,
correct partial windows and PCGrad projections, validation/calibration separation,
stable masked attention, EOS handling, and one shared encoder pass for joint inference.
Tests establish arithmetic/call-count behavior, not GPU throughput or model quality.

The visualization script previously manufactured a plausible-looking confusion
matrix and synthetic embeddings. Confusion plots now require explicit observed
counts with their recorded label order. Synthetic surfaces/embedding illustrations
require `--illustrations`, carry visible synthetic titles and distinct filenames,
and are not enabled by `--all` alone. Existing historical figures are retained;
this change does not retroactively validate them.

## Verification and remaining boundaries

During review, the full local Python suite passed 193 tests;
the web suite passed 37 unit tests. Final totals and hosted CI are recorded with the
release in `docs/deployment.md`. Unit tests use synthetic data and temporary files;
visualization tests no longer write into the project's historical outputs directory.
No historical model metrics were recomputed.

CI now installs a pinned CPU PyTorch wheel and test dependencies instead of resolving
the entire research environment and downloading CUDA libraries on a CPU runner.
The preceding green job spent nearly nine minutes installing dependencies and 26 seconds
running tests. The new environment cache and eliminated duplicate PR/push workflows
reduce repeated work; hosted timing must be checked before claiming a CI speedup.

Native device testing was unavailable: no iOS simulators were configured and no
Android SDK was installed. T3 browser checks cover controls and responsive geometry;
desktop snapshots were inspected. Narrow-viewport snapshot capture was intermittent.
Safari-specific file downloads, CUDA/AMP/compile behavior, full-checkpoint prediction
parity, exact optimizer/RNG resume, and research-quality evaluation remain unverified.
