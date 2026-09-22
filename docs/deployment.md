# Initial book-discovery release

Deployed 22 September 2026: **https://leximind-five.vercel.app**

- Vercel project: `leximind` in `oliverperrins-projects`.
- Source branch: `feat/books-first-revival`.
- Deployed application commit: `26bfd42` (later documentation-only changes do not
  affect the application).
- Deployment: `dpl_5oMy5Lhn3Bu2bCJD7u4GRdMH9MFQ`, production, ready.
- Runtime: Next.js 16.3.5, Node.js 24; 89 pre-rendered book pages.
- Catalogue: 89 identified works, 84 source descriptions, source links and covers.
- Deployment contains only `web/` and its public catalogue, about 519 KB of source.
  Training data, checkpoints, research artifacts and local credentials are excluded.

## Verification

- 27 Python catalogue/provenance tests passed in an isolated environment without
  model dependencies.
- 15 web unit tests passed, covering ranking constraints and publication provenance.
- 10 browser tests passed across desktop and mobile: search, genre filters, empty
  states, favourites, persistence, hiding/restoring, modal keyboard focus, shareable
  pages, source links and 404s.
- Web lint, TypeScript and local/cloud production builds passed. Python Ruff checks
  and formatting passed across the repository. Legacy formatting edits were checked
  against unchanged Python syntax trees.
- Production dependency audit reported no known vulnerabilities at this check.
- Live public check: homepage HTTP 200; title search and book navigation work;
  correct Open Library work link; unknown book HTTP 404; no browser JavaScript errors
  during the checked flow. Desktop/mobile layouts were inspected visually.
- Legacy Gradio interface constructed in an isolated environment with 89 canonical
  books, 500 historical paper examples, and no displayed unvalidated tones. The
  existing remote Hugging Face Space was not redeployed.
- Historical result integrity audit verified archived files and current local
  checkpoint/data hashes. It did not recompute model metrics.

Tests establish software behavior, not recommendation relevance or model quality.
No training, model evaluations, teacher calls or research experiments were run.

## Known scope

Search/recommendation is a deterministic content-based baseline using weighted
TF-IDF, source subjects and genres, favourites, and diversity. Neural embeddings,
validated mood annotations, accounts, cross-device sync and research comparisons are
future work. Mood publication fails closed until an evidence schema is reviewed.
Open Library community metadata can contain errors despite source-identity checks.

Preferences are local to the current browser and origin. Preview and production
domains have separate shelves. Covers are loaded from Open Library.

## Re-deploy

From `web/`, run the validation commands in `web/README.md`, then:

```sh
npx vercel deploy --prod
```

The local project association is in ignored `.vercel/` files. On a new machine, link
to the existing project first with `vercel link --project leximind --scope
oliverperrins-projects`. There are no application secrets or database migrations.
This first deployment was made with the CLI. A Git-connected deployment should set
the repository's root directory to `web` before enabling automatic builds.
