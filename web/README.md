# LexiMind web

A books-first Next.js application: catalogue search, genre and source-topic filters, content-based recommendations, source-linked book pages, and a personal browser reading shelf. Research models and training jobs are not required to browse or build the website.

## Local development

Use Node.js 24 (also selected for Vercel), then run:

```sh
cd web
npm ci
npm run dev
```

Open http://localhost:3000. `npm run build` creates the production build; `npm start` serves it locally. Fonts are bundled. Covers are requested from Open Library with a title-card fallback when they cannot load.

## Data and recommendations

`data/books.json` is the generated catalogue. `lib/catalog.ts` validates identities and source records before the app builds. The offline catalogue importer and audit live in the parent repository; do not edit generated book identities by hand. Source attribution is visible in each book’s details and shareable `/books/OL…W` page.

`lib/recommendations.ts` is a deterministic content-based baseline using descriptions, subjects, genres, and explicit user favourites, with author/genre variety. The website does not claim semantic-model quality or use unvalidated research emotion labels. No model training, paid APIs, account credentials, or runtime inference services are needed.

Save, favourite, and hide preferences are stored under `leximind.reading-shelf.v1` in browser local storage. They persist on the same browser and origin, are not sent to a server, and do not sync across devices. Shelf actions read the latest stored state before writing; where available, the browser’s Web Locks API serializes writes from multiple tabs. Changes from another tab, including clearing site data, update an open shelf. Browsers without Web Locks use serialized local actions and latest-state reads, but cannot guarantee simultaneous cross-tab writes are atomic.

Storage failures leave the current session usable and show an explanation. Failed actions are replayed against the newest stored preferences on a later successful write. Unreadable original storage is never overwritten automatically; export the current session’s shelf to keep a copy.

### Shelf backup and transfer

Open **My shelf → Back up or move your shelf** to export or import a local JSON file. Export format `leximind.reading-shelf`, version `1`, contains the export date and the three preference lists. Files are parsed locally and are never uploaded. Imports are limited to 256 KB and 2,000 valid Open Library work IDs per list.

Import always merges; it never replaces or truncates existing entries. Duplicates are removed. Existing visible saved/favourite books cannot be hidden by an import, and already hidden books remain hidden. Valid IDs not present in this catalogue are retained in storage and future exports, with an explanation in the shelf. Invalid, oversized, and unsupported-version files leave the shelf unchanged.

Useful URLs:

- `/`: discover and search books.
- `/books/<Open Library work ID>`: shareable static book page.
- `/?book=<work ID>`: open a book’s interactive details in discovery.
- `/?similar=<work ID>`: browse recommendations related to a book.
- `/?subject=<URL-encoded source subject>`: browse an exact source topic. Repeated `subject` parameters match any selected topic, combined with other facets.
- `/about`: project, sources, recommendation approach, and storage explanation.

## Validation

```sh
npm run lint
npm run typecheck
npm run test:unit
npm run build
npx playwright install chromium
npm run test:e2e
```

The end-to-end suite serves the production build on port 3012, tests desktop and mobile browser layouts, and covers search, filters, empty states, book details and keyboard focus, source-linked book URLs, recommendations, shelf persistence, hidden-book restoration, invalid saved data, versioned export/import, unknown-ID preservation, source-topic entry points, and cross-tab storage clearing. Run `npm run build` before it after changing application code. For agent-led interactive checks, use the native T3 preview tools when available; the checked-in Playwright suite also runs in CI. Playwright reports and screenshots are ignored by Git.

## Vercel

Import the repository and set **Root Directory** to `web`, **Framework Preset** to Next.js, and **Node.js Version** to 24.x. Use the default `npm run build` build command. No environment variables or database configuration are required for this version. The lockfile pins dependencies for reproducible installs; Vercel should install dependencies with `npm ci`.

Book pages and the home catalogue are generated at build time. Rebuild/redeploy after refreshing `data/books.json`. Browser shelves are origin-specific: a preview deployment’s shelf will not appear automatically on a production domain.
