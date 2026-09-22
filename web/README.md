# LexiMind web

A books-first Next.js application: catalogue search, genre filters, content-based recommendations, source-linked book pages, and a personal browser reading shelf. Research models and training jobs are not required to browse or build the website.

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

Save, favourite, and hide preferences are stored under `leximind.reading-shelf.v1` in browser local storage. They persist on the same browser and origin, are not sent to a server, and do not sync across devices. Storage failures leave the current session usable and show an explanation.

Useful URLs:

- `/`: discover and search books.
- `/books/<Open Library work ID>`: shareable static book page.
- `/?book=<work ID>`: open a book’s interactive details in discovery.
- `/?similar=<work ID>`: browse recommendations related to a book.
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

The end-to-end suite serves the production build on port 3012, tests desktop and mobile browser layouts, and covers search, filters, empty states, book details and keyboard focus, source-linked book URLs, recommendations, shelf persistence, hidden-book restoration, and invalid saved data. Run `npm run build` before it after changing application code. Playwright reports and screenshots are ignored by Git.

## Vercel

Import the repository and set **Root Directory** to `web`, **Framework Preset** to Next.js, and **Node.js Version** to 24.x. Use the default `npm run build` build command. No environment variables or database configuration are required for this version. The lockfile pins dependencies for reproducible installs; Vercel should install dependencies with `npm ci`.

Book pages and the home catalogue are generated at build time. Rebuild/redeploy after refreshing `data/books.json`. Browser shelves are origin-specific: a preview deployment’s shelf will not appear automatically on a production domain.
