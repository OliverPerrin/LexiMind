# Book-discovery deployment

Production: **https://leximind-five.vercel.app** — verified 27 September 2026.

- Application commit: `167e4f1560e3f0a0c2339c39fb37acf3cfbff3a6` on `main`.
- Vercel deployment: `dpl_4oQvT1uaox3s6AziamCzwzLj7rfc`, production, ready.
- Project: `leximind` in `oliverperrins-projects`; Next.js 16.3.5, Node.js 24.
- Catalogue: **120 works, 113 descriptions**, including 18 reviewed modern additions.
  All previous 102 work IDs remain valid. Publication corrections link to separate
  author/publisher evidence; edition dates are not assumed to be original dates.
- Only `web/` and its attributed catalogue are deployed. Training data, checkpoints,
  local credentials and the BGC archive are excluded.

## Verification

- 441 Python tests and 67 subtests, 38 web unit tests and 18 desktop/narrow-viewport
  browser tests passed, plus lint, formatting, type checks and production builds.
- All three [main CI jobs](https://github.com/OliverPerrin/LexiMind/actions/runs/36309938425)
  passed. The offline catalogue rebuild was byte-identical.
- Live T3 checks confirmed the 120-book catalogue, Gone Girl search, its book page
  and details dialog, the corrected 2012 date and publisher source link. The checked
  390-pixel layout had no horizontal overflow. This is not native-device testing.
- No model training, research experiments, real-checkpoint evaluation or paid
  teacher calls ran. The custom transformer and local Gradio interface are preserved;
  the remote Hugging Face Space was not redeployed.

Search and recommendations use the existing content-based baseline. These software
checks do not establish recommendation relevance or model quality. Metadata access
is not permission to train on a book's full text; candidate sources and restrictions
are recorded in [dataset decisions](research/dataset_decisions.md).

## Re-deploy

From `web/`, run the checks in [web/README.md](../web/README.md), then:

```sh
npx vercel deploy --prod
```

The project association is in ignored `.vercel/` files. On a new machine, use
`vercel link --project leximind --scope oliverperrins-projects` first. There are no
application secrets or database migrations. Deployments currently use the CLI;
set the repository root to `web` before enabling Git-connected builds.

Preferences are local to each browser and origin; preview and production shelves
are separate. Previous releases remain in Git history. Changes are recorded in
[the book-discovery release](https://github.com/OliverPerrin/LexiMind/pull/2) and
[the modern-books update](https://github.com/OliverPerrin/LexiMind/pull/6).
