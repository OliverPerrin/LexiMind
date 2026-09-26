---
title: LexiMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: scripts/demo_gradio.py
pinned: false
---

# LexiMind

**Find your next book through the subjects, stories, and books you already love.**

LexiMind is a book-discovery website and a small-model research project. The web app
has a source-attributed catalogue, search, genre filters, content-based recommendations,
and a reading list that stays in your browser. The Python project preserves the
original multi-task transformer and its experimental record.

[Live website](https://leximind-five.vercel.app) · [Product direction](docs/product.md) · [Catalogue provenance](data/catalog/README.md) ·
[Historical results](docs/RESULTS.md) · [Research plan](docs/research_plan_2026.md)

## Book discovery

The Next.js application lives in `web/`. Its initial catalogue comes from identified
Open Library works, with authors, descriptions, subjects, covers, and source links.
It includes:

- Search by title, author, or what you want to read.
- Genre browsing and “more like this” recommendations.
- Browse the source subjects linked from each book.
- Favourites that influence recommendations, saved books, and hide/restore controls.
- Local browser persistence, with no account or cross-device sync.
- Versioned shelf exports and safe imports that merge with existing preferences.
- Accessible book details and an explanation of where the information comes from.

Recommendations currently use weighted TF-IDF text similarity, metadata overlap, and
modest diversity across authors and genres. This is a content-based baseline; no
research-model quality claim is implied. There are no generated descriptions or
randomly assigned mood labels. Mood browsing will appear only after supported labels
are added.

### Run locally

Use Node.js 24 and npm:

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`. To validate and build:

```bash
npm run lint
npm run typecheck
npm run test:unit
npm run build
npx playwright install chromium
npm run test:e2e
```

The website deploys from `web/` on Vercel without a GPU, model API key, or database.
See [web/README.md](web/README.md) for deployment and browser tests.
The first production deployment and its verification are recorded in
[docs/deployment.md](docs/deployment.md).

### Catalogue maintenance

The deployed snapshot is `web/data/books.json`. The importer uses cached, identified,
rate-limited Open Library API requests and records source hashes and review decisions.
The current catalogue has 102 works and 96 source descriptions. Its publication receipt
is verified before website builds; the original 89-work snapshot remains archived.
See [data/catalog/README.md](data/catalog/README.md) for the exact rebuild command.
Do not regenerate it from the old Gutenberg/Goodreads title-only matches.

## Research status

**Training and research experiments are paused as of 22 September 2026.**
Documentation, artifact preservation, and software validation can continue; no new
benchmark or training result is claimed by the website release.

Phase 1 implemented a custom PyTorch encoder-decoder initialized from FLAN-T5, with
summarization, multi-label emotion detection, and topic classification heads. Training
and evaluation code, local checkpoints, a completed seed-17 training history, and
BERT baseline artifacts remain available. The original paper drafts are historical
and must not be cited: several comparison tables lacked supporting runs.

The current audit also distinguishes report values from reproducibility. In particular,
the test report names a checkpoint path without a recorded hash; that checkpoint is
not byte-identical to the seed-17 checkpoint. Dataset and configuration drift are
recorded explicitly in [docs/RESULTS.md](docs/RESULTS.md).

Phase 2 asks which post-training recipe best combines heterogeneous tasks under a
specified budget. The proposed first comparison is joint training versus specialists
and adapter merging, with backbone feasibility, distillation, and other arms to be
settled after research resumes. It has not been run.

- [Research plan and current resume state](docs/research_plan_2026.md)
- [Draft evaluation protocol](docs/eval_protocol.md)
- [Initial related-work review](docs/related_work.md)
- [Historical artifact manifest](research/results/manifest.json)
- [Generated historical result tables](research/results/historical_tables.md)
- [Phase 1 architecture](docs/architecture.md)
- [Archived draft audit](docs/archive/README.md)

### Python development

The catalogue and provenance tooling are lightweight; model development is separate:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-quality.txt
python -m pytest tests/test_catalog tests/test_research -q
```

Software tests for the model stack use `requirements-test.txt` plus a suitable PyTorch
wheel. CI installs a pinned CPU wheel and runs with Hugging Face networking disabled;
the tests use synthetic components and temporary artifacts, not research checkpoints.
See [the September code review](docs/code_review_2026-09-26.md) for fixes, measured
software performance, and the remaining research validation boundaries.

For the legacy model environment, use Poetry with `pyproject.toml` and an appropriate
PyTorch installation for the target machine. The original model dependencies are not
yet a pinned, cross-platform reproduction environment. Do not treat installing them
as reproducing the historical result.

## Layout

```text
web/                    Next.js book discovery app and deployable catalogue
src/catalog/            Source identity checks and Open Library import helpers
src/models/             Original encoder, decoder, attention and task heads
src/training/           Original multi-task trainer, metrics and PCGrad
src/inference/          Legacy checkpoint inference pipeline
src/api/                Legacy text-analysis API
scripts/                Catalogue, provenance, evaluation and training tooling
research/results/       Small archived reports and provenance manifest
configs/                Model/training configurations and research preparation
```

The Hugging Face [model](https://huggingface.co/OliverPerrin/LexiMind-Model) and
[legacy Space](https://huggingface.co/spaces/OliverPerrin/LexiMind) remain historical
project resources. Their published contents may precede this catalogue repair.

## License

Code is MIT licensed; see [LICENSE](LICENSE). Third-party book descriptions, covers,
and source datasets retain their own source terms and attribution.

Built by Oliver Perrin · Originally an undergraduate research project at Appalachian
State University.
