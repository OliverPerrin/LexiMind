---
title: LexiMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: scripts/demo_gradio.py
pinned: false
---

<h1 align="center">LexiMind</h1>

<p align="center">Find your next book through subjects, genres and the books you already like.</p>

<p align="center">
  <a href="https://leximind-five.vercel.app"><strong>Live site</strong></a> ·
  <a href="https://huggingface.co/spaces/OliverPerrin/LexiMind">Hugging Face Space</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/research/README.md">Research notes</a> ·
  <a href="docs/research/visuals.html">Visual archive</a>
</p>

<p align="center">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" />
  <img alt="Next.js 16" src="https://img.shields.io/badge/Next.js-16-black?style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square" />
</p>

[![LexiMind book discovery site](docs/screenshot.png)](https://leximind-five.vercel.app)

---

### What it is

LexiMind is two things in one repository:

- **A book-discovery website.** Search by title, author or a description of what you want to read. Browse by genre and subject, open "more like this" on any book, and keep favourites and a reading list. No account is needed; your shelf stays in your browser and can be exported and imported.
- **A transformer written from scratch in PyTorch.** Attention, encoder and decoder layers, T5 layer normalisation, relative-position bias, generation with a key-value cache, and task heads are all implemented in `src/models/`. Pretrained FLAN-T5 weights are loaded into these modules, and one shared encoder serves summarisation, emotion classification and topic classification.

The website does not depend on the model. It runs without a GPU, an API key or a database.

### Quick start

Requires Node.js 24.

```sh
cd web
npm ci
npm run dev
```

Open http://localhost:3000.

### How recommendations work

Recommendations are content-based: weighted TF-IDF similarity over descriptions, overlap in subjects and genres, and a small adjustment so results are not dominated by one author or genre. This is a baseline, not a neural search model, and the scores are ranking values, not ratings.

The catalogue holds 120 works identified in Open Library, 113 of them with sourced descriptions. No description is generated. Source records, hashes and review decisions are kept by the [catalogue importer](data/catalog/README.md).

### The model

| Part | Where |
| --- | --- |
| Attention, encoder, decoder, feed-forward, T5 layer norm | `src/models/` |
| FLAN-T5 weight transfer (small, base, large) | `src/models/factory.py`, `configs/model/` |
| Multi-task training, metrics, PCGrad | `src/training/` |
| Checkpoint loading and shared-encoder inference | `src/inference/` |

**Status:** bounded local training runs are active. Source-backed book continuation and missing-word experiments run on the M5 using the native transformer and LoRA; the [research notes](docs/research/README.md) record their results and limits. The formal book-field/model-recipe and recommendation studies still need reviewed labels and evaluation evidence. Earlier undergraduate results remain [historical evidence](docs/RESULTS.md) and have not been reproduced.

The [interactive research visual archive](docs/research/visuals.html) collects the charts with their PRs and pinned evidence. Download the HTML and open it locally; GitHub displays its source rather than rendering the gallery.

The Hugging Face Space runs `scripts/demo_gradio.py`. It uses the same book catalogue and stored historical outputs; it does not run the research model.

### Development

Website checks, from `web/`:

```sh
npm run lint
npm run typecheck
npm run test:unit
npm run build
npx playwright install chromium && npm run test:e2e
```

Python checks for the catalogue and research tooling:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-quality.txt
python -m pytest tests/test_catalog tests/test_research -q
```

Model tests need a PyTorch build for your machine plus `requirements-test.txt`. They use synthetic inputs and check the software, not model quality. Deployment is covered in [docs/deployment.md](docs/deployment.md); only `web/` is deployed to Vercel.

### Layout

```text
web/                  Book-discovery application
src/catalog/          Open Library import and source identity checks
src/models/           From-scratch transformer and FLAN-T5 weight transfer
src/training/         Multi-task optimisation, metrics and PCGrad
src/inference/        Checkpoint loading and prediction
src/research/         Data provenance and study preparation
scripts/              Catalogue, reconstruction and model tools
docs/                 Product, architecture and research notes
```

### Licence

Code is [MIT licensed](LICENSE). Third-party text, covers, datasets and pretrained weights keep their own terms. LexiMind began as an undergraduate research project at Appalachian State University and is built by [Oliver Perrin](https://github.com/OliverPerrin).
