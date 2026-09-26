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

**Discover books through subjects, genres, and books you already enjoy.**

LexiMind combines a book-discovery website with research on a deliberately
from-scratch PyTorch transformer. The website is usable independently of the
model; current research work reconstructs source data and prepares controlled
multi-task comparisons.

[Live website](https://leximind-five.vercel.app) ·
[Product](docs/product.md) · [Architecture](docs/architecture.md) ·
[Research preparation](docs/research/README.md)

## Book discovery

The Next.js app in `web/` provides search, genre and subject browsing, “more like
this,” favourites, a reading list, hide/restore controls, and shelf export/import.
Preferences stay in browser local storage; there are no accounts or cross-device
sync. Recommendations use weighted TF-IDF, metadata overlap, and modest author
and genre diversity. They are a content-based baseline, not a model-quality result.

The catalogue contains 102 identified Open Library works with 96 sourced
descriptions. There are no generated descriptions or inferred book-mood labels.
Source records, hashes, and review decisions are retained by the
[catalogue importer](data/catalog/README.md).

Use Node.js 24:

```sh
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`. Software checks:

```sh
npm run lint
npm run typecheck
npm run test:unit
npm run build
npx playwright install chromium
npm run test:e2e
```

Deploy only `web/` to Vercel. The website needs no GPU, model API key, or database.
See [web/README.md](web/README.md) and [deployment notes](docs/deployment.md).

## Gradio demo

The existing [Hugging Face Space](https://huggingface.co/spaces/OliverPerrin/LexiMind)
remains available. Its code is `scripts/demo_gradio.py`, with the Docker setup and
minimal `requirements-demo.txt` retained. It uses the same attributed book catalogue
and historical paper outputs; it does not run the research model.

## Model and research

The model architecture is implemented in `src/models/`: attention, encoder and
decoder layers, feed-forward blocks, T5 normalization, relative-position bias,
generation with a KV cache, and task heads. Hugging Face supplies pretrained
FLAN-T5 weights and tokenization; the research forward pass uses LexiMind's own
modules. This implementation is an intentional part of the project.

A shared encoder supports summarization, multilabel emotion classification, and
topic classification. Training, evaluation, inference, and numerical component
tests remain available. The website does not load these checkpoints, and a
Reddit-comment emotion score does not establish a book's atmosphere.

**Training and research experiments remain paused.** Current work concerns
source reconstruction, data identity, and the design of two independent studies:
model-recipe comparisons and book-recommendation relevance. Neither has new results.
[Research preparation](docs/research/README.md) records what is ready, what remains
unresolved, and the read-only verification commands.

## Python development

Catalogue and preparation tests use lightweight dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-quality.txt
python -m pytest tests/test_catalog tests/test_research -q
```

Source reconstruction adds only the small data environment:

```sh
python -m pip install -r requirements-data.txt
```

For model software tests, install an appropriate PyTorch wheel and
`requirements-test.txt`. These tests use synthetic inputs; they do not reproduce
a research result or authorize training. Model/data environments and future run
artifacts need their own pinned provenance. `pyproject.toml` retains the model's
core dependencies, with separate optional extras for data preparation, Gradio,
quantization, and profiling. These software environments are not frozen experiment
recipes.

## Layout

```text
web/                    Book-discovery application and deployment snapshot
src/catalog/            Source identity, Open Library import, publication checks
src/models/             From-scratch transformer and FLAN-T5 weight transfer
src/training/           Multi-task optimization, metrics, and PCGrad
src/inference/          Checkpoint loading and shared-encoder prediction
src/research/           Preparation, provenance, and admission contracts
scripts/                Catalogue, reconstruction, and model tools
docs/research/           Current research decisions and source reviews
research/preparation/    Source-linked preparation evidence
```

Code is [MIT licensed](LICENSE). Third-party text, covers, datasets, and pretrained
weights retain their own terms. Originally an undergraduate research project at
Appalachian State University, built by Oliver Perrin.
