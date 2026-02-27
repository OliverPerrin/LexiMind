---
title: LexiMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: scripts/demo_gradio.py
pinned: false
---

<!-- markdownlint-disable MD025 -->
# LexiMind

A multi-task NLP system for literary and academic text understanding. LexiMind performs **abstractive summarization**, **topic classification**, and **emotion detection** using a single encoder-decoder transformer initialized from [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) (272M parameters).

**[Live Demo](https://huggingface.co/spaces/OliverPerrin/LexiMind)** · **[Model](https://huggingface.co/OliverPerrin/LexiMind-Model)** · **[Discovery Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery)** · **[Research Paper](docs/research_paper.tex)**

## What It Does

| Task | Description | Metric |
| ------ | ------------- | -------- |
| **Summarization** | Generates back-cover style book descriptions and paper abstracts from source text | BERTScore F1: **0.830** |
| **Topic Classification** | Classifies passages into 7 categories | Accuracy: **85.2%** |
| **Emotion Detection** | Identifies emotions from 28 fine-grained labels (multi-label) | Sample-avg F1: **0.199** |

**Topic labels:** Arts · Business · Fiction · History · Philosophy · Science · Technology

The model is trained on literary text (Project Gutenberg + Goodreads descriptions), academic papers (arXiv), and emotion-annotated Reddit comments (GoEmotions). For summarization, it learns to produce descriptive summaries—what a book *is about*—rather than plot recaps, by pairing Gutenberg full texts with Goodreads descriptions and arXiv bodies with their abstracts.

## Architecture

LexiMind is a **custom Transformer implementation** that loads pre-trained weights from FLAN-T5-base via a factory module. The architecture is reimplemented from scratch for transparency, not wrapped from HuggingFace.

| Component | Detail |
| ----------- | -------- |
| Backbone | Encoder-Decoder Transformer (272M params) |
| Encoder / Decoder | 12 layers each |
| Hidden Dim | 768, 12 attention heads |
| Position Encoding | T5-style relative position bias |
| Normalization | RMSNorm (Pre-LN) |
| Attention | FlashAttention via PyTorch 2.0 SDPA |
| Summarization Head | Full decoder with language modeling head |
| Classification Heads | Linear layers on mean-pooled encoder states |

### Multi-Task Training

All three tasks share the encoder. Summarization uses the full encoder-decoder; topic and emotion classification branch off the encoder with lightweight linear heads. Training uses round-robin scheduling (one batch per task per step), fixed loss weights (summarization=1.0, emotion=1.0, topic=0.3), and early stopping.

## Training Data

| Task | Source | Train Samples |
| ------ | -------- | --------------- |
| Summarization | Gutenberg + Goodreads (literary) | ~4K |
| Summarization | arXiv body → abstract (academic) | ~45K |
| Topic | 20 Newsgroups + Gutenberg + arXiv metadata | 3,402 |
| Emotion | GoEmotions (Reddit comments, 28 labels) | 43,410 |

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- NVIDIA GPU with CUDA (for training; CPU works for inference)

### Installation

```bash
git clone https://github.com/OliverPerrin/LexiMind.git
cd LexiMind
poetry install
```

### Download Data

```bash
poetry run python scripts/download_data.py
```

Downloads Goodreads descriptions, arXiv papers, GoEmotions, 20 Newsgroups, and Gutenberg texts.

### Training

```bash
# Full training (~45-60 min on RTX 4070 12GB)
poetry run python scripts/train.py training=full

# Quick dev run (~10-15 min)
poetry run python scripts/train.py training=dev

# Medium run (~30-45 min)
poetry run python scripts/train.py training=medium

# Override parameters
poetry run python scripts/train.py training.optimizer.lr=5e-5

# Resume from checkpoint
poetry run python scripts/train.py training=full resume_from=checkpoints/epoch_5.pt
```

Training uses BFloat16 mixed precision, gradient checkpointing, `torch.compile`, and cosine LR decay with warmup. Experiments are tracked with MLflow (`mlflow ui` to browse).

### Evaluation

```bash
# Full evaluation (ROUGE, BERTScore, topic accuracy, emotion F1)
poetry run python scripts/evaluate.py

# Skip BERTScore for faster runs
poetry run python scripts/evaluate.py --skip-bertscore

# Single task
poetry run python scripts/evaluate.py --summarization-only
```

### Inference

```bash
# Command-line
poetry run python scripts/inference.py "Your text to analyze"

# Gradio web demo
poetry run python scripts/demo_gradio.py
```

### Docker

```bash
docker build -t leximind .
docker run -p 7860:7860 leximind
```

## Project Structure

```text
configs/
├── config.yaml              # Main Hydra config
├── data/datasets.yaml       # Dataset paths and tokenizer settings
├── model/                   # Architecture configs (base, small, large)
└── training/                # Training configs (dev, medium, full)

src/
├── models/
│   ├── encoder.py           # Transformer Encoder with Pre-LN RMSNorm
│   ├── decoder.py           # Transformer Decoder with KV-cache
│   ├── attention.py         # Multi-Head Attention + T5 relative position bias
│   ├── feedforward.py       # Gated feed-forward network
│   ├── positional_encoding.py  # Sinusoidal & learned position encodings
│   ├── t5_layer_norm.py     # T5-style RMSNorm
│   ├── heads.py             # Task-specific classification heads
│   ├── multitask.py         # Multi-task model combining all components
│   └── factory.py           # Model builder with FLAN-T5 weight loading
├── data/
│   ├── dataset.py           # Dataset classes for all tasks
│   ├── dataloader.py        # Multi-task dataloader with round-robin sampling
│   └── tokenization.py      # Tokenizer wrapper
├── training/
│   ├── trainer.py           # Training loop with AMP, grad accumulation, early stopping
│   ├── metrics.py           # ROUGE, BERTScore, F1, accuracy computation
│   └── utils.py             # Checkpointing, logging utilities
├── inference/
│   ├── pipeline.py          # End-to-end inference pipeline
│   └── factory.py           # Model loading for inference
├── api/                     # FastAPI REST endpoint
└── utils/                   # Shared utilities

scripts/
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation with all metrics
├── inference.py             # CLI inference
├── demo_gradio.py           # Gradio web UI
├── download_data.py         # Dataset downloader
├── export_model.py          # Model export utilities
├── export_tokenizer.py      # Tokenizer export
├── preprocess_data.py       # Data preprocessing
├── process_books.py         # Gutenberg text processing
├── eval_rouge.py            # ROUGE-only evaluation
└── visualize_training.py    # Training curve plotting

tests/                       # Pytest suite (data, models, training, inference, utils)
docs/                        # Research paper and architecture notes
artifacts/                   # Tokenizer files and label definitions
checkpoints/                 # Saved model checkpoints
```

## Code Quality

```bash
poetry run ruff check .     # Linting
poetry run mypy .           # Type checking
poetry run pytest           # Test suite
poetry run pre-commit run --all-files  # All checks
```

## Key Results

From the research paper ([docs/research_paper.tex](docs/research_paper.tex)):

- **Multi-task learning helps topic classification** (+3.2% accuracy over single-task) because the small topic dataset (3.4K) benefits from shared encoder representations trained on the larger summarization corpus (49K).
- **Summarization is robust to MTL**—quality stays comparable whether trained alone or jointly.
- **Emotion detection shows slight negative transfer** (−0.02 F1), likely due to domain mismatch between Reddit-sourced emotion labels and literary/academic text.
- **FLAN-T5 pre-training is essential**—random initialization produces dramatically worse results on all tasks.

See the paper for full ablations, per-class breakdowns, and discussion of limitations.

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

---

Built by Oliver Perrin · Appalachian State University · 2025–2026
