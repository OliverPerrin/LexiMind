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

A multi-task NLP system for literary and academic text understanding. LexiMind jointly performs **abstractive summarization**, **topic classification**, and **multi-label emotion detection** using a single encoder-decoder transformer initialized from [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) (272M parameters).

**[Live Demo](https://huggingface.co/spaces/OliverPerrin/LexiMind)** · **[Model](https://huggingface.co/OliverPerrin/LexiMind-Model)** · **[Discovery Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery)** · **[Research Paper](docs/research_paper.tex)**

## Results

| Task | Metric | Score |
| ---- | ------ | ----- |
| Summarization | ROUGE-1 / ROUGE-L | 0.309 / 0.185 |
| Summarization (academic) | ROUGE-1 | 0.319 |
| Summarization (literary) | ROUGE-1 | 0.206 |
| Topic Classification | Accuracy (95% CI) | 85.7% (80.4–91.0%) |
| Emotion Detection | Sample-avg F1 | 0.352 |
| Emotion Detection (tuned thresholds) | Sample-avg F1 / Macro F1 | 0.503 / 0.294 |

Trained for 8 epochs on an RTX 4070 12GB (~9 hours) with BFloat16 mixed precision, `torch.compile`, and cosine LR decay.

## Key Findings

From my research paper:

- **Naive MTL produces mixed results**: topic classification benefits (+3.7% accuracy), but emotion detection suffers negative transfer (−0.02 F1) under mean pooling with round-robin scheduling.
- **Learned attention pooling + temperature sampling eliminates negative transfer entirely**: emotion F1 improves from 0.199 → 0.352 (+77%), surpassing the single-task baseline (0.218).
- **Summarization is robust to MTL** — quality remains stable across configurations.
- **FLAN-T5 pre-training is essential** — random initialization produces dramatically worse results on all tasks.
- **Domain gap matters**: academic summaries (ROUGE-1: 0.319) substantially outperform literary (0.206), driven by an 11:1 training data imbalance.

## Architecture

LexiMind is a **from-scratch PyTorch Transformer** that loads pre-trained FLAN-T5-base weights layer by layer via a custom factory module — no HuggingFace model wrappers.

| Component | Detail |
| --------- | ------ |
| Backbone | Encoder-Decoder Transformer (272M params) |
| Encoder / Decoder | 12 layers each, 768d, 12 attention heads |
| Normalization | RMSNorm (Pre-LN, T5-style) |
| Attention | FlashAttention via PyTorch SDPA + T5 relative position bias |
| FFN | Gated-GELU (wi\_0, wi\_1, wo) |
| Summarization | Full decoder → language modeling head |
| Emotion (28-class multi-label) | Learned attention pooling → linear head |
| Topic (7-class) | Mean pooling → linear head |

### Multi-Task Training

All three tasks share the encoder. Summarization uses the full encoder-decoder; classification heads branch off the encoder output. Key training details:

- **Temperature-based task sampling** (α=0.5): allocates training steps proportional to dataset size, preventing large tasks from dominating
- **Attention pooling** for emotion: a learned query attends over encoder outputs, focusing on emotionally salient tokens rather than averaging the full sequence
- **Fixed loss weights**: summarization=1.0, emotion=1.0, topic=0.3 (reduced to prevent overfitting on the small topic dataset)
- **Frozen encoder layers 0–3**: preserves FLAN-T5's language understanding in lower layers
- **Gradient conflict diagnostics**: optional inter-task gradient cosine similarity monitoring

See [docs/architecture.md](docs/architecture.md) for full implementation details, weight loading tables, and training configuration rationale.

## Training Data

| Task | Source | Samples |
| ---- | ------ | ------- |
| Summarization | Gutenberg + Goodreads descriptions (literary) | ~4K |
| Summarization | arXiv body → abstract (academic) | ~45K |
| Topic | Gutenberg + arXiv metadata → 7 categories | 3,402 |
| Emotion | GoEmotions — Reddit comments, 28 labels | 43,410 |

For summarization, the model learns to produce descriptive summaries — what a book *is about* — rather than plot recaps, by pairing Gutenberg full texts with Goodreads descriptions and arXiv papers with their abstracts.

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (for training; CPU works for inference)

### Installation

```bash
git clone https://github.com/OliverPerrin/LexiMind.git
cd LexiMind
pip install -r requirements.txt
```

### Training

```bash
# Full training (~9 hours on RTX 4070 12GB)
python scripts/train.py training=full

# Quick dev run
python scripts/train.py training=dev

# Override parameters
python scripts/train.py training=full training.optimizer.lr=5e-5

# Resume from checkpoint
python scripts/train.py training=full resume_from=checkpoints/epoch_5.pt
```

Experiments are tracked with MLflow (`mlflow ui` to browse).

### Evaluation

```bash
python scripts/evaluate.py
python scripts/evaluate.py --skip-bertscore    # faster
python scripts/evaluate.py --tune-thresholds   # per-class threshold tuning
```

### Inference

```bash
# Command-line
python scripts/inference.py "Your text to analyze"

# Gradio web demo
python scripts/demo_gradio.py
```

### Profiling

```bash
# Profile GPU usage (CUDA kernels, memory, Chrome trace)
python scripts/profile_training.py
```

### Docker

```bash
docker build -t leximind .
docker run -p 7860:7860 leximind
```

## Project Structure

```text
src/
├── models/          # Encoder, decoder, attention, FFN, heads, factory
├── data/            # Datasets, dataloaders, tokenization, cross-task dedup
├── training/        # Trainer (AMP, grad accum, temperature sampling), metrics
├── inference/       # Pipeline + factory for checkpoint loading
├── api/             # FastAPI REST endpoint
└── utils/           # Device detection, checkpointing, label I/O

scripts/
├── train.py                    # Hydra training entry point
├── evaluate.py                 # Full evaluation suite
├── inference.py                # CLI inference
├── demo_gradio.py              # Gradio discovery demo
├── profile_training.py         # PyTorch profiler
├── train_multiseed.py          # Multi-seed training with aggregation
├── visualize_training.py       # Training curve visualization
├── download_data.py            # Dataset downloader
└── build_discovery_dataset.py  # Pre-compute discovery dataset

configs/             # Hydra configs (model, training, data)
docs/                # Research paper + architecture documentation
tests/               # Pytest suite
```

## Code Quality

```bash
ruff check .                     # Linting
mypy src/ scripts/ tests/        # Type checking
pytest                           # Tests
pre-commit run --all-files       # All checks
```

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

---

Built by Oliver Perrin · Appalachian State University · 2025–2026
