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

**Train Short, Infer Long: Multi-Task NLP for Long-Document Content Discovery.**

LexiMind studies how to train a multi-task NLP model on short-form data (paper abstracts, single-sentence emotion examples, paragraph-level topics) and deploy it on long documents (full books, full papers) via chunking and aggregation. We compare against single-task FLAN-T5 baselines and zero-shot frontier-LLM baselines on the same eval set.

The model is a from-scratch encoder-decoder transformer initialized from [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) (272M parameters), jointly trained on **abstractive summarization**, **multi-label emotion detection** (28 classes), and **single-label topic classification** (7 classes).

**[Live Demo](https://huggingface.co/spaces/OliverPerrin/LexiMind)** · **[Model](https://huggingface.co/OliverPerrin/LexiMind-Model)** · **[Discovery Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery)** · **[Working Paper](docs/research_paper.tex)**

## Status

Working paper in progress. Numbers below are from a prior training configuration (8 epochs, frozen encoder layers 0–3, label smoothing 0.1) and will be replaced once the corrected campaign (5 epochs, full encoder fine-tuning, no smoothing) and single-task baselines complete.

## Preliminary Results

| Task | Metric | Score |
| ---- | ------ | ----- |
| Summarization | ROUGE-1 / ROUGE-L | 0.309 / 0.185 |
| Summarization (academic) | ROUGE-1 | 0.319 |
| Summarization (literary) | ROUGE-1 | 0.206 |
| Topic Classification | Accuracy (95% CI) | 85.7% (80.4–91.0%) |
| Emotion Detection | Sample-avg F1 | 0.352 |
| Emotion Detection (tuned thresholds) | Sample-avg F1 / Macro F1 | 0.503 / 0.294 |

Trained for 8 epochs on an RTX 4070 12GB (~9 hours) with BFloat16 mixed precision, `torch.compile`, and cosine LR decay. New campaign (5 epochs, full encoder unfrozen) targets ~7 hours per run.

## Research Questions

The working paper investigates four falsifiable claims:

1. **Does MTL beat single-task at deployment?** Single-task FLAN-T5 baselines (`single_summarization`, `single_emotion`, `single_topic` configs) train on the same data with identical hyperparameters except the task head. We measure delta vs the joint-MTL configuration.
2. **What aggregation strategy works for chunked long-document inference?** We compare mean / max / attention-weighted / length-weighted aggregation when running the trained classification heads over chunked book input.
3. **Which evaluation pitfalls inflate reported MTL gains?** Controlled demonstrations of threshold contamination (tuning thresholds on the same val split used for early stopping) and other bugs we caught in our own pipeline.
4. **Cost vs quality tradeoff against zero-shot frontier LLMs.** Same eval set, scored against Claude / GPT-4 zero-shot, with API-cost and latency reported.

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
- **Full encoder fine-tuning** (no frozen layers): the prior 4-layer freeze handicapped the most encoder-demanding task (summarization)
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
# Multi-task training (~7 hours on RTX 4070 12GB)
python scripts/train.py training=full

# Single-task baselines for MTL comparison
python scripts/train.py training=single_summarization
python scripts/train.py training=single_emotion
python scripts/train.py training=single_topic

# Quick dev run
python scripts/train.py training=dev

# Override parameters
python scripts/train.py training=full training.optimizer.lr=5e-5

# Multi-seed campaign for paper headline numbers
python scripts/train_multiseed.py --seeds 17 42 123 --config training=full
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
├── train.py                    # Hydra training entry point (MTL + single-task baselines)
├── evaluate.py                 # Full evaluation suite
├── inference.py                # CLI inference
├── demo_gradio.py              # Gradio discovery demo
├── profile_training.py         # PyTorch profiler
├── train_multiseed.py          # Multi-seed training with aggregation
├── train_bert_baseline.py      # BERT single-task baselines (architecture comparison)
├── visualize_training.py       # Training curve visualization
├── download_data.py            # Dataset downloader
└── build_discovery_dataset.py  # Pre-compute discovery dataset

configs/training/
├── full.yaml                   # Joint MTL training (paper headline config)
├── single_summarization.yaml   # FLAN-T5 single-task summarization baseline
├── single_emotion.yaml         # FLAN-T5 single-task emotion baseline
├── single_topic.yaml           # FLAN-T5 single-task topic baseline
├── medium.yaml                 # Mid-size dev config
└── dev.yaml                    # Quick smoke-test config

docs/                # Research paper + architecture documentation
tests/               # Pytest suite
```

## Long-Document Deployment (in progress)

Coming Weeks 3–4 of the campaign:

- **Chunked inference module** for long-document classification (chunk → per-chunk logits → mean / max / attention-weighted / length-weighted aggregation)
- **Hierarchical summarization** for back-cover-blurb generation (chunk → per-chunk summary → meta-summary)
- **Held-out Gutenberg book evaluation**

This is the deployment scenario the paper studies — short-context model, long-document target.

## Code Quality

```bash
ruff check .                     # Linting
mypy src/ scripts/ tests/        # Type checking
pytest                           # Tests
pre-commit run --all-files       # All checks
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by Oliver Perrin · Appalachian State University · 2025–2026
