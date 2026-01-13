---
title: LexiMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: scripts/demo_gradio.py
pinned: false
---

## LexiMind: A Multi-Task NLP Model

LexiMind is a state-of-the-art Natural Language Processing model designed for complex document understanding. It features a **custom-built Transformer architecture** initialized with weights from Google's **FLAN-T5**, combining the flexibility of from-scratch implementation with the power of modern pre-trained models.

The model performs three sophisticated tasks simultaneously: **text summarization**, **emotion classification**, and **topic clustering**.

This project is built with industry-standard MLOps practices, including configuration management with Hydra, experiment tracking with MLflow, and containerization with Docker, making it a reproducible and scalable solution.

## Core Features

* **Abstractive Summarization:** Generates concise, coherent summaries of long-form text using encoder-decoder attention. Trained on CNN/DailyMail (news) and BookSum (literary).
* **Emotion Classification:** Identifies 28 emotions from Google's GoEmotions dataset (admiration, amusement, anger, joy, love, etc.).
* **Topic Classification:** Classifies documents into 4 categories (World, Sports, Business, Sci/Tech) using AG News.

## Model Architecture

LexiMind implements a **from-scratch Transformer** with modern architectural choices:

### Custom Transformer Features

* **Pre-Layer Normalization (Pre-LN):** RMSNorm applied before each sublayer for stable training
* **FlashAttention:** Via PyTorch 2.0's `scaled_dot_product_attention` for efficient computation
* **Learned Positional Embeddings:** Trainable position representations
* **Multi-Head Attention:** 12 heads with 768-dimensional representations
* **RMSNorm:** Modern normalization without bias (more efficient than LayerNorm)

### Pre-trained Weight Initialization

The model loads weights from **Google's FLAN-T5-base**, which provides:

* Strong language understanding from instruction-tuning
* Excellent performance on summarization and classification tasks
* Encoder-decoder architecture matching our custom implementation

### Multi-Task Learning

A shared encoder-decoder backbone with task-specific heads:

* **Summarization Head:** Language modeling head with weight tying
* **Emotion Head:** Mean-pooled classification with dropout
* **Topic Head:** Mean-pooled classification with dropout

## Technical Specifications

| Component | Specification |
| --------- | -------------- |
| Architecture | Encoder-Decoder Transformer |
| Pre-trained Base | google/flan-t5-base |
| Hidden Dimension | 768 |
| Encoder Layers | 12 |
| Decoder Layers | 12 |
| Attention Heads | 12 |
| FFN Dimension | 2048 |
| Normalization | RMSNorm (Pre-LN) |
| Position Encoding | Learned Embeddings |
| Max Sequence Length | 512 tokens |

## Getting Started

### Prerequisites

* Python 3.10+
* Poetry for dependency management
* Docker (for containerized deployment)
* An NVIDIA GPU with CUDA support (for training and accelerated inference)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/OliverPerrin/LexiMind.git
   cd LexiMind
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

3. **Download datasets:**

   ```bash
   poetry run python scripts/download_data.py
   ```

   This downloads CNN/DailyMail, BookSum, GoEmotions, AG News, and Gutenberg books.

## Usage

### Configuration

All training and model parameters are managed via Hydra. Configurations are located in the `configs/` directory.

Available configurations:

* `model=base` - FLAN-T5-base (default, 12 layers)
* `model=small` - Smaller model for testing (no pretrained weights)
* `model=large` - FLAN-T5-large (24 layers, requires more VRAM)
* `training=dev` - Quick development run (~10-15 min)
* `training=medium` - Balanced training (~45-60 min on RTX 4070)
* `training=full` - Full training run (~3-4 hours, or ~24h for max data)

### Training

```bash
# Default training with FLAN-T5-base
poetry run python scripts/train.py

# Quick development run
poetry run python scripts/train.py training=dev

# Medium training run (recommended for RTX 4070)
poetry run python scripts/train.py training=medium

# Override parameters
poetry run python scripts/train.py training.optimizer.lr=5e-5

# Resume from a checkpoint
poetry run python scripts/train.py training=full resume_from=checkpoints/epoch_5.pt
```

Experiments are automatically tracked with MLflow. View results with `mlflow ui`.

### Evaluation

```bash
# Run inference on test data
poetry run python scripts/inference.py "Your text to analyze"
```

### Inference & Demo

```bash
# Command-line inference
poetry run python scripts/inference.py "Your text to analyze"

# Gradio web demo
poetry run python scripts/demo_gradio.py
```

## Docker

```bash
# Build
docker build -t leximind .

# Run demo
docker run -p 7860:7860 leximind
```

## Project Structure

```text
├── configs/            # Hydra configuration files
│   ├── model/          # Model architectures (base, small, large)
│   ├── training/       # Training configs (dev, medium, full)
│   └── data/           # Dataset paths
├── data/
│   └── processed/      # Training data (downloaded via scripts/download_data.py)
│       ├── summarization/  # CNN/DailyMail + BookSum
│       ├── emotion/        # GoEmotions (28 labels)
│       ├── topic/          # AG News (4 categories)
│       └── books/          # Gutenberg prose chunks
├── src/
│   ├── models/         # Custom Transformer implementation
│   │   ├── encoder.py  # TransformerEncoder with Pre-LN RMSNorm
│   │   ├── decoder.py  # TransformerDecoder with KV-cache
│   │   ├── attention.py # Multi-Head Attention with FlashAttention
│   │   └── factory.py  # Model building with FLAN-T5 weight loading
│   ├── data/           # Dataset classes and dataloaders
│   ├── training/       # Trainer with AMP and gradient accumulation
│   └── inference/      # Inference pipeline
├── scripts/
│   ├── train.py        # Main training script
│   ├── download_data.py # Dataset downloader
│   ├── inference.py    # CLI inference
│   └── demo_gradio.py  # Web demo
└── tests/              # Unit tests
```

## Code Quality

* **Ruff:** Fast linting and formatting
* **MyPy:** Static type checking
* **Pytest:** Full test suite covering data, models, and training
* **Pre-commit hooks:** Automated quality checks

```bash
# Install hooks
poetry run pre-commit install

# Lint
poetry run ruff check .

# Type check
poetry run mypy .

# Tests
poetry run pytest
```

## Performance Optimizations

* **torch.compile:** JIT compilation with Inductor backend
* **Mixed Precision:** bfloat16 training on Ampere/Ada GPUs
* **TF32:** Enabled for RTX 30xx/40xx series
* **KV-Cache:** Efficient autoregressive decoding
* **FlashAttention:** Memory-efficient attention via SDPA

## License

MIT License - see [LICENSE](LICENSE) for details.
