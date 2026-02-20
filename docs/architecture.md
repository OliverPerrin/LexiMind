# LexiMind Architecture

## Overview

LexiMind couples a from-scratch Transformer implementation with a modern data and inference stack. The project consists of three major layers:

1. **Data & Tokenization** – HuggingFace tokenizer wrapper with tensor-aware batching and T5-specific decoder input preparation.
2. **Model Composition** – the bespoke encoder/decoder stack with task heads assembled via `MultiTaskModel`, plus `models.factory.build_multitask_model` to rebuild the network from configuration files.
3. **Inference & Serving** – a multi-task pipeline capable of summarization, emotion, and topic classification; surfaced through a CLI and Gradio UI.

## Custom Transformer Stack

The custom Transformer is designed with **modern architectural choices** while maintaining compatibility with pre-trained weights from Google's **FLAN-T5**.

### Architecture Highlights

- **Pre-Layer Normalization (Pre-LN):** RMSNorm applied *before* each sublayer for stable training
- **RMSNorm:** More efficient than LayerNorm (no mean computation, no bias parameters)
- **FlashAttention:** Via PyTorch 2.0's `F.scaled_dot_product_attention` for O(N) memory
- **Learned Positional Embeddings:** Trainable position representations (randomly initialized)
- **Multi-Head Attention:** 12 heads with optional LoRA adapters and RoPE support

### Weight Loading from FLAN-T5

The `factory.py` module loads weights from FLAN-T5-base, which uses a compatible Pre-LN architecture:

- **Token embeddings:** Shared between encoder and decoder
- **Attention projections:** Q, K, V, O weights (bias initialized to zero since T5 has no attention bias)
- **FFN weights:** `wi_1` → `linear1`, `wo` → `linear2` (T5 uses gated FFN; we use the up/down projections)
- **RMSNorm weights:** Direct transfer (both use RMSNorm without bias)
- **LM head:** Loaded from T5's `lm_head`

**Note:** T5 uses *relative position bias* computed in attention, not absolute embeddings. Our learned positional embeddings are randomly initialized and train quickly during fine-tuning.

### File Structure

- `src/models/encoder.py` – TransformerEncoder with Pre-LN RMSNorm blocks
- `src/models/decoder.py` – TransformerDecoder with KV-cache for efficient generation
- `src/models/attention.py` – Multi-Head Attention with FlashAttention, LoRA, and RoPE support
- `src/models/heads.py` – ClassificationHead (mean pooling) and LMHead (with weight tying)
- `src/models/multitask.py` – Routes inputs to task-specific heads
- `src/models/factory.py` – Builds models and loads FLAN-T5 weights

## Data, Tokenization, and Datasets

- `src/data/tokenization.py` wraps `AutoTokenizer` (configured for FLAN-T5) to provide tensor-aware batching and helper utilities for decoder input shifting.
- `src/data/dataset.py` and `src/data/dataloader.py` define strongly typed dataset containers and task-specific collators.
- `scripts/download_data.py` fetches and processes training data from HuggingFace datasets.

### Training Datasets

| Task | Dataset | Size | Labels |
| ---- | ------- | ---- | ------ |
| Summarization | BookSum + arXiv | ~90K | Text→Summary |
| Emotion | GoEmotions | ~43K | 28 emotions (multi-label) |
| Topic | Books + Papers | 3.4K | 7 categories (Arts, Business, Fiction, History, Philosophy, Science, Technology) |
| Books | Gutenberg (prose chunks) | ~30K | Literary text |

### T5 Tokenizer Differences

- **Vocab size:** 32,128 tokens (SentencePiece)
- **Special tokens:** pad=0, eos=1 (no explicit BOS; decoder starts with pad token)
- **Subword tokenization:** Unigram-based (vs BART's BPE)

## Training Pipeline

- `src/training/trainer.py` coordinates multi-task optimization with:
  - Mixed precision training (bfloat16 on Ampere/Ada GPUs)
  - Gradient accumulation for larger effective batch sizes
  - Per-task loss weighting and label smoothing
  - Early stopping based on validation loss
  - Cosine learning rate schedule with warmup
- **torch.compile:** JIT compilation with Inductor backend for 20-40% speedup
- Metrics in `src/training/metrics.py` include accuracy, multi-label F1, and ROUGE-like overlap

## Inference & Serving

- `src/inference/pipeline.py` exposes summarization, emotion, and topic predictions with shared pre-processing, generation, and thresholding logic.
- `src/inference/factory.py` rebuilds the full pipeline using the exported tokenizer artifact
- The CLI (`scripts/inference.py`) drives the pipeline from the command line
- Gradio demo (`scripts/demo_gradio.py`) provides an interactive web interface

## Key Decisions

- **Custom Transformer + Pre-trained Weights:** Building from scratch demonstrates deep understanding while leveraging FLAN-T5's language knowledge
- **Pre-LN RMSNorm:** Modern architecture used by LLaMA, T5 v1.1, and other 2023-2025 models
- **Simplified Training:** Removed NaN detection and gradient monitoring (Windows workarounds no longer needed on WSL/Linux)
- **Clean Dataset Pipeline:** AG News (4 clean categories) instead of Yahoo Answers (10 messy categories); BookSum for literary summarization
- **Tokenizer Artifact Preference:** Inference favors `artifacts/hf_tokenizer` for reproducibility
