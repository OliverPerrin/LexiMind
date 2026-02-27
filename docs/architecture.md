# LexiMind Architecture

## Overview

LexiMind is a **272M parameter encoder-decoder transformer** initialized from Google's FLAN-T5-base, trained jointly on three tasks: abstractive summarization, topic classification, and multi-label emotion detection. The project spans data preparation, custom model architecture, multi-task training, evaluation, and a Gradio-based discovery demo.

## Model Architecture

### Backbone: Custom Transformer (FLAN-T5-base Initialization)

The model is a from-scratch PyTorch implementation of a T5-style encoder-decoder. We do **not** use HuggingFace's `T5ForConditionalGeneration` — instead, every component (attention, FFN, normalization, positional encoding) is implemented manually in `src/models/`, then FLAN-T5 weights are loaded layer by layer in `src/models/factory.py`.

```text
Input Text
    │
    ▼
┌─────────────────────────────────────────────┐
│  Shared Encoder (12 layers, 768d, 12 heads) │
│  ┌────────────────────────────────────────┐  │
│  │ Layers 0-3: FROZEN (FLAN-T5 weights)  │  │
│  │ Layers 4-11: TRAINABLE (fine-tuned)   │  │
│  └────────────────────────────────────────┘  │
│  Pre-LN RMSNorm │ T5 Relative Position Bias │
│  FlashAttention (SDPA) │ Gated-GELU FFN      │
└────────────┬──────────────┬──────────────┬───┘
             │              │              │
    ┌────────▼────────┐  ┌─▼──────────┐  ┌▼───────────┐
    │   Decoder       │  │  Attention  │  │    Mean     │
    │  (12 layers)    │  │  Pooling    │  │   Pooling   │
    │  Causal + Cross │  │  (learned)  │  │             │
    │  Attention      │  │      │      │  │      │      │
    │       │         │  │  MLP 768→   │  │  Linear     │
    │  LM Head        │  │  384→28     │  │  768→7      │
    │  (tied weights) │  │             │  │             │
    └────────┬────────┘  └─────┬───────┘  └──────┬─────┘
             │                 │                  │
      Summarization       Emotion (28)       Topic (7)
      (generative)       (multi-label)     (single-label)
```

### Encoder

**File**: `src/models/encoder.py` (317 lines)

- 12 transformer layers, 768-dimensional, 12 attention heads
- **Pre-Layer Normalization (Pre-LN)** using T5-style RMSNorm — normalization applied *before* each sublayer, not after. This is the modern standard (LLaMA, T5 v1.1+, PaLM).
- **T5 Relative Position Bias**: Bucketed log-linear position bias computed in the attention layer. Bidirectional (encoder attends in both directions). Shared across layers (computed once, passed to all layers).
- **FlashAttention**: Via PyTorch 2.0's `F.scaled_dot_product_attention`, which automatically selects the optimal kernel (Flash, memory-efficient, or math fallback). **Note**: T5 does NOT scale attention scores by 1/√d_k — the `scale_scores=False` flag preserves this behavior.
- **Gated-GELU FFN**: Two linear projections (gate + up) element-wise multiplied, then a down projection. Matches T5's `DenseGatedGeluDense`.
- **Gradient checkpointing**: Optional per-layer activation recomputation to reduce VRAM (enabled in our full training config, saves ~2-3 GB).
- Bottom 4 layers are frozen during fine-tuning to preserve FLAN-T5's general language representations.

The encoder processes all input text and produces contextualized representations that are consumed by all three task heads.

### Decoder (Summarization Only)

**File**: `src/models/decoder.py` (749 lines)

- 12 transformer layers, 768-dimensional, 12 attention heads
- ~136M parameters — roughly half the total model
- **Masked self-attention** (causal mask prevents attending to future positions)
- **Cross-attention** to encoder outputs (allows decoder to attend to the full input)
- **KV-cache** for efficient autoregressive generation — incremental key/value computation avoids recomputing previous positions
- **Greedy decoding** with:
  - No-repeat n-gram blocking (`no_repeat_ngram_size=3`)
  - Repetition penalty (1.2x)
  - Length penalty
  - Min/max length constraints
- **LM Head**: Linear projection from 768d → 32,128 vocab. **Weight-tied** with decoder token embeddings (reduces parameters and improves coherence).

The decoder is exclusive to summarization. Classification tasks only use the encoder.

### Task Heads

**File**: `src/models/heads.py` (221 lines)

#### Emotion Head (Attention Pooling + MLP)

- **AttentionPooling**: A single linear layer (`nn.Linear(768, 1, bias=False)`) serves as a learned query. It computes softmax attention weights over all encoder positions, producing a weighted sum. This allows the model to focus on emotionally salient tokens (e.g., "grateful", "hilarious") rather than averaging the entire 512-token sequence. Padding is masked before softmax.
- **2-layer MLP**: 768 → 384 (GELU) → 28. The hidden layer provides nonlinear feature transformation before the 28-way multi-label output.
- **Loss**: BCEWithLogitsLoss (binary cross-entropy per class)
- **Inference threshold**: 0.3 (lowered from default 0.5 because 28-class multi-label predictions have lower per-class confidence)

#### Topic Head (Mean Pooling + Linear)

- **Mean pooling** over encoder positions (attention-mask-aware)
- **Single linear layer**: 768 → 7
- **Loss**: CrossEntropyLoss
- **Task weight**: 0.3 (reduced to prevent overfitting on the small 3.4K dataset)

#### Summarization Head (Decoder + LM Head)

- Full decoder (described above) + weight-tied LM head
- **Loss**: CrossEntropyLoss with label smoothing (0.1) and `-100` ignore index for padding
- **Task weight**: 1.0

### Multi-Task Router

**File**: `src/models/multitask.py` (263 lines)

The `MultiTaskModel` class routes `forward(task, inputs)` calls to the correct head:

- **Classification** (`emotion`, `topic`): encoder → pool → classify
- **Generation** (`summarization`): encoder → decoder → LM head

A `memory.clone()` call between encoder and decoder output prevents CUDA Graph buffer reuse issues when using `torch.compile`.

### Weight Loading from FLAN-T5

**File**: `src/models/factory.py` (571 lines)

Weights are transferred from HuggingFace's `google/flan-t5-base` checkpoint layer by layer:

| FLAN-T5 Component | Our Component |
| --- | --- |
| `shared.weight` | `encoder.embed_tokens.weight` and `decoder.embed_tokens.weight` |
| `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}` | `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight` |
| `encoder.block.{i}.layer.1.DenseReluDense.wi_0/wi_1/wo` | `encoder.layers.{i}.ffn.gate/up_proj/down_proj.weight` |
| `encoder.block.{i}.layer.{0,1}.layer_norm.weight` | `encoder.layers.{i}.norm{1,2}.weight` |
| `encoder.block.0.layer.0.SelfAttention.relative_attention_bias` | `encoder.layers.0.self_attn.attn.position_bias.relative_attention_bias` |
| `lm_head.weight` | `summarization_head.projection.weight` |

Vocab size mismatch (T5: 32,100 → ours: 32,128) is handled by zero-padding the embedding matrix.

### Available but Unused Components

These are implemented but not activated in the current configuration:

- **LoRA adapters** on Q and V projections in `MultiHeadAttention` — for parameter-efficient fine-tuning
- **Rotary Position Embeddings (RoPE)** — alternative to T5's relative position bias
- **4-bit/8-bit quantization** via bitsandbytes — for inference on constrained hardware
- **TokenClassificationHead** — for NER/POS tasks
- **ProjectionHead** — for contrastive/representation learning
- **LLaMA weight loading** — `_load_llama_weights()` for loading Gemma/LLaMA checkpoints

## Tokenization

**File**: `src/data/tokenization.py` (157 lines)

Wraps HuggingFace's `AutoTokenizer` configured for FLAN-T5:

- **SentencePiece** (Unigram) tokenizer, 32,128 vocabulary
- Special tokens: `pad=0`, `eos=1`, no explicit BOS (decoder starts with pad token, per T5 convention)
- Max sequence length: 512 tokens (encoder), 128 tokens (decoder during validation generation)
- Classification tasks use a reduced max length of 256 tokens (sufficient for classification, saves compute)

## Datasets

**File**: `src/data/dataset.py` (316 lines), `src/data/dataloader.py` (174 lines)

| Task | Dataset Source | Train Size | Val Size | Test Size |
| ------ | --------------- | ----------- | --------- | ---------- |
| Summarization | arXiv abstracts (~45K) + Goodreads book descriptions (~4K) | ~49K | ~2.7K | ~2.7K |
| Emotion | GoEmotions (Reddit comments, 28 labels) | ~43K | ~5.4K | — |
| Topic | arXiv categories + Gutenberg subjects → 7 classes | ~3.2K | ~189 | — |

**Cross-task deduplication**: `deduplicate_across_tasks()` uses MD5 fingerprinting on normalized text prefixes (200 chars) to detect and remove overlapping documents between summarization and topic datasets (both draw from arXiv and Gutenberg).

**Data pipeline**: Each task has a typed `Dataset` class and a corresponding `Collator` that handles tokenization, padding, and label preparation. Collators are passed to PyTorch `DataLoader` instances created by factory functions (`build_*_dataloader`).

## Training

**File**: `src/training/trainer.py` (527 lines)

### Training Loop

Each epoch iterates through batches using **temperature-based task sampling**:

1. **Sample task** with probability p_i proportional to n_i^0.5 where n_i is dataset size
   - Summarization (~49K): ~45% of steps
   - Emotion (~43K): ~43% of steps
   - Topic (~3.4K): ~12% of steps
2. **Forward pass** under `torch.autocast(dtype=bfloat16)` mixed precision
3. **Compute task-specific loss** with task weight (summ=1.0, emotion=1.0, topic=0.3)
4. **Backward pass** and accumulate gradients (4 accumulation steps → effective batch size 40)
5. **Optimizer step** every 4 batches: clip gradients (max norm 1.0), AdamW step, cosine LR step

### Training Configuration (full.yaml)

| Parameter | Value | Rationale |
| ----------- | ------- | ----------- |
| Batch size | 10 | Fits ~10GB VRAM on RTX 4070 12GB |
| Gradient accumulation | 4 | Effective batch size 40 |
| Learning rate | 3e-5 | Standard for fine-tuning T5 |
| Weight decay | 0.01 | Standard AdamW regularization |
| Warmup steps | 300 | ~0.5 epochs of linear warmup |
| Max epochs | 8 | Val loss still improving at epoch 8 |
| LR schedule | Cosine | Decays to 0.1x base LR, flattens near step 8000 |
| Early stopping | Patience 3 | Never triggered (val loss monotonically decreased) |
| Label smoothing | 0.1 | Summarization cross-entropy only |
| Task weights | summ=1.0, emot=1.0, topic=0.3 | Reduced topic weight to prevent overfitting |
| Task sampling | Temperature (alpha=0.5) | Square-root proportional sampling |
| Frozen encoder layers | 0-3 | Preserves FLAN-T5's general language knowledge |
| Gradient checkpointing | Enabled | Saves ~2-3 GB VRAM |
| torch.compile | Both encoder and decoder | ~20-40% speedup via Inductor backend |

### Mixed Precision

The RTX 4070 (Ada Lovelace, compute capability 8.9) has dedicated BF16 tensor cores:

- All forward/backward passes run under `torch.autocast("cuda", dtype=torch.bfloat16)`
- BF16 has the same exponent range as FP32 (8 bits), so no GradScaler is needed (unlike FP16)
- Loss computation and softmax remain in FP32 (handled automatically by autocast)
- Encoder/decoder layers include `clamp(min=-65504, max=65504)` stability guards (carried over from HuggingFace T5)

### Optimizer

- **Fused AdamW**: CUDA-native fused kernel (`torch.optim.AdamW(fused=True)`), ~5-10% faster than standard AdamW
- Betas: (0.9, 0.98) — slightly faster momentum decay than default
- Epsilon: 1e-6

### Gradient Conflict Diagnostics (Available, Disabled)

The trainer includes `_compute_gradient_conflicts()` which:

1. Computes per-task gradients independently
2. Flattens all parameter gradients into a single vector per task
3. Computes pairwise cosine similarity between task gradient vectors
4. Logs cosine similarity and binary conflict flags to MLflow

This is a **diagnostic only** — it does not modify gradients (unlike PCGrad/CAGrad). Disabled by default (`gradient_conflict_frequency: 0`) because it requires extra backward passes per measurement.

### MLflow Tracking

Training metrics (losses, accuracy, F1, ROUGE, learning rate) are logged to MLflow with a SQLite backend (`mlruns.db`). This enables experiment comparison across training runs.

## Evaluation

**File**: `scripts/evaluate.py` (538 lines), `src/training/metrics.py` (452 lines)

### Metrics

| Task | Metrics |
| ------ | --------- |
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L (`rouge-score` library), BLEU-4 (NLTK), optional BERTScore |
| Emotion | Sample-averaged F1, macro F1, micro F1, per-class P/R/F1, per-class threshold tuning |
| Topic | Accuracy, macro F1, per-class P/R/F1, confusion matrix |
| All | Bootstrap 95% confidence intervals (1000 resamples), paired bootstrap test |

### Per-Class Threshold Tuning (Emotion)

For multi-label classification, different emotion classes have very different base rates and prediction confidence. The tuning procedure:

1. For each of the 28 emotion classes independently
2. Sweep threshold tau in {0.1, 0.2, ..., 0.9}
3. Select the threshold that maximizes per-class F1 on the validation set
4. Re-compute all metrics with the tuned thresholds

This improved macro F1 from 0.143 (default 0.5 threshold) to 0.294.

### BERTScore

Available via `--include-bertscore` flag in evaluation (opt-in). Uses `roberta-large` for semantic similarity. Not included in primary evaluation due to computational cost and difficulty interpreting absolute values.

## Inference

**File**: `src/inference/pipeline.py` (217 lines), `src/inference/factory.py` (91 lines)

`InferencePipeline` loads a trained checkpoint and runs all three tasks:

- **Summarization**: Greedy decode with KV-cache, no-repeat trigram blocking, repetition penalty 1.2
- **Emotion**: Sigmoid probabilities → threshold at 0.3 → emit labels above threshold
- **Topic**: Softmax → argmax → emit top label with confidence score

`create_inference_pipeline()` reconstructs the full pipeline from checkpoint + labels + tokenizer artifacts.

## Serving

### Gradio Demo

**File**: `scripts/demo_gradio.py` (507 lines)

A discovery interface for browsing pre-analyzed books and papers. Loads a pre-computed discovery dataset (not live inference) from HuggingFace Hub (`OliverPerrin/LexiMind-Discovery`). Users can browse by topic, emotion, or keyword search.

### FastAPI

**Files**: `src/api/app.py` (18 lines), `src/api/routes.py` (49 lines)

Minimal REST API with a single `/summarize` endpoint that runs all three tasks and returns JSON results. Uses dependency injection for the inference pipeline.

### CLI

**File**: `scripts/inference.py` (108 lines)

Command-line interface accepting text from arguments or file, running batch prediction, and printing JSON output.

### Profiling

**File**: `scripts/profile_training.py`

Wraps a few training steps with `torch.profiler` to capture:

- CUDA kernel timing (per-operator breakdown)
- GPU memory usage (peak allocations)
- CPU/GPU overlap and idle time
- Chrome trace (viewable in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev))
- CUDA stacks for flamegraph generation

```bash
python scripts/profile_training.py                     # 20 steps by default
PROFILE_STEPS=40 python scripts/profile_training.py    # custom step count
```

Outputs go to `outputs/profile/` — TensorBoard traces, Chrome trace JSON, and stack files.

## Training Results (8 Epochs, RTX 4070, ~9 Hours)

| Epoch | Train Loss | Val Loss | Summ Val Loss | Emotion Val F1 | Topic Val Acc |
| ------- | ----------- | --------- | --------------- | ---------------- | --------------- |
| 1 | 6.106 | 4.298 | 3.815 | 0.197 | 70.4% |
| 2 | 5.528 | 4.027 | 3.739 | 0.301 | 84.7% |
| 3 | 5.379 | 3.973 | 3.700 | 0.347 | 84.2% |
| 4 | 5.303 | 3.951 | 3.677 | 0.404 | 85.7% |
| 5 | 5.208 | 3.940 | 3.665 | 0.431 | 86.3% |
| 6 | 5.231 | 3.925 | 3.658 | 0.452 | 87.3% |
| 7 | 5.154 | 3.928 | 3.655 | 0.458 | 85.7% |
| 8 | 5.178 | 3.925 | 3.653 | 0.459 | 85.7% |

Key observations:

- Early stopping never triggered (val loss monotonically decreased through all 8 epochs)
- Topic val accuracy plateaued at epoch 2 (~85%), while topic train accuracy reached 98% — overfitting expected on 3.4K samples
- Emotion F1 improved steadily across all 8 epochs (0.197 → 0.459), showing attention pooling continues learning throughout
- Summarization loss plateaued after epoch 5 (~3.66)
- Train loss was lowest at epoch 7 (5.154), slightly higher at epoch 8 (5.178) — normal variance
- LR schedule cosine curve flattens near step 8000 (0.1x floor)

## Final Evaluation Results

| Task | Metric | Value | 95% CI |
| ------ | -------- | ------- | -------- |
| Summarization | ROUGE-1 | 0.310 | [0.306, 0.313] |
| Summarization | ROUGE-2 | 0.091 | — |
| Summarization | ROUGE-L | 0.185 | — |
| Summarization | BLEU-4 | 0.024 | — |
| Emotion | Sample F1 | 0.352 | [0.340, 0.366] |
| Emotion | Macro F1 | 0.143 | — |
| Emotion | Micro F1 | 0.443 | — |
| Emotion (tuned) | Macro F1 | 0.294 | — |
| Emotion (tuned) | Sample F1 | 0.503 | — |
| Topic | Accuracy | 85.7% | [80.4%, 91.0%] |
| Topic | Macro F1 | 0.854 | — |

Per-domain summarization: Academic ROUGE-1=0.319, Literary ROUGE-1=0.206.

## Project Structure

```text
LexiMind/
├── configs/                 # Hydra configuration
│   ├── config.yaml          # Main config (seeds, paths, device)
│   ├── data/datasets.yaml   # Data paths
│   ├── model/               # Model configs (base, small, large)
│   └── training/            # Training configs (full, medium, dev)
├── src/
│   ├── models/
│   │   ├── encoder.py       # TransformerEncoder (12 Pre-LN layers)
│   │   ├── decoder.py       # TransformerDecoder with KV-cache
│   │   ├── attention.py     # MultiHeadAttention, FlashAttention, T5 relative pos bias, LoRA, RoPE
│   │   ├── heads.py         # AttentionPooling, ClassificationHead, LMHead
│   │   ├── multitask.py     # MultiTaskModel (task routing)
│   │   ├── feedforward.py   # Gated-GELU / SwiGLU / ReLU FFN
│   │   ├── positional_encoding.py  # Sinusoidal + Learned positional encodings
│   │   ├── t5_layer_norm.py # RMSNorm (T5-style)
│   │   └── factory.py       # Model construction + FLAN-T5 weight loading
│   ├── data/
│   │   ├── tokenization.py  # HuggingFace tokenizer wrapper
│   │   ├── dataset.py       # Typed datasets + JSONL loaders + cross-task dedup
│   │   └── dataloader.py    # Task-specific collators + DataLoader factories
│   ├── training/
│   │   ├── trainer.py       # Multi-task trainer (AMP, gradient accum, temperature sampling)
│   │   └── metrics.py       # ROUGE, BLEU, BERTScore, F1 variants, bootstrap CI
│   ├── inference/
│   │   ├── pipeline.py      # Multi-task inference pipeline
│   │   └── factory.py       # Pipeline reconstruction from artifacts
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   └── routes.py        # REST endpoints
│   └── utils/
│       ├── core.py          # Device detection, seed setting
│       ├── io.py            # Checkpoint save/load
│       └── labels.py        # Label metadata I/O
├── scripts/
│   ├── train.py             # Hydra-based training entry point
│   ├── evaluate.py          # Full evaluation with all metrics
│   ├── inference.py         # CLI inference
│   ├── demo_gradio.py       # Gradio discovery demo
│   ├── visualize_training.py # Training visualization suite
│   ├── profile_training.py  # PyTorch profiler for GPU analysis
│   ├── download_data.py     # Data preparation from HuggingFace
│   └── build_discovery_dataset.py  # Pre-compute discovery dataset
├── artifacts/               # Tokenizer + label exports
├── checkpoints/             # Model checkpoints (best.pt + per-epoch)
├── outputs/                 # Evaluation reports, training history, visualizations
└── docs/                    # Architecture docs + research paper
```
