---
title: LexiMind
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: scripts/demo_gradio.py
pinned: false
license: mit
short_description: Multi-task transformer for document understanding
---

# LexiMind

LexiMind is a multitask transformer that performs document summarization, multi-label emotion detection, and topic classification in a single Gradio experience. The project packages the training code, inference pipeline, and visual analytics needed to explore model behavior.

## Run The Demo Locally

```bash
pip install -r requirements.txt
python scripts/demo_gradio.py
```

The Gradio space expects the following assets to be available at runtime:

- `checkpoints/best.pt` – multitask model weights
- `artifacts/hf_tokenizer/` – tokenizer files (or adjust the `tokenizer_dir` argument)
- `data/labels.json` – label metadata for emotion and topic heads

## Features

- 📝 **Text Summarization** with adjustable compression
- 😊 **Emotion Detection** with visualization
- 🏷️ **Topic Prediction** with confidence scores
- 🔥 **Attention Heatmap** visualization

## Project Structure

```
.
├── configs/                 # YAML presets for data, model, and training runs
├── scripts/
│   ├── demo_gradio.py       # Hugging Face Space entry point
│   ├── train.py             # Training CLI
│   └── inference.py         # Batch inference utility
├── src/
│   ├── data/                # Tokenization, datasets, and dataloaders
│   ├── inference/           # Pipeline orchestration for multitask heads
│   ├── models/              # Encoder/decoder/backbone modules
│   ├── training/            # Trainer, callbacks, metrics, and losses
│   └── visualization/       # Attention, embeddings, and metric plots
├── tests/                   # Pytest suites for API, data, inference, models, training
├── artifacts/               # Saved tokenizer assets
├── checkpoints/             # Pretrained multitask checkpoints
└── data/                    # Raw, processed, and cached datasets
```

## Usage

Enter your text, adjust the compression slider, and click "Analyze" to see the results!

## Repository

GitHub: [OliverPerrin/LexiMind](https://github.com/OliverPerrin/LexiMind)

HuggingFace: [OliverPerrin/LexiMind](https://huggingface.co/spaces/OliverPerrin/LexiMind)
