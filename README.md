# LexiMind (Inference Edition)

LexiMind now ships as a focused inference sandbox for the custom multitask Transformer found in
`src/models`. Training, dataset downloaders, and legacy scripts have been removed so it is easy to
load a checkpoint, run the Streamlit demo, and experiment with summarization, emotion
classification, and topic cues on your own text.

## What Stays
- Transformer encoder/decoder and task heads under `src/models`
- Unit tests for the model stack (`tests/test_models`)
- Streamlit UI (`src/ui/streamlit_app.py`) wired to the inference helpers in `src/api/inference`

## What Changed
- Hugging Face tokenizers provide all tokenization (see `TextPreprocessor`)
- Training, dataset downloaders, and CLI scripts have been removed
- Scikit-learn powers light text normalization (stop-word removal optional)
- Requirements trimmed to inference-only dependencies

## Quick Start
```bash
git clone https://github.com/OliverPerrin/LexiMind.git
cd LexiMind
pip install -r requirements.txt

# Optional extras via setup.py packaging metadata
pip install .[web]   # installs streamlit + plotly
pip install .[api]   # installs fastapi
pip install .[all]   # installs both groups

streamlit run src/ui/streamlit_app.py
```

Configure the Streamlit app via the sidebar to point at your tokenizer directory and model
checkpoint (defaults assume `artifacts/hf_tokenizer` and `checkpoints/best.pt`).

## Minimal Project Map
```
src/
├── api/       # load_models + helpers
├── data/      # TextPreprocessor using Hugging Face + sklearn
├── inference/ # thin summarizer facade
├── models/    # core Transformer architecture (untouched)
└── ui/        # Streamlit interface
```

Everything outside `src/` now holds optional assets such as checkpoints, tokenizer exports, and
documentation stubs.

## Loading a Checkpoint Programmatically
```python
from src.api.inference import load_models, summarize_text

models = load_models({
    "checkpoint_path": "checkpoints/best.pt",
    "tokenizer_path": "artifacts/hf_tokenizer",
    "hf_tokenizer_name": "facebook/bart-base",
})

summary, _ = summarize_text("Paste any article here.", models=models)
print(summary)
```

## License
GPL-3.0

## Author
Oliver Perrin · oliver.t.perrin@gmail.com
