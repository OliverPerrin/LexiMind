# API & CLI Documentation

## FastAPI Service
The FastAPI application is defined in `src/api/app.py` and wires routes from
`src/api/routes.py`. All dependencies resolve through `src/api/dependencies.py`, which lazily constructs the shared inference pipeline.

### POST `/summarize`
- **Request Body** (`SummaryRequest`):
  ```json
  {
    "text": "Your input document"
  }
  ```
- **Response** (`SummaryResponse`):
  ```json
  {
    "summary": "Generated abstractive summary",
    "emotion_labels": ["joy", "surprise"],
    "emotion_scores": [0.91, 0.63],
    "topic": "news",
    "topic_confidence": 0.82
  }
  ```
- **Behaviour:**
  1. Text is preprocessed through `TextPreprocessor` (with optional sklearn transformer if configured).
  2. The multitask model generates a summary via greedy decoding.
  3. Emotion and topic heads produce logits which are converted to probabilities and mapped to
     human-readable labels using `artifacts/labels.json`.
  4. Results are returned as structured JSON suitable for a future Gradio interface.

### Error Handling
- If the checkpoint or label metadata is missing, the dependency raises an HTTP 503 error with
  an explanatory message.
- Validation errors (missing `text`) are handled automatically by FastAPI/Pydantic.

## Command-Line Interface
`scripts/inference.py` provides a CLI that mirrors the API behaviour.

### Usage
```bash
python scripts/inference.py "Document to analyse" \
  --checkpoint checkpoints/best.pt \
  --labels artifacts/labels.json \
  --tokenizer artifacts/hf_tokenizer \
  --model-config configs/model/base.yaml \
  --device cpu
```

Options:
- `text` – zero or more positional arguments. If omitted, use `--file` to point to a newline
  delimited text file.
- `--file` – optional path containing one text per line.
- `--checkpoint` – path to the trained model weights.
- `--labels` – JSON containing emotion/topic vocabularies (defaults to `artifacts/labels.json`).
- `--tokenizer` – optional tokenizer directory; defaults to the exported artifact if present.
- `--model-config` – YAML describing the architecture.
- `--device` – `cpu` or `cuda`. Passing `cuda` attempts to run inference on GPU.
- `--summary-max-length` – overrides the default maximum generation length.

### Output
The CLI prints a JSON array where each entry contains the original text, summary, emotion labels
with scores, and topic prediction. This format is identical to the REST response, facilitating
integration tests and future Gradio UI rendering.

## Future Gradio UI
- The planned UI will call the same inference pipeline and display results interactively.
- Given the response schema, the UI can show:
  - Generated summary text.
  - Emotion chips with probability bars.
  - Topic confidence gauges.
  - Placeholder panel for attention heatmaps and explanations.
- Once implemented, documentation updates will add a `docs/ui.md` section and screenshots under
  `docs/images/`.

## Testing
- `tests/test_api/test_routes.py` stubs the pipeline to ensure response fields and dependency
  overrides behave as expected.
- `tests/test_inference/test_pipeline.py` validates pipeline methods end-to-end with dummy models,
  guaranteeing API and CLI consumers receive consistent payload shapes.
