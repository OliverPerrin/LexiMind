# LexiMind Architecture

## Overview
LexiMind couples a from-scratch Transformer implementation with a modern data and inference stack. The project consists of three major layers:

1. **Data & Preprocessing** – lightweight text cleaning built on top of scikit-learn
   primitives and a Hugging Face tokenizer wrapper with deterministic batching helpers.
2. **Model Composition** – the bespoke encoder/decoder stack with task heads assembled via
   `MultiTaskModel`, plus `models.factory.build_multitask_model` to rebuild the network from
   configuration files.
3. **Inference & Serving** – a multi-task pipeline capable of summarization, emotion, and topic classification; surfaced through a CLI and FastAPI service with plans for a Gradio UI.

## Custom Transformer Stack
- `src/models/encoder.py` and `src/models/decoder.py` implement Pre-LayerNorm Transformer
  blocks with explicit positional encoding, masking logic, and incremental decoding support.
- `src/models/heads.py` provides modular output heads. Summarization uses an `LMHead` tied to
  the decoder embedding weights; emotion and topic tasks use `ClassificationHead` instances.
- `src/models/multitask.py` routes inputs to the correct head, computes task-specific losses,
  and exposes a single forward API used by the trainer and inference pipeline.
- `src/models/factory.py` rebuilds the encoder, decoder, and heads directly from YAML config
  and tokenizer metadata so inference rebuilds the exact architecture used in training.

## Data, Tokenization, and Preprocessing
- `src/data/tokenization.py` wraps `AutoTokenizer` to provide tensor-aware batching and helper
  utilities for decoder input shifting, BOS/EOS resolution, and vocab size retrieval.
- `src/data/preprocessing.py` introduces `TextPreprocessor`, layering a `BasicTextCleaner` with
  optional scikit-learn transformers (via `sklearn_transformer`) before tokenization. This keeps
  the default cleaning minimal while allowing future reuse of `sklearn.preprocessing` utilities
  without changing calling code.
- `src/data/dataset.py` and `src/data/dataloader.py` define strongly typed dataset containers and
  collators that encode inputs with the shared tokenizer and set up task-specific labels (multi-label
  emotions, categorical topics, seq2seq summaries).

## Training Pipeline
- `src/training/trainer.py` coordinates multi-task optimization with per-task loss functions, gradient clipping, and shared tokenizer decoding for metric computation.
- Metrics in `src/training/metrics.py` include accuracy, multi-label F1, and a ROUGE-like overlap score for summarization. These metrics mirror the trainer outputs logged per task.
- Label vocabularies are serialized to `artifacts/labels.json` after training so inference can decode class indices consistently.

## Inference & Serving
- `src/inference/pipeline.py` exposes summarization, emotion, and topic predictions with shared pre-processing, generation, and thresholding logic. It expects label vocabularies from the serialized metadata file.
- `src/inference/factory.py` rebuilds the full pipeline by loading the tokenizer (preferring the exported tokenizer artifact), reconstructing the model via the factory helpers, restoring checkpoints, and injecting label metadata.
- The CLI (`scripts/inference.py`) drives the pipeline from the command line. The FastAPI app (`src/api/routes.py`) exposes the `/summarize` endpoint that returns summaries, emotion labels + scores, and topic predictions. Test coverage in `tests/test_inference` and `tests/test_api` validates both layers with lightweight stubs.

## Gradio UI Roadmap
- The inference pipeline returns structured outputs that are already suitable for a web UI.
- Planned steps for a Gradio demo:
  1. Wrap `InferencePipeline.batch_predict` inside Gradio callbacks for text input.
  2. Display summaries alongside emotion tag chips and topic confidence bars.
  3. Surface token-level attention visualizations by extending the pipeline to emit decoder attention maps (hooks already exist in the decoder).
- Documentation and code paths were structured to keep the Gradio integration isolated in a future `src/ui/gradio_app.py` module without altering core logic.

## Key Decisions
- **Custom Transformer Preservation** – all modeling remains on the bespoke encoder/decoder, satisfying the constraint to avoid Hugging Face model classes while still leveraging their tokenizer implementation.
- **Tokenizer Artifact Preference** – inference automatically favors the exported tokenizer in `artifacts/hf_tokenizer`, guaranteeing consistent vocabularies between training and serving.
- **Sklearn-friendly Preprocessing** – the text preprocessor now accepts an optional
  `TransformerMixin` so additional normalization (lemmatization, custom token filters, etc.) can be injected using familiar scikit-learn tooling without rewriting the batching code.
- **Documentation Alignment** – the `docs/` folder mirrors the structure requested, capturing design reasoning and paving the way for future diagrams in `docs/images`.
