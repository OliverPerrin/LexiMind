# Training Procedure

## Data Sources
- **Summarization** – expects JSONL files with `source` and `summary` fields under
  `data/processed/summarization`.
- **Emotion Classification** – multi-label samples loaded from JSONL files with
  `text` and `emotions` arrays. The dataset owns a `MultiLabelBinarizer` for consistent encoding.
- **Topic Classification** – single-label categorical samples with `text` and `topic` fields, encoded via `LabelEncoder`.

Paths and tokenizer defaults are configured in `configs/data/datasets.yaml`. The tokenizer section chooses the Hugging Face backbone (`google/flan-t5-base` by default) and maximum length. Gutenberg book downloads are controlled via the `downloads.books` list (each entry includes `name`, `url`, and `output`).

## Dataloaders & Collators
- `SummarizationCollator` encodes encoder/decoder inputs, prepares decoder input IDs via `Tokenizer.prepare_decoder_inputs`, and masks padding tokens with `-100` for loss computation. Note: FLAN-T5 uses `pad_token_id=0` and `decoder_start_token_id=0`.
- `EmotionCollator` applies the dataset's `MultiLabelBinarizer`, returning dense float tensors suitable for `BCEWithLogitsLoss`.
- `TopicCollator` emits integer class IDs via the dataset's `LabelEncoder` for `CrossEntropyLoss`.

These collators keep all tokenization centralized, reducing duplication and making it easy to swap in additional sklearn transformations through `TextPreprocessor` should we wish to extend cleaning or normalization.

## Model Assembly
- `src/models/factory.build_multitask_model` rebuilds the encoder, decoder, and heads from the tokenizer metadata and YAML config. This factory is used both during training and inference to eliminate drift between environments.
- Pretrained weights are loaded from FLAN-T5 using `_load_t5_weights()`, which transfers:
  - Shared token embeddings (with proper scaling)
  - Attention projections (q, k, v, o) for all encoder/decoder layers
  - FFN weights (wi_0, wi_1 for gated activation, wo for output)
  - Layer normalization parameters (mapped from T5's RMSNorm)
- The model wraps:
  - Transformer encoder/decoder stacks with **Pre-LN RMSNorm** architecture.
  - LM head tied to decoder embeddings for summarization.
  - Mean-pooled classification heads for emotion and topic tasks.

## Optimisation Loop
- `src/training/trainer.Trainer` orchestrates multi-task training.
  - Cross-entropy is used for summarization (seq2seq logits vs. shifted labels).
  - `BCEWithLogitsLoss` handles multi-label emotions.
  - `CrossEntropyLoss` handles topic classification.
- Gradient clipping ensures stability, and per-task weights can be configured via
  `TrainerConfig.task_weights` to balance gradients if needed.
- Metrics tracked per task:
  - **Summarization** – ROUGE-like overlap metric (`training.metrics.rouge_like`).
  - **Emotion** – micro F1 score for multi-label predictions.
  - **Topic** – categorical accuracy.

## Checkpoints & Artifacts
- `src/utils/io.save_state` stores model weights; checkpoints live under `checkpoints/`.
- `artifacts/labels.json` captures the ordered emotion/topic vocabularies immediately after
  training. This file is required for inference so class indices map back to human-readable labels.
- The tokenizer is exported to `artifacts/hf_tokenizer/` for reproducible vocabularies using `scripts/export_tokenizer.py`.

## Running Training
1. Ensure processed datasets are available (see `data/processed/` structure).
2. Export the FLAN-T5 tokenizer: `python scripts/export_tokenizer.py`
3. Choose a configuration (e.g., `configs/training/dev.yaml`) for hyperparameters and data splits.
4. Instantiate the tokenizer via `TokenizerConfig` and build datasets/dataloaders.
5. Use `build_multitask_model` to construct the model with FLAN-T5 weights, create an optimizer, and run
   `Trainer.fit(train_loaders, val_loaders)`.
6. Save checkpoints and update `artifacts/labels.json` with the dataset label order.

```bash
# Quick start
python scripts/export_tokenizer.py          # Export FLAN-T5 tokenizer
python scripts/train.py training=dev        # Run dev training (2 epochs)
python scripts/train.py training=medium     # Run medium training (5 epochs)
python scripts/train.py training=full       # Run full training (10 epochs)
```

## Why FLAN-T5?
LexiMind's custom Transformer uses **Pre-LN (normalization before sublayers)** with **RMSNorm**. This modern architecture choice provides:
- Better gradient flow during training
- Improved training stability
- Faster convergence

FLAN-T5 uses the same Pre-LN RMSNorm architecture, making weight transfer straightforward. Previously used BART (Post-LN LayerNorm) had a fundamental architectural mismatch that caused training issues.

> **Note:** T5's relative position bias is NOT transferred. The model uses learned positional encodings which train from scratch. This is fine since positional information is task-specific.

## Future Enhancements
- Integrate curriculum scheduling or task-balanced sampling once empirical results dictate.
- Capture attention maps during training to support visualization in the planned Gradio UI.
- Leverage the optional `sklearn_transformer` hook in `TextPreprocessor` for lemmatization or domain-specific normalization when datasets require it.
- Experiment with FLAN-T5-large for improved performance on longer sequences.
