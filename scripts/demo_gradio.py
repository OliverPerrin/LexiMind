"""
Gradio demo for LexiMind multi-task NLP model.

Provides a simple web interface for the three core tasks:
- Summarization: Generates concise summaries of input text
- Emotion Detection: Identifies emotional content with confidence scores
- Topic Classification: Categorizes text into predefined topics

Author: Oliver Perrin
Date: 2025-12-04
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

# --------------- Path Setup ---------------
# Ensure local src package is importable when running script directly

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download

from src.inference.factory import create_inference_pipeline
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# --------------- Constants ---------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"

SAMPLE_TEXT = (
    "Artificial intelligence is rapidly transforming technology. "
    "Machine learning algorithms process vast amounts of data, identifying "
    "patterns with unprecedented accuracy. From healthcare to finance, AI is "
    "revolutionizing industries worldwide. However, ethical considerations "
    "around privacy and bias remain critical challenges."
)

# --------------- Pipeline Management ---------------

_pipeline = None


def get_pipeline():
    """Lazy-load the inference pipeline, downloading checkpoint if needed."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    checkpoint_path = Path("checkpoints/best.pt")

    # Download from HuggingFace Hub if checkpoint doesn't exist locally
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="OliverPerrin/LexiMind-Model",
            filename="best.pt",
            local_dir="checkpoints",
            local_dir_use_symlinks=False,
        )

    _pipeline, _ = create_inference_pipeline(
        tokenizer_dir="artifacts/hf_tokenizer/",
        checkpoint_path="checkpoints/best.pt",
        labels_path="artifacts/labels.json",
    )
    return _pipeline


# --------------- Core Functions ---------------


def analyze(text: str) -> str:
    """
    Run all three tasks on input text.

    Returns markdown-formatted results for display in Gradio.
    """
    if not text or not text.strip():
        return "Please enter some text to analyze."

    try:
        pipe = get_pipeline()

        # Run each task
        summary = pipe.summarize([text], max_length=128)[0].strip() or "(empty)"
        emotions = pipe.predict_emotions([text], threshold=0.5)[0]
        topic = pipe.predict_topics([text])[0]

        # Format emotion results
        if emotions.labels:
            emotion_str = ", ".join(
                f"{lbl} ({score:.1%})"
                for lbl, score in zip(emotions.labels, emotions.scores, strict=True)
            )
        else:
            emotion_str = "No strong emotions detected"

        return f"""## Summary
{summary}

## Detected Emotions
{emotion_str}

## Topic
{topic.label} ({topic.confidence:.1%})
"""
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}"


def get_metrics() -> str:
    """Load evaluation metrics from JSON and format as markdown tables."""
    if not EVAL_REPORT_PATH.exists():
        return "No evaluation report found. Run `scripts/evaluate.py` first."

    try:
        with open(EVAL_REPORT_PATH) as f:
            r = json.load(f)

        # Build overall metrics table
        lines = [
            "## Model Performance\n",
            "| Task | Metric | Score |",
            "|------|--------|-------|",
            f"| Summarization | ROUGE-Like | {r['summarization']['rouge_like']:.4f} |",
            f"| Summarization | BLEU | {r['summarization']['bleu']:.4f} |",
            f"| Emotion | F1 Macro | {r['emotion']['f1_macro']:.4f} |",
            f"| Topic | Accuracy | {r['topic']['accuracy']:.4f} |",
            "",
            "### Topic Classification Details\n",
            "| Label | Precision | Recall | F1 |",
            "|-------|-----------|--------|-----|",
        ]

        # Add per-class metrics
        for label, metrics in r["topic"]["classification_report"].items():
            if isinstance(metrics, dict) and "precision" in metrics:
                lines.append(
                    f"| {label} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['f1-score']:.3f} |"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"Error loading metrics: {e}"


# --------------- Gradio Interface ---------------

with gr.Blocks(title="LexiMind Demo") as demo:
    gr.Markdown(
        "# LexiMind NLP Demo\n"
        "Multi-task model: summarization, emotion detection, topic classification."
    )

    with gr.Tab("Analyze"):
        text_input = gr.Textbox(label="Input Text", lines=6, value=SAMPLE_TEXT)
        analyze_btn = gr.Button("Analyze", variant="primary")
        output = gr.Markdown(label="Results")
        analyze_btn.click(fn=analyze, inputs=text_input, outputs=output)

    with gr.Tab("Metrics"):
        gr.Markdown(get_metrics())


# --------------- Entry Point ---------------

if __name__ == "__main__":
    get_pipeline()  # Pre-load to fail fast if checkpoint missing
    demo.launch(server_name="0.0.0.0", server_port=7860)
