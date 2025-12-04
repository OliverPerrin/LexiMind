"""Minimal Gradio demo for LexiMind multitask model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download

from src.inference.factory import create_inference_pipeline
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        checkpoint_path = Path("checkpoints/best.pt")
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


def analyze(text: str) -> str:
    """Run all three tasks and return results as formatted text."""
    if not text or not text.strip():
        return "Please enter some text to analyze."

    try:
        pipe = get_pipeline()

        # Summarization
        summary = pipe.summarize([text], max_length=128)[0].strip() or "(empty)"

        # Emotion detection
        emotions = pipe.predict_emotions([text], threshold=0.5)[0]
        if emotions.labels:
            emotion_str = ", ".join(
                f"{lbl} ({score:.1%})"
                for lbl, score in zip(emotions.labels, emotions.scores, strict=True)
            )
        else:
            emotion_str = "No strong emotions detected"

        # Topic classification
        topic = pipe.predict_topics([text])[0]
        topic_str = f"{topic.label} ({topic.confidence:.1%})"

        return f"""## Summary
{summary}

## Detected Emotions
{emotion_str}

## Topic
{topic_str}
"""
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}"


def get_metrics() -> str:
    """Load evaluation metrics as markdown."""
    if not EVAL_REPORT_PATH.exists():
        return "No evaluation report found. Run `scripts/evaluate.py` first."

    try:
        with open(EVAL_REPORT_PATH) as f:
            r = json.load(f)

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
        for k, v in r["topic"]["classification_report"].items():
            if isinstance(v, dict) and "precision" in v:
                lines.append(
                    f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} |"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"Error loading metrics: {e}"


SAMPLE = """Artificial intelligence is rapidly transforming technology. Machine learning algorithms process vast amounts of data, identifying patterns with unprecedented accuracy. From healthcare to finance, AI is revolutionizing industries worldwide. However, ethical considerations around privacy and bias remain critical challenges."""

with gr.Blocks(title="LexiMind Demo") as demo:
    gr.Markdown(
        "# LexiMind NLP Demo\nMulti-task model: summarization, emotion detection, topic classification."
    )

    with gr.Tab("Analyze"):
        text_input = gr.Textbox(label="Input Text", lines=6, value=SAMPLE)
        analyze_btn = gr.Button("Analyze", variant="primary")
        output = gr.Markdown(label="Results")
        analyze_btn.click(fn=analyze, inputs=text_input, outputs=output)

    with gr.Tab("Metrics"):
        gr.Markdown(get_metrics())

if __name__ == "__main__":
    get_pipeline()  # Pre-load
    demo.launch(server_name="0.0.0.0", server_port=7860)
