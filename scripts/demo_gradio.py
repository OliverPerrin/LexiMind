"""
Gradio demo for LexiMind multi-task NLP model.

Showcases the model's capabilities across three tasks:
- Summarization: Generates concise summaries of input text
- Emotion Detection: Multi-label emotion classification
- Topic Classification: Categorizes text into news topics

Author: Oliver Perrin
Date: 2025-12-04
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

# --------------- Path Setup ---------------

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

SAMPLE_TEXTS = [
    (
        "Artificial intelligence is rapidly transforming technology. "
        "Machine learning algorithms process vast amounts of data, identifying "
        "patterns with unprecedented accuracy. From healthcare to finance, AI is "
        "revolutionizing industries worldwide."
    ),
    (
        "The team's incredible comeback in the final quarter left fans in tears of joy. "
        "After trailing by 20 points, they scored three consecutive touchdowns to secure "
        "their first championship victory in over a decade."
    ),
    (
        "Global markets tumbled today as investors reacted to rising inflation concerns. "
        "The Federal Reserve hinted at potential interest rate hikes, sending shockwaves "
        "through technology and banking sectors."
    ),
]

# --------------- Pipeline Management ---------------

_pipeline = None


def get_pipeline():
    """Lazy-load the inference pipeline, downloading checkpoint if needed."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

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


# --------------- Core Functions ---------------


def analyze(text: str) -> tuple[str, str, str]:
    """Run all three tasks and return formatted results."""
    if not text or not text.strip():
        return "Enter text above", "", ""

    try:
        pipe = get_pipeline()

        # Run tasks
        summary = pipe.summarize([text], max_length=128)[0].strip() or "(empty)"
        emotions = pipe.predict_emotions([text], threshold=0.5)[0]
        topic = pipe.predict_topics([text])[0]

        # Format emotions
        if emotions.labels:
            emotion_str = " • ".join(
                f"**{lbl}** ({score:.0%})"
                for lbl, score in zip(emotions.labels, emotions.scores, strict=True)
            )
        else:
            emotion_str = "No strong emotions detected"

        # Format topic
        topic_str = f"**{topic.label}** ({topic.confidence:.0%})"

        return summary, emotion_str, topic_str

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}", "", ""


def load_metrics() -> str:
    """Load evaluation metrics and format as markdown."""
    if not EVAL_REPORT_PATH.exists():
        return "No evaluation report found."

    try:
        with open(EVAL_REPORT_PATH) as f:
            r = json.load(f)

        return f"""
### Overall Performance

| Task | Metric | Score |
|------|--------|-------|
| **Emotion** | F1 Macro | **{r["emotion"]["f1_macro"]:.1%}** |
| **Topic** | Accuracy | **{r["topic"]["accuracy"]:.1%}** |
| **Summarization** | ROUGE-Like | {r["summarization"]["rouge_like"]:.1%} |
| **Summarization** | BLEU | {r["summarization"]["bleu"]:.1%} |

### Topic Classification (per-class)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Business | {r["topic"]["classification_report"]["Business"]["precision"]:.1%} | {r["topic"]["classification_report"]["Business"]["recall"]:.1%} | {r["topic"]["classification_report"]["Business"]["f1-score"]:.1%} |
| Sci/Tech | {r["topic"]["classification_report"]["Sci/Tech"]["precision"]:.1%} | {r["topic"]["classification_report"]["Sci/Tech"]["recall"]:.1%} | {r["topic"]["classification_report"]["Sci/Tech"]["f1-score"]:.1%} |
| Sports | {r["topic"]["classification_report"]["Sports"]["precision"]:.1%} | {r["topic"]["classification_report"]["Sports"]["recall"]:.1%} | {r["topic"]["classification_report"]["Sports"]["f1-score"]:.1%} |
| World | {r["topic"]["classification_report"]["World"]["precision"]:.1%} | {r["topic"]["classification_report"]["World"]["recall"]:.1%} | {r["topic"]["classification_report"]["World"]["f1-score"]:.1%} |
"""
    except Exception as e:
        return f"Error loading metrics: {e}"


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind Demo",
    theme=gr.themes.Soft(),
    css=".output-box { min-height: 80px; }",
) as demo:
    gr.Markdown(
        """
        # 🧠 LexiMind
        ### Multi-Task Transformer for Document Analysis
        
        A custom encoder-decoder Transformer trained on summarization, emotion detection,
        and topic classification. Built from scratch with PyTorch.
        """
    )

    # --------------- Try It Tab ---------------
    with gr.Tab("🚀 Try It"):
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    placeholder="Enter text to analyze...",
                    value=SAMPLE_TEXTS[0],
                )
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary", scale=2)
                    gr.Examples(
                        examples=[[t] for t in SAMPLE_TEXTS],
                        inputs=text_input,
                        label="Examples",
                    )

            with gr.Column(scale=2):
                summary_out = gr.Textbox(label="📝 Summary", lines=3, elem_classes="output-box")
                emotion_out = gr.Markdown(label="😊 Emotions")
                topic_out = gr.Markdown(label="📂 Topic")

        analyze_btn.click(
            fn=analyze,
            inputs=text_input,
            outputs=[summary_out, emotion_out, topic_out],
        )

    # --------------- Metrics Tab ---------------
    with gr.Tab("📊 Metrics"):
        gr.Markdown(load_metrics())
        gr.Markdown("### Confusion Matrix")
        gr.Image(str(OUTPUTS_DIR / "topic_confusion_matrix.png"), label="Topic Classification")

    # --------------- Architecture Tab ---------------
    with gr.Tab("🔧 Architecture"):
        gr.Markdown(
            """
            ### Model Architecture
            
            - **Base**: Custom Transformer (encoder-decoder)
            - **Initialized from**: FLAN-T5-base weights
            - **Encoder**: 6 layers, 768 hidden dim, 12 attention heads
            - **Decoder**: 6 layers with cross-attention
            - **Task Heads**: Classification heads for emotion/topic
            
            ### Training
            
            - **Optimizer**: AdamW with cosine LR schedule
            - **Mixed Precision**: bfloat16 with TF32
            - **Compilation**: torch.compile with inductor backend
            """
        )
        with gr.Row():
            gr.Image(
                str(OUTPUTS_DIR / "attention_visualization.png"),
                label="Self-Attention Pattern",
            )
            gr.Image(
                str(OUTPUTS_DIR / "positional_encoding_heatmap.png"),
                label="Positional Encodings",
            )

    # --------------- About Tab ---------------
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ### About LexiMind
            
            LexiMind is a multi-task NLP model designed to demonstrate end-to-end
            machine learning engineering skills:
            
            - **Custom Transformer** implementation from scratch
            - **Multi-task learning** with shared encoder
            - **Production-ready** inference pipeline
            - **Comprehensive evaluation** with multiple metrics
            
            ### Links
            
            - 🔗 [GitHub Repository](https://github.com/OliverPerrin/LexiMind)
            - 🤗 [HuggingFace Space](https://huggingface.co/spaces/OliverPerrin/LexiMind)
            
            ### Author
            
            **Oliver Perrin** - Machine Learning Engineer
            """
        )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    get_pipeline()  # Pre-load to fail fast if checkpoint missing
    demo.launch(server_name="0.0.0.0", server_port=7860)
