"""
Gradio demo for LexiMind multi-task NLP model.

Showcases the model's capabilities across three tasks:
- Summarization: Generates concise summaries of input text
- Emotion Detection: Multi-label emotion classification
- Topic Classification: Categorizes text into topics

Author: Oliver Perrin
Date: 2025-12-05
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
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"

SAMPLE_TEXTS = [
    "Global markets tumbled today as investors reacted to rising inflation concerns. The Federal Reserve hinted at potential interest rate hikes, sending shockwaves through technology and banking sectors. Analysts predict continued volatility as economic uncertainty persists.",
    "Scientists at MIT have developed a breakthrough quantum computing chip that operates at room temperature. This advancement could revolutionize drug discovery, cryptography, and artificial intelligence. The research team published their findings in Nature.",
    "The championship game ended in dramatic fashion as the underdog team scored in the final seconds to secure victory. Fans rushed the field in celebration, marking the team's first title in 25 years.",
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
        )

    _pipeline, _ = create_inference_pipeline(
        tokenizer_dir="artifacts/hf_tokenizer/",
        checkpoint_path="checkpoints/best.pt",
        labels_path="artifacts/labels.json",
        model_config_path="configs/model/base.yaml",
    )
    return _pipeline


# --------------- Core Functions ---------------


def analyze(text: str) -> tuple[str, str, str]:
    """Run all three tasks and return formatted results."""
    if not text or not text.strip():
        return "Please enter text above to analyze.", "", ""

    try:
        pipe = get_pipeline()

        # Run tasks
        summary = pipe.summarize([text], max_length=128)[0].strip()
        if not summary:
            summary = "(Unable to generate summary)"

        emotions = pipe.predict_emotions([text], threshold=0.3)[0]  # Lower threshold
        topic = pipe.predict_topics([text])[0]

        # Format emotions with emoji
        emotion_emoji = {
            "joy": "😊",
            "love": "❤️",
            "anger": "😠",
            "fear": "😨",
            "sadness": "😢",
            "surprise": "😲",
            "neutral": "😐",
            "admiration": "🤩",
            "amusement": "😄",
            "annoyance": "😤",
            "approval": "👍",
            "caring": "🤗",
            "confusion": "😕",
            "curiosity": "🤔",
            "desire": "😍",
            "disappointment": "😞",
            "disapproval": "👎",
            "disgust": "🤢",
            "embarrassment": "😳",
            "excitement": "🎉",
            "gratitude": "🙏",
            "grief": "😭",
            "nervousness": "��",
            "optimism": "🌟",
            "pride": "🦁",
            "realization": "💡",
            "relief": "😌",
            "remorse": "😔",
        }

        if emotions.labels:
            emotion_parts = []
            for lbl, score in zip(emotions.labels[:5], emotions.scores[:5], strict=False):
                emoji = emotion_emoji.get(lbl.lower(), "•")
                emotion_parts.append(f"{emoji} **{lbl.title()}** ({score:.0%})")
            emotion_str = "\n".join(emotion_parts)
        else:
            emotion_str = "😐 No strong emotions detected"

        # Format topic
        topic_str = f"**{topic.label}**\n\nConfidence: {topic.confidence:.0%}"

        return summary, emotion_str, topic_str

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}", "", ""


def load_metrics() -> str:
    """Load evaluation metrics and format as markdown."""
    # Load evaluation report
    eval_metrics = {}
    if EVAL_REPORT_PATH.exists():
        try:
            with open(EVAL_REPORT_PATH) as f:
                eval_metrics = json.load(f)
        except Exception:
            pass

    # Load training history
    train_metrics = {}
    if TRAINING_HISTORY_PATH.exists():
        try:
            with open(TRAINING_HISTORY_PATH) as f:
                train_metrics = json.load(f)
        except Exception:
            pass

    # Get final validation metrics
    val_final = train_metrics.get("val_epoch_3", {})

    md = """
## 📈 Model Performance

### Training Results (3 Epochs)

| Task | Metric | Final Score |
|------|--------|-------------|
| **Topic Classification** | Accuracy | **{topic_acc:.1%}** |
| **Emotion Detection** | F1 (training) | {emo_f1:.1%} |
| **Summarization** | ROUGE-like | {rouge:.1%} |

### Evaluation Results

| Metric | Value |
|--------|-------|
| Topic Accuracy | **{eval_topic:.1%}** |
| Emotion F1 (macro) | {eval_emo:.1%} |
| ROUGE-like | {eval_rouge:.1%} |
| BLEU | {eval_bleu:.3f} |

---

### Topic Classification Details

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
""".format(
        topic_acc=val_final.get("topic_accuracy", 0),
        emo_f1=val_final.get("emotion_f1", 0),
        rouge=val_final.get("summarization_rouge_like", 0),
        eval_topic=eval_metrics.get("topic", {}).get("accuracy", 0),
        eval_emo=eval_metrics.get("emotion", {}).get("f1_macro", 0),
        eval_rouge=eval_metrics.get("summarization", {}).get("rouge_like", 0),
        eval_bleu=eval_metrics.get("summarization", {}).get("bleu", 0),
    )

    # Add per-class metrics
    topic_report = eval_metrics.get("topic", {}).get("classification_report", {})
    for cat, metrics in topic_report.items():
        if cat in ["macro avg", "weighted avg", "micro avg"]:
            continue
        if isinstance(metrics, dict):
            md += f"| {cat} | {metrics.get('precision', 0):.1%} | {metrics.get('recall', 0):.1%} | {metrics.get('f1-score', 0):.1%} |\n"

    return md


def get_viz_path(filename: str) -> str | None:
    """Get visualization path if file exists."""
    path = OUTPUTS_DIR / filename
    return str(path) if path.exists() else None


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind - Multi-Task NLP",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🧠 LexiMind
        ### Multi-Task Transformer for Document Analysis
        
        A custom encoder-decoder Transformer trained on **summarization**, **emotion detection** (28 classes),
        and **topic classification** (10 categories). Built from scratch with PyTorch.
        
        > ⚠️ **Note**: Summarization is experimental - the model works best on news-style articles.
        """
    )

    # --------------- Try It Tab ---------------
    with gr.Tab("🚀 Try It"):
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Input Text",
                    lines=6,
                    placeholder="Enter or paste text to analyze (works best with news articles)...",
                    value=SAMPLE_TEXTS[0],
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    size="sm",
                )

                gr.Markdown("**Sample Texts** (click to use):")
                with gr.Row():
                    sample1_btn = gr.Button("📰 Markets", size="sm", variant="secondary")
                    sample2_btn = gr.Button("🔬 Science", size="sm", variant="secondary")
                    sample3_btn = gr.Button("🏆 Sports", size="sm", variant="secondary")

                sample1_btn.click(fn=lambda: SAMPLE_TEXTS[0], outputs=text_input)
                sample2_btn.click(fn=lambda: SAMPLE_TEXTS[1], outputs=text_input)
                sample3_btn.click(fn=lambda: SAMPLE_TEXTS[2], outputs=text_input)

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                summary_out = gr.Textbox(
                    label="📝 Summary",
                    lines=3,
                    interactive=False,
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**😊 Emotions**")
                        emotion_out = gr.Markdown(value="*Run analysis*")
                    with gr.Column():
                        gr.Markdown("**📂 Topic**")
                        topic_out = gr.Markdown(value="*Run analysis*")

        analyze_btn.click(
            fn=analyze,
            inputs=text_input,
            outputs=[summary_out, emotion_out, topic_out],
        )

    # --------------- Metrics Tab ---------------
    with gr.Tab("📊 Metrics"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(load_metrics())
            with gr.Column(scale=1):
                confusion_path = get_viz_path("topic_confusion_matrix.png")
                if confusion_path:
                    gr.Image(confusion_path, label="Confusion Matrix", show_label=True)

    # --------------- Visualizations Tab ---------------
    with gr.Tab("🎨 Visualizations"):
        gr.Markdown("### Model Internals")

        with gr.Row():
            attn_path = get_viz_path("attention_visualization.png")
            if attn_path:
                gr.Image(attn_path, label="Self-Attention Pattern")

            pos_path = get_viz_path("positional_encoding_heatmap.png")
            if pos_path:
                gr.Image(pos_path, label="Positional Encodings")

        with gr.Row():
            multi_path = get_viz_path("multihead_attention_visualization.png")
            if multi_path:
                gr.Image(multi_path, label="Multi-Head Attention")

            single_path = get_viz_path("single_vs_multihead.png")
            if single_path:
                gr.Image(single_path, label="Single vs Multi-Head Comparison")

    # --------------- Architecture Tab ---------------
    with gr.Tab("🔧 Architecture"):
        gr.Markdown(
            """
            ### Model Architecture
            
            | Component | Configuration |
            |-----------|---------------|
            | **Base** | Custom Transformer (encoder-decoder) |
            | **Initialization** | FLAN-T5-base weights |
            | **Encoder** | 6 layers, 768 hidden dim, 12 heads |
            | **Decoder** | 6 layers with cross-attention |
            | **Activation** | Gated-GELU |
            | **Position** | Relative position bias |
            
            ### Training Configuration
            
            | Setting | Value |
            |---------|-------|
            | **Optimizer** | AdamW (lr=2e-5, wd=0.01) |
            | **Scheduler** | Cosine with 1000 warmup steps |
            | **Batch Size** | 14 × 3 accumulation = 42 effective |
            | **Precision** | TF32 (Ampere GPU) |
            | **Compilation** | torch.compile (inductor) |
            
            ### Datasets
            
            | Task | Dataset | Size |
            |------|---------|------|
            | **Summarization** | CNN/DailyMail + BookSum | ~110K |
            | **Emotion** | GoEmotions | ~43K (28 labels) |
            | **Topic** | Yahoo Answers | ~200K (10 classes) |
            """
        )

    # --------------- About Tab ---------------
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ### About LexiMind
            
            LexiMind is a **portfolio project** demonstrating end-to-end machine learning engineering:
            
            ✅ Custom Transformer implementation from scratch  
            ✅ Multi-task learning with shared encoder  
            ✅ Production-ready inference pipeline  
            ✅ Comprehensive evaluation and visualization  
            ✅ CI/CD with GitHub Actions  
            
            ### Known Limitations
            
            - **Summarization** quality is limited (needs more training epochs)
            - **Emotion detection** has low F1 due to class imbalance in GoEmotions
            - Best results on **news-style text** (training domain)
            
            ### Links
            
            - 🔗 [GitHub Repository](https://github.com/OliverPerrin/LexiMind)
            - 🤗 [Model on HuggingFace](https://huggingface.co/OliverPerrin/LexiMind-Model)
            
            ---
            
            **Built by Oliver Perrin** | December 2025
            """
        )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    get_pipeline()  # Pre-load to fail fast if checkpoint missing
    demo.launch(server_name="0.0.0.0", server_port=7860)
