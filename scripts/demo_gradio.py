"""
LexiMind Demo - Multi-Task NLP Model

A streamlined demo for showcasing the model to recruiters and engineers.
Focuses on: live inference, model metrics, and training visualizations.

Author: Oliver Perrin
Date: 2025-12-05, Updated: 2026-01-13
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

# --------------- Path Setup ---------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download

from src.inference.factory import create_inference_pipeline

# --------------- Constants ---------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"

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


# --------------- Core Analysis Function ---------------


def analyze_text(text: str) -> tuple[str, str, str]:
    """Run all three tasks and return formatted results."""
    if not text or not text.strip():
        return "Enter text above to analyze.", "", ""

    try:
        pipe = get_pipeline()

        # Run tasks
        summary = pipe.summarize([text], max_length=150)[0].strip()
        if not summary:
            summary = "(Unable to generate summary)"

        emotions = pipe.predict_emotions([text], threshold=0.3)[0]
        topic = pipe.predict_topics([text])[0]

        # Format emotions with scores
        if emotions.labels:
            paired = list(zip(emotions.labels, emotions.scores, strict=False))
            paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)[:5]
            emotion_lines = [f"• **{lbl.title()}**: {score:.0%}" for lbl, score in paired_sorted]
            emotion_str = "\n".join(emotion_lines)
        else:
            emotion_str = "No strong emotions detected"

        # Format topic
        topic_str = f"**{topic.label}** ({topic.confidence:.0%})"

        return summary, emotion_str, topic_str

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}", "", ""


# --------------- Sample Texts ---------------

SAMPLES = {
    "business": """Global markets tumbled today as investors reacted to rising inflation concerns. 
The Federal Reserve hinted at potential interest rate hikes, sending shockwaves through technology 
and banking sectors. Analysts predict continued volatility as economic uncertainty persists. 
Major indices fell by over 2%, with tech stocks leading the decline.""",

    "science": """Scientists at MIT have developed a breakthrough quantum computing chip that 
operates at room temperature. This advancement could revolutionize drug discovery, cryptography, 
and artificial intelligence. The research team published their findings in Nature, demonstrating 
stable qubit operations for over 100 microseconds.""",

    "sports": """The championship game ended in dramatic fashion as the underdog team scored in 
the final seconds to secure victory. Fans rushed the field in celebration, marking the team's 
first title in 25 years. The winning goal came from a rookie player who had only joined the 
team this season.""",
}


# --------------- Gradio Interface ---------------

with gr.Blocks(title="LexiMind") as demo:
    
    gr.Markdown(
        """
        # LexiMind
        ### Multi-Task Transformer for Document Understanding
        
        A custom 272M parameter encoder-decoder model trained jointly on three NLP tasks.
        Built from scratch in PyTorch, initialized from FLAN-T5-base weights.
        """
    )

    with gr.Tabs():
        # ===================== TAB 1: LIVE DEMO =====================
        with gr.Tab("Try It"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Input Text",
                        lines=6,
                        placeholder="Paste a news article or any text to analyze...",
                    )
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze", variant="primary")
                        clear_btn = gr.Button("Clear")
                    
                    gr.Markdown("**Quick samples:**")
                    with gr.Row():
                        btn_business = gr.Button("Business", size="sm")
                        btn_science = gr.Button("Science", size="sm")
                        btn_sports = gr.Button("Sports", size="sm")
                
                with gr.Column(scale=2):
                    summary_output = gr.Textbox(label="Generated Summary", lines=4, interactive=False)
                    with gr.Row():
                        emotions_output = gr.Markdown(label="Detected Emotions")
                        topic_output = gr.Markdown(label="Topic Classification")
            
            # Event handlers
            analyze_btn.click(analyze_text, inputs=[text_input], outputs=[summary_output, emotions_output, topic_output])
            clear_btn.click(lambda: ("", "", "", ""), outputs=[text_input, summary_output, emotions_output, topic_output])
            btn_business.click(lambda: SAMPLES["business"], outputs=[text_input])
            btn_science.click(lambda: SAMPLES["science"], outputs=[text_input])
            btn_sports.click(lambda: SAMPLES["sports"], outputs=[text_input])

        # ===================== TAB 2: METRICS =====================
        with gr.Tab("Metrics"):
            gr.Markdown("### Training Results")
            
            # Load metrics from training history
            metrics_md = "| Task | Metric | Score |\n|------|--------|-------|\n"
            if TRAINING_HISTORY_PATH.exists():
                with open(TRAINING_HISTORY_PATH) as f:
                    history = json.load(f)
                # Get latest validation epoch
                val_keys = [k for k in history.keys() if k.startswith("val_epoch")]
                if val_keys:
                    latest = sorted(val_keys)[-1]
                    val = history[latest]
                    metrics_md += f"| Topic Classification | Accuracy | **{val.get('topic_accuracy', 0):.1%}** |\n"
                    metrics_md += f"| Emotion Detection | F1 Score | {val.get('emotion_f1', 0):.1%} |\n"
                    metrics_md += f"| Summarization | ROUGE-like | {val.get('summarization_rouge_like', 0):.1%} |\n"
            
            gr.Markdown(metrics_md)
            
            gr.Markdown("### Training Visualizations")
            with gr.Row():
                loss_curve = OUTPUTS_DIR / "training_loss_curve.png"
                if loss_curve.exists():
                    gr.Image(str(loss_curve), label="Loss Curves", show_label=True)
                
                task_metrics = OUTPUTS_DIR / "task_metrics.png"
                if task_metrics.exists():
                    gr.Image(str(task_metrics), label="Task Metrics", show_label=True)
            
            with gr.Row():
                lr_schedule = OUTPUTS_DIR / "learning_rate_schedule.png"
                if lr_schedule.exists():
                    gr.Image(str(lr_schedule), label="LR Schedule", show_label=True)
                
                dynamics = OUTPUTS_DIR / "training_dynamics.png"
                if dynamics.exists():
                    gr.Image(str(dynamics), label="Training Dynamics", show_label=True)

        # ===================== TAB 3: ARCHITECTURE =====================
        with gr.Tab("Architecture"):
            gr.Markdown(
                """
                ### Model Architecture
                
                | Component | Details |
                |-----------|---------|
                | **Type** | Encoder-Decoder Transformer |
                | **Parameters** | 272.5M |
                | **Encoder** | 12 layers, 768 dim, 12 heads |
                | **Decoder** | 12 layers, causal attention |
                | **Normalization** | Pre-LN with RMSNorm |
                | **Position** | T5 Relative Position Bias |
                | **Initialization** | FLAN-T5-base weights |
                
                ### Task Heads
                
                | Task | Output | Loss Function |
                |------|--------|---------------|
                | **Summarization** | Seq2seq generation | Cross-entropy + label smoothing |
                | **Emotion** | 28-class multi-label | Binary cross-entropy |
                | **Topic** | 4-class single-label | Cross-entropy |
                
                ### Training Data
                
                | Dataset | Task | Size |
                |---------|------|------|
                | CNN/DailyMail | Summarization | ~100K |
                | GoEmotions | Emotion | ~43K |
                | AG News | Topic | ~120K |
                
                ### Links
                
                - **Code**: [github.com/OliverPerrin/LexiMind](https://github.com/OliverPerrin/LexiMind)
                - **Model**: [huggingface.co/OliverPerrin/LexiMind-Model](https://huggingface.co/OliverPerrin/LexiMind-Model)
                
                ---
                *Oliver Perrin • Appalachian State University • 2025-2026*
                """
            )
            
            # Show embedding visualization if exists
            embedding_viz = OUTPUTS_DIR / "embedding_space.png"
            if embedding_viz.exists():
                gr.Image(str(embedding_viz), label="Embedding Space (t-SNE)")


# --------------- Entry Point ---------------

if __name__ == "__main__":
    logger.info("Loading inference pipeline...")
    get_pipeline()
    logger.info("Starting Gradio server...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
