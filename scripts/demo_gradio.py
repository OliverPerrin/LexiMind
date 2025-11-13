"""
Gradio Demo interface for LexiMind NLP pipeline.
Showcases summarization, emotion detection, and topic prediction.
"""
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import gradio as gr
from gradio.themes import Soft
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure

# Add project root to the path, going up two folder levels from this file
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.factory import create_inference_pipeline
from src.inference.pipeline import EmotionPrediction, InferencePipeline, TopicPrediction
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_pipeline: InferencePipeline | None = None  # Global pipeline instance
_label_metadata = None  # Cached label metadata


def get_pipeline() -> InferencePipeline:
    """Lazy Loading and Caching the inference pipeline"""
    global _pipeline, _label_metadata
    if _pipeline is None:
        try:
            logger.info("Loading inference pipeline...")
            pipeline, label_metadata = create_inference_pipeline(
                tokenizer_dir="artifacts/hf_tokenizer/",
                checkpoint_path="checkpoints/best.pt",
                labels_path="artifacts/labels.json",
            )
            _pipeline = pipeline
            _label_metadata = label_metadata
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise RuntimeError("Could not initialize inference pipeline. Check logs for details.")
    return _pipeline

def count_tokens(text: str) -> str:
    """Count tokens in the input text."""
    if not text:
        return "Tokens: 0"
    try: 
        pipeline = get_pipeline()
        token_count = len(pipeline.tokenizer.encode(text))
        return f"Tokens: {token_count}"
    except Exception as e:
        logger.error(f"Token counting error: {e}")
        return "Token count unavailable"
            
def map_compression_to_length(compression: int, max_model_length: int = 512):
    """
    Map Compression slider (20-80%) to max summary length.
    Higher compression = shorter summary output.
    """
    # Invert, 20% compression = 80% of max length
    ratio = (100 - compression) / 100
    return int(ratio * max_model_length)

def predict(text: str, compression: int):
    """Run the full pipeline and prepare Gradio outputs."""
    hidden_download = gr.update(value=None, visible=False)
    if not text or not text.strip():
        return (
            "Please enter some text to analyze.",
            None,
            "No topic prediction available",
            None,
            hidden_download,
        )
    try:
        pipeline = get_pipeline()
        max_len = map_compression_to_length(compression)
        logger.info("Generating summary with max length of %s", max_len)

        summary = pipeline.summarize([text], max_length=max_len)[0]
        emotions = pipeline.predict_emotions([text])[0]
        topic = pipeline.predict_topics([text])[0]

        summary_html = format_summary(text, summary)
        emotion_plot = create_emotion_plot(emotions)
        topic_output = format_topic(topic)
        attention_fig = create_attention_heatmap(text, summary, pipeline)
        download_bytes = prepare_download(text, summary, emotions, topic)
        download_update = gr.update(value=download_bytes, visible=True)

        return summary_html, emotion_plot, topic_output, attention_fig, download_update

    except Exception as exc:  # pragma: no cover - surfaced in UI
        logger.error("Prediction error: %s", exc, exc_info=True)
        error_msg = "Prediction failed. Check logs for details."
        return error_msg, None, "Error", None, hidden_download


def format_summary(original: str, summary: str) -> str:
    """Format original and summary text for display."""
    return f"""
    <div style="padding: 10px; border-radius: 5px;">
        <h3>Original Text</h3>
        <p style="background-color: #f0f0f0; padding: 10px; border-radius: 3px;">
            {original}
        </p>
        <h3>Summary</h3>
        <p style="background-color: #e6f3ff; padding: 10px; border-radius: 3px;">
            {summary}
        </p>
    </div>
    """


def create_emotion_plot(
    emotions: EmotionPrediction | dict[str, Sequence[float] | Sequence[str]]
) -> Figure | None:
    """Create a horizontal bar chart for emotion predictions."""
    if isinstance(emotions, EmotionPrediction):
        labels = list(emotions.labels)
        scores = list(emotions.scores)
    else:
        labels = list(emotions.get("labels", []))
        scores = list(emotions.get("scores", []))

    if not labels or not scores:
        return None

    df = pd.DataFrame({"Emotion": labels, "Probability": scores})
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(labels))
    bars = ax.barh(df["Emotion"], df["Probability"], color=colors)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Emotion", fontsize=12)
    ax.set_title("Emotion Detection Results", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2%}",
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
    plt.tight_layout()
    return fig


def format_topic(topic: TopicPrediction | dict[str, float | str]) -> str:
    """Format topic prediction output as markdown."""
    if isinstance(topic, TopicPrediction):
        label = topic.label
        score = topic.confidence
    else:
        label = str(topic.get("label", "Unknown"))
        score = float(topic.get("score", 0.0))

    return f"""
    ### Predicted Topic

    **{label}**

    Confidence: {score:.2%}
    """

def _clean_tokens(tokens: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for token in tokens:
        item = token.replace("Ġ", " ").replace("▁", " ")
        cleaned.append(item.strip() if item.strip() else token)
    return cleaned


def create_attention_heatmap(text: str, summary: str, pipeline: InferencePipeline) -> Figure | None:
    """Generate a seaborn heatmap of decoder cross-attention averaged over heads."""
    if not summary:
        return None
    try:
        batch = pipeline.preprocessor.batch_encode([text])
        batch = pipeline._batch_to_device(batch)
        src_ids = batch.input_ids
        src_mask = batch.attention_mask
        encoder_mask = None
        if src_mask is not None:
            encoder_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

        with torch.inference_mode():
            memory = pipeline.model.encoder(src_ids, mask=encoder_mask)
            target_enc = pipeline.tokenizer.batch_encode([summary])
            target_ids = target_enc["input_ids"].to(pipeline.device)
            target_mask = target_enc["attention_mask"].to(pipeline.device)
            target_len = int(target_mask.sum().item())
            decoder_inputs = pipeline.tokenizer.prepare_decoder_inputs(target_ids)
            decoder_inputs = decoder_inputs[:, :target_len].to(pipeline.device)
            target_ids = target_ids[:, :target_len]
            memory_mask = src_mask.to(pipeline.device) if src_mask is not None else None
            _, attn_list = pipeline.model.decoder(
                decoder_inputs,
                memory,
                memory_mask=memory_mask,
                collect_attn=True,
            )
        if not attn_list:
            return None
        cross_attn = attn_list[-1]["cross"]  # (B, heads, T, S)
        attn_matrix = cross_attn.mean(dim=1)[0].detach().cpu().numpy()

        source_len = batch.lengths[0]
        attn_matrix = attn_matrix[:target_len, :source_len]

        source_ids = src_ids[0, :source_len].tolist()
        target_id_list = target_ids[0].tolist()

        special_ids = {
            pipeline.tokenizer.pad_token_id,
            pipeline.tokenizer.bos_token_id,
            pipeline.tokenizer.eos_token_id,
        }
        keep_indices = [index for index, token_id in enumerate(target_id_list) if token_id not in special_ids]
        if not keep_indices:
            return None

        pruned_matrix = attn_matrix[keep_indices, :]
        tokenizer_impl = pipeline.tokenizer.tokenizer
        convert_tokens = getattr(tokenizer_impl, "convert_ids_to_tokens", None)
        if convert_tokens is None:
            logger.warning("Tokenizer does not expose convert_ids_to_tokens; skipping attention heatmap.")
            return None

        summary_tokens_raw = convert_tokens([target_id_list[index] for index in keep_indices])
        source_tokens_raw = convert_tokens(source_ids)

        summary_tokens = _clean_tokens(summary_tokens_raw)
        source_tokens = _clean_tokens(source_tokens_raw)

        height = max(4.0, 0.4 * len(summary_tokens))
        width = max(6.0, 0.4 * len(source_tokens))
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(
            pruned_matrix,
            cmap="mako",
            xticklabels=source_tokens,
            yticklabels=summary_tokens,
            ax=ax,
            cbar_kws={"label": "Attention"},
        )
        ax.set_xlabel("Input Tokens")
        ax.set_ylabel("Summary Tokens")
        ax.set_title("Cross-Attention (decoder last layer)")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig

    except Exception as exc:
        logger.error("Unable to build attention heatmap: %s", exc, exc_info=True)
        return None


def prepare_download(
    text: str,
    summary: str,
    emotions: EmotionPrediction | dict[str, Sequence[float] | Sequence[str]],
    topic: TopicPrediction | dict[str, float | str],
) -> bytes:
    """Prepare JSON data buffer for download."""
    if isinstance(emotions, EmotionPrediction):
        emotion_payload = {
            "labels": list(emotions.labels),
            "scores": list(emotions.scores),
        }
    else:
        emotion_payload = emotions

    if isinstance(topic, TopicPrediction):
        topic_payload = {
            "label": topic.label,
            "confidence": topic.confidence,
        }
    else:
        topic_payload = topic

    payload = {
        "original_text": text,
        "summary": summary,
        "emotions": emotion_payload,
        "topic": topic_payload,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# Sample data for the demo
SAMPLE_TEXT = """
Artificial intelligence is rapidly transforming the technology landscape. 
Machine learning algorithms are now capable of processing vast amounts of data, 
identifying patterns, and making predictions with unprecedented accuracy. 
From healthcare diagnostics to financial forecasting, AI applications are 
revolutionizing industries worldwide. However, ethical considerations around 
privacy, bias, and transparency remain critical challenges that must be addressed 
as these technologies continue to evolve.
"""

def create_interface() -> gr.Blocks:
    with gr.Blocks(title="LexiMind Demo", theme=Soft()) as demo:
        gr.Markdown("""
        # LexiMind NLP Pipeline Demo
        
        **Full pipleine for text summarization, emotion detection, and topic prediction.**
        
        Enter text below and adjust compressoin to see the results.
        """)
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_text = gr.Textbox(
                    label="Enter text",
                    placeholder="Paste or type your text here...",
                    lines=10,
                    value=SAMPLE_TEXT
                )
                token_count = gr.Textbox(
                    label="Token Count",
                    value="Tokens: 0",
                    interactive=False
                )
                compression = gr.Slider(
                    minimum=20,
                    maximum=80,
                    value=50,
                    step=5,
                    label="Compression %",
                    info="Higher = shorter summary"
                )
                predict_btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
            # Right column - Outputs
            with gr.Column(scale=2):
                gr.Markdown("### Result")
                with gr.Tabs():
                    with gr.TabItem("Summary"):
                        summary_output = gr.HTML(label="Summary")
                    with gr.TabItem("Emotions"):
                        emotion_output = gr.Plot(label="Emotion Analysis")
                    with gr.TabItem("Topic"):
                        topic_output = gr.Markdown(label="Topic Prediction")
                    with gr.TabItem("Attention Heatmap"):
                        attention_output = gr.Plot(label="Attention Weights")
                        gr.Markdown("*Visualizes which parts of the input the model focused on.*")
                # Download section
                gr.Markdown("### Export Results")
                download_btn = gr.DownloadButton(
                    "Download Results (JSON)",
                    visible=False,
                )
            # Event Handlers
            input_text.change(
                fn=count_tokens,
                inputs=[input_text],
                outputs=[token_count]
            )
            predict_btn.click(
                fn=predict,
                inputs=[input_text, compression],
                outputs=[summary_output, emotion_output, topic_output, attention_output, download_btn],
            )
            # Examples
            gr.Examples(
                examples=[
                    [SAMPLE_TEXT, 50],
                    [
                        "Climate change poses significant risks to global ecosystems. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent. Scientists urge immediate action to reduce carbon emissions and transition to renewable energy sources.",
                        40,
                    ],
                ],
                inputs=[input_text, compression],
                label="Try these examples:",
            )
        return demo


demo = create_interface()
app = demo


if __name__ == "__main__":
    try:
        get_pipeline()
        demo.queue().launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    except Exception as e:
        logger.error("Failed to launch demo: %s", e, exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

        
