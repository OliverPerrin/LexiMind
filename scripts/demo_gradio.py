"""
Minimal Gradio demo for the LexiMind multitask model.
Shows raw model outputs without any post-processing tricks.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Iterable, Sequence

import gradio as gr
from gradio.themes import Soft
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure

# Make local packages importable when running the script directly
SCRIPT_DIR = Path(__file__).resolve().parent


def guess_project_root(script_dir: Path) -> Path:
    """Attempt to locate the LexiMind repo root even when deployed under /app."""
    markers = ("pyproject.toml", "setup.py", "README.md")
    candidates = [script_dir]
    candidates.extend(script_dir.parents)
    candidates.extend(
        [
            script_dir / "LexiMind",
            script_dir.parent / "LexiMind",
            Path("/LexiMind"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if any((candidate / marker).exists() for marker in markers):
            return candidate

    return script_dir.parent


PROJECT_ROOT = guess_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Resolve ROUGE report path with fallback
_env_path = os.environ.get("ROUGE_REPORT_PATH")
if _env_path and Path(_env_path).exists():
    ROUGE_REPORT_PATH = Path(_env_path)
else:
    ROUGE_REPORT_PATH = OUTPUTS_DIR / "rouge_validation.json"

from src.inference.factory import create_inference_pipeline
from src.inference.pipeline import EmotionPrediction, InferencePipeline, TopicPrediction
from src.utils.logging import configure_logging, get_logger
from huggingface_hub import hf_hub_download

configure_logging()
logger = get_logger(__name__)

_pipeline: InferencePipeline | None = None

VISUALIZATION_DIR = OUTPUTS_DIR
VISUALIZATION_ASSETS: list[tuple[str, str]] = [
    ("attention_visualization.png", "Attention weights (single head)"),
    ("multihead_attention_visualization.png", "Multi-head attention comparison"),
    ("single_vs_multihead.png", "Single vs multi-head attention"),
    ("positional_encoding_heatmap.png", "Positional encoding heatmap"),
]


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Loading inference pipeline ...")
        
        # Download checkpoint if not found locally
        checkpoint_path = Path("checkpoints/best.pt")
        if not checkpoint_path.exists():
            logger.info("Checkpoint not found locally. Downloading from Hugging Face Hub...")
            try:
                # Ensure checkpoints directory exists
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download from the model repository
                # NOTE: Replace 'OliverPerrin/LexiMind-Model' with your actual model repo ID
                downloaded_path = hf_hub_download(
                    repo_id="OliverPerrin/LexiMind-Model",
                    filename="best.pt",
                    local_dir="checkpoints",
                    local_dir_use_symlinks=False
                )
                logger.info(f"Checkpoint downloaded to {downloaded_path}")
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                # Fallback or re-raise will happen in create_inference_pipeline
        
        _pipeline, _ = create_inference_pipeline(
            tokenizer_dir="artifacts/hf_tokenizer/",
            checkpoint_path="checkpoints/best.pt",
            labels_path="artifacts/labels.json",
        )
        logger.info("Pipeline loaded")
    return _pipeline


def map_compression_to_length(compression: int, max_model_length: int = 512) -> int:
    ratio = (100 - compression) / 100
    return max(16, int(ratio * max_model_length))


def count_tokens(text: str) -> str:
    if not text:
        return "Tokens: 0"
    try:
        pipeline = get_pipeline()
        return f"Tokens: {len(pipeline.tokenizer.encode(text))}"
    except Exception as exc:  # pragma: no cover - surfaced in UI
        logger.error("Token counting failed: %s", exc, exc_info=True)
        return "Token count unavailable"


def predict(text: str, compression: int):
    hidden_download = gr.update(value=None, visible=False)
    if not text or not text.strip():
        return (
            "Please enter text to analyze.",
            None,
            "No topic prediction available.",
            None,
            hidden_download,
        )

    try:
        pipeline = get_pipeline()
        max_len = map_compression_to_length(compression)
        logger.info("Generating summary with max length %s", max_len)

        summary = pipeline.summarize([text], max_length=max_len)[0].strip()
        # Use a higher threshold to filter out weak/wrong predictions on out-of-domain text
        emotions = pipeline.predict_emotions([text], threshold=0.6)[0]
        topic = pipeline.predict_topics([text])[0]

        fallback_summary = None
        summary_notice = ""
        summary_source = summary
        if not summary:
            fallback_summary = generate_fallback_summary(text)
            summary_source = fallback_summary
            summary_notice = (
                "<p style=\"color: #b45309; margin-top: 8px;\">"
                "Model returned an empty summary, so a simple extractive fallback is shown instead." "</p>"
            )

        summary_html = format_summary(text, summary_source, notice=summary_notice)
        emotion_plot = create_emotion_plot(emotions)
        topic_markdown = format_topic(topic)
        heatmap_source = summary if summary else fallback_summary
        if heatmap_source:
            attention_fig = create_attention_heatmap(text, heatmap_source, pipeline)
        else:
            attention_fig = render_message_figure("Attention heatmap unavailable: summary was empty.")

        download_path = prepare_download(
            text,
            summary_source,
            emotions,
            topic,
            neural_summary=summary or None,
            fallback_summary=fallback_summary,
        )
        download_update = gr.update(value=download_path, visible=True)

        return summary_html, emotion_plot, topic_markdown, attention_fig, download_update

    except Exception as exc:  # pragma: no cover - surfaced in UI
        logger.error("Prediction error: %s", exc, exc_info=True)
        return "Prediction failed. Check logs for details.", None, "Error", None, hidden_download


def format_summary(original: str, summary: str, *, notice: str = "") -> str:
    if not summary:
        summary = "(Model returned an empty summary. Consider retraining the summarization head.)"

    return f"""
    <div style=\"padding: 12px; border-radius: 6px; background-color: #fafafa; color: #222;\">
        <h3 style=\"margin-top: 0; color: #222;\">Original Text</h3>
        <p style=\"background-color: #f0f0f0; padding: 10px; border-radius: 4px; white-space: pre-wrap; color: #222;\">
            {original}
        </p>
        <h3 style=\"color: #222;\">Summary</h3>
        <p style=\"background-color: #e6f3ff; padding: 10px; border-radius: 4px; white-space: pre-wrap; color: #111;\">
            {summary}
        </p>
        {notice}
        <p style=\"margin-top: 12px; color: #6b7280; font-size: 0.9rem;\">
            Outputs are shown exactly as produced by the checkpoint.
        </p>
    </div>
    """.strip()


def create_emotion_plot(emotions: EmotionPrediction) -> Figure | None:
    if not emotions.labels:
        return render_message_figure("No emotions cleared the model threshold.")

    df = pd.DataFrame({"Emotion": emotions.labels, "Probability": emotions.scores}).sort_values(
        "Probability", ascending=True
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("crest", len(df))
    bars = ax.barh(df["Emotion"], df["Probability"], color=colors)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Scores")
    ax.set_xlim(0, 1)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2%}", va="center")
    plt.tight_layout()
    return fig


def format_topic(topic: TopicPrediction | dict[str, float | str]) -> str:
    if isinstance(topic, TopicPrediction):
        label = topic.label
        confidence = topic.confidence
    else:
        label = str(topic.get("label", "Unknown"))
        confidence = float(topic.get("score", 0.0))
    return f"""
    ### Predicted Topic

    **{label}**

    Confidence: {confidence:.2%}
    """.strip()


def _clean_tokens(tokens: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for token in tokens:
        item = token.replace("Ġ", " ").replace("▁", " ")
        cleaned.append(item.strip() if item.strip() else token)
    return cleaned


def create_attention_heatmap(text: str, summary: str, pipeline: InferencePipeline) -> Figure | None:
    try:
        batch = pipeline.preprocessor.batch_encode([text])
        batch = pipeline._batch_to_device(batch)
        src_ids = batch.input_ids
        src_mask = batch.attention_mask
        encoder_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None

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
        cross_attn = attn_list[-1]["cross"]
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
        keep_indices = [idx for idx, token_id in enumerate(target_id_list) if token_id not in special_ids]
        if not keep_indices:
            return None

        pruned_matrix = attn_matrix[keep_indices, :]
        tokenizer_impl = pipeline.tokenizer.tokenizer
        convert_tokens = getattr(tokenizer_impl, "convert_ids_to_tokens", None)
        if convert_tokens is None:
            return None

        summary_tokens_raw = convert_tokens([target_id_list[idx] for idx in keep_indices])
        source_tokens_raw = convert_tokens(source_ids)

        summary_tokens = _clean_tokens(summary_tokens_raw)
        source_tokens = _clean_tokens(source_tokens_raw)

        # Cap the visualization to prevent massive heatmaps
        max_tokens = 40
        if len(summary_tokens) > max_tokens:
            summary_tokens = summary_tokens[:max_tokens]
            pruned_matrix = pruned_matrix[:max_tokens, :]
        if len(source_tokens) > max_tokens:
            source_tokens = source_tokens[:max_tokens]
            pruned_matrix = pruned_matrix[:, :max_tokens]

        height = max(4.0, 0.3 * len(summary_tokens))
        width = max(6.0, 0.3 * len(source_tokens))
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
        return render_message_figure("Unable to render attention heatmap for this example.")


def render_message_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def prepare_download(
    text: str,
    summary: str,
    emotions: EmotionPrediction | dict[str, Sequence[float] | Sequence[str]],
    topic: TopicPrediction | dict[str, float | str],
    *,
    neural_summary: str | None = None,
    fallback_summary: str | None = None,
) -> str:
    if isinstance(emotions, EmotionPrediction):
        emotion_payload = {
            "labels": list(emotions.labels),
            "scores": list(emotions.scores),
        }
    else:
        emotion_payload = {
            "labels": list(emotions.get("labels", [])),
            "scores": list(emotions.get("scores", [])),
        }

    if isinstance(topic, TopicPrediction):
        topic_payload = {"label": topic.label, "confidence": topic.confidence}
    else:
        topic_payload = {
            "label": str(topic.get("label", topic.get("topic", "Unknown"))),
            "confidence": float(topic.get("confidence", topic.get("score", 0.0))),
        }

    payload = {
        "original_text": text,
        "summary": summary,
        "neural_summary": neural_summary,
        "fallback_summary": fallback_summary,
        "emotions": emotion_payload,
        "topic": topic_payload,
    }

    with NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        return handle.name


def load_visualization_gallery() -> tuple[list[tuple[str, str]], str]:
    """Collect visualization images produced by model tests."""
    items: list[tuple[str, str]] = []
    missing: list[str] = []
    for filename, label in VISUALIZATION_ASSETS:
        path = VISUALIZATION_DIR / filename
        if path.exists():
            items.append((str(path), label))
        else:
            missing.append(filename)

    if items:
        status = f"Loaded {len(items)} visualization(s) from {VISUALIZATION_DIR}."
    else:
        status = (
            "No visualization PNGs found in the outputs/ directory. "
            "Ensure tests/test_models/* have produced the PNGs and that they are available on this host."
        )

    if missing:
        status += f" Missing files: {', '.join(missing)}."

    return items, status


def generate_fallback_summary(text: str, max_chars: int = 320) -> str:
    content = text.strip()
    if not content:
        return "(Input text was empty.)"

    sentences = re.split(r"(?<=[.!?])\s+", content)
    fragments: list[str] = []
    total = 0
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if sentence.endswith(('.', '!', '?')) else f"{sentence}."
        if total + len(candidate) > max_chars and fragments:
            break
        fragments.append(candidate)
        total += len(candidate)

    if not fragments:
        return content[:max_chars]
    return " ".join(fragments)


def load_rouge_metrics():
    columns = ["metric", "precision", "recall", "f1"]
    empty = pd.DataFrame(columns=columns)

    if not ROUGE_REPORT_PATH.exists():
        return empty, {
            "error": f"ROUGE report not found at {ROUGE_REPORT_PATH}",
            "hint": "Run scripts/eval_rouge.py then deploy/copy outputs/rouge_validation.json with the app.",
        }

    try:
        with ROUGE_REPORT_PATH.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception as exc:  # pragma: no cover - surfaced in UI
        logger.error("Failed to read ROUGE report: %s", exc, exc_info=True)
        return empty, {"error": f"Unable to parse report: {exc}", "report_path": str(ROUGE_REPORT_PATH)}

    rows: list[dict[str, object]] = []
    metrics_data = report.get("metrics", {})
    if not metrics_data:
        logger.warning("ROUGE report found but 'metrics' key is missing or empty.")
    
    for metric_name, components in metrics_data.items():
        rows.append(
            {
                "metric": metric_name,
                "precision": round(float(components.get("precision", 0.0)), 4),
                "recall": round(float(components.get("recall", 0.0)), 4),
                "f1": round(float(components.get("fmeasure", 0.0)), 4),
            }
        )

    table = pd.DataFrame(rows, columns=columns) if rows else empty
    
    # Clean up path for display
    display_path = str(ROUGE_REPORT_PATH)
    if "/app/" in display_path:
        display_path = display_path.replace("/app/", "/LexiMind/")
        
    metadata = {
        "num_examples": report.get("num_examples"),
        "config": report.get("config"),
        "report_path": display_path,
        "last_updated": datetime.fromtimestamp(ROUGE_REPORT_PATH.stat().st_mtime).isoformat(),
    }
    return table, metadata


SAMPLE_TEXT = (
    "Artificial intelligence is rapidly transforming the technology landscape. "
    "Machine learning algorithms are now capable of processing vast amounts of data, "
    "identifying patterns, and making predictions with unprecedented accuracy. "
    "From healthcare diagnostics to financial forecasting, AI applications are "
    "revolutionizing industries worldwide. However, ethical considerations around "
    "privacy, bias, and transparency remain critical challenges that must be addressed "
    "as these technologies continue to evolve."
)


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="LexiMind Demo", theme=Soft()) as demo:
        gr.Markdown(
            """
            # LexiMind NLP Demo

            This demo streams the raw outputs from the saved LexiMind checkpoint.
            Results may be noisy; retraining is recommended for production use.
            """
        )

        initial_visuals, initial_visual_status = load_visualization_gallery()
        initial_metrics, initial_metrics_meta = load_rouge_metrics()

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Input Text",
                    lines=10,
                    value=SAMPLE_TEXT,
                    placeholder="Paste or type your text here...",
                )
                token_box = gr.Textbox(label="Token Count", value="Tokens: 0", interactive=False)
                compression = gr.Slider(
                    minimum=20,
                    maximum=80,
                    value=50,
                    step=5,
                    label="Compression %",
                    info="Higher values request shorter summaries.",
                )
                analyze_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Summary"):
                        summary_output = gr.HTML(label="Summary")
                    with gr.TabItem("Emotions"):
                        emotion_output = gr.Plot(label="Emotion Probabilities")
                    with gr.TabItem("Topic"):
                        topic_output = gr.Markdown(label="Topic Prediction")
                    with gr.TabItem("Attention"):
                        attention_output = gr.Plot(label="Attention Heatmap")
                        gr.Markdown("*Shows decoder attention if a summary is available.*")
                    with gr.TabItem("Model Visuals"):
                        visuals = gr.Gallery(
                            label="Test Visualizations",
                            value=initial_visuals,
                            columns=2,
                            height=400,
                            interactive=False,
                            type="filepath"
                        )
                        gr.Markdown(
                            "These PNGs come from the visualization-focused tests in `tests/test_models` and are consumed as-is."
                        )
                        visuals_notice = gr.Markdown(initial_visual_status)
                        refresh_visuals = gr.Button("Refresh Visuals")
                    with gr.TabItem("Metrics"):
                        rouge_table = gr.Dataframe(
                            value=initial_metrics,
                            headers=["metric", "precision", "recall", "fmeasure"],
                            datatype=["str", "number", "number", "number"],
                            interactive=False,
                            label="ROUGE Scores",
                        )
                        rouge_meta = gr.JSON(
                            value=initial_metrics_meta,
                            label="ROUGE Run Metadata",
                        )
                        refresh_metrics = gr.Button("Refresh Metrics")
                gr.Markdown("### Download Results")
                download_btn = gr.DownloadButton("Download JSON", visible=False)

        input_text.change(fn=count_tokens, inputs=[input_text], outputs=[token_box])
        analyze_btn.click(
            fn=predict,
            inputs=[input_text, compression],
            outputs=[summary_output, emotion_output, topic_output, attention_output, download_btn],
        )
        refresh_visuals.click(
            fn=load_visualization_gallery,
            inputs=None,
            outputs=[visuals, visuals_notice],
        )
        refresh_metrics.click(fn=load_rouge_metrics, inputs=None, outputs=[rouge_table, rouge_meta])
        return demo


demo = create_interface()
app = demo


if __name__ == "__main__":
    try:
        get_pipeline()
        demo.queue().launch(share=False, allowed_paths=[str(OUTPUTS_DIR)])
    except Exception as exc:  # pragma: no cover - surfaced in console
        logger.error("Failed to launch demo: %s", exc, exc_info=True)
        raise

