"""
LexiMind -- Discover Books & Papers

Browse literary works and research papers analyzed by a multi-task transformer.
Find your next read by topic, emotion, or keyword -- with AI-generated summaries.

Author: Oliver Perrin
Date: 2026-01-14
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message=".*parameter in the Blocks constructor will be removed.*")

import gradio as gr
from datasets import Dataset, load_dataset

# --------------- Load Dataset from HuggingFace Hub ---------------

print("Loading discovery dataset from HuggingFace Hub...")
_hf_token = os.environ.get("HF_TOKEN")
_dataset: Dataset = load_dataset("OliverPerrin/LexiMind-Discovery", split="train", token=_hf_token)  # type: ignore[assignment]
print(f"Loaded {len(_dataset)} items")

# Convert to list of dicts, excluding social media posts
ALL_ITEMS: list[dict[str, Any]] = [
    dict(row) for row in _dataset if row.get("source_type") != "social"
]

# Extract unique topics and emotions from the dataset (what model predicted)
TOPICS: list[str] = sorted(set(str(item["topic"]) for item in ALL_ITEMS if item.get("topic")))
EMOTIONS: list[str] = sorted(
    {
        str(item["emotion"])
        for item in ALL_ITEMS
        if item.get("emotion") and item["emotion"] != "neutral"
    }
)

# Group by source type
BOOKS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "literary"]
PAPERS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "academic"]

print(f"Topics ({len(TOPICS)}): {TOPICS}")
print(f"Emotions ({len(EMOTIONS)}): {EMOTIONS}")
print(f"Books: {len(BOOKS)}, Papers: {len(PAPERS)}")

# --------------- Load Evaluation Metrics ---------------

METRICS: dict[str, Any] = {}
_metrics_path = Path(__file__).parent.parent / "outputs" / "evaluation_report.json"
if _metrics_path.exists():
    try:
        with open(_metrics_path) as f:
            METRICS = json.load(f)
        print(f"Loaded evaluation metrics from {_metrics_path}")
    except Exception as e:
        print(f"Warning: Could not load metrics: {e}")


# --------------- Card Formatting ---------------

ITEMS_PER_PAGE = 25


def _format_book_card(item: dict) -> str:
    """Format a literary work as a discovery card."""
    title = item.get("title", "Untitled")
    topic = item.get("topic", "")
    emotion = item.get("emotion", "neutral")
    emotion_conf = item.get("emotion_confidence", 0)

    gen_summary = (item.get("generated_summary") or "").strip()
    ref_summary = (item.get("reference_summary") or "").strip()

    # Build metadata line
    parts = ["Book"]
    if topic:
        parts.append(f"Topic: {topic}")
    if emotion != "neutral" and emotion_conf > 0.3:
        parts.append(f"Emotion: {emotion.title()}")
    meta_line = " | ".join(parts)

    card = f"### {title}\n\n"
    card += f"*{meta_line}*\n\n"

    if gen_summary:
        card += f"> {gen_summary}\n\n"
    elif ref_summary:
        card += f"> {ref_summary}\n\n"

    if gen_summary and ref_summary:
        card += (
            f"<details>\n<summary>Original Goodreads Description</summary>"
            f"\n\n{ref_summary}\n\n</details>\n\n"
        )

    card += "---\n\n"
    return card


def _format_paper_card(item: dict) -> str:
    """Format a research paper as a discovery card."""
    title = item.get("title", "Untitled")
    topic = item.get("topic", "")
    emotion = item.get("emotion", "neutral")
    emotion_conf = item.get("emotion_confidence", 0)

    gen_summary = (item.get("generated_summary") or "").strip()
    ref_summary = (item.get("reference_summary") or "").strip()

    # Clean up arXiv titles (often start lowercase or have trailing ...)
    display_title = title.strip().rstrip(".")
    if display_title and display_title[0].islower():
        display_title = display_title[0].upper() + display_title[1:]

    # Build metadata line
    parts = ["Paper"]
    if topic:
        parts.append(f"Topic: {topic}")
    if emotion != "neutral" and emotion_conf > 0.3:
        parts.append(f"Emotion: {emotion.title()}")
    meta_line = " | ".join(parts)

    card = f"### {display_title}\n\n"
    card += f"*{meta_line}*\n\n"

    if gen_summary:
        card += f"> {gen_summary}\n\n"
    elif ref_summary:
        card += f"> {ref_summary}\n\n"

    if gen_summary and ref_summary:
        card += (
            f"<details>\n<summary>Original Abstract</summary>\n\n{ref_summary}\n\n</details>\n\n"
        )

    card += "---\n\n"
    return card


def _format_card(item: dict) -> str:
    """Route to the appropriate card formatter."""
    source_type = item.get("source_type", "")
    if source_type == "literary":
        return _format_book_card(item)
    elif source_type == "academic":
        return _format_paper_card(item)
    return ""


# --------------- Browse Functions ---------------


def browse_by_topic(topic: str, source_filter: str) -> str:
    """Browse items filtered by topic and source type."""
    if topic == "All Topics":
        items = list(ALL_ITEMS)
    else:
        items = [i for i in ALL_ITEMS if i.get("topic") == topic]

    if source_filter == "Books Only":
        items = [i for i in items if i.get("source_type") == "literary"]
    elif source_filter == "Papers Only":
        items = [i for i in items if i.get("source_type") == "academic"]

    if not items:
        return "No items found for this selection."

    books = [i for i in items if i.get("source_type") == "literary"]
    papers = [i for i in items if i.get("source_type") == "academic"]

    result = f"Showing **{len(items)}** results"
    if topic != "All Topics":
        result += f" in **{topic}**"
    result += f" -- {len(books)} books, {len(papers)} papers\n\n---\n\n"

    if source_filter != "Papers Only" and books:
        if source_filter == "All":
            result += f"## Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if source_filter != "Books Only" and papers:
        if source_filter == "All":
            result += f"## Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    return result


def browse_by_emotion(emotion: str, source_filter: str) -> str:
    """Browse items filtered by emotion and source type."""
    if emotion == "All Emotions":
        items = [i for i in ALL_ITEMS if i.get("emotion") != "neutral"]
    else:
        items = [i for i in ALL_ITEMS if i.get("emotion") == emotion.lower()]

    if source_filter == "Books Only":
        items = [i for i in items if i.get("source_type") == "literary"]
    elif source_filter == "Papers Only":
        items = [i for i in items if i.get("source_type") == "academic"]

    if not items:
        return (
            "No items found with a detected emotion for this selection.\n\n"
            "Most literary and academic texts are classified as neutral. "
            "Try browsing by topic instead, or select a different emotion."
        )

    books = [i for i in items if i.get("source_type") == "literary"]
    papers = [i for i in items if i.get("source_type") == "academic"]

    header = emotion if emotion != "All Emotions" else "any detected emotion"
    result = f"Showing **{len(items)}** results with **{header}**\n\n---\n\n"

    if source_filter != "Papers Only" and books:
        if source_filter == "All":
            result += f"## Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if source_filter != "Books Only" and papers:
        if source_filter == "All":
            result += f"## Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    return result


def search_items(query: str) -> str:
    """Search items by text content using word-boundary matching."""
    if not query or len(query) < 2:
        return "Enter at least 2 characters to search."

    pattern = re.compile(r"\b" + re.escape(query) + r"\b", re.IGNORECASE)
    matches = [
        item
        for item in ALL_ITEMS
        if pattern.search(item.get("text", ""))
        or pattern.search(item.get("reference_summary", ""))
        or pattern.search(item.get("generated_summary", ""))
        or pattern.search(item.get("title", ""))
    ]

    if not matches:
        return f'No results found for "{query}".'

    books = [i for i in matches if i.get("source_type") == "literary"]
    papers = [i for i in matches if i.get("source_type") == "academic"]

    result = f'Found **{len(matches)}** results for **"{query}"**\n\n---\n\n'

    if books:
        result += f"## Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if papers:
        result += f"## Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    return result


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind -- Discover Books & Papers",
    theme=gr.themes.Soft(),
    css="""
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        'Helvetica Neue', Arial, sans-serif !important; }
    .result-box { max-height: 800px; overflow-y: auto; }
    h3 { margin-top: 0.5em !important; margin-bottom: 0.2em !important; }
    blockquote {
        border-left: 3px solid #6366f1 !important;
        padding-left: 1em !important;
        color: #374151 !important;
    }
    """,
) as demo:
    gr.Markdown(
        "# LexiMind\n"
        "### Discover Your Next Read\n\n"
        "Browse **{book_count} books** and **{paper_count} research papers** "
        "analyzed by a multi-task AI model. Each item has an AI-generated "
        "summary, a topic classification, and an emotion label.\n\n"
        "Use the tabs below to filter by topic or emotion, or search by keyword.".format(
            book_count=len(BOOKS), paper_count=len(PAPERS)
        )
    )

    with gr.Tabs():
        # -- Browse by Topic --
        with gr.Tab("By Topic"):
            gr.Markdown("Select a topic to explore related books and papers.")
            with gr.Row():
                topic_dropdown = gr.Dropdown(
                    choices=["All Topics"] + TOPICS,
                    value="All Topics",
                    label="Topic",
                    interactive=True,
                    scale=2,
                )
                source_filter_topic = gr.Radio(
                    choices=["All", "Books Only", "Papers Only"],
                    value="All",
                    label="Show",
                    interactive=True,
                    scale=1,
                )

            topic_results = gr.Markdown(
                value=browse_by_topic("All Topics", "All"),
                elem_classes=["result-box"],
            )

            topic_dropdown.change(
                fn=browse_by_topic,
                inputs=[topic_dropdown, source_filter_topic],
                outputs=[topic_results],
            )
            source_filter_topic.change(
                fn=browse_by_topic,
                inputs=[topic_dropdown, source_filter_topic],
                outputs=[topic_results],
            )

        # -- Browse by Emotion --
        with gr.Tab("By Emotion"):
            gr.Markdown(
                "Find books and papers where the model detected a specific emotion in the text."
            )
            with gr.Row():
                emotion_dropdown = gr.Dropdown(
                    choices=["All Emotions"] + [e.title() for e in EMOTIONS],
                    value="All Emotions",
                    label="Emotion",
                    interactive=True,
                    scale=2,
                )
                source_filter_emotion = gr.Radio(
                    choices=["All", "Books Only", "Papers Only"],
                    value="All",
                    label="Show",
                    interactive=True,
                    scale=1,
                )

            emotion_results = gr.Markdown(
                value=browse_by_emotion("All Emotions", "All"),
                elem_classes=["result-box"],
            )

            emotion_dropdown.change(
                fn=lambda e, f: browse_by_emotion(e, f),
                inputs=[emotion_dropdown, source_filter_emotion],
                outputs=[emotion_results],
            )
            source_filter_emotion.change(
                fn=lambda e, f: browse_by_emotion(e, f),
                inputs=[emotion_dropdown, source_filter_emotion],
                outputs=[emotion_results],
            )

        # -- Search --
        with gr.Tab("Search"):
            gr.Markdown("Search across all books and papers by keyword.")

            search_input = gr.Textbox(
                placeholder="e.g. quantum, Shakespeare, neural network, gravity...",
                label="Search",
                interactive=True,
            )

            search_results = gr.Markdown(
                value="Enter at least 2 characters to search.",
                elem_classes=["result-box"],
            )

            search_input.change(
                fn=search_items,
                inputs=[search_input],
                outputs=[search_results],
            )

        # -- Metrics --
        with gr.Tab("Metrics"):
            gr.Markdown("### Model Evaluation\n\nComputed on held-out validation data.")

            gr.Markdown("#### Summarization")

            if METRICS.get("summarization"):
                summ = METRICS["summarization"]
                summ_md = (
                    "| Metric | Score |\n"
                    "|--------|-------|\n"
                    "| ROUGE-1 | {rouge1:.4f} |\n"
                    "| ROUGE-2 | {rouge2:.4f} |\n"
                    "| ROUGE-L | {rougeL:.4f} |\n"
                    "| BLEU-4 | {bleu4:.4f} |\n"
                ).format(
                    rouge1=summ.get("rouge_rouge1", summ.get("rouge1", 0)),
                    rouge2=summ.get("rouge_rouge2", summ.get("rouge2", 0)),
                    rougeL=summ.get("rouge_rougeL", summ.get("rougeL", 0)),
                    bleu4=summ.get("bleu4", 0),
                )
                gr.Markdown(summ_md)
            else:
                gr.Markdown("Summarization metrics not available. Run the evaluation script.")

            gr.Markdown("#### Topic Classification")

            if METRICS.get("topic"):
                topic_m = METRICS["topic"]
                topic_md = (
                    "| Metric | Score |\n"
                    "|--------|-------|\n"
                    "| Accuracy | {accuracy:.2%} |\n"
                    "| Macro F1 | {f1:.4f} |\n"
                ).format(
                    accuracy=topic_m.get("accuracy", 0),
                    f1=topic_m.get("f1", topic_m.get("macro_f1", 0)),
                )
                gr.Markdown(topic_md)
            else:
                gr.Markdown("Topic classification metrics not available.")

            gr.Markdown("#### Emotion Detection")

            if METRICS.get("emotion"):
                emotion_m = METRICS["emotion"]
                emotion_md = (
                    "| Metric | Score |\n"
                    "|--------|-------|\n"
                    "| Sample-avg F1 | {sample_f1:.4f} |\n"
                    "| Macro F1 | {macro_f1:.4f} |\n"
                    "| Micro F1 | {micro_f1:.4f} |\n\n"
                    "28-label multi-label classification trained on GoEmotions."
                ).format(
                    sample_f1=emotion_m.get(
                        "sample_avg_f1", emotion_m.get("f1", emotion_m.get("multilabel_f1", 0))
                    ),
                    macro_f1=emotion_m.get("macro_f1", 0),
                    micro_f1=emotion_m.get("micro_f1", 0),
                )
                gr.Markdown(emotion_md)
            else:
                gr.Markdown("Emotion detection metrics not available.")

            gr.Markdown("#### Discovery Dataset")

            gr.Markdown(
                "| Content | Count |\n"
                "|---------|-------|\n"
                f"| Literary Works | {len(BOOKS)} |\n"
                f"| Research Papers | {len(PAPERS)} |\n"
                f"| **Total** | **{len(ALL_ITEMS)}** |\n"
                f"| Unique Topics | {len(TOPICS)} |\n"
                f"| Unique Emotions | {len(EMOTIONS)} |"
            )

        # -- About --
        with gr.Tab("About"):
            gr.Markdown(
                "### About LexiMind\n\n"
                "LexiMind is a **272M parameter encoder-decoder transformer** "
                "(FLAN-T5-base) trained jointly on three tasks:\n\n"
                "| Task | What it does | Training data |\n"
                "|------|-------------|---------------|\n"
                "| **Summarization** | Generates back-cover blurbs for books and "
                "abstracts for papers | ~49K pairs (arXiv + Project Gutenberg/Goodreads) |\n"
                "| **Topic Classification** | Assigns one of 7 topics | 3.4K samples |\n"
                "| **Emotion Detection** | Detects up to 28 emotions | "
                "43K GoEmotions samples |\n\n"
                "The summaries shown here are **generated by the model** from the "
                "original full text -- not copied from any source. "
                'The "Original Description" / "Original Abstract" in the expandable '
                "sections are the human-written references for comparison.\n\n"
                "#### Architecture\n\n"
                "- Custom from-scratch Transformer (not HuggingFace wrappers)\n"
                "- Shared encoder with task-specific heads: decoder for summarization, "
                "attention pooling for emotion, mean pooling for topic\n"
                "- Trained in ~9 hours on a single RTX 4070 12GB\n\n"
                "[GitHub](https://github.com/OliverPerrin/LexiMind) | "
                "[Model](https://huggingface.co/OliverPerrin/LexiMind-Model) | "
                "[Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery) | "
                "[Paper](https://github.com/OliverPerrin/LexiMind/blob/main/docs/research_paper.tex)"
                "\n\n*Oliver Perrin -- Appalachian State University -- 2025-2026*"
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
