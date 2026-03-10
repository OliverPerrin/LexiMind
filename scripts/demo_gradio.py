"""
LexiMind — Discover Books & Papers

Browse literary works and research papers analyzed by a multi-task transformer.
Find your next read by topic, emotion, or keyword — with AI-generated summaries.

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

# Convert to list of dicts for easier filtering
ALL_ITEMS: list[dict[str, Any]] = [dict(row) for row in _dataset]

# Extract unique topics and emotions FROM THE DATASET (what model predicted)
DATASET_TOPICS: list[str] = sorted(
    set(str(item["topic"]) for item in ALL_ITEMS if item.get("topic"))
)
DATASET_EMOTIONS: list[str] = sorted(
    {
        str(item["emotion"])
        for item in ALL_ITEMS
        if item.get("emotion") and item["emotion"] != "neutral"
    }
)

# Load ALL possible labels from labels.json (what the model CAN predict)
_labels_path = Path(__file__).parent.parent / "artifacts" / "labels.json"
if _labels_path.exists():
    with open(_labels_path) as f:
        _labels = json.load(f)
    ALL_TOPICS: list[str] = _labels.get("topic", DATASET_TOPICS)
    ALL_EMOTIONS: list[str] = _labels.get("emotion", DATASET_EMOTIONS)
else:
    ALL_TOPICS = DATASET_TOPICS
    ALL_EMOTIONS = DATASET_EMOTIONS

# Use dataset-observed values for dropdown filtering
TOPICS = DATASET_TOPICS
EMOTIONS = DATASET_EMOTIONS

# Group by source type — only books, papers, social (no blog posts)
BOOKS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "literary"]
PAPERS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "academic"]
SOCIAL: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "social"]

# Discoverable items = books + papers (what users want to browse)
DISCOVERABLE: list[dict[str, Any]] = BOOKS + PAPERS

print(f"Dataset Topics ({len(TOPICS)}): {TOPICS}")
print(f"Dataset Emotions ({len(EMOTIONS)}): {EMOTIONS}")
print(f"Books: {len(BOOKS)}, Papers: {len(PAPERS)}, Social: {len(SOCIAL)}")

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


# --------------- Helpers ---------------

# Friendly source labels
_SOURCE_LABELS = {
    "academic": "Research Paper",
    "literary": "Book",
    "social": "Social Post",
}

# Friendly dataset labels
_DATASET_LABELS = {
    "arxiv": "arXiv",
    "gutenberg": "Project Gutenberg",
    "goemotions": "GoEmotions (Reddit)",
}


def _get_best_summary(item: dict) -> str:
    """Return the best available summary for display.

    Prefers the AI-generated summary; falls back to the reference summary.
    For social posts (short Reddit comments), return empty since the text IS
    the content.
    """
    if item.get("source_type") == "social":
        return ""
    gen = (item.get("generated_summary") or "").strip()
    if gen:
        return gen
    return (item.get("reference_summary") or "").strip()


def _format_book_card(item: dict) -> str:
    """Format a literary work as a discovery card."""
    title = item.get("title", "Untitled")
    topic = item.get("topic", "")
    topic_conf = item.get("topic_confidence", 0)  # noqa: F841
    emotion = item.get("emotion", "neutral")
    emotion_conf = item.get("emotion_confidence", 0)

    gen_summary = (item.get("generated_summary") or "").strip()
    ref_summary = (item.get("reference_summary") or "").strip()

    # Build tags line
    tags = []
    if topic:
        tags.append(f"📂 {topic}")
    if emotion != "neutral" and emotion_conf > 0.3:
        tags.append(f"💭 {emotion.title()}")
    tags_line = " · ".join(tags)

    card = f"### 📖 {title}\n\n"
    if tags_line:
        card += f"{tags_line}\n\n"

    # Show the AI-generated summary prominently as a blurb
    if gen_summary:
        card += f"> {gen_summary}\n\n"
    elif ref_summary:
        card += f"> {ref_summary}\n\n"

    # Show reference (original Goodreads description) as comparison
    if gen_summary and ref_summary:
        card += f"<details>\n<summary>Original Goodreads Description</summary>\n\n{ref_summary}\n\n</details>\n\n"

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

    tags = []
    if topic:
        tags.append(f"📂 {topic}")
    if emotion != "neutral" and emotion_conf > 0.3:
        tags.append(f"💭 {emotion.title()}")
    tags_line = " · ".join(tags)

    card = f"### 📄 {display_title}\n\n"
    if tags_line:
        card += f"{tags_line}\n\n"

    # Show AI-generated abstract prominently
    if gen_summary:
        card += f"> {gen_summary}\n\n"
    elif ref_summary:
        card += f"> {ref_summary}\n\n"

    # Show original abstract as comparison
    if gen_summary and ref_summary:
        card += (
            f"<details>\n<summary>Original Abstract</summary>\n\n{ref_summary}\n\n</details>\n\n"
        )

    card += "---\n\n"
    return card


def _format_social_card(item: dict) -> str:
    """Format a social media post as a compact card."""
    text = (item.get("text") or "").strip()
    emotion = item.get("emotion", "neutral")
    emotion_conf = item.get("emotion_confidence", 0)
    topic = item.get("topic", "")

    # Display emotion as the primary feature for social posts
    emotion_display = emotion.title() if emotion != "neutral" else ""
    tags = []
    if emotion_display:
        tags.append(f"💭 {emotion_display} ({emotion_conf:.0%})")
    if topic:
        tags.append(f"📂 {topic}")
    tags_line = " · ".join(tags)

    card = f"> {text}\n\n"
    if tags_line:
        card += f"*{tags_line}*\n\n"
    card += "---\n\n"
    return card


def _format_card(item: dict) -> str:
    """Route to the appropriate card formatter."""
    source_type = item.get("source_type", "")
    if source_type == "literary":
        return _format_book_card(item)
    elif source_type == "academic":
        return _format_paper_card(item)
    elif source_type == "social":
        return _format_social_card(item)
    return ""


# --------------- Browse Functions ---------------

ITEMS_PER_PAGE = 25


def browse_by_topic(topic: str, source_filter: str) -> str:
    """Browse items filtered by topic and source type."""
    if topic == "All Topics":
        items = DISCOVERABLE
    else:
        items = [i for i in DISCOVERABLE if i.get("topic") == topic]

    if source_filter == "Books Only":
        items = [i for i in items if i.get("source_type") == "literary"]
    elif source_filter == "Papers Only":
        items = [i for i in items if i.get("source_type") == "academic"]

    if not items:
        return "No items found for this selection."

    books = [i for i in items if i.get("source_type") == "literary"]
    papers = [i for i in items if i.get("source_type") == "academic"]

    result = f"Found **{len(items)}** items"
    if topic != "All Topics":
        result += f" in **{topic}**"
    result += f" — {len(books)} books, {len(papers)} papers\n\n---\n\n"

    if source_filter != "Papers Only" and books:
        if source_filter == "All":
            result += f"## 📚 Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if source_filter != "Books Only" and papers:
        if source_filter == "All":
            result += f"## 📄 Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    return result


def browse_by_emotion(emotion: str, source_filter: str) -> str:
    """Browse items filtered by emotion and source type."""
    if emotion == "All Emotions":
        items = [i for i in DISCOVERABLE if i.get("emotion") != "neutral"]
    else:
        items = [i for i in DISCOVERABLE if i.get("emotion") == emotion.lower()]

    if source_filter == "Books Only":
        items = [i for i in items if i.get("source_type") == "literary"]
    elif source_filter == "Papers Only":
        items = [i for i in items if i.get("source_type") == "academic"]

    if not items:
        return "No items found with a detected emotion for this selection.\n\nMost literary and academic texts are classified as **Neutral** — try the Social tab to browse emotion-rich content."

    books = [i for i in items if i.get("source_type") == "literary"]
    papers = [i for i in items if i.get("source_type") == "academic"]

    header = emotion if emotion != "All Emotions" else "any detected emotion"
    result = f"Found **{len(items)}** items with **{header}**\n\n---\n\n"

    if source_filter != "Papers Only" and books:
        if source_filter == "All":
            result += f"## 📚 Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if source_filter != "Books Only" and papers:
        if source_filter == "All":
            result += f"## 📄 Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    return result


def browse_social(emotion: str) -> str:
    """Browse social media posts by emotion — these showcase emotion detection."""
    if emotion == "All Emotions":
        items = SOCIAL
    else:
        items = [i for i in SOCIAL if i.get("emotion") == emotion.lower()]

    if not items:
        return "No social posts found for this emotion."

    result = f"**{len(items)}** social media posts"
    if emotion != "All Emotions":
        result += f" expressing **{emotion}**"
    result += "\n\n*Short Reddit comments analyzed for emotion — these demonstrate the model's 28-label emotion detection.*\n\n---\n\n"

    for item in items[:50]:
        result += _format_social_card(item)

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
    social = [i for i in matches if i.get("source_type") == "social"]

    result = f'Found **{len(matches)}** results for **"{query}"**\n\n---\n\n'

    if books:
        result += f"## 📚 Books ({len(books)})\n\n"
        for item in books[:ITEMS_PER_PAGE]:
            result += _format_book_card(item)

    if papers:
        result += f"## 📄 Research Papers ({len(papers)})\n\n"
        for item in papers[:ITEMS_PER_PAGE]:
            result += _format_paper_card(item)

    if social:
        result += f"## 💬 Social Posts ({len(social)})\n\n"
        for item in social[:10]:
            result += _format_social_card(item)

    return result


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind — Discover Books & Papers",
    theme=gr.themes.Soft(),
    css="""
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important; }
    .result-box { max-height: 800px; overflow-y: auto; }
    h3 { margin-top: 0.5em !important; margin-bottom: 0.2em !important; }
    blockquote { border-left: 3px solid #6366f1 !important; padding-left: 1em !important; color: #374151 !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🧠 LexiMind
        ### Discover Your Next Read
        
        Browse **{book_count} books** and **{paper_count} research papers** — each analyzed by a multi-task transformer that generates summaries, classifies topics, and detects emotions.
        
        *Summaries are AI-generated: back-cover blurbs for books, abstracts for papers.*
        """.format(
            book_count=len(BOOKS),
            paper_count=len(PAPERS),
        )
    )

    with gr.Tabs():
        # ── Browse by Topic ──
        with gr.Tab("📂 By Topic"):
            gr.Markdown("*Select a topic to explore related books and papers*")
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

        # ── Browse by Emotion ──
        with gr.Tab("💭 By Emotion"):
            gr.Markdown("*Find books and papers that evoke specific emotions*")
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

        # ── Social Posts (emotion showcase) ──
        with gr.Tab("💬 Social"):
            gr.Markdown(
                "*Short Reddit comments analyzed for emotion — showcasing the model's 28-label emotion detection.*"
            )

            social_emotion_dropdown = gr.Dropdown(
                choices=["All Emotions"] + [e.title() for e in EMOTIONS],
                value="All Emotions",
                label="Filter by Emotion",
                interactive=True,
            )

            social_results = gr.Markdown(
                value=browse_social("All Emotions"),
                elem_classes=["result-box"],
            )

            social_emotion_dropdown.change(
                fn=browse_social,
                inputs=[social_emotion_dropdown],
                outputs=[social_results],
            )

        # ── Search ──
        with gr.Tab("🔍 Search"):
            gr.Markdown("*Search through all books, papers, and posts by keyword*")

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

        # ── Metrics ──
        with gr.Tab("📊 Metrics"):
            gr.Markdown("### Model Evaluation\n\n*Computed on held-out validation data.*")

            # Summarization Metrics
            gr.Markdown("#### Summarization")

            if METRICS.get("summarization"):
                summ = METRICS["summarization"]
                summ_md = """
| Metric | Score |
|--------|-------|
| **ROUGE-1** | {rouge1:.4f} |
| **ROUGE-2** | {rouge2:.4f} |
| **ROUGE-L** | {rougeL:.4f} |
| **BLEU-4** | {bleu4:.4f} |
""".format(
                    rouge1=summ.get("rouge_rouge1", summ.get("rouge1", 0)),
                    rouge2=summ.get("rouge_rouge2", summ.get("rouge2", 0)),
                    rougeL=summ.get("rouge_rougeL", summ.get("rougeL", 0)),
                    bleu4=summ.get("bleu4", 0),
                )
                gr.Markdown(summ_md)
            else:
                gr.Markdown("*Summarization metrics not available. Run evaluation script.*")

            # Topic Classification Metrics
            gr.Markdown("#### Topic Classification")

            if METRICS.get("topic"):
                topic_m = METRICS["topic"]
                topic_md = """
| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.2%} |
| **Macro F1** | {f1:.4f} |
""".format(
                    accuracy=topic_m.get("accuracy", 0),
                    f1=topic_m.get("f1", topic_m.get("macro_f1", 0)),
                )
                gr.Markdown(topic_md)
            else:
                gr.Markdown("*Topic classification metrics not available.*")

            # Emotion Detection Metrics
            gr.Markdown("#### Emotion Detection")

            if METRICS.get("emotion"):
                emotion_m = METRICS["emotion"]
                emotion_md = """
| Metric | Score |
|--------|-------|
| **Sample-avg F1** | {sample_f1:.4f} |
| **Macro F1** | {macro_f1:.4f} |
| **Micro F1** | {micro_f1:.4f} |

*28-label multi-label classification from GoEmotions.*
""".format(
                    sample_f1=emotion_m.get(
                        "sample_avg_f1", emotion_m.get("f1", emotion_m.get("multilabel_f1", 0))
                    ),
                    macro_f1=emotion_m.get("macro_f1", 0),
                    micro_f1=emotion_m.get("micro_f1", 0),
                )
                gr.Markdown(emotion_md)
            else:
                gr.Markdown("*Emotion detection metrics not available.*")

            # Dataset Statistics
            gr.Markdown("#### Discovery Dataset")

            gr.Markdown(f"""
| Content | Count |
|---------|-------|
| Literary Works | {len(BOOKS)} |
| Research Papers | {len(PAPERS)} |
| Social Posts | {len(SOCIAL)} |
| **Total** | **{len(ALL_ITEMS)}** |
| Unique Topics | {len(TOPICS)} |
| Unique Emotions | {len(EMOTIONS)} |
""")

        # ── About ──
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### About LexiMind
                
                LexiMind is a **272M parameter encoder-decoder transformer** (FLAN-T5-base) trained jointly on three tasks:
                
                | Task | What it does | Training data |
                |------|-------------|---------------|
                | **Summarization** | Generates back-cover blurbs for books and abstracts for papers | ~49K pairs (arXiv + Project Gutenberg/Goodreads) |
                | **Topic Classification** | Assigns one of 7 topics | 3.4K samples |
                | **Emotion Detection** | Detects up to 28 emotions | 43K GoEmotions samples |
                
                The summaries you see here are **generated by the model** from the original full text — not copied from any source. The "Original Description" / "Original Abstract" shown in the expandable sections are the human-written references for comparison.
                
                #### Key Architecture Details
                
                - Custom from-scratch Transformer implementation (not HuggingFace wrappers)
                - Shared encoder with task-specific heads: decoder for summarization, attention pooling for emotion, mean pooling for topic
                - Trained in ~9 hours on a single RTX 4070 12GB
                
                [GitHub](https://github.com/OliverPerrin/LexiMind) · [Model](https://huggingface.co/OliverPerrin/LexiMind-Model) · [Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery) · [Paper](https://github.com/OliverPerrin/LexiMind/blob/main/docs/research_paper.tex)
                
                *Oliver Perrin · Appalachian State University · 2025–2026*
                """
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
