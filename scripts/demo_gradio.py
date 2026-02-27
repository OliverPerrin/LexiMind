"""
LexiMind - Book & Paper Discovery

Browse books and research papers by topic or emotion.
Pre-analyzed summaries help you find what to read next.

Author: Oliver Perrin
Date: 2026-01-14
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
from datasets import Dataset, load_dataset

# --------------- Load Dataset from HuggingFace Hub ---------------

print("Loading discovery dataset from HuggingFace Hub...")
_dataset: Dataset = load_dataset("OliverPerrin/LexiMind-Discovery", split="train")  # type: ignore[assignment]
print(f"Loaded {len(_dataset)} items")

# Convert to list of dicts for easier filtering
ALL_ITEMS: list[dict[str, Any]] = [dict(row) for row in _dataset]

# Extract unique topics and emotions FROM THE DATASET (what model predicted)
DATASET_TOPICS: list[str] = sorted(set(str(item["topic"]) for item in ALL_ITEMS if item.get("topic")))
DATASET_EMOTIONS: list[str] = sorted(set(str(item["emotion"]) for item in ALL_ITEMS if item.get("emotion")))

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

# Group by source type
BOOKS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "literary"]
PAPERS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "academic"]

print(f"Dataset Topics ({len(TOPICS)}): {TOPICS}")
print(f"Dataset Emotions ({len(EMOTIONS)}): {EMOTIONS}")
print(f"All Model Topics ({len(ALL_TOPICS)}): {ALL_TOPICS}")
print(f"All Model Emotions ({len(ALL_EMOTIONS)}): {ALL_EMOTIONS}")
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


# --------------- Filter Functions ---------------


def get_items_by_topic(topic: str) -> list[dict]:
    """Get all items matching a topic."""
    if topic == "All":
        return ALL_ITEMS
    return [item for item in ALL_ITEMS if item.get("topic") == topic]


def get_items_by_emotion(emotion: str) -> list[dict]:
    """Get all items matching an emotion."""
    if emotion == "All":
        return ALL_ITEMS
    return [item for item in ALL_ITEMS if item.get("emotion") == emotion]


def format_item_card(item: dict) -> str:
    """Format an item as a markdown card."""
    title = item.get("title", "Unknown")
    source_type = item.get("source_type", "unknown")
    dataset_name = item.get("dataset", "").title()
    
    # Icon based on type
    if source_type == "academic":
        type_label = "Research Paper"
    else:
        type_label = "Literature"
    
    # Topic and emotion with confidence
    topic = item.get("topic", "Unknown")
    topic_conf = item.get("topic_confidence", 0)
    emotion = item.get("emotion", "Unknown")
    emotion_conf = item.get("emotion_confidence", 0)
    
    # Summary - check if using reference or generated
    use_reference = item.get("use_reference_summary", False)
    if use_reference or source_type == "literary":
        summary = item.get("reference_summary", "")
        summary_label = "**Book Description:**"
    else:
        summary = item.get("generated_summary", "")
        summary_label = "**AI-Generated Description:**"
    
    if not summary:
        summary = "No summary available."
    
    # Truncate summary if too long
    if len(summary) > 400:
        summary = summary[:400].rsplit(' ', 1)[0] + "..."
    
    # Preview of original text
    text_preview = item.get("text", "")[:400] + "..." if len(item.get("text", "")) > 400 else item.get("text", "")
    
    return f"""### **{title}**

<small>*{type_label}* from {dataset_name}</small>

**Topic:** {topic} ({topic_conf:.0%}) | **Emotion:** {emotion.title()} ({emotion_conf:.0%})

{summary_label}
> {summary}

<details>
<summary>View Original Text</summary>

{text_preview}

</details>

---
"""


def browse_by_topic(topic: str) -> str:
    """Browse items filtered by topic."""
    items = get_items_by_topic(topic)
    if not items:
        return "No items found for this topic."
    
    # Group by type
    literary = [i for i in items if i.get("source_type") == "literary"]
    academic = [i for i in items if i.get("source_type") == "academic"]
    
    result = f"## {topic if topic != 'All' else 'All Topics'}\n\n"
    result += f"*Found {len(items)} items ({len(literary)} literary, {len(academic)} academic)*\n\n"
    
    if literary:
        result += "### Literary Works\n\n"
        for item in literary[:25]:  # Limit to avoid huge pages
            result += format_item_card(item)
    
    if academic:
        result += "### Academic Papers\n\n"
        for item in academic[:25]:
            result += format_item_card(item)
    
    return result


def browse_by_emotion(emotion: str) -> str:
    """Browse items filtered by emotion."""
    items = get_items_by_emotion(emotion)
    if not items:
        return "No items found for this emotion."
    
    literary = [i for i in items if i.get("source_type") == "literary"]
    academic = [i for i in items if i.get("source_type") == "academic"]
    
    result = f"## Feeling {emotion.title() if emotion != 'All' else 'All Emotions'}?\n\n"
    result += f"*Found {len(items)} items ({len(literary)} literary, {len(academic)} academic)*\n\n"
    
    if literary:
        result += "### Literary Works\n\n"
        for item in literary[:25]:
            result += format_item_card(item)
    
    if academic:
        result += "### Academic Papers\n\n"
        for item in academic[:25]:
            result += format_item_card(item)
    
    return result


def search_items(query: str) -> str:
    """Search items by text content."""
    if not query or len(query) < 3:
        return "Enter at least 3 characters to search."
    
    query_lower = query.lower()
    matches = [
        item for item in ALL_ITEMS
        if query_lower in item.get("text", "").lower()
        or query_lower in item.get("generated_summary", "").lower()
        or query_lower in item.get("title", "").lower()
    ]
    
    if not matches:
        return f"No results found for '{query}'."
    
    result = f"## Search Results for '{query}'\n\n"
    result += f"*Found {len(matches)} matching items*\n\n"
    
    for item in matches[:30]:
        result += format_item_card(item)
    
    return result


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind",
    theme=gr.themes.Soft(),
    css="""
    .result-box { max-height: 700px; overflow-y: auto; }
    h3 { margin-top: 0.5em !important; }
    """
) as demo:
    
    gr.Markdown(
        """
        # LexiMind
        ### Discover Books & Papers by Topic, Emotion, or Keyword
        
        Browse **{total_count}** texts — {lit_count} classic books and {paper_count} research papers — analyzed by a multi-task transformer.
        
        ---
        """.format(
            total_count=len(ALL_ITEMS),
            lit_count=len(BOOKS),
            paper_count=len(PAPERS)
        )
    )
    
    with gr.Tabs():
        # ===================== TAB 1: BROWSE BY TOPIC =====================
        with gr.Tab("By Topic"):
            gr.Markdown("*Select a topic to explore related books and papers*")
            
            topic_dropdown = gr.Dropdown(
                choices=["All"] + TOPICS,
                value="All",
                label="Select Topic",
                interactive=True,
            )
            
            topic_results = gr.Markdown(
                value=browse_by_topic("All"),
                elem_classes=["result-box"],
            )
            
            topic_dropdown.change(
                fn=browse_by_topic,
                inputs=[topic_dropdown],
                outputs=[topic_results],
            )
        
        # ===================== TAB 2: BROWSE BY EMOTION =====================
        with gr.Tab("By Emotion"):
            gr.Markdown("*Find books and papers that evoke specific emotions*")
            
            emotion_dropdown = gr.Dropdown(
                choices=["All"] + [e.title() for e in EMOTIONS],
                value="All",
                label="Select Emotion",
                interactive=True,
            )
            
            emotion_results = gr.Markdown(
                value=browse_by_emotion("All"),
                elem_classes=["result-box"],
            )
            
            emotion_dropdown.change(
                fn=lambda e: browse_by_emotion(e.lower() if e != "All" else "All"),
                inputs=[emotion_dropdown],
                outputs=[emotion_results],
            )
        
        # ===================== TAB 3: SEARCH =====================
        with gr.Tab("Search"):
            gr.Markdown("*Search through all books and papers by keyword*")
            
            search_input = gr.Textbox(
                placeholder="Enter keywords to search...",
                label="Search",
                interactive=True,
            )
            
            search_results = gr.Markdown(
                value="Enter at least 3 characters to search.",
                elem_classes=["result-box"],
            )
            
            search_input.change(
                fn=search_items,
                inputs=[search_input],
                outputs=[search_results],
            )
        
        # ===================== TAB 4: METRICS =====================
        with gr.Tab("Metrics"):
            gr.Markdown(
                """
                ### Evaluation Metrics
                
                Computed on held-out validation data.
                """
            )
            
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
                topic = METRICS["topic"]
                topic_md = """
| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.2%} |
| **Macro F1** | {f1:.4f} |
""".format(
                    accuracy=topic.get("accuracy", 0),
                    f1=topic.get("f1", topic.get("macro_f1", 0)),
                )
                gr.Markdown(topic_md)
            else:
                gr.Markdown("*Topic classification metrics not available.*")
            
            # Emotion Detection Metrics
            gr.Markdown("#### Emotion Detection")
            
            if METRICS.get("emotion"):
                emotion = METRICS["emotion"]
                emotion_md = """
| Metric | Score |
|--------|-------|
| **Sample-avg F1** | {sample_f1:.4f} |
| **Macro F1** | {macro_f1:.4f} |
| **Micro F1** | {micro_f1:.4f} |

*28-label multi-label classification from GoEmotions.*
""".format(
                    sample_f1=emotion.get("sample_avg_f1", emotion.get("f1", emotion.get("multilabel_f1", 0))),
                    macro_f1=emotion.get("macro_f1", 0),
                    micro_f1=emotion.get("micro_f1", 0),
                )
                gr.Markdown(emotion_md)
            else:
                gr.Markdown("*Emotion detection metrics not available.*")
            
            # Dataset Statistics
            gr.Markdown("#### Dataset Statistics")
            
            gr.Markdown(f"""
| Statistic | Value |
|-----------|-------|
| Total Items | {len(ALL_ITEMS)} |
| Literary Works | {len(BOOKS)} |
| Academic Papers | {len(PAPERS)} |
| Topics | {len(TOPICS)} |
| Emotions | {len(EMOTIONS)} |
""")
        
        # ===================== TAB 5: ABOUT =====================
        with gr.Tab("About"):
            gr.Markdown(
                """
                ### About LexiMind
                
                A **272M parameter encoder-decoder transformer** (FLAN-T5-base) trained on three tasks:
                
                - **Summarization**: Generate back-cover style descriptions from full text
                - **Topic Classification**: 7 categories (Fiction, Science, History, Philosophy, Arts, Business, Technology)
                - **Emotion Detection**: 28 emotions via GoEmotions
                
                Training data: ~49K summarization pairs (arXiv + Goodreads), 43K emotion samples, 3.4K topic samples.
                
                [GitHub](https://github.com/OliverPerrin/LexiMind) | [Model](https://huggingface.co/OliverPerrin/LexiMind-Model) | [Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery)
                
                *Oliver Perrin — Appalachian State University — 2025-2026*
                """
            )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

