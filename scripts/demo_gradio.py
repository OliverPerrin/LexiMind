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

# Extract unique topics and emotions
TOPICS: list[str] = sorted(set(str(item["topic"]) for item in ALL_ITEMS if item.get("topic")))
EMOTIONS: list[str] = sorted(set(str(item["emotion"]) for item in ALL_ITEMS if item.get("emotion")))

# Group by source type
BOOKS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "literary"]
PAPERS: list[dict[str, Any]] = [item for item in ALL_ITEMS if item.get("source_type") == "academic"]

print(f"Topics: {TOPICS}")
print(f"Emotions: {EMOTIONS}")
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
        icon = "📄"
        type_label = "Research Paper"
    else:
        icon = "📖"
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
        summary_label = "📚 **Book Description** (Goodreads-style):"
    else:
        summary = item.get("generated_summary", "")
        summary_label = "🤖 **AI-Generated Description:**"
    
    if not summary:
        summary = "No summary available."
    
    # Truncate summary if too long
    if len(summary) > 400:
        summary = summary[:400].rsplit(' ', 1)[0] + "..."
    
    # Preview of original text
    text_preview = item.get("text", "")[:400] + "..." if len(item.get("text", "")) > 400 else item.get("text", "")
    
    # Confidence badges
    topic_badge = "🟢" if topic_conf > 0.6 else "🟡" if topic_conf > 0.3 else "🔴"
    emotion_badge = "🟢" if emotion_conf > 0.6 else "🟡" if emotion_conf > 0.3 else "🔴"
    
    return f"""### {icon} **{title}**

<small>*{type_label}* from {dataset_name}</small>

| Topic | Emotion |
|-------|---------|
| {topic_badge} {topic} ({topic_conf:.0%}) | {emotion_badge} {emotion.title()} ({emotion_conf:.0%}) |

{summary_label}
> {summary}

<details>
<summary>📜 View Original Text</summary>

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
        result += "### 📖 Literary Works\n\n"
        for item in literary[:25]:  # Limit to avoid huge pages
            result += format_item_card(item)
    
    if academic:
        result += "### 📄 Academic Papers\n\n"
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
        result += "### 📖 Literary Works\n\n"
        for item in literary[:25]:
            result += format_item_card(item)
    
    if academic:
        result += "### 📄 Academic Papers\n\n"
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
        # 📚 LexiMind - Literary Discovery
        ### Find Books & Research Papers by Topic or Emotional Tone
        
        Explore **{total_count}** items analyzed by the LexiMind multi-task transformer:
        
        | Source | Count | Description |
        |--------|-------|-------------|
        | 📖 Literature | {lit_count} | Classic books with Goodreads-style descriptions |
        | 📄 Research | {paper_count} | Scientific papers from arXiv |
        
        **Model Capabilities:**
        - 🏷️ **Topic Classification**: Fiction, Science, History, Philosophy, Arts, Business, Technology
        - 💭 **Emotion Detection**: 28 emotions (joy, sadness, anger, fear, surprise, love, etc.)
        - 📝 **Book Descriptions**: Back-cover style summaries of what texts are about
        
        ---
        """.format(
            total_count=len(ALL_ITEMS),
            lit_count=len(BOOKS),
            paper_count=len(PAPERS)
        )
    )
    
    with gr.Tabs():
        # ===================== TAB 1: BROWSE BY TOPIC =====================
        with gr.Tab("🏷️ Browse by Topic"):
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
        with gr.Tab("💭 Browse by Emotion"):
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
        with gr.Tab("🔍 Search"):
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
        with gr.Tab("📊 Model Metrics"):
            gr.Markdown(
                """
                ### Evaluation Metrics
                
                LexiMind is evaluated using comprehensive metrics across all three tasks.
                Metrics are computed on held-out validation data.
                """
            )
            
            # Summarization Metrics
            gr.Markdown("#### 📝 Summarization Metrics")
            
            if METRICS.get("summarization"):
                summ = METRICS["summarization"]
                summ_md = """
| Metric | Score | Description |
|--------|-------|-------------|
| **ROUGE-1** | {rouge1:.4f} | Unigram overlap with reference |
| **ROUGE-2** | {rouge2:.4f} | Bigram overlap with reference |
| **ROUGE-L** | {rougeL:.4f} | Longest common subsequence |
| **BLEU-4** | {bleu4:.4f} | 4-gram precision score |
| **BERTScore F1** | {bertscore:.4f} | Semantic similarity (contextual) |

*Note: For back-cover style descriptions, BERTScore is more meaningful than ROUGE 
since descriptions paraphrase rather than quote the source text.*
""".format(
                    rouge1=summ.get("rouge_rouge1", summ.get("rouge1", 0)),
                    rouge2=summ.get("rouge_rouge2", summ.get("rouge2", 0)),
                    rougeL=summ.get("rouge_rougeL", summ.get("rougeL", 0)),
                    bleu4=summ.get("bleu4", 0),
                    bertscore=summ.get("bertscore_f1", 0),
                )
                gr.Markdown(summ_md)
            else:
                gr.Markdown("*Summarization metrics not available. Run evaluation script.*")
            
            # Topic Classification Metrics
            gr.Markdown("#### 🏷️ Topic Classification Metrics")
            
            if METRICS.get("topic"):
                topic = METRICS["topic"]
                topic_md = """
| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.2%} |
| **Macro F1** | {f1:.4f} |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
""".format(
                    accuracy=topic.get("accuracy", 0),
                    f1=topic.get("f1", topic.get("macro_f1", 0)),
                    precision=topic.get("precision", 0),
                    recall=topic.get("recall", 0),
                )
                gr.Markdown(topic_md)
            else:
                gr.Markdown("*Topic classification metrics not available.*")
            
            # Emotion Detection Metrics
            gr.Markdown("#### 💭 Emotion Detection Metrics")
            
            if METRICS.get("emotion"):
                emotion = METRICS["emotion"]
                emotion_md = """
| Metric | Score |
|--------|-------|
| **Multi-label F1** | {f1:.4f} |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |

*Emotion detection uses 28 labels from GoEmotions. Multiple emotions can be assigned to each text.*
""".format(
                    f1=emotion.get("f1", emotion.get("multilabel_f1", 0)),
                    precision=emotion.get("precision", 0),
                    recall=emotion.get("recall", 0),
                )
                gr.Markdown(emotion_md)
            else:
                gr.Markdown("*Emotion detection metrics not available.*")
            
            # Dataset Statistics
            gr.Markdown("#### 📈 Dataset Statistics")
            gr.Markdown(f"""
| Statistic | Value |
|-----------|-------|
| Total Items | {len(ALL_ITEMS)} |
| Literary Works | {len(BOOKS)} |
| Academic Papers | {len(PAPERS)} |
| Unique Topics | {len(TOPICS)} |
| Unique Emotions | {len(EMOTIONS)} |
""")
        
        # ===================== TAB 5: ABOUT =====================
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### About LexiMind
                
                LexiMind is a **272M parameter encoder-decoder transformer** trained on three tasks:
                
                | Task | Description |
                |------|-------------|
                | **Book Descriptions** | Generate back-cover style descriptions of what books are about |
                | **Topic Classification** | Categorize into Fiction, Science, Technology, Philosophy, History, Business, Arts |
                | **Emotion Detection** | Identify emotional tones (28 emotions from GoEmotions) |
                
                ### Architecture
                
                - **Base:** FLAN-T5-base (Google)
                - **Encoder:** 12 layers, 768 dim, 12 attention heads
                - **Decoder:** 12 layers with causal attention
                - **Position:** T5 relative position bias
                - **Training:** Multi-task learning with task-specific heads
                
                ### Training Data
                
                | Dataset | Task | Description |
                |---------|------|-------------|
                | Goodreads (711k+ blurbs) | Book Descriptions | Back-cover style descriptions matched with Gutenberg texts |
                | arXiv | Paper Abstracts | Scientific paper summarization |
                | 20 Newsgroups + Gutenberg | Topic Classification | Multi-domain topic categorization |
                | GoEmotions | Emotion Detection | 28-class multi-label emotion classification |
                
                ### Key Design Decision
                
                LexiMind generates **back-cover style descriptions** (what a book is about) rather than 
                plot summaries (what happens in the book). This is achieved by training on Goodreads 
                descriptions paired with Project Gutenberg book texts.
                
                ### Evaluation Metrics
                
                - **ROUGE-1/2/L**: Lexical overlap (expected range: 0.15-0.25 for descriptions)
                - **BLEU-4**: N-gram precision
                - **BERTScore**: Semantic similarity using contextual embeddings (key metric for paraphrasing)
                
                ### Links
                
                - 🔗 [GitHub](https://github.com/OliverPerrin/LexiMind)
                - 🤗 [Model](https://huggingface.co/OliverPerrin/LexiMind-Model)
                - 📊 [Discovery Dataset](https://huggingface.co/datasets/OliverPerrin/LexiMind-Discovery)
                
                ---
                *Built by Oliver Perrin • Appalachian State University • 2025-2026*
                """
            )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

