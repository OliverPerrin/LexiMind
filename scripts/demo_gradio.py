"""
LexiMind - Book & Paper Discovery

Browse books and research papers by topic or emotion.
Pre-analyzed summaries help you find what to read next.

Author: Oliver Perrin
Date: 2026-01-13
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

# --------------- Load Catalog ---------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"

# For HF Space, catalog may be in root
if not CATALOG_PATH.exists():
    CATALOG_PATH = Path("data/catalog.json")

with open(CATALOG_PATH) as f:
    CATALOG = json.load(f)

BOOKS = CATALOG["books"]
PAPERS = CATALOG["papers"]
ALL_ITEMS = BOOKS + PAPERS
TOPICS = CATALOG["topics"]
EMOTIONS = CATALOG["emotions"]


# --------------- Filter Functions ---------------


def get_items_by_topic(topic: str) -> list[dict]:
    """Get all items matching a topic."""
    if topic == "All":
        return ALL_ITEMS
    return [item for item in ALL_ITEMS if item.get("topic") == topic]


def get_items_by_emotion(emotion: str) -> list[dict]:
    """Get all items containing an emotion."""
    if emotion == "All":
        return ALL_ITEMS
    return [item for item in ALL_ITEMS if emotion.lower() in [e.lower() for e in item.get("emotions", [])]]


def format_item_card(item: dict) -> str:
    """Format an item as a markdown card."""
    is_book = "author" in item
    
    if is_book:
        author_line = f"**{item['author']}** ({item['year']})"
    else:
        authors = item.get("authors", ["Unknown"])
        author_line = f"**{', '.join(authors)}** ({item['year']})"
    
    emotions = ", ".join(e.title() for e in item.get("emotions", []))
    
    return f"""### {item['title']}

{author_line}

📚 **Topic:** {item['topic']} &nbsp;|&nbsp; 💭 **Emotions:** {emotions}

{item['summary']}

---
"""


def browse_by_topic(topic: str) -> str:
    """Browse items filtered by topic."""
    items = get_items_by_topic(topic)
    if not items:
        return "No items found for this topic."
    
    # Group by type
    books = [i for i in items if "author" in i]
    papers = [i for i in items if "authors" in i]
    
    result = f"## {topic if topic != 'All' else 'All Topics'}\n\n"
    result += f"*Found {len(items)} items ({len(books)} books, {len(papers)} papers)*\n\n"
    
    if books:
        result += "### 📖 Books\n\n"
        for item in books:
            result += format_item_card(item)
    
    if papers:
        result += "### 📄 Research Papers\n\n"
        for item in papers:
            result += format_item_card(item)
    
    return result


def browse_by_emotion(emotion: str) -> str:
    """Browse items filtered by emotion."""
    items = get_items_by_emotion(emotion)
    if not items:
        return "No items found for this emotion."
    
    books = [i for i in items if "author" in i]
    papers = [i for i in items if "authors" in i]
    
    result = f"## Feeling {emotion.title() if emotion != 'All' else 'All Emotions'}?\n\n"
    result += f"*Found {len(items)} items ({len(books)} books, {len(papers)} papers)*\n\n"
    
    if books:
        result += "### 📖 Books\n\n"
        for item in books:
            result += format_item_card(item)
    
    if papers:
        result += "### 📄 Research Papers\n\n"
        for item in papers:
            result += format_item_card(item)
    
    return result


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind",
    theme=gr.themes.Soft(),
    css="""
    .result-box { max-height: 600px; overflow-y: auto; }
    """
) as demo:
    
    gr.Markdown(
        """
        # 📚 LexiMind
        ### Discover Books & Papers by Topic or Emotion
        
        Browse a curated collection of classic literature and research papers.
        Each item has been analyzed for **topic classification**, **emotional tone**, 
        and includes a **generated summary** to help you decide what to read next.
        
        ---
        """
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
        
        # ===================== TAB 3: ABOUT =====================
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### About LexiMind
                
                LexiMind is a **272M parameter encoder-decoder transformer** trained on three tasks:
                
                | Task | Description |
                |------|-------------|
                | **Summarization** | Generate concise summaries of long texts |
                | **Topic Classification** | Categorize into Fiction, Science, Technology, Philosophy, History, Business, Arts |
                | **Emotion Detection** | Identify emotional tones (28 emotions from GoEmotions) |
                
                ### Architecture
                
                - **Base:** FLAN-T5-base (Google)
                - **Encoder:** 12 layers, 768 dim, 12 attention heads
                - **Decoder:** 12 layers with causal attention
                - **Position:** T5 relative position bias
                - **Training:** Multi-task learning with task-specific heads
                
                ### Training Data
                
                | Dataset | Task |
                |---------|------|
                | BookSum + arXiv | Summarization |
                | 20 Newsgroups + Gutenberg | Topic Classification |
                | GoEmotions | Emotion Detection |
                
                ### Links
                
                - 🔗 [GitHub](https://github.com/OliverPerrin/LexiMind)
                - 🤗 [Model](https://huggingface.co/OliverPerrin/LexiMind-Model)
                
                ---
                *Built by Oliver Perrin • Appalachian State University • 2025-2026*
                """
            )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
