"""
Gradio demo for LexiMind multi-task NLP model.

Redesigned to showcase the model's capabilities on training data:
- Browse classic literature and news articles
- Filter by topic and emotion
- View real-time summaries and classifications
- Compare model outputs across different texts

Author: Oliver Perrin
Date: 2025-12-05, Updated: 2026-01-12
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

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
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BOOKS_DIR = DATA_DIR / "books"
SUMMARIZATION_DIR = DATA_DIR / "summarization"

EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"

# Emotion display with emojis
EMOTION_EMOJI = {
    "joy": "😊", "love": "❤️", "anger": "😠", "fear": "😨",
    "sadness": "😢", "surprise": "😲", "neutral": "😐",
    "admiration": "🤩", "amusement": "😄", "annoyance": "😤",
    "approval": "👍", "caring": "🤗", "confusion": "😕",
    "curiosity": "🤔", "desire": "😍", "disappointment": "😞",
    "disapproval": "👎", "disgust": "🤢", "embarrassment": "😳",
    "excitement": "🎉", "gratitude": "🙏", "grief": "😭",
    "nervousness": "😰", "optimism": "🌟", "pride": "🦁",
    "realization": "💡", "relief": "😌", "remorse": "😔",
}

# Topic display with emojis
TOPIC_EMOJI = {
    "World": "🌍", "Sports": "🏆", "Business": "💼", 
    "Sci/Tech": "🔬", "Science & Mathematics": "🔬",
    "Education & Reference": "📚", "Entertainment & Music": "🎬",
    "Health": "🏥", "Family & Relationships": "👨‍👩‍👧",
    "Society & Culture": "🏛️", "Politics & Government": "🗳️",
    "Computers & Internet": "💻",
}

# --------------- Data Loading ---------------


def load_books_data() -> list[dict[str, Any]]:
    """Load book paragraphs from JSONL files."""
    books = []
    library_path = BOOKS_DIR / "library.json"
    
    if library_path.exists():
        with open(library_path) as f:
            library = json.load(f)
            
        for book_info in library.get("books", []):
            title = book_info["title"]
            jsonl_name = book_info["filename"].replace(".txt", ".jsonl")
            jsonl_path = BOOKS_DIR / jsonl_name
            
            if jsonl_path.exists():
                paragraphs = []
                with open(jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            para = json.loads(line)
                            # Only include paragraphs with substantial content
                            if para.get("token_count", 0) > 50:
                                paragraphs.append(para)
                
                if paragraphs:
                    books.append({
                        "title": title,
                        "paragraphs": paragraphs[:20],  # Limit to first 20 substantial paragraphs
                        "word_count": book_info.get("word_count", 0),
                    })
    
    return books


def load_news_data(split: str = "validation", max_items: int = 100) -> list[dict[str, Any]]:
    """Load news articles from summarization dataset."""
    articles = []
    data_path = SUMMARIZATION_DIR / f"{split}.jsonl"
    
    if data_path.exists():
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= max_items:
                    break
                if line.strip():
                    article = json.loads(line)
                    # Only include articles with reasonable length
                    source = article.get("source", "")
                    if len(source) > 200:
                        articles.append({
                            "text": source,
                            "reference_summary": article.get("summary", ""),
                            "id": i,
                        })
    
    return articles


# Cache the loaded data
_books_cache: list[dict] | None = None
_news_cache: list[dict] | None = None


def get_books() -> list[dict]:
    global _books_cache
    if _books_cache is None:
        _books_cache = load_books_data()
    return _books_cache


def get_news() -> list[dict]:
    global _news_cache
    if _news_cache is None:
        _news_cache = load_news_data()
    return _news_cache


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


# --------------- Core Analysis Functions ---------------


def analyze_text(text: str) -> tuple[str, str, str]:
    """Run all three tasks and return formatted results."""
    if not text or not text.strip():
        return "Please enter or select text to analyze.", "", ""

    try:
        pipe = get_pipeline()

        # Run tasks
        summary = pipe.summarize([text], max_length=150)[0].strip()
        if not summary:
            summary = "(Unable to generate summary)"

        emotions = pipe.predict_emotions([text], threshold=0.3)[0]
        topic = pipe.predict_topics([text])[0]

        # Format emotions
        if emotions.labels:
            emotion_parts = []
            for lbl, score in zip(emotions.labels[:5], emotions.scores[:5], strict=False):
                emoji = EMOTION_EMOJI.get(lbl.lower(), "•")
                emotion_parts.append(f"{emoji} **{lbl.title()}** ({score:.0%})")
            emotion_str = "\n".join(emotion_parts)
        else:
            emotion_str = "😐 No strong emotions detected"

        # Format topic
        topic_emoji = TOPIC_EMOJI.get(topic.label, "📄")
        topic_str = f"{topic_emoji} **{topic.label}**\n\nConfidence: {topic.confidence:.0%}"

        return summary, emotion_str, topic_str

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return f"Error: {e}", "", ""


# --------------- Book Browser Functions ---------------


def get_book_titles() -> list[str]:
    """Get list of available book titles."""
    books = get_books()
    return [b["title"] for b in books]


def get_book_excerpt(title: str, paragraph_idx: int = 0) -> str:
    """Get a specific paragraph from a book."""
    books = get_books()
    for book in books:
        if book["title"] == title:
            paragraphs = book["paragraphs"]
            if 0 <= paragraph_idx < len(paragraphs):
                text = paragraphs[paragraph_idx].get("text", "")
                return str(text) if text else ""
    return ""


def get_book_info(title: str) -> str:
    """Get book metadata."""
    books = get_books()
    for book in books:
        if book["title"] == title:
            num_paras = len(book["paragraphs"])
            word_count = book["word_count"]
            return f"**{title}**\n\n📖 {word_count:,} words | {num_paras} excerpts available"
    return ""


def on_book_select(title: str) -> tuple[str, str, int]:
    """Handle book selection - return first excerpt and info."""
    info = get_book_info(title)
    excerpt = get_book_excerpt(title, 0)
    return info, excerpt, 0


def on_paragraph_change(title: str, idx: int) -> str:
    """Handle paragraph slider change."""
    return get_book_excerpt(title, int(idx))


def get_max_paragraphs(title: str) -> int:
    """Get the number of paragraphs for a book."""
    books = get_books()
    for book in books:
        if book["title"] == title:
            return len(book["paragraphs"]) - 1
    return 0


# --------------- News Browser Functions ---------------


def get_random_news() -> tuple[str, str]:
    """Get a random news article and its reference summary."""
    news = get_news()
    if news:
        article = random.choice(news)
        return article["text"], article.get("reference_summary", "")
    return "", ""


def get_news_by_index(idx: int) -> tuple[str, str]:
    """Get news article by index."""
    news = get_news()
    if 0 <= idx < len(news):
        article = news[idx]
        return article["text"], article.get("reference_summary", "")
    return "", ""


# --------------- Metrics Loading ---------------


def load_metrics() -> str:
    """Load evaluation metrics and format as markdown."""
    eval_metrics = {}
    if EVAL_REPORT_PATH.exists():
        try:
            with open(EVAL_REPORT_PATH) as f:
                eval_metrics = json.load(f)
        except Exception:
            pass

    train_metrics = {}
    if TRAINING_HISTORY_PATH.exists():
        try:
            with open(TRAINING_HISTORY_PATH) as f:
                train_metrics = json.load(f)
        except Exception:
            pass

    val_final = train_metrics.get("val_epoch_3", {})

    md = """
## 📈 Model Performance

### Training Results

| Task | Metric | Score |
|------|--------|-------|
| **Topic Classification** | Accuracy | **{topic_acc:.1%}** |
| **Emotion Detection** | F1 | {emo_f1:.1%} |
| **Summarization** | ROUGE-like | {rouge:.1%} |

### Evaluation Results

| Metric | Value |
|--------|-------|
| Topic Accuracy | **{eval_topic:.1%}** |
| Emotion F1 (macro) | {eval_emo:.1%} |
| ROUGE-like | {eval_rouge:.1%} |
| BLEU | {eval_bleu:.3f} |
""".format(
        topic_acc=val_final.get("topic_accuracy", 0),
        emo_f1=val_final.get("emotion_f1", 0),
        rouge=val_final.get("summarization_rouge_like", 0),
        eval_topic=eval_metrics.get("topic", {}).get("accuracy", 0),
        eval_emo=eval_metrics.get("emotion", {}).get("f1_macro", 0),
        eval_rouge=eval_metrics.get("summarization", {}).get("rouge_like", 0),
        eval_bleu=eval_metrics.get("summarization", {}).get("bleu", 0),
    )

    return md


# --------------- Gradio Interface ---------------

with gr.Blocks(
    title="LexiMind - Multi-Task NLP",
    theme=gr.themes.Soft(),
    css="""
    .book-card { padding: 10px; border-radius: 8px; background: #f0f4f8; }
    .results-panel { min-height: 200px; }
    """
) as demo:
    gr.Markdown(
        """
        # 🧠 LexiMind
        ### Multi-Task Transformer for Document Analysis
        
        Explore classic literature and news articles with AI-powered analysis:
        - 📝 **Summarization** - Generate concise summaries
        - 😊 **Emotion Detection** - Identify emotional tones  
        - 📂 **Topic Classification** - Categorize by subject
        
        > Built with a custom Transformer initialized from FLAN-T5 weights.
        """
    )

    # ===================== TAB 1: EXPLORE BOOKS =====================
    with gr.Tab("📚 Explore Books"):
        gr.Markdown(
            """
            ### Classic Literature Collection
            Browse excerpts from classic novels and see how LexiMind analyzes them.
            Select a book, navigate through excerpts, and click **Analyze** to run the model.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                book_dropdown = gr.Dropdown(
                    choices=get_book_titles(),
                    label="📖 Select a Book",
                    value=get_book_titles()[0] if get_book_titles() else None,
                )
                book_info = gr.Markdown(elem_classes=["book-card"])
                
                para_slider = gr.Slider(
                    minimum=0,
                    maximum=19,
                    step=1,
                    value=0,
                    label="📄 Excerpt Number",
                    info="Navigate through different parts of the book"
                )
                
                analyze_book_btn = gr.Button("🔍 Analyze This Excerpt", variant="primary")
            
            with gr.Column(scale=2):
                book_excerpt = gr.Textbox(
                    label="📜 Book Excerpt",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                )
                
                with gr.Row():
                    with gr.Column():
                        book_summary = gr.Textbox(
                            label="📝 Generated Summary",
                            lines=4,
                            interactive=False,
                        )
                    with gr.Column():
                        with gr.Row():
                            book_emotions = gr.Markdown(
                                label="😊 Emotions",
                                value="*Click Analyze*",
                            )
                            book_topic = gr.Markdown(
                                label="📂 Topic", 
                                value="*Click Analyze*",
                            )

        # Book event handlers
        book_dropdown.change(
            fn=on_book_select,
            inputs=[book_dropdown],
            outputs=[book_info, book_excerpt, para_slider],
        )
        
        para_slider.change(
            fn=on_paragraph_change,
            inputs=[book_dropdown, para_slider],
            outputs=[book_excerpt],
        )
        
        analyze_book_btn.click(
            fn=analyze_text,
            inputs=[book_excerpt],
            outputs=[book_summary, book_emotions, book_topic],
        )
        
        # Initialize with first book
        demo.load(
            fn=on_book_select,
            inputs=[book_dropdown],
            outputs=[book_info, book_excerpt, para_slider],
        )

    # ===================== TAB 2: EXPLORE NEWS =====================
    with gr.Tab("📰 Explore News"):
        gr.Markdown(
            """
            ### CNN/DailyMail News Articles
            Explore news articles from the training dataset. Compare the model's 
            generated summary with the original human-written summary.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                news_slider = gr.Slider(
                    minimum=0,
                    maximum=99,
                    step=1,
                    value=0,
                    label="📰 Article Number",
                )
                random_news_btn = gr.Button("🎲 Random Article", variant="secondary")
                analyze_news_btn = gr.Button("🔍 Analyze Article", variant="primary")
                
                gr.Markdown("### Reference Summary")
                gr.Markdown("*Original human-written summary from the dataset:*")
                reference_summary = gr.Textbox(
                    label="",
                    lines=4,
                    interactive=False,
                    show_label=False,
                )
            
            with gr.Column(scale=2):
                news_text = gr.Textbox(
                    label="📰 News Article",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                )
                
                with gr.Row():
                    with gr.Column():
                        news_summary = gr.Textbox(
                            label="📝 LexiMind Summary",
                            lines=4,
                            interactive=False,
                        )
                    with gr.Column():
                        with gr.Row():
                            news_emotions = gr.Markdown(
                                label="😊 Emotions",
                                value="*Click Analyze*",
                            )
                            news_topic = gr.Markdown(
                                label="📂 Topic",
                                value="*Click Analyze*",
                            )

        # News event handlers
        news_slider.change(
            fn=get_news_by_index,
            inputs=[news_slider],
            outputs=[news_text, reference_summary],
        )
        
        random_news_btn.click(
            fn=get_random_news,
            outputs=[news_text, reference_summary],
        )
        
        analyze_news_btn.click(
            fn=analyze_text,
            inputs=[news_text],
            outputs=[news_summary, news_emotions, news_topic],
        )
        
        # Initialize with first article
        demo.load(
            fn=lambda: get_news_by_index(0),
            outputs=[news_text, reference_summary],
        )

    # ===================== TAB 3: FREE TEXT =====================
    with gr.Tab("✏️ Free Text"):
        gr.Markdown(
            """
            ### Try Your Own Text
            Enter any text to analyze. Note that the model performs best on 
            **news-style articles** and **literary prose** similar to the training data.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                free_text_input = gr.Textbox(
                    label="📝 Enter Text",
                    lines=8,
                    placeholder="Paste or type your text here...\n\nThe model works best with news articles or literary passages.",
                )
                
                with gr.Row():
                    analyze_free_btn = gr.Button("🔍 Analyze", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("**Sample texts:**")
                with gr.Row():
                    sample1 = gr.Button("📈 Business News", size="sm")
                    sample2 = gr.Button("🔬 Science News", size="sm")
                    sample3 = gr.Button("🏆 Sports News", size="sm")
            
            with gr.Column(scale=2):
                free_summary = gr.Textbox(
                    label="📝 Summary",
                    lines=4,
                    interactive=False,
                )
                with gr.Row():
                    free_emotions = gr.Markdown(value="*Enter text and click Analyze*")
                    free_topic = gr.Markdown(value="")

        # Sample texts
        SAMPLES = {
            "business": "Global markets tumbled today as investors reacted to rising inflation concerns. The Federal Reserve hinted at potential interest rate hikes, sending shockwaves through technology and banking sectors. Analysts predict continued volatility as economic uncertainty persists. Major indices fell by over 2%, with tech stocks leading the decline.",
            "science": "Scientists at MIT have developed a breakthrough quantum computing chip that operates at room temperature. This advancement could revolutionize drug discovery, cryptography, and artificial intelligence. The research team published their findings in Nature, demonstrating stable qubit operations for over 100 microseconds.",
            "sports": "The championship game ended in dramatic fashion as the underdog team scored in the final seconds to secure victory. Fans rushed the field in celebration, marking the team's first title in 25 years. The winning goal came from a rookie player who had only joined the team this season.",
        }
        
        sample1.click(fn=lambda: SAMPLES["business"], outputs=free_text_input)
        sample2.click(fn=lambda: SAMPLES["science"], outputs=free_text_input)
        sample3.click(fn=lambda: SAMPLES["sports"], outputs=free_text_input)
        clear_btn.click(fn=lambda: ("", "", "", ""), outputs=[free_text_input, free_summary, free_emotions, free_topic])
        
        analyze_free_btn.click(
            fn=analyze_text,
            inputs=[free_text_input],
            outputs=[free_summary, free_emotions, free_topic],
        )

    # ===================== TAB 4: METRICS =====================
    with gr.Tab("📊 Metrics"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(load_metrics())
            with gr.Column(scale=1):
                confusion_path = OUTPUTS_DIR / "topic_confusion_matrix.png"
                if confusion_path.exists():
                    gr.Image(str(confusion_path), label="Topic Confusion Matrix")

    # ===================== TAB 5: ABOUT =====================
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ### About LexiMind
            
            LexiMind is a **multi-task NLP system** built from scratch with PyTorch,
            demonstrating end-to-end machine learning engineering.
            
            #### 🏗️ Architecture
            
            - **Custom Transformer** encoder-decoder (12 layers each)
            - **Pre-LN with RMSNorm** for training stability
            - **T5 Relative Position Bias** for sequence modeling
            - **FLAN-T5-base** weight initialization
            - **Task-specific heads**: LM head (summarization), Classification heads (emotion, topic)
            
            #### 📚 Training Data
            
            | Task | Dataset | Description |
            |------|---------|-------------|
            | Summarization | CNN/DailyMail | ~100K news articles with summaries |
            | Emotion | GoEmotions | Multi-label emotion classification |
            | Topic | AG News | 4-class news categorization |
            | Books | Project Gutenberg | 8 classic novels for evaluation |
            
            #### ⚠️ Known Limitations
            
            - **Domain-specific**: Best results on news articles and literary text
            - **Summarization quality**: Limited by model size and training data
            - **Generalization**: May struggle with very different text styles
            
            #### 🔗 Links
            
            - [GitHub Repository](https://github.com/OliverPerrin/LexiMind)
            - [Model on HuggingFace](https://huggingface.co/OliverPerrin/LexiMind-Model)
            - [HuggingFace Space](https://huggingface.co/spaces/OliverPerrin/LexiMind)
            
            ---
            
            **Built by Oliver Perrin** | Appalachian State University | 2025-2026
            """
        )


# --------------- Entry Point ---------------

if __name__ == "__main__":
    # Pre-load pipeline and data
    logger.info("Loading inference pipeline...")
    get_pipeline()
    logger.info("Loading book data...")
    get_books()
    logger.info("Loading news data...")
    get_news()
    logger.info("Starting Gradio server...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
