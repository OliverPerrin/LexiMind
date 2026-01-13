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
import logging
import random
import sys
from pathlib import Path
from typing import Any

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
# Demo data is stored in artifacts/demo_data (committed to git)
# Full data in data/processed/ is gitignored
DEMO_DATA_DIR = PROJECT_ROOT / "artifacts" / "demo_data"
BOOKS_DIR = DEMO_DATA_DIR
NEWS_FILE = DEMO_DATA_DIR / "news_samples.jsonl"

EVAL_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"

# Emotion display - clean labels without emojis for research aesthetic
EMOTION_LABELS = {
    "joy": "Joy", "love": "Love", "anger": "Anger", "fear": "Fear",
    "sadness": "Sadness", "surprise": "Surprise", "neutral": "Neutral",
    "admiration": "Admiration", "amusement": "Amusement", "annoyance": "Annoyance",
    "approval": "Approval", "caring": "Caring", "confusion": "Confusion",
    "curiosity": "Curiosity", "desire": "Desire", "disappointment": "Disappointment",
    "disapproval": "Disapproval", "disgust": "Disgust", "embarrassment": "Embarrassment",
    "excitement": "Excitement", "gratitude": "Gratitude", "grief": "Grief",
    "nervousness": "Nervousness", "optimism": "Optimism", "pride": "Pride",
    "realization": "Realization", "relief": "Relief", "remorse": "Remorse",
}

# Topic display - clean labels
TOPIC_LABELS = {
    "World": "World News", "Sports": "Sports", "Business": "Business/Finance", 
    "Sci/Tech": "Science & Technology", "Science & Mathematics": "Science",
    "Education & Reference": "Education", "Entertainment & Music": "Entertainment",
    "Health": "Health", "Family & Relationships": "Lifestyle",
    "Society & Culture": "Society", "Politics & Government": "Politics",
    "Computers & Internet": "Technology",
}

# --------------- Data Loading ---------------


def load_books_data() -> list[dict[str, Any]]:
    """Load book data including pre-computed summaries from library.json."""
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
                        "summary": book_info.get("summary", ""),
                        "topic": book_info.get("topic", ""),
                        "emotions": book_info.get("emotions", []),
                    })
    
    return books


def load_news_data(max_items: int = 100) -> list[dict[str, Any]]:
    """Load news articles from demo data samples."""
    articles = []
    
    if NEWS_FILE.exists():
        with open(NEWS_FILE) as f:
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

        # Format emotions - sort by score descending, clean presentation
        if emotions.labels:
            paired = list(zip(emotions.labels, emotions.scores, strict=False))
            paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)[:5]
            
            emotion_parts = []
            for lbl, score in paired_sorted:
                clean_label = EMOTION_LABELS.get(lbl.lower(), lbl.title())
                # Use progress bar style
                bar_width = int(score * 10)
                bar = "█" * bar_width + "░" * (10 - bar_width)
                emotion_parts.append(f"**{clean_label}**  \n`{bar}` {score:.0%}")
            emotion_str = "\n\n".join(emotion_parts)
        else:
            emotion_str = "*No strong emotions detected*"

        # Format topic - clean presentation
        clean_topic = TOPIC_LABELS.get(topic.label, topic.label)
        topic_str = f"**{clean_topic}**\n\nConfidence: {topic.confidence:.0%}"

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
            topic = book.get("topic", "Unknown")
            return f"**{title}**\n\n{word_count:,} words | {num_paras} excerpts\n\nTopic: {topic}"
    return ""


def get_book_summary_and_emotions(title: str) -> tuple[str, str, str]:
    """Get pre-computed book summary and emotions from library.json."""
    books = get_books()
    for book in books:
        if book["title"] == title:
            # Get summary
            summary = book.get("summary", "No summary available.")
            
            # Get topic
            topic = book.get("topic", "Unknown")
            clean_topic = TOPIC_LABELS.get(topic, topic)
            topic_str = f"**{clean_topic}**"
            
            # Get emotions - already sorted by score in library.json
            emotions = book.get("emotions", [])
            if emotions:
                emotion_parts = []
                for emo in emotions[:5]:
                    label = emo.get("label", "")
                    score = emo.get("score", 0)
                    clean_label = EMOTION_LABELS.get(label.lower(), label.title())
                    bar_width = int(score * 10)
                    bar = "█" * bar_width + "░" * (10 - bar_width)
                    emotion_parts.append(f"**{clean_label}**  \n`{bar}` {score:.0%}")
                emotion_str = "\n\n".join(emotion_parts)
            else:
                emotion_str = "*No strong emotions detected*"
            
            return summary, emotion_str, topic_str
    return "No summary available.", "", ""


def on_book_select(title: str) -> tuple[str, str, str, str, str]:
    """Handle book selection - return book info, excerpt, and pre-computed analysis."""
    info = get_book_info(title)
    excerpt = get_book_excerpt(title, 0)
    summary, emotions, topic = get_book_summary_and_emotions(title)
    return info, excerpt, summary, emotions, topic


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
## Model Performance

### Training Results (Final Epoch)

| Task | Metric | Score |
|------|--------|-------|
| **Topic Classification** | Accuracy | **{topic_acc:.1%}** |
| **Emotion Detection** | F1 (macro) | {emo_f1:.1%} |
| **Summarization** | ROUGE-like | {rouge:.1%} |

### Evaluation Set Results

| Metric | Value |
|--------|-------|
| Topic Accuracy | **{eval_topic:.1%}** |
| Emotion F1 (macro) | {eval_emo:.1%} |
| ROUGE-like | {eval_rouge:.1%} |
| BLEU | {eval_bleu:.3f} |

*Training on CNN/DailyMail, GoEmotions, and AG News datasets.*
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

# Custom yellow/black research theme
lexitheme = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#e5e5e5",
    body_text_color_dark="#e5e5e5",
    block_background_fill="#171717",
    block_background_fill_dark="#171717",
    block_border_color="#262626",
    block_border_color_dark="#262626",
    block_label_text_color="#fbbf24",
    block_label_text_color_dark="#fbbf24",
    block_title_text_color="#fbbf24",
    block_title_text_color_dark="#fbbf24",
    input_background_fill="#262626",
    input_background_fill_dark="#262626",
    input_border_color="#404040",
    input_border_color_dark="#404040",
    button_primary_background_fill="#fbbf24",
    button_primary_background_fill_dark="#fbbf24",
    button_primary_text_color="#0a0a0a",
    button_primary_text_color_dark="#0a0a0a",
    button_primary_background_fill_hover="#f59e0b",
    button_primary_background_fill_hover_dark="#f59e0b",
    button_secondary_background_fill="#262626",
    button_secondary_background_fill_dark="#262626",
    button_secondary_text_color="#fbbf24",
    button_secondary_text_color_dark="#fbbf24",
    button_secondary_border_color="#fbbf24",
    button_secondary_border_color_dark="#fbbf24",
    slider_color="#fbbf24",
    slider_color_dark="#fbbf24",
)

with gr.Blocks(
    title="LexiMind | Multi-Task NLP Research Demo",
    theme=lexitheme,
    css="""
    .book-card { padding: 16px; border-radius: 8px; background: #1f1f1f; border: 1px solid #333; }
    .book-card p { color: #e5e5e5 !important; }
    .results-panel { min-height: 200px; }
    .metric-box { background: #1f1f1f; border: 1px solid #fbbf24; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #fbbf24 !important; }
    .prose { color: #d4d4d4 !important; }
    a { color: #fbbf24 !important; }
    """
) as demo:
    gr.Markdown(
        """
        # LexiMind
        ## Multi-Task Transformer for Document Understanding
        
        A custom encoder-decoder architecture initialized from FLAN-T5-base, trained jointly on:
        
        | Task | Dataset | Output |
        |------|---------|--------|
        | **Abstractive Summarization** | CNN/DailyMail | Condensed text |
        | **Emotion Detection** | GoEmotions | 28-class multi-label |
        | **Topic Classification** | AG News | 4-class single-label |
        
        ---
        """
    )

    # ===================== TAB 1: EXPLORE BOOKS =====================
    with gr.Tab("Literature"):
        gr.Markdown(
            """
            ### Classic Literature Collection
            Select a book to view pre-computed analysis (summary, emotions, topic).
            Browse excerpts and run real-time inference on individual passages.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                book_dropdown = gr.Dropdown(
                    choices=get_book_titles(),
                    label="Select Book",
                    value=get_book_titles()[0] if get_book_titles() else None,
                )
                book_info = gr.Markdown(elem_classes=["book-card"])
                
                gr.Markdown("---")
                gr.Markdown("**Browse Excerpts**")
                para_slider = gr.Slider(
                    minimum=0,
                    maximum=19,
                    step=1,
                    value=0,
                    label="Excerpt Index",
                    info="Navigate through passages"
                )
                
                analyze_excerpt_btn = gr.Button("Analyze Excerpt", variant="secondary", size="sm")
            
            with gr.Column(scale=2):
                gr.Markdown("#### Summary")
                book_summary = gr.Textbox(
                    label="",
                    lines=4,
                    interactive=False,
                    show_label=False,
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Detected Emotions")
                        book_emotions = gr.Markdown(value="*Select a book*")
                    with gr.Column():
                        gr.Markdown("#### Topic Classification")
                        book_topic = gr.Markdown(value="*Select a book*")
                
                gr.Markdown("---")
                gr.Markdown("#### Current Excerpt")
                book_excerpt = gr.Textbox(
                    label="",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                    show_label=False,
                )
                
                with gr.Row():
                    with gr.Column():
                        excerpt_summary = gr.Textbox(
                            label="Excerpt Summary",
                            lines=3,
                            interactive=False,
                        )
                    with gr.Column():
                        excerpt_emotions = gr.Markdown(value="*Click Analyze Excerpt*")

        # Book event handlers
        book_dropdown.change(
            fn=on_book_select,
            inputs=[book_dropdown],
            outputs=[book_info, book_excerpt, book_summary, book_emotions, book_topic],
        )
        
        para_slider.change(
            fn=on_paragraph_change,
            inputs=[book_dropdown, para_slider],
            outputs=[book_excerpt],
        )
        
        analyze_excerpt_btn.click(
            fn=analyze_text,
            inputs=[book_excerpt],
            outputs=[excerpt_summary, excerpt_emotions, book_topic],
        )
        
        # Initialize with first book
        demo.load(
            fn=on_book_select,
            inputs=[book_dropdown],
            outputs=[book_info, book_excerpt, book_summary, book_emotions, book_topic],
        )

    # ===================== TAB 2: EXPLORE NEWS =====================
    with gr.Tab("News Articles"):
        gr.Markdown(
            """
            ### CNN/DailyMail Dataset
            Browse news articles from the training data. Compare generated summaries
            against human-written reference summaries.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                news_slider = gr.Slider(
                    minimum=0,
                    maximum=99,
                    step=1,
                    value=0,
                    label="Article Index",
                )
                random_news_btn = gr.Button("Random Article", variant="secondary")
                analyze_news_btn = gr.Button("Run Inference", variant="primary")
                
                gr.Markdown("### Reference Summary")
                gr.Markdown("*Human-written summary from dataset:*")
                reference_summary = gr.Textbox(
                    label="",
                    lines=4,
                    interactive=False,
                    show_label=False,
                )
            
            with gr.Column(scale=2):
                news_text = gr.Textbox(
                    label="Source Article",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                )
                
                with gr.Row():
                    with gr.Column():
                        news_summary = gr.Textbox(
                            label="Generated Summary",
                            lines=4,
                            interactive=False,
                        )
                    with gr.Column():
                        with gr.Row():
                            news_emotions = gr.Markdown(
                                label="Emotions",
                                value="*Run inference*",
                            )
                            news_topic = gr.Markdown(
                                label="Topic",
                                value="*Run inference*",
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
    with gr.Tab("Custom Input"):
        gr.Markdown(
            """
            ### Interactive Inference
            Enter any text for real-time analysis. Best results on news-style articles
            and literary prose matching training distribution.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                free_text_input = gr.Textbox(
                    label="Input Text",
                    lines=8,
                    placeholder="Paste or type text here...\n\nOptimal for: news articles, literary passages, formal writing.",
                )
                
                with gr.Row():
                    analyze_free_btn = gr.Button("Run Inference", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                gr.Markdown("**Sample Inputs**")
                with gr.Row():
                    sample1 = gr.Button("Business", size="sm")
                    sample2 = gr.Button("Science", size="sm")
                    sample3 = gr.Button("Sports", size="sm")
            
            with gr.Column(scale=2):
                free_summary = gr.Textbox(
                    label="Generated Summary",
                    lines=4,
                    interactive=False,
                )
                with gr.Row():
                    free_emotions = gr.Markdown(value="*Enter text and run inference*")
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
    with gr.Tab("Metrics"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(load_metrics())
            with gr.Column(scale=1):
                confusion_path = OUTPUTS_DIR / "topic_confusion_matrix.png"
                if confusion_path.exists():
                    gr.Image(str(confusion_path), label="Topic Confusion Matrix")

    # ===================== TAB 5: ABOUT =====================
    with gr.Tab("Architecture"):
        gr.Markdown(
            """
            ### Model Architecture
            
            LexiMind is a **multi-task encoder-decoder Transformer** built from scratch in PyTorch,
            demonstrating end-to-end ML engineering from data processing to deployment.
            
            #### Architecture Details
            
            | Component | Specification |
            |-----------|---------------|
            | **Encoder** | 12-layer Transformer, 768-dim, 12 heads |
            | **Decoder** | 12-layer Transformer with causal masking |
            | **Normalization** | Pre-LN with RMSNorm |
            | **Position Encoding** | T5 Relative Position Bias |
            | **Initialization** | FLAN-T5-base pretrained weights |
            | **Parameters** | 272.5M total |
            
            #### Task Heads
            
            | Head | Architecture | Loss |
            |------|--------------|------|
            | **Summarization** | LM head (shared embeddings) | Cross-entropy w/ label smoothing |
            | **Emotion** | Mean-pooled encoder → Linear(768, 28) | BCE (multi-label) |
            | **Topic** | Mean-pooled encoder → Linear(768, 4) | Cross-entropy |
            
            #### Training Data
            
            | Task | Dataset | Size |
            |------|---------|------|
            | Summarization | CNN/DailyMail | ~100K articles |
            | Emotion | GoEmotions | ~43K samples |
            | Topic | AG News | ~120K articles |
            
            #### Known Limitations
            
            - **Domain specificity**: Optimized for news/literary text
            - **Emotion task**: Multi-label F1 limited by class imbalance in GoEmotions
            - **Summarization**: Constrained by 512 token context
            
            ---
            
            **Repository**: [github.com/OliverPerrin/LexiMind](https://github.com/OliverPerrin/LexiMind)  
            **Model**: [huggingface.co/OliverPerrin/LexiMind-Model](https://huggingface.co/OliverPerrin/LexiMind-Model)
            
            *Oliver Perrin • Appalachian State University • 2025-2026*
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
