#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""
Dataset download script for LexiMind.

Focus: Books, Academic Papers, Technical Writing
- NO news articles (overdone, dated)
- YES literary text, research, technical writing

Datasets:
- Goodreads + Gutenberg for book descriptions (back-cover style blurbs)
- arXiv for academic paper summarization (abstracts)
- Project Gutenberg for literary language modeling
- GoEmotions for emotion classification (28 labels)
- Custom topic classification: Fiction, Science, Technology, etc.

Key Design Decision:
Book "summarization" uses Goodreads descriptions which describe what the book
is ABOUT (like a back cover), not plot summaries. This trains the model to
answer "What is this book about?" rather than "What happens in this book?"

Usage:
    python scripts/download_data.py              # Download all
    python scripts/download_data.py --task summarization  # Just summarization
    python scripts/download_data.py --max-arxiv 50000

Author: Oliver Perrin
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# ============== LABEL DEFINITIONS ==============

# 28 emotions from GoEmotions - works for all text types
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# New topic labels for books + papers + blogs
TOPIC_LABELS = [
    "Fiction",           # Novels, short stories, literary fiction
    "Science",           # Physics, chemistry, biology, nature
    "Technology",        # CS, engineering, programming, AI/ML
    "Philosophy",        # Ethics, logic, metaphysics, epistemology
    "History",           # Historical texts, biographies, memoirs
    "Psychology",        # Mind, behavior, self-help, mental health
    "Business",          # Economics, finance, entrepreneurship
    "Arts",              # Music, visual arts, film, architecture
]

# arXiv category → our topic mapping
ARXIV_CATEGORY_MAP = {
    # Computer Science
    "cs.AI": "Technology", "cs.CL": "Technology", "cs.CV": "Technology",
    "cs.LG": "Technology", "cs.NE": "Technology", "cs.RO": "Technology",
    "cs.SE": "Technology", "cs.PL": "Technology", "cs.DB": "Technology",
    "cs.DS": "Technology", "cs.CR": "Technology", "cs.DC": "Technology",
    "cs.HC": "Technology", "cs.IR": "Technology", "cs.IT": "Technology",
    "cs.MA": "Technology", "cs.MM": "Technology", "cs.NI": "Technology",
    "cs.OS": "Technology", "cs.PF": "Technology", "cs.SY": "Technology",
    # Physics
    "physics": "Science", "astro-ph": "Science", "cond-mat": "Science",
    "gr-qc": "Science", "hep-ex": "Science", "hep-lat": "Science",
    "hep-ph": "Science", "hep-th": "Science", "math-ph": "Science",
    "nlin": "Science", "nucl-ex": "Science", "nucl-th": "Science",
    "quant-ph": "Science",
    # Math
    "math": "Science",
    # Biology/Medicine
    "q-bio": "Science", "stat": "Science",
    # Economics/Finance
    "econ": "Business", "q-fin": "Business",
    # Electrical Engineering
    "eess": "Technology",
}

# Gutenberg subject → our topic mapping
GUTENBERG_SUBJECT_MAP = {
    "fiction": "Fiction", "novel": "Fiction", "stories": "Fiction",
    "poetry": "Arts", "drama": "Arts", "plays": "Arts",
    "science": "Science", "physics": "Science", "chemistry": "Science",
    "biology": "Science", "nature": "Science", "astronomy": "Science",
    "philosophy": "Philosophy", "ethics": "Philosophy", "logic": "Philosophy",
    "history": "History", "biography": "History", "memoir": "History",
    "psychology": "Psychology", "mind": "Psychology",
    "economics": "Business", "business": "Business", "finance": "Business",
    "art": "Arts", "music": "Arts", "architecture": "Arts",
    "technology": "Technology", "engineering": "Technology",
}


def write_jsonl(records: list[dict[str, Any]], path: Path, desc: str = "Writing") -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=desc, leave=False):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):,} samples → {path}")


# ============== ENGLISH LANGUAGE FILTER ==============

# Common English words for detection
ENGLISH_WORDS = {
    "the", "and", "of", "to", "a", "in", "that", "is", "was", "he", "she", "it",
    "for", "with", "as", "his", "her", "they", "be", "at", "on", "have", "had",
    "this", "but", "not", "from", "by", "or", "an", "said", "were", "been",
    "would", "could", "which", "their", "there", "what", "when", "who", "will",
    "more", "if", "no", "out", "so", "up", "into", "than", "them", "can", "only",
    "other", "new", "some", "very", "just", "over", "such", "also", "its", "then",
}

# Non-English language patterns
NON_ENGLISH_PATTERNS = [
    # French
    r"\b(le|la|les|un|une|des|du|et|est|sont|dans|pour|avec|sur|qui|que|ce|cette|nous|vous|ils|elles|je|tu|il|elle|être|avoir)\b",
    # German
    r"\b(der|die|das|ein|eine|und|ist|nicht|mit|von|zu|den|dem|auf|für|als|auch|oder|nach|bei|nur|noch|wie|mehr|aber|wenn|hat|kann|ich|sie|er|wir|ihr|es|sich|sein)\b",
    # Spanish
    r"\b(el|la|los|las|un|una|que|por|para|con|del|al|es|en|se|no|más|como|pero|su|sus|le|lo|te|me|nos)\b",
    # Italian
    r"\b(il|lo|la|gli|le|che|per|con|del|della|di|da|non|sono|anche|più|ma|se|mi|ti|ci)\b",
    # Latin
    r"\b(et|in|ad|cum|de|ex|per|pro|sub|ab|ante|post|inter|contra|super|trans|apud)\b",
]

# ============== TEXT QUALITY FILTERS ==============

# Patterns that indicate garbage/metadata text
GARBAGE_PATTERNS = [
    r"^Page \d+:",           # Page corrections
    r"changed to",           # Errata
    r"Punctuation has been", # Editorial notes
    r"^\[.*\]$",             # Bracketed notes
    r"^Note\.?[-—]",         # Notes
    r"^follows:",            # "as follows:"
    r"CHAPTER [IVXLC]+\.",   # Chapter headers only
    r"^\*\*\*",              # Project Gutenberg markers
    r"^End of.*Project",     # End markers
    r"^Produced by",         # Production credits
    r"transcriber",          # Transcriber notes
    r"eBook",                # eBook references
    r"©|copyright",          # Copyright notices
    r"^INDEX",               # Index pages
    r"^\d+\.\s+\w+,\s+\d+",  # Index entries like "1. Name, 234"
    r"(syn\.|var\.|sp\.)",   # Botanical abbreviations
    r"[A-Z][a-z]+aceae",     # Botanical family names
    r"\(\s*syn\s+",          # Synonym references
]

# Patterns that indicate technical manuals/instructions (not narrative)
TECHNICAL_PATTERNS = [
    r"\d+\.\s+It\s+(is|has|can)",  # Numbered features "1. It is a..."
    r"^\d+(st|nd|rd|th)\.",        # "1st. 2nd. 3rd."
    r"Mesh\.?\s*\d+",              # Mesh sizes (pottery)
    r"\d+\s*(oz|lb|kg|g|ml|mm|cm|inch)",  # Measurements
    r"Parts?\s*:?\s*\d+",          # "Parts: 50"
    r"Method of Using",            # Instructions
    r"How to\s+\w+",               # How-to guides
    r"Step\s+\d+",                 # Step-by-step
    r"wire.*address",              # Business instructions
    r"orders?\s+should\s+be",      # Order instructions  
    r"specifications?",            # Technical specs
    r"(Front|Back)\s+Focus",       # Camera terms
    r"Rack and Pinion",            # Mechanical terms
]

# Shakespeare and plays to exclude (model hallucinates on Early Modern English)
EXCLUDED_TITLES = {
    # Shakespeare
    "King Lear", "Hamlet", "Macbeth", "Othello", "Romeo and Juliet",
    "A Midsummer Night's Dream", "The Tempest", "Julius Caesar",
    "The Merchant of Venice", "Twelfth Night", "Much Ado About Nothing",
    "As You Like It", "The Taming of the Shrew", "Antony and Cleopatra",
    "Coriolanus", "Cymbeline", "Timon of Athens", "Troilus and Cressida",
    "Measure for Measure", "All's Well That Ends Well", "Pericles",
    "The Winter's Tale", "The Comedy of Errors", "Two Gentlemen of Verona",
    "Love's Labour's Lost", "The Merry Wives of Windsor", "Henry IV",
    "Henry V", "Henry VI", "Henry VIII", "Richard II", "Richard III",
    "King John", "Titus Andronicus",
    # French plays
    "Tartuffe", "Phaedra", "Cyrano de Bergerac", "Cyrano De Bergerac",
    "Le Misanthrope", "The School for Wives", "The Miser", "The Imaginary Invalid",
    "Andromaque", "Britannicus", "Bérénice", "Le Cid",
    # Greek/Roman plays
    "Oedipus Rex", "Oedipus the King", "Antigone", "Electra", "Medea",
    "The Bacchae", "The Oresteia", "Agamemnon", "Prometheus Bound",
    # Other classic plays
    "The Importance of Being Earnest", "Pygmalion", "Doctor Faustus",
    "Waiting for Godot", "Death of a Salesman", "A Streetcar Named Desire",
    "The Glass Menagerie", "Our Town", "Long Day's Journey Into Night",
    "Who's Afraid of Virginia Woolf", "The Crucible", "Cat on a Hot Tin Roof",
    # Verse/poetic epics
    "Idylls of the King", "Paradise Lost", "Paradise Regained",
    "The Divine Comedy", "Inferno", "Purgatorio", "Paradiso",
    "The Faerie Queene", "Beowulf",
}


def is_technical_manual(text: str) -> bool:
    """Check if text appears to be a technical manual/instructions."""
    for pattern in TECHNICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def is_quality_text(text: str) -> bool:
    """Check if text is quality content (not metadata/garbage)."""
    # Check for garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Reject technical manuals/instructions
    if is_technical_manual(text):
        return False
    
    # Must have reasonable length
    if len(text) < 300:
        return False
    
    # Must have sentences (not just fragments)
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) < 4:
        return False
    
    # Check for too many special characters
    special_ratio = len(re.findall(r'[^\w\s.,!?\'"()-]', text)) / max(len(text), 1)
    if special_ratio > 0.08:
        return False
    
    return True


def is_excluded_title(title: str) -> bool:
    """Check if title should be excluded (plays, epics, etc.)."""
    title_lower = title.lower()
    return any(excluded.lower() in title_lower for excluded in EXCLUDED_TITLES)


def is_play_text(text: str) -> bool:
    """Check if text appears to be a play/dramatic format."""
    play_patterns = [
        r"^(SCENE|ACT|Enter|Exit|Exeunt)\s",
        r"^\[.*\]$",  # Stage directions
        r"^[A-Z]{2,}\.\s",  # Character names like "HAMLET."
        r"Alarum|Flourish|Sennet",  # Stage directions
    ]
    lines = text.split('\n')[:10]
    play_indicators = 0
    for line in lines:
        for pattern in play_patterns:
            if re.search(pattern, line.strip()):
                play_indicators += 1
    return play_indicators >= 2


def is_english_text(text: str, min_ratio: float = 0.08, max_foreign: int = 5) -> bool:
    """
    Check if text is primarily English.
    
    Args:
        text: Text to check
        min_ratio: Minimum ratio of common English words
        max_foreign: Maximum number of foreign word matches before rejecting
    
    Returns:
        True if text appears to be English
    """
    if not text or len(text) < 100:
        return False
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) < 20:
        return False
    
    # Check for excessive non-English words
    for pattern in NON_ENGLISH_PATTERNS:
        matches = len(re.findall(pattern, text_lower))
        if matches > max_foreign:
            return False
    
    # Check for sufficient English words
    english_count = sum(1 for w in words if w.strip(".,!?;:'\"") in ENGLISH_WORDS)
    ratio = english_count / len(words)
    
    return ratio >= min_ratio


def normalize_title(title: str) -> str:
    """Normalize a book title for matching."""
    # Remove common prefixes/suffixes
    title = re.sub(r'^(The|A|An)\s+', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*\([^)]*\)\s*', '', title)  # Remove parentheticals
    title = re.sub(r'\s*:.+$', '', title)  # Remove subtitles
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    return title.lower().strip()


# ============== SUMMARIZATION: BOOKS + ARXIV ==============

def download_goodreads_descriptions() -> dict[str, dict]:
    """
    Download Goodreads book descriptions - back-cover style blurbs.
    
    These are "what the book is about" descriptions, not plot summaries.
    Returns dict mapping normalized title -> {title, description}
    """
    print("\n📚 Loading Goodreads book descriptions...")
    
    descriptions = {}
    
    # Try multiple sources
    datasets_to_try = [
        "booksouls/goodreads-book-descriptions",
        "Skelebor/book_titles_and_descriptions_en_clean",
    ]
    
    for ds_name in datasets_to_try:
        try:
            print(f"    Loading {ds_name}...")
            ds = load_dataset(ds_name, split="train")
            
            for item in tqdm(ds, desc="Goodreads", leave=False):
                title = item.get("title", "")
                description = item.get("description", "")
                
                if not title or not description:
                    continue
                
                # Skip very short descriptions (not useful for training)
                if len(description) < 100:
                    continue
                
                # Skip very long descriptions (truncate later)
                if len(description) > 2000:
                    description = description[:2000]
                
                # Skip plays and excluded titles
                if is_excluded_title(title):
                    continue
                
                # Skip non-English descriptions
                if not is_english_text(description):
                    continue
                
                norm_title = normalize_title(title)
                if norm_title and norm_title not in descriptions:
                    descriptions[norm_title] = {
                        "title": title,
                        "description": description,
                    }
            
            print(f"    Loaded {len(descriptions):,} descriptions from {ds_name}")
        except Exception as e:
            print(f"    {ds_name} failed: {e}")
    
    print(f"    Total: {len(descriptions):,} unique book descriptions")
    return descriptions


def download_book_descriptions(
    goodreads_descriptions: dict[str, dict],
    max_samples: int = 20000
) -> list[dict[str, Any]]:
    """
    Download book description data by matching Gutenberg texts with Goodreads descriptions.
    
    This gives us (book_excerpt, book_description) training pairs where descriptions
    are back-cover style "what is this book about" blurbs, not plot summaries.
    """
    print("\n📖 Matching Gutenberg books with Goodreads descriptions...")
    
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    matched_titles = set()
    skipped_quality = 0
    skipped_play = 0
    
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    for i in tqdm(indices, desc="Matching books", leave=False):
        if len(records) >= max_samples:
            break
        
        item = gutenberg[i]
        text = item.get("TEXT", "") or item.get("text", "")
        metadata_raw = item.get("METADATA", "") or "{}"
        
        # Parse metadata
        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        # Get title
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        if not title:
            continue
        
        # Check if we have a Goodreads description for this book
        norm_title = normalize_title(title)
        if norm_title not in goodreads_descriptions:
            continue
        
        # Skip if already matched this book
        if norm_title in matched_titles:
            continue
        
        goodreads_data = goodreads_descriptions[norm_title]
        
        # Skip plays and excluded titles
        if is_excluded_title(title):
            skipped_play += 1
            continue
        
        if not text or len(text) < 2000:
            continue
        
        # Get a clean excerpt from the book (skip front matter)
        paragraphs = re.split(r'\n\s*\n', text)
        excerpt_parts = []
        total_len = 0
        
        for para in paragraphs[10:]:  # Skip front matter
            para = para.strip()
            if len(para) < 100:
                continue
            
            # Quality check on paragraph
            if not is_english_text(para):
                continue
            if is_play_text(para):
                skipped_play += 1
                break
            if not is_quality_text(para) and len(para) > 300:
                skipped_quality += 1
                continue
            
            excerpt_parts.append(para)
            total_len += len(para)
            
            if total_len >= 3000:
                break
        
        if total_len < 1000:
            continue
        
        book_excerpt = "\n\n".join(excerpt_parts)[:4000]
        matched_titles.add(norm_title)
        
        records.append({
            "source": book_excerpt,
            "summary": goodreads_data["description"][:800],  # Back-cover blurbs are shorter
            "type": "literary",
            "title": goodreads_data["title"],
        })
    
    print(f"    Matched {len(records):,} books with descriptions")
    print(f"    Skipped: {skipped_quality} quality, {skipped_play} plays")
    
    return records


# Keep BookSum for additional literary training (chapter summaries are still useful)
def download_booksum(max_samples: int = 20000) -> list[dict[str, Any]]:
    """Download BookSum - literary chapter summarization (English only, quality filtered).
    
    Note: These are chapter-level plot summaries, useful as supplementary training data.
    The primary book training comes from Goodreads descriptions (back-cover style).
    """
    print("\n📖 Loading BookSum (supplementary literary data)...")
    
    all_records: list[dict[str, Any]] = []
    booksum = load_dataset("kmfoda/booksum")
    
    for split_name in booksum.keys():
        split = str(split_name)
        data = booksum[split_name]
        limit = max_samples if "train" in split else max_samples // 10
        indices = random.sample(range(len(data)), min(len(data), limit))
        
        records = []
        skipped_language = 0
        skipped_excluded = 0
        skipped_play = 0
        
        for i in tqdm(indices, desc=f"BookSum {split}", leave=False):
            item = data[i]
            chapter = item.get("chapter", "")
            summary = item.get("summary_text") or item.get("summary", "")
            
            # Extract book title from book_id (e.g., "The Last of the Mohicans.chapters 1-2")
            book_id = item.get("book_id", "")
            book_title = book_id.split(".")[0] if "." in book_id else book_id
            chapter_name = item.get("summary_id", "") or item.get("summary_name", "")
            
            if not (chapter and summary and len(chapter) > 300):
                continue
            
            # Filter: excluded titles (Shakespeare, plays, etc.)
            if is_excluded_title(book_title):
                skipped_excluded += 1
                continue
            
            # Filter: play text format
            if is_play_text(chapter):
                skipped_play += 1
                continue
            
            # Filter: English only
            if not is_english_text(chapter):
                skipped_language += 1
                continue
            
            # Filter: quality text
            if not is_quality_text(chapter):
                continue
            
            records.append({
                "source": chapter[:4000],
                "summary": summary,
                "type": "literary",
                "split": split,
                "title": book_title,
                "chapter": chapter_name,
            })
        all_records.extend(records)
        print(f"    {split}: {len(records):,} (skipped {skipped_language} non-English, {skipped_excluded} excluded, {skipped_play} plays)")
    
    return all_records


def clean_arxiv_text(text: str) -> str:
    """Clean arXiv LaTeX-style text to make it more readable."""
    import re
    # Remove LaTeX math placeholders
    text = re.sub(r'@xmath\d+', '', text)
    text = re.sub(r'@xcite', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()


def extract_paper_title(abstract: str) -> str:
    """Extract a meaningful title from the first sentence of an abstract."""
    # Clean the abstract first
    abstract = clean_arxiv_text(abstract)
    
    # Get the first sentence (up to first period, question mark, or newline)
    first_sentence = re.split(r'[.!?\n]', abstract)[0].strip()
    
    # Truncate if too long
    if len(first_sentence) > 100:
        # Try to cut at a natural word boundary
        first_sentence = first_sentence[:100].rsplit(' ', 1)[0] + '...'
    
    # Capitalize first letter
    if first_sentence:
        first_sentence = first_sentence[0].upper() + first_sentence[1:]
    
    return first_sentence or "Untitled Paper"


def download_arxiv_summarization(max_samples: int = 50000) -> list[dict[str, Any]]:
    """
    Download arXiv papers for academic summarization only (English only).
    Note: This dataset doesn't have categories, so can't be used for topic classification.
    
    Returns: summarization_records
    """
    print("\n🎓 Loading arXiv (academic papers for summarization)...")
    
    print("  Loading dataset (this may take a minute)...")
    arxiv = load_dataset("ccdv/arxiv-summarization", split="train")
    
    summ_records: list[dict[str, Any]] = []
    skipped_language = 0
    
    indices = list(range(len(arxiv)))
    random.shuffle(indices)
    
    print("  Processing papers...")
    for i in tqdm(indices[:max_samples * 2], desc="arXiv", leave=False):
        if len(summ_records) >= max_samples:
            break
            
        item = arxiv[i]
        
        # Get abstract and article
        abstract = item.get("abstract", "")
        article = item.get("article", "")
        
        if not abstract or len(abstract) < 100:
            continue
        
        # Clean LaTeX artifacts
        abstract = clean_arxiv_text(abstract)
        article = clean_arxiv_text(article)
        
        # Skip if still has too many weird characters after cleaning
        if '@' in abstract or '@' in article[:500]:
            continue
        
        # Filter: English only
        if not is_english_text(article[:1000]):
            skipped_language += 1
            continue
        
        # Summarization: article → abstract
        if article and len(article) > 500:
            # Extract title from abstract
            paper_title = extract_paper_title(abstract)
            
            summ_records.append({
                "source": article[:4000],
                "summary": abstract,
                "type": "academic",
                "title": paper_title,
            })
    
    print(f"    Summarization: {len(summ_records):,} (skipped {skipped_language} non-English)")
    
    return summ_records


def download_topics_from_datasets(max_samples: int = 50000) -> list[dict[str, Any]]:
    """
    Download topic classification data from multiple sources with real categories.
    
    Sources:
    - 20 Newsgroups (classic topic classification)
    - Wikipedia (article categories)
    """
    print("\n📂 Loading topic classification datasets...")
    
    records: list[dict[str, Any]] = []
    
    # 20 Newsgroups - classic topic dataset
    print("  Loading 20 Newsgroups...")
    try:
        newsgroups = load_dataset("SetFit/20_newsgroups", split="train")
        
        # Map 20 newsgroups categories to our 8 topics
        newsgroup_map = {
            # Science
            "sci.crypt": "Science", "sci.electronics": "Science",
            "sci.med": "Science", "sci.space": "Science",
            # Technology  
            "comp.graphics": "Technology", "comp.os.ms-windows.misc": "Technology",
            "comp.sys.ibm.pc.hardware": "Technology", "comp.sys.mac.hardware": "Technology",
            "comp.windows.x": "Technology",
            # Philosophy/Religion
            "alt.atheism": "Philosophy", "soc.religion.christian": "Philosophy",
            "talk.religion.misc": "Philosophy",
            # History/Politics
            "talk.politics.guns": "History", "talk.politics.mideast": "History",
            "talk.politics.misc": "History",
            # Business
            "misc.forsale": "Business",
            # Sports/Recreation
            "rec.autos": "Arts", "rec.motorcycles": "Arts",
            "rec.sport.baseball": "Arts", "rec.sport.hockey": "Arts",
        }
        
        for item in tqdm(newsgroups, desc="20 Newsgroups", leave=False):
            if len(records) >= max_samples:
                break
            label_name = item.get("label_text", "")
            text = item.get("text", "")
            
            if label_name in newsgroup_map and text and len(text) > 100:
                records.append({
                    "text": text[:1500],
                    "topic": newsgroup_map[label_name],
                    "source": "newsgroups",
                })
        
        print(f"    20 Newsgroups: {len(records):,}")
    except Exception as e:
        print(f"    20 Newsgroups failed: {e}")
    
    # Add from Gutenberg for Fiction
    gutenberg_topics = download_gutenberg_topics(max_samples // 4)
    records.extend(gutenberg_topics)
    
    # Add from scientific papers abstract dataset for more Science/Tech
    print("  Loading scientific papers...")
    try:
        sci_papers = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
        sci_count = 0
        for item in tqdm(sci_papers, desc="Scientific papers", leave=False, total=max_samples//4):
            if sci_count >= max_samples // 4:
                break
            abstract = item.get("abstract", "")
            if abstract and len(abstract) > 100:
                # Alternate between Science and Technology
                topic = "Science" if sci_count % 2 == 0 else "Technology"
                records.append({
                    "text": abstract[:1500],
                    "topic": topic,
                    "source": "scientific_papers",
                })
                sci_count += 1
        print(f"    Scientific papers: {sci_count:,}")
    except Exception as e:
        print(f"    Scientific papers failed: {e}")
    
    return records


def download_summarization(max_books: int = 20000, max_arxiv: int = 50000) -> None:
    """Download all summarization data (books + arxiv, NO news).
    
    Book data now uses Goodreads descriptions (back-cover blurbs) instead of
    plot summaries. This trains the model to describe "what the book is about"
    rather than summarizing the plot.
    """
    print("\n📝 Downloading Summarization Data...")
    out_dir = OUTPUT_DIR / "summarization"
    
    all_records: list[dict[str, Any]] = []
    
    # Goodreads descriptions - primary book training data (back-cover style)
    goodreads_descriptions = download_goodreads_descriptions()
    book_records = download_book_descriptions(goodreads_descriptions, max_books)
    all_records.extend(book_records)
    
    # Optional: Add some BookSum for additional literary variety
    # These are chapter summaries, not back-cover style, so keep limited
    # booksum_records = download_booksum(max_books // 4)
    # all_records.extend(booksum_records)
    
    # arXiv - academic (abstracts are already "what is this paper about")
    arxiv_summ = download_arxiv_summarization(max_arxiv)
    all_records.extend(arxiv_summ)
    
    # Shuffle and split
    random.shuffle(all_records)
    
    # Split by original split if available, else 90/5/5
    train_records = [r for r in all_records if r.get("split", "train") == "train" or "split" not in r]
    val_records = [r for r in all_records if r.get("split") == "validation"]
    test_records = [r for r in all_records if r.get("split") == "test"]
    
    # If no split info, do 90/5/5
    if len(val_records) < 100:
        n = len(train_records)
        random.shuffle(train_records)
        val_records = train_records[int(n*0.9):int(n*0.95)]
        test_records = train_records[int(n*0.95):]
        train_records = train_records[:int(n*0.9)]
    
    # Remove split key before saving
    for r in train_records + val_records + test_records:
        r.pop("split", None)
    
    write_jsonl(train_records, out_dir / "train.jsonl", "train")
    write_jsonl(val_records, out_dir / "validation.jsonl", "val")
    write_jsonl(test_records, out_dir / "test.jsonl", "test")
    
    # Print breakdown
    literary_count = sum(1 for r in train_records + val_records + test_records if r.get("type") == "literary")
    academic_count = sum(1 for r in train_records + val_records + test_records if r.get("type") == "academic")
    print(f"\n  ✓ Total summarization: {len(train_records) + len(val_records) + len(test_records):,}")
    print(f"    Literary (book descriptions): {literary_count:,}")
    print(f"    Academic (paper abstracts): {academic_count:,}")


# ============== TOPIC CLASSIFICATION ==============

def download_topics(max_samples: int = 50000) -> None:
    """
    Download topic classification data from multiple sources.
    
    Sources:
    - 20 Newsgroups (classic topic dataset)
    - Gutenberg books (Fiction)
    - Scientific papers (Science, Technology)
    """
    print("\n📂 Downloading Topic Classification...")
    out_dir = OUTPUT_DIR / "topic"
    
    # Get topic records from various sources
    all_records = download_topics_from_datasets(max_samples)
    
    # Balance topics
    topic_counts: dict[str, list] = {t: [] for t in TOPIC_LABELS}
    for r in all_records:
        topic = r.get("topic")
        if topic in topic_counts:
            topic_counts[topic].append(r)
    
    # Print distribution before balancing
    print("\n  Topic distribution (before balancing):")
    for topic, records in topic_counts.items():
        print(f"    {topic}: {len(records):,}")
    
    # Balance to min count (with some tolerance) - only from topics that have data
    counts_with_data = [len(v) for v in topic_counts.values() if v]
    if not counts_with_data:
        print("  ⚠️ No topic data found!")
        return
    
    min_count = min(counts_with_data)
    target_count = min(min_count, max_samples // len(TOPIC_LABELS))
    
    balanced: list[dict[str, Any]] = []
    for topic, records in topic_counts.items():
        if records:
            random.shuffle(records)
            balanced.extend(records[:target_count])
    
    random.shuffle(balanced)
    
    # Split 90/5/5
    n = len(balanced)
    train_records = balanced[:int(n*0.9)]
    val_records = balanced[int(n*0.9):int(n*0.95)]
    test_records = balanced[int(n*0.95):]
    
    write_jsonl(train_records, out_dir / "train.jsonl", "train")
    write_jsonl(val_records, out_dir / "validation.jsonl", "val")
    write_jsonl(test_records, out_dir / "test.jsonl", "test")
    
    # Save labels - only labels that have data
    used_labels = [t for t in TOPIC_LABELS if topic_counts.get(t)]
    (out_dir / "labels.json").write_text(json.dumps(used_labels, indent=2))
    print(f"\n  ✓ {len(used_labels)} topic labels with data: {used_labels}")


def download_gutenberg_topics(max_samples: int = 30000) -> list[dict[str, Any]]:
    """Extract topic-labeled samples from Gutenberg books (English only)."""
    print("\n📚 Loading Gutenberg for topic classification...")
    
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        print("  Trying pg19...")
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    skipped_language = 0
    
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    for i in tqdm(indices, desc="Gutenberg topics", leave=False):
        if len(records) >= max_samples:
            break
        
        item = gutenberg[i]
        text = item.get("TEXT", "") or item.get("text", "")
        metadata = item.get("METADATA", {}) or {}
        
        if not text or len(text) < 1000:
            continue
        
        # Try to determine topic from metadata
        subjects = ""
        if isinstance(metadata, dict):
            subjects = str(metadata.get("subjects", "")).lower()
            subjects += " " + str(metadata.get("subject", "")).lower()
            subjects += " " + str(metadata.get("category", "")).lower()
        
        topic = None
        for keyword, mapped_topic in GUTENBERG_SUBJECT_MAP.items():
            if keyword in subjects:
                topic = mapped_topic
                break
        
        # Default fiction for novels without clear subject
        if not topic and ("novel" in subjects or not subjects.strip()):
            topic = "Fiction"
        
        if topic:
            # Get a clean paragraph as sample
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs[5:]:  # Skip front matter
                para = para.strip()
                if 200 < len(para) < 1500 and para.count('.') >= 2:
                    # Filter: English only
                    if not is_english_text(para):
                        skipped_language += 1
                        break
                    
                    records.append({
                        "text": para,
                        "topic": topic,
                        "source": "gutenberg",
                    })
                    break
    
    print(f"    Gutenberg topics: {len(records):,} (skipped {skipped_language} non-English)")
    return records


# ============== EMOTIONS (unchanged) ==============

def download_emotions() -> None:
    """Download GoEmotions for emotion classification."""
    print("\n😊 Downloading Emotions (GoEmotions)...")
    out_dir = OUTPUT_DIR / "emotion"
    
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    for split_name in ds.keys():
        split = str(split_name)
        data = ds[split_name]
        
        records: list[dict[str, Any]] = []
        for item in tqdm(data, desc=split, leave=False):
            text = item.get("text", "")
            label_ids = item.get("labels", [])
            if text and label_ids:
                emotions = [EMOTION_LABELS[i] for i in label_ids if 0 <= i < len(EMOTION_LABELS)]
                if emotions:
                    records.append({"text": text, "emotions": emotions})
        write_jsonl(records, out_dir / f"{split}.jsonl", split)
    
    (out_dir / "labels.json").write_text(json.dumps(EMOTION_LABELS, indent=2))
    print(f"  ✓ {len(EMOTION_LABELS)} emotion labels saved")


# ============== GUTENBERG BOOKS (for language modeling) ==============

GUTENBERG_JUNK_PATTERNS = [
    r"Project Gutenberg", r"www\.gutenberg\.org", r"This ebook is for",
    r"Gutenberg License", r"^\*\*\* START OF", r"^\*\*\* END OF",
    r"Produced by", r"Transcriber's Note", r"TABLE OF CONTENTS",
    r"^\s*CHAPTER\s+[IVXLC\d]+", r"^\s*Chapter\s+[IVXLC\d]+",
    r"^\s*BOOK\s+[IVXLC\d]+", r"^\s*PREFACE\s*$", r"^\s*INTRODUCTION\s*$",
    r"E-text prepared by", r"Internet Archive", r"Distributed Proofreaders",
]
GUTENBERG_JUNK_REGEX = re.compile("|".join(GUTENBERG_JUNK_PATTERNS), re.IGNORECASE)


def is_clean_prose(text: str) -> bool:
    """Check if text is clean literary prose (English only)."""
    if len(text) < 300 or len(text) > 3000:
        return False
    if GUTENBERG_JUNK_REGEX.search(text):
        return False
    if text.count('.') < 2:
        return False
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if uppercase_ratio > 0.3:
        return False
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.1:
        return False
    # English filter
    if not is_english_text(text):
        return False
    return True


def download_gutenberg(max_samples: int = 30000) -> None:
    """Download Gutenberg books for language modeling (English only)."""
    print("\n📚 Downloading Gutenberg Books (English only)...")
    out_dir = OUTPUT_DIR / "books"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gutenberg = load_dataset("sedthh/gutenberg_english", split="train")
    except Exception:
        gutenberg = load_dataset("pg19", split="train")
    
    records: list[dict[str, Any]] = []
    indices = list(range(len(gutenberg)))
    random.shuffle(indices)
    
    for i in tqdm(indices, desc="Books", leave=False):
        if len(records) >= max_samples:
            break
        
        item = gutenberg[i]
        text = item.get("TEXT", "") or item.get("text", "")
        metadata_raw = item.get("METADATA", "") or "{}"
        
        # Parse metadata - it's stored as JSON string
        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        # Extract title and author
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        author = metadata.get("author", "") if isinstance(metadata, dict) else ""
        if not title:
            title = item.get("title", f"Unknown Book #{i}")
        
        if not text or len(text) < 1000:
            continue
        
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para = para.strip()
            if is_clean_prose(para):
                records.append({
                    "text": para, 
                    "title": title, 
                    "author": author,
                    "type": "gutenberg"
                })
                if len(records) >= max_samples:
                    break
    
    random.shuffle(records)
    n = len(records)
    write_jsonl(records[:int(n*0.9)], out_dir / "train.jsonl", "train")
    write_jsonl(records[int(n*0.9):int(n*0.95)], out_dir / "validation.jsonl", "val")
    write_jsonl(records[int(n*0.95):], out_dir / "test.jsonl", "test")


# ============== MAIN ==============

def main() -> None:
    parser = argparse.ArgumentParser(description="Download LexiMind datasets")
    parser.add_argument(
        "--task",
        choices=["all", "summarization", "emotion", "topic", "gutenberg"],
        default="all",
        help="Dataset to download"
    )
    parser.add_argument("--max-books", type=int, default=40000, help="Max BookSum samples")
    parser.add_argument("--max-arxiv", type=int, default=50000, help="Max arXiv samples")
    parser.add_argument("--max-gutenberg", type=int, default=30000, help="Max Gutenberg chunks")
    parser.add_argument("--max-topics", type=int, default=50000, help="Max topic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("LexiMind Dataset Download")
    print("Books + Academic Papers + Topic Classification")
    print("=" * 60)
    
    if args.task in ["all", "summarization"]:
        download_summarization(args.max_books, args.max_arxiv)
    if args.task in ["all", "emotion"]:
        download_emotions()
    if args.task in ["all", "topic"]:
        download_topics(args.max_topics)
    if args.task in ["all", "gutenberg"]:
        download_gutenberg(args.max_gutenberg)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
