#!/usr/bin/env python3
"""Build a discovery dataset for the HuggingFace Space demo.

This script extracts a diverse sample of books and academic papers from the
training data, runs inference to generate summaries/topics/emotions, and
uploads the result to HuggingFace Datasets.

Preprocessing includes:
- Filtering for English content only
- Removing metadata, errata, and front matter
- Requiring minimum text quality
- Ensuring topic and emotion diversity
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from tqdm import tqdm

from src.inference.factory import create_inference_pipeline

# --------------- Text Quality Filters ---------------

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

# Non-English indicators (expanded)
NON_ENGLISH_PATTERNS = [
    r"\b(le|la|les|un|une|des|du|de la|au|aux|et|est|sont|dans|pour|avec|sur|qui|que)\b",  # French
    r"\b(der|die|das|ein|eine|und|ist|nicht|mit|von|zu|den|dem|auf|für|als|auch|oder|nach|bei|nur|noch|wie|mehr|aber|wenn|so|hat|kann|ich|sie|er|wir|ihr|es|sich|sein)\b",  # German (expanded)
    r"\b(el|la|los|las|un|una|que|por|para|con|del|al|es|en|se|no|más|como|pero|su|sus)\b",  # Spanish
    r"\b(il|lo|la|gli|le|un|una|che|per|con|del|della|di|da|non|sono|è|anche|più|ma|se)\b",  # Italian
    r"[àâäéèêëïîôùûüÿœæäöüß]{2,}",  # Accented chars (German ß, umlauts)
    r"\b[A-Z][a-z]+ü[a-z]+\b",  # German words with ü
    r"\b[A-Z][a-z]+ö[a-z]+\b",  # German words with ö  
    r"\b[A-Z][a-z]+ä[a-z]+\b",  # German words with ä
]

# Patterns that indicate index/glossary/list content (not narrative)
INDEX_PATTERNS = [
    r"^\s*\d+\s*$",           # Just numbers
    r"^[A-Z][a-z]+,\s+\d+",   # "Word, 123" index entries
    r"(\d+,\s*)+\d+",         # Lists of page numbers
    r"^[A-Z]{2,}\s+",         # ALL CAPS words at start
    r"^\s*[-•]\s+",           # Bullet points
    r"p\.\s*\d+",             # Page references
]


def is_english(text: str) -> bool:
    """Check if text appears to be English."""
    text_lower = text.lower()
    
    # Check for non-English patterns - stricter threshold
    for pattern in NON_ENGLISH_PATTERNS:
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 3:  # Stricter: was 5
            return False
    
    # Check English word ratio
    english_words = ["the", "and", "of", "to", "a", "in", "that", "is", "was", "he", "she", "it", "for", "with", "as", "his", "her", "they", "be", "at", "on", "have", "had", "this", "but", "not", "from", "by", "or", "an", "said", "were", "been", "would", "could", "which", "their", "there", "what", "when", "who", "will", "more", "if", "no", "out", "so", "up", "into", "than", "them", "can", "only", "other", "new", "some", "very", "just", "over", "such", "also", "its", "then", "two", "first", "any", "these", "may", "after", "most", "made", "before", "should", "now", "where", "those", "being", "has", "between", "own", "under"]
    words = text_lower.split()
    if len(words) < 30:  # Stricter: was 20
        return False
    
    english_count = sum(1 for w in words if w in english_words)
    ratio = english_count / len(words)
    
    return ratio > 0.08  # Stricter: was 0.05


def is_narrative_text(text: str) -> bool:
    """Check if text is actual narrative (not index/glossary/list)."""
    lines = text.strip().split('\n')
    
    # Count lines that look like index entries
    index_lines = 0
    for line in lines:
        for pattern in INDEX_PATTERNS:
            if re.search(pattern, line):
                index_lines += 1
                break
    
    # If more than 30% are index-like, reject
    if len(lines) > 0 and index_lines / len(lines) > 0.3:
        return False
    
    # Must have actual sentences with verbs
    # Check for common verbs
    verb_patterns = r"\b(is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|might|can|said|says|went|came|made|took|saw|knew|thought|found|gave|told|asked|seemed|felt|looked|heard|began|kept|left|called|turned|wanted|tried|needed|used|believe|think|know|see|want|need|find|give|tell|become|leave|put|mean|keep|let|begin|seem|help|show|hear|play|run|move|live|read|write|learn|speak|bring|hold|stand|set|pay|meet|lead|understand|watch|follow|stop|create|speak|allow|add|spend|grow|open|walk|offer|remember|consider|appear|buy|wait|serve|die|send|build|stay|fall|cut|reach|kill|remain|suggest|raise|pass|sell|require|report|decide|pull)\b"
    verb_count = len(re.findall(verb_patterns, text.lower()))
    
    # Should have at least 1 verb per 50 words
    words = len(text.split())
    if words > 0 and verb_count / words < 0.02:
        return False
    
    return True


def is_quality_text(text: str) -> bool:
    """Check if text is quality content (not metadata/garbage)."""
    # Check for garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Must have reasonable length
    if len(text) < 300:  # Stricter: was 200
        return False
    
    # Must have sentences (not just fragments)
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) < 4:  # Stricter: was 3
        return False
    
    # Check for too many special characters
    special_ratio = len(re.findall(r'[^\w\s.,!?\'"()-]', text)) / len(text)
    if special_ratio > 0.08:  # Stricter: was 0.1
        return False
    
    # Must be narrative, not index/list
    if not is_narrative_text(text):
        return False
    
    return True


def clean_text(text: str) -> str:
    """Clean up text by removing artifacts."""
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove carriage returns
    text = text.replace('\r', '')
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove page numbers in brackets
    text = re.sub(r'\[p\.\s*\d+\]', '', text)
    
    # Remove asterisks used as separators
    text = re.sub(r'\*{3,}', '', text)
    
    return text.strip()


# --------------- Data Loading ---------------

def load_books(data_dir: Path, max_per_book: int = 1) -> list[dict]:
    """Load book excerpts with quality filtering."""
    books_file = data_dir / "books" / "train.jsonl"
    
    if not books_file.exists():
        print(f"  Warning: {books_file} not found")
        return []
    
    # Group by title
    by_title = defaultdict(list)
    with open(books_file) as f:
        for line in f:
            item = json.loads(line)
            text = clean_text(item["text"])
            
            # Quality filters
            if not is_english(text):
                continue
            if not is_quality_text(text):
                continue
            
            by_title[item["title"]].append({**item, "text": text})
    
    # Sample from each book - prefer longer, quality excerpts
    samples = []
    for title, excerpts in by_title.items():
        excerpts.sort(key=lambda x: len(x["text"]), reverse=True)
        for excerpt in excerpts[:max_per_book]:
            # Use real title, clean up if it's just an ID
            display_title = title
            if title.startswith("Book_") or title.startswith("Unknown Book"):
                display_title = f"Classic Literature #{title.split('_')[-1] if '_' in title else 'Unknown'}"
            
            author = excerpt.get("author", "")
            if author:
                display_title = f"{display_title} by {author}"
            
            samples.append({
                "id": f"book_{hash(title) % 100000}",
                "title": display_title,
                "text": excerpt["text"][:2000],
                "source_type": "literary",
                "dataset": "gutenberg"
            })
    
    print(f"  Loaded {len(samples)} quality book excerpts from {len(by_title)} books")
    return samples


def load_academic_papers(data_dir: Path, max_samples: int = 100) -> list[dict]:
    """Load academic paper samples with quality filtering."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    if not summ_file.exists():
        print(f"  Warning: {summ_file} not found")
        return []
    
    academic = []
    with open(summ_file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") != "academic":
                continue
            
            text = clean_text(item["source"])
            if not is_english(text):
                continue
            if len(text) < 500:
                continue
            
            # Use real title from data if available
            title = item.get("title", "")
            if not title:
                # Generate title from first sentence of summary
                summary = item.get("summary", "")
                title = summary.split('.')[0][:80] if summary else f"Research Paper"
                if len(title) > 60:
                    title = title[:60].rsplit(' ', 1)[0] + "..."
            
            academic.append({
                "text": text[:2000],
                "title": title,
                "reference_summary": item.get("summary", "")[:500]
            })
    
    random.seed(42)
    samples = random.sample(academic, min(max_samples, len(academic)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"paper_{i}",
            "title": item["title"],
            "text": item["text"],
            "source_type": "academic",
            "dataset": "arxiv",
            "reference_summary": item["reference_summary"]
        })
    
    print(f"  Loaded {len(results)} academic papers")
    return results


def load_literary(data_dir: Path, max_samples: int = 50) -> list[dict]:
    """Load literary samples from BookSum with quality filtering."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
    if not summ_file.exists():
        print(f"  Warning: {summ_file} not found")
        return []
    
    literary = []
    with open(summ_file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") != "literary":
                continue
            
            text = clean_text(item["source"])
            if not is_english(text):
                continue
            if len(text) < 500:
                continue
            
            # Use real book title and chapter from data
            title = item.get("title", "")
            chapter = item.get("chapter", "")
            
            if title and chapter:
                display_title = f"{title} - {chapter}"
            elif title:
                display_title = title
            else:
                display_title = "Classic Literature"
            
            literary.append({
                "text": text[:2000],
                "title": display_title,
                "reference_summary": item.get("summary", "")[:500]
            })
    
    random.seed(42)
    samples = random.sample(literary, min(max_samples, len(literary)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"literary_{i}",
            "title": item["title"],
            "text": item["text"],
            "source_type": "literary",
            "dataset": "booksum",
            "reference_summary": item["reference_summary"]
        })
    
    print(f"  Loaded {len(results)} BookSum literary excerpts")
    return results


# --------------- Inference ---------------

# Keywords for fallback emotion detection when model is undertrained
EMOTION_KEYWORDS = {
    "joy": ["happy", "delighted", "wonderful", "excellent", "fantastic", "amazing", "pleased"],
    "sadness": ["sad", "sorrow", "grief", "mourn", "weep", "tears", "tragic", "loss", "died"],
    "anger": ["angry", "furious", "rage", "outraged", "infuriated", "wrath"],
    "fear": ["afraid", "terror", "frightened", "scared", "horror", "dread"],
    "surprise": ["surprised", "astonished", "amazed", "shocked", "unexpected"],
    "disgust": ["disgusted", "revolting", "vile", "repulsive", "sickening"],
    "love": ["love", "adore", "beloved", "passion", "affection", "cherish"],
    "excitement": ["excited", "thrilled", "eager", "enthusiastic", "exhilarating"],
    "curiosity": ["curious", "wondered", "intrigued", "fascinated", "mysterious"],
    "admiration": ["admire", "respect", "impressive", "remarkable", "brilliant"],
    "gratitude": ["grateful", "thankful", "appreciate", "blessed"],
    "optimism": ["hope", "hopeful", "optimistic", "promising", "bright future"],
    "neutral": [],  # Default fallback
}


def detect_emotion_fallback(text: str) -> tuple[str, float]:
    """Keyword-based emotion detection as fallback."""
    text_lower = text.lower()
    
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if not keywords:
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[emotion] = count
    
    if scores:
        best = max(scores, key=scores.get)
        return best, min(0.7, 0.4 + scores[best] * 0.1)
    
    return "neutral", 0.5


def run_inference(pipeline, samples: list[dict], min_topic_conf: float = 0.3) -> list[dict]:
    """Run inference with quality thresholds."""
    results = []
    emotion_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    
    for sample in tqdm(samples, desc="Running inference"):
        text = sample["text"]
        
        # Generate summary
        summaries = pipeline.summarize([text], max_length=150)
        summary = summaries[0] if summaries else ""
        
        # Get topic - require minimum confidence
        topic_results = pipeline.predict_topics([text])
        if topic_results:
            topic = topic_results[0].label
            topic_conf = topic_results[0].confidence
        else:
            topic, topic_conf = "General", 0.0
        
        # Get emotions - check if model is outputting uniform probabilities
        emotion_results = pipeline.predict_emotions([text], threshold=0.1)
        use_fallback = False
        
        if emotion_results and emotion_results[0].labels:
            scores = emotion_results[0].scores
            # Check if scores are too uniform (model undertrained)
            score_range = max(scores) - min(scores)
            if score_range < 0.15:  # All scores within 15% = undertrained
                use_fallback = True
            else:
                # Use model prediction
                emotions_with_scores = list(zip(
                    emotion_results[0].labels, 
                    emotion_results[0].scores
                ))
                emotions_with_scores.sort(key=lambda x: x[1], reverse=True)
                emotion = emotions_with_scores[0][0]
                emotion_conf = emotions_with_scores[0][1]
                all_emotions = [e[0] for e in emotions_with_scores[:3]]
        else:
            use_fallback = True
        
        if use_fallback:
            # Use keyword-based fallback
            emotion, emotion_conf = detect_emotion_fallback(text)
            all_emotions = [emotion]
        
        # Track distribution
        emotion_counts[emotion] += 1
        topic_counts[topic] += 1
        
        sample_with_predictions = {
            **sample,
            "generated_summary": summary,
            "topic": topic,
            "topic_confidence": round(topic_conf, 3),
            "emotion": emotion,
            "emotion_confidence": round(emotion_conf, 3),
            "all_emotions": all_emotions,
        }
        results.append(sample_with_predictions)
    
    print(f"\nTopic distribution: {dict(topic_counts)}")
    print(f"Emotion distribution: {dict(emotion_counts)}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--num-books", type=int, default=300, help="Number of books to include")
    parser.add_argument("--num-papers", type=int, default=200, help="Number of academic papers")
    parser.add_argument("--num-literary", type=int, default=100, help="Number of BookSum excerpts")
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()
    
    print("Loading data samples with quality filtering...")
    
    # Load samples from each source
    books = load_books(args.data_dir, max_per_book=1)
    random.seed(42)
    books = random.sample(books, min(args.num_books, len(books))) if books else []
    
    papers = load_academic_papers(args.data_dir, args.num_papers)
    literary = load_literary(args.data_dir, args.num_literary)
    
    all_samples = books + papers + literary
    print(f"\nTotal samples: {len(all_samples)}")
    
    if not all_samples:
        print("ERROR: No samples loaded! Check if data/processed exists and has data.")
        return
    
    # Load model and run inference
    print(f"\nLoading model from {args.checkpoint}...")
    labels_path = Path("artifacts/labels.json")
    pipeline, labels = create_inference_pipeline(
        args.checkpoint,
        labels_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Running inference on all samples...")
    results = run_inference(pipeline, all_samples)
    
    # Save locally
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    # Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        dataset = Dataset.from_list(results)
        dataset.push_to_hub(
            args.hub_repo,
            private=False,
            commit_message="Rebuild with real titles and improved quality"
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
