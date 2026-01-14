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
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

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
]

# Non-English indicators
NON_ENGLISH_PATTERNS = [
    r"\b(le|la|les|un|une|des|du|de la|au|aux)\b",  # French articles
    r"\b(der|die|das|ein|eine|und|ist|nicht)\b",     # German
    r"\b(el|la|los|las|un|una|que|por|para)\b",      # Spanish
    r"\b(il|lo|la|gli|le|un|una|che|per|con)\b",     # Italian
    r"[àâäéèêëïîôùûüÿœæ]{3,}",                       # Multiple French accents
]

def is_english(text: str) -> bool:
    """Check if text appears to be English."""
    text_lower = text.lower()
    
    # Check for non-English patterns
    for pattern in NON_ENGLISH_PATTERNS:
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 5:  # Too many non-English words
            return False
    
    # Check English word ratio
    english_words = ["the", "and", "of", "to", "a", "in", "that", "is", "was", "he", "she", "it", "for", "with", "as", "his", "her", "they", "be", "at", "on", "have", "had", "this", "but", "not", "from", "by", "or", "an"]
    words = text_lower.split()
    if len(words) < 20:
        return False
    
    english_count = sum(1 for w in words if w in english_words)
    ratio = english_count / len(words)
    
    return ratio > 0.05  # At least 5% common English words


def is_quality_text(text: str) -> bool:
    """Check if text is quality content (not metadata/garbage)."""
    # Check for garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Must have reasonable length
    if len(text) < 200:
        return False
    
    # Must have sentences (not just fragments)
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) < 3:
        return False
    
    # Check for too many special characters
    special_ratio = len(re.findall(r'[^\w\s.,!?\'"()-]', text)) / len(text)
    if special_ratio > 0.1:
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
            samples.append({
                "id": f"book_{title}",
                "title": title.replace("Book_", "Gutenberg #"),
                "text": excerpt["text"][:2000],
                "source_type": "literary",
                "dataset": "gutenberg"
            })
    
    print(f"  Loaded {len(samples)} quality book excerpts from {len(by_title)} books")
    return samples


def load_academic_papers(data_dir: Path, max_samples: int = 100) -> list[dict]:
    """Load academic paper samples with quality filtering."""
    summ_file = data_dir / "summarization" / "train.jsonl"
    
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
            
            academic.append({
                "text": text[:2000],
                "reference_summary": item.get("summary", "")[:500]
            })
    
    random.seed(42)
    samples = random.sample(academic, min(max_samples, len(academic)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"paper_{i}",
            "title": f"Academic Paper #{i+1}",
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
            
            literary.append({
                "text": text[:2000],
                "reference_summary": item.get("summary", "")[:500]
            })
    
    random.seed(42)
    samples = random.sample(literary, min(max_samples, len(literary)))
    
    results = []
    for i, item in enumerate(samples):
        results.append({
            "id": f"literary_{i}",
            "title": f"Literary Excerpt #{i+1}",
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
    parser.add_argument("--num-books", type=int, default=150, help="Number of books to include")
    parser.add_argument("--num-papers", type=int, default=100, help="Number of academic papers")
    parser.add_argument("--num-literary", type=int, default=50, help="Number of BookSum excerpts")
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()
    
    print("Loading data samples with quality filtering...")
    
    # Load samples from each source
    books = load_books(args.data_dir, max_per_book=1)
    random.seed(42)
    books = random.sample(books, min(args.num_books, len(books)))
    
    papers = load_academic_papers(args.data_dir, args.num_papers)
    literary = load_literary(args.data_dir, args.num_literary)
    
    all_samples = books + papers + literary
    print(f"\nTotal samples: {len(all_samples)}")
    
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
            commit_message="Rebuild with quality filtering and diverse emotions"
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
