
import sys
from pathlib import Path
import torch
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.factory import create_inference_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    print("Loading pipeline...")
    try:
        pipeline, _ = create_inference_pipeline(
            tokenizer_dir="artifacts/hf_tokenizer/",
            checkpoint_path="checkpoints/best.pt",
            labels_path="artifacts/labels.json",
        )
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return

    text = (
        "Artificial intelligence is rapidly transforming the technology landscape. "
        "Machine learning algorithms are now capable of processing vast amounts of data, "
        "identifying patterns, and making predictions with unprecedented accuracy. "
        "From healthcare diagnostics to financial forecasting, AI applications are "
        "revolutionizing industries worldwide. However, ethical considerations around "
        "privacy, bias, and transparency remain critical challenges that must be addressed "
        "as these technologies continue to evolve."
    )

    print("\n--- Testing Summarization ---")
    try:
        summary = pipeline.summarize([text], max_length=64)
        print(f"Summary: '{summary[0]}'")
    except Exception as e:
        print(f"Summarization failed: {e}")

    print("\n--- Testing Emotion ---")
    try:
        emotions = pipeline.predict_emotions([text])
        print(f"Emotions: {emotions[0]}")
    except Exception as e:
        print(f"Emotion prediction failed: {e}")

    print("\n--- Testing Topic ---")
    try:
        topic = pipeline.predict_topics([text])
        print(f"Topic: {topic[0]}")
    except Exception as e:
        print(f"Topic prediction failed: {e}")

if __name__ == "__main__":
    test_pipeline()
