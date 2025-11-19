
import sys
from pathlib import Path
import torch
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import ModelConfig
from src.data.tokenization import Tokenizer, TokenizerConfig
from src.models.factory import build_multitask_model
from src.utils.io import load_state
from src.utils.labels import load_label_metadata
from src.inference.pipeline import InferencePipeline, InferenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_pipeline():
    labels = load_label_metadata("artifacts/labels.json")
    tokenizer = Tokenizer(TokenizerConfig(pretrained_model_name="artifacts/hf_tokenizer"))
    
    for heads in [4, 8, 16]:
        print(f"\n============================================")
        print(f"Testing num_heads={heads}")
        print(f"============================================")
        try:
            cfg = ModelConfig(num_attention_heads=heads)
            model = build_multitask_model(
                tokenizer,
                num_emotions=labels.emotion_size,
                num_topics=labels.topic_size,
                config=cfg,
            )
            load_state(model, "checkpoints/best.pt")
            
            # Tie weights (as per my previous fix)
            if hasattr(model.decoder, "output_projection") and hasattr(model.decoder, "embedding"):
                model.decoder.output_projection.weight = model.decoder.embedding.weight
                
            pipeline = InferencePipeline(
                model=model,
                tokenizer=tokenizer,
                config=InferenceConfig(device="cpu"),
                emotion_labels=labels.emotion,
                topic_labels=labels.topic,
                device="cpu"
            )
            
            text = "Artificial intelligence is rapidly transforming the technology landscape."
            summary = pipeline.summarize([text], max_length=20)
            print(f"Summary: '{summary[0]}'")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_pipeline()
