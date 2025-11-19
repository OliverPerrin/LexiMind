
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

def debug_pipeline():
    print("Loading pipeline...")
    pipeline, _ = create_inference_pipeline(
        tokenizer_dir="artifacts/hf_tokenizer/",
        checkpoint_path="checkpoints/best.pt",
        labels_path="artifacts/labels.json",
    )
    
    tokenizer = pipeline.tokenizer
    print(f"BOS ID: {tokenizer.bos_token_id}")
    print(f"EOS ID: {tokenizer.eos_token_id}")
    print(f"PAD ID: {tokenizer.pad_token_id}")
    
    text = "Artificial intelligence is rapidly transforming the technology landscape."
    
    print("\n--- Input Analysis ---")
    encoded = tokenizer.encode(text)
    print(f"Encoded input: {encoded}")
    print(f"Decoded input: {tokenizer.decode(encoded)}")
    
    print("\n--- Model Generation Debug ---")
    # Manually run the summarization steps
    batch = pipeline.preprocessor.batch_encode([text])
    batch = pipeline._batch_to_device(batch)
    
    src_ids = batch.input_ids
    src_mask = batch.attention_mask
    
    print(f"Source IDs shape: {src_ids.shape}")
    print(f"Source IDs: {src_ids}")
    
    with torch.inference_mode():
        encoder_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2) if src_mask is not None else None
        memory = pipeline.model.encoder(src_ids, mask=encoder_mask)
        
        # Try decoding with BOS as start
        print("\n--- Decoding with BOS start ---")
        generated_bos = pipeline.model.decoder.greedy_decode(
            memory=memory,
            max_len=20,
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            device=pipeline.device,
            min_len=0
        )
        print(f"Generated IDs (BOS start): {generated_bos.tolist()}")
        print(f"Decoded (BOS start): {tokenizer.decode_batch(generated_bos.tolist())}")

        # Try decoding with [BOS, FirstContentToken] start
        print("\n--- Decoding with [BOS, FirstContentToken] start ---")
        bos_id = tokenizer.bos_token_id
        first_content_id = src_ids[0, 1] # Skip BOS in input
        print(f"First content token ID: {first_content_id} ({tokenizer.decode([first_content_id])})")
        
        generated = torch.tensor([[bos_id, first_content_id]], dtype=torch.long, device=pipeline.device)
        
        for _ in range(20):
            logits = pipeline.model.decoder.forward(generated, memory, collect_attn=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        print(f"Generated IDs ([BOS, Content] start): {generated.tolist()}")
        print(f"Decoded ([BOS, Content] start): {tokenizer.decode_batch(generated.tolist())}")


if __name__ == "__main__":
    debug_pipeline()
