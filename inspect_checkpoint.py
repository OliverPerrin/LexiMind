
import torch
import sys
from pathlib import Path

def inspect_checkpoint():
    path = "checkpoints/best.pt"
    print(f"Loading {path}...")
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Keys found: {len(state_dict)}")
        
        print("\n--- Head Keys ---")
        head_keys = [k for k in state_dict.keys() if "head" in k]
        for k in sorted(head_keys):
            print(k)
            
        print("\n--- Decoder Keys (Sample) ---")
        decoder_keys = [k for k in state_dict.keys() if "decoder" in k][:10]
        for k in sorted(decoder_keys):
            print(k)

        print("\n--- Checking for Cross Attention ---")
        if "decoder.layers.0.cross_attn.W_Q.weight" in state_dict:
            print("Found decoder.layers.0.cross_attn.W_Q.weight")
        else:
            print("MISSING decoder.layers.0.cross_attn.W_Q.weight")

    except Exception as e:
        print(f"Failed to load: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
