"""Export the FLAN-T5 tokenizer to the artifacts directory for reproducible inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tokenizer to artifacts directory")
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-base",
        help="HuggingFace model name for the tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hf_tokenizer",
        help="Output directory for tokenizer files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Saving tokenizer to {output_dir}...")
    tokenizer.save_pretrained(str(output_dir))

    # Print tokenizer info
    print("\nTokenizer saved successfully!")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  BOS token: {tokenizer.bos_token} (id={getattr(tokenizer, 'bos_token_id', 'N/A')})")

    print("\nFiles created:")
    for file in sorted(output_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
