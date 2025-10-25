#!/usr/bin/env python
"""Script to train the BPE tokenizer on WikiText-103."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokenizer import train_tokenizer, test_tokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer/wikimini_32k",
        help="Output directory for tokenizer",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Training BPE Tokenizer")
    print("="*60)
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min frequency: {args.min_frequency}")
    print()
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
        show_progress=True,
    )
    
    # Test tokenizer
    print("\nTesting tokenizer...")
    test_tokenizer(tokenizer)
    
    print("\nTokenizer training complete!")
    print(f"Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()