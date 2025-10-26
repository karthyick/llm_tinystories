#!/usr/bin/env python3
"""
Train a custom tokenizer with 10K vocabulary for TinyStories.

This follows the proven approach from research:
- ALL successful implementations use 8K-10K vocab
- Results in 3-5× more exposure for article tokens
- Reduces model size (smaller embedding matrix)
- Better compression for TinyStories text
"""

import sys
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json

def train_tinystories_tokenizer(
    vocab_size=10000,
    output_dir="./tokenizer/tinystories_10k",
    min_frequency=2,
    dataset_split="train",
    max_samples=100000  # Sample 100K stories for training
):
    """
    Train a custom BPE tokenizer optimized for TinyStories.

    Args:
        vocab_size: Target vocabulary size (default: 10000)
        output_dir: Where to save the trained tokenizer
        min_frequency: Minimum frequency for a token to be included
        dataset_split: Which dataset split to use
        max_samples: Maximum number of samples to use for training
    """

    print("=" * 70)
    print("TRAINING CUSTOM TINYSTORIES TOKENIZER")
    print("=" * 70)
    print(f"Target vocabulary size: {vocab_size:,}")
    print(f"Output directory: {output_dir}")
    print(f"Min frequency: {min_frequency}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load TinyStories dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split=dataset_split, streaming=True)

    # Sample stories for training
    print(f"Sampling {max_samples:,} stories...")
    stories = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        stories.append(example['text'])
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1:,} stories...")

    print(f"✅ Loaded {len(stories):,} stories")
    print()

    # Initialize BPE tokenizer
    print("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<|endoftext|>",
            "<|padding|>",
            "<unk>",
        ],
        show_progress=True,
    )

    # Train tokenizer
    print(f"Training tokenizer on {len(stories):,} stories...")
    print("This may take a few minutes...")
    print()

    tokenizer.train_from_iterator(stories, trainer=trainer)

    # Add post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<|padding|>"),
        pad_token="<|padding|>"
    )

    # Enable truncation
    tokenizer.enable_truncation(max_length=512)

    # Save tokenizer
    print(f"Saving tokenizer to {output_dir}...")
    tokenizer_file = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    # Save config
    config = {
        "vocab_size": vocab_size,
        "model_type": "BPE",
        "dataset": "roneneldan/TinyStories",
        "min_frequency": min_frequency,
        "training_samples": len(stories),
        "special_tokens": {
            "pad_token": "<|padding|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<unk>",
        }
    }

    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Tokenizer saved to {output_dir}")
    print()

    # Test tokenizer
    print("=" * 70)
    print("TOKENIZER VERIFICATION")
    print("=" * 70)

    # Reload tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # Get actual vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    print()

    # Test encoding/decoding
    test_stories = [
        "Once upon a time there was a little girl named Lily.",
        "She was 3 years old and lived in a small house.",
        "One day, Lily found a big red ball in the park.",
    ]

    print("Test Encodings:")
    print("-" * 70)
    for story in test_stories:
        encoded = tokenizer.encode(story)
        decoded = tokenizer.decode(encoded.ids)

        print(f"Original:  {story}")
        print(f"Tokens:    {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
        print(f"Num tokens: {len(encoded.ids)}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {'✅' if decoded.strip() == story.strip() else '❌'}")
        print()

    # Check for article tokens
    print("Article Token Check:")
    print("-" * 70)

    articles = [" a", " the", " an"]
    for article in articles:
        encoded = tokenizer.encode(article)
        if len(encoded.ids) > 0:
            token_id = encoded.ids[0]
            decoded = tokenizer.decode([token_id])
            print(f"'{article}' → Token {token_id:5d} → '{decoded}' {'✅' if decoded.strip() == article.strip() else '⚠️'}")
        else:
            print(f"'{article}' → NOT FOUND ❌")

    print()
    print("=" * 70)
    print("✅ TOKENIZER TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Update your config to use this tokenizer:")
    print(f"   tokenizer_path: {output_dir}")
    print("2. Delete old cached data:")
    print("   rm -rf ./data/cache/*")
    print("3. Start training with the new tokenizer")
    print()
    print("Expected benefits:")
    print("  • 3-5× more exposure for article tokens")
    print("  • Smaller model (reduced embedding matrix)")
    print("  • Better compression for TinyStories")
    print("  • Articles will appear naturally in generation!")
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train custom TinyStories tokenizer")
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="Target vocabulary size (default: 10000)")
    parser.add_argument("--output_dir", type=str, default="./tokenizer/tinystories_10k",
                       help="Output directory for tokenizer")
    parser.add_argument("--min_frequency", type=int, default=2,
                       help="Minimum token frequency (default: 2)")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Max stories to use for training (default: 100000)")

    args = parser.parse_args()

    train_tinystories_tokenizer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency,
        max_samples=args.max_samples
    )
