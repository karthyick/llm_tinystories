#!/usr/bin/env python3
"""
Download and use a pre-trained TinyStories tokenizer from Hugging Face.

This downloads a tokenizer in the correct format (Hugging Face Tokenizers)
that's compatible with your codebase.
"""

from transformers import AutoTokenizer
import os
from pathlib import Path

def download_tinystories_tokenizer(output_dir="./tokenizer/tinystories_hf"):
    """
    Download a pre-trained TinyStories tokenizer from Hugging Face.

    Args:
        output_dir: Where to save the tokenizer
    """

    print("=" * 70)
    print("DOWNLOADING TINYSTORIES TOKENIZER FROM HUGGING FACE")
    print("=" * 70)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Try to download from roneneldan's models (original TinyStories authors)
    model_id = "roneneldan/TinyStories-33M"

    print(f"Downloading tokenizer from: {model_id}")
    print("This may take a minute...")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Save in the format your code expects
        tokenizer.save_pretrained(output_dir)

        print("✓ Tokenizer downloaded successfully!")
        print()

        # Get vocabulary info
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size:,}")
        print()

        # Test tokenizer
        test_texts = [
            "Once upon a time there was a little girl named Lily.",
            "She was 3 years old and lived in a small house.",
        ]

        print("Test Encodings:")
        print("-" * 70)
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)

            print(f"Original:  {text}")
            print(f"Tokens:    {encoded[:15]}{'...' if len(encoded) > 15 else ''}")
            print(f"Num tokens: {len(encoded)}")
            print(f"Decoded:   {decoded}")
            print()

        # Check for articles
        print("Article Token Check:")
        print("-" * 70)
        articles = [" a", " the", " an"]
        for article in articles:
            encoded = tokenizer.encode(article, add_special_tokens=False)
            if len(encoded) > 0:
                token_id = encoded[0]
                decoded = tokenizer.decode([token_id])
                print(f"'{article}' → Token {token_id:5d} → '{decoded}' {'✓' if article.strip() in decoded.strip() else '⚠️'}")
            else:
                print(f"'{article}' → NOT FOUND ✗")

        print()
        print("=" * 70)
        print("✓ DOWNLOAD COMPLETE!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Delete old cache:")
        print("   Remove-Item -Path './data/cache' -Recurse -Force")
        print()
        print("2. Update your config to use this tokenizer:")
        print(f"   tokenizer_path: {output_dir}")
        print()
        print("3. Start training:")
        print("   python train.py --config your_config.yaml")
        print()

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        print()
        print("Alternative: Train custom tokenizer")
        print("  python train_custom_tokenizer.py --vocab_size 10000")
        print()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TinyStories tokenizer from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./tokenizer/tinystories_hf",
                       help="Output directory for tokenizer")

    args = parser.parse_args()

    success = download_tinystories_tokenizer(args.output_dir)

    if not success:
        print("If download continues to fail, train a custom tokenizer instead:")
        print("  python train_custom_tokenizer.py --vocab_size 10000")
