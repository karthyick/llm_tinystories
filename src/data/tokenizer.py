"""Tokenizer training and loading utilities for WikiMini model.

This module provides functions to:
1. Train a BPE tokenizer on WikiText-103
2. Load a trained tokenizer from disk
3. Test tokenizer functionality
"""

import os
from pathlib import Path
from typing import Optional, List
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


def train_tokenizer(
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str = "./tokenizer/wikimini_32k",
    show_progress: bool = True,
) -> Tokenizer:
    """Train a BPE tokenizer on WikiText-103 dataset.

    Args:
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for tokens
        output_dir: Directory to save the trained tokenizer
        show_progress: Whether to show progress during training

    Returns:
        Trained tokenizer
    """
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenization (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer
    special_tokens = [
        "<unk>",  # Unknown token
        "<s>",    # Begin of sentence
        "</s>",   # End of sentence
        "<pad>",  # Padding token
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )

    # Load WikiText-103 dataset
    logger.info("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Create iterator for training
    def batch_iterator(batch_size: int = 1000):
        """Yield batches of text for training."""
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield batch["text"]

    # Train tokenizer
    logger.info("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Add post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<pad>"),
        pad_token="<pad>",
    )

    # Enable truncation
    tokenizer.enable_truncation(max_length=2048)

    # Save tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_file = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    logger.info(f"Tokenizer saved to {tokenizer_file}")

    # Save config
    config = {
        "vocab_size": vocab_size,
        "model_type": "BPE",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    }

    import json
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_file}")

    return tokenizer


def load_tokenizer(tokenizer_path: str, return_wrapper: bool = True):
    """Load a trained tokenizer from disk.

    Args:
        tokenizer_path: Path to the tokenizer directory or file
        return_wrapper: If True, returns TokenizerWrapper (default), else raw Tokenizer

    Returns:
        Loaded tokenizer (wrapped by default for compatibility)
    """
    tokenizer_path = Path(tokenizer_path)

    # Handle both directory and file paths
    if tokenizer_path.is_dir():
        tokenizer_file = tokenizer_path / "tokenizer.json"
    else:
        tokenizer_file = tokenizer_path

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    logger.info(f"Loading tokenizer from {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # Return wrapped version for easier use (supports len(), etc.)
    if return_wrapper:
        return TokenizerWrapper(tokenizer)

    return tokenizer


def test_tokenizer(tokenizer: Tokenizer) -> None:
    """Test tokenizer with sample text.

    Args:
        tokenizer: Tokenizer to test
    """
    print("\n" + "="*70)
    print(" "*25 + "Tokenizer Test")
    print("="*70)

    # Get vocab info
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nVocabulary size: {vocab_size:,}")

    # Test special tokens
    print("\nSpecial tokens:")
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token:8s} -> ID {token_id}")

    # Test encoding/decoding
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "WikiText-103 is a large-scale language modeling benchmark.",
    ]

    print("\nEncoding/Decoding tests:")
    print("-" * 70)

    for i, text in enumerate(test_texts, 1):
        # Encode
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        ids = encoding.ids

        # Decode
        decoded = tokenizer.decode(ids)

        print(f"\nTest {i}:")
        print(f"  Original: {text}")
        print(f"  Tokens:   {len(tokens)}")
        print(f"  IDs:      {ids[:10]}..." if len(ids) > 10 else f"  IDs:      {ids}")
        print(f"  Decoded:  {decoded}")

        # Check round-trip
        if decoded.strip() == text.strip():
            print("  ✅ Round-trip successful")
        else:
            print("  ⚠️  Round-trip differs slightly (common with BPE)")

    # Test batch encoding
    print("\n\nBatch encoding test:")
    print("-" * 70)
    encodings = tokenizer.encode_batch(test_texts)
    print(f"  Batch size: {len(encodings)}")
    print(f"  Token counts: {[len(enc.ids) for enc in encodings]}")

    print("\n" + "="*70)
    print(" "*25 + "✅ Test Complete")
    print("="*70 + "\n")


# Wrapper class for compatibility with HuggingFace-style interface
class TokenizerWrapper:
    """Wrapper to make tokenizers.Tokenizer compatible with expected interface."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._vocab_size = tokenizer.get_vocab_size()

        # Get special token IDs - support multiple formats
        # Try standard format first, then TinyStories custom format
        self.pad_token_id = (
            tokenizer.token_to_id("<pad>") or
            tokenizer.token_to_id("<|padding|>") or
            0  # Fallback to 0 if not found
        )
        self.bos_token_id = (
            tokenizer.token_to_id("<s>") or
            tokenizer.token_to_id("<|startoftext|>")
        )
        self.eos_token_id = (
            tokenizer.token_to_id("</s>") or
            tokenizer.token_to_id("<|endoftext|>")
        )
        self.unk_token_id = tokenizer.token_to_id("<unk>")

    def __call__(self, text, **kwargs):
        """Encode text (callable interface)."""
        if isinstance(text, str):
            return self.tokenizer.encode(text).ids
        elif isinstance(text, list):
            return [self.tokenizer.encode(t).ids for t in text]

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self):
        """Return vocabulary size."""
        return self._vocab_size

    @property
    def vocab_size(self):
        """Vocabulary size property."""
        return self._vocab_size


def create_tokenizer_wrapper(tokenizer_path: str) -> TokenizerWrapper:
    """Create a wrapped tokenizer for easier use.

    Args:
        tokenizer_path: Path to tokenizer directory or file

    Returns:
        TokenizerWrapper instance
    """
    tokenizer = load_tokenizer(tokenizer_path, return_wrapper=False)
    return TokenizerWrapper(tokenizer)
