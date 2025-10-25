#!/usr/bin/env python3
"""
Dataset Pipeline Root Cause Analysis

Tests every aspect of how training data is prepared and fed to the model.
This will reveal if articles are being lost during data processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data.tokenizer import load_tokenizer
from src.data.dataset import TinyStoriesDataset
from datasets import load_dataset
from collections import Counter
import numpy as np

print("="*80)
print("DATASET PIPELINE ROOT CAUSE ANALYSIS")
print("="*80)


def test_1_raw_data_has_articles():
    """Test 1: Verify raw TinyStories data has articles."""
    print("\n" + "="*80)
    print("TEST 1: Raw TinyStories Data Quality")
    print("="*80)

    print("\nLoading TinyStories dataset (100 samples)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    article_counts = {'a': 0, 'the': 0, 'an': 0}
    total_words = 0
    samples_checked = 0

    for i, sample in enumerate(dataset):
        if i >= 100:
            break

        text = sample['text'].lower()
        words = text.split()
        total_words += len(words)

        article_counts['a'] += words.count('a')
        article_counts['the'] += words.count('the')
        article_counts['an'] += words.count('an')

        samples_checked += 1

    print(f"\n✅ Checked {samples_checked} samples, {total_words} words")
    print(f"\nArticle statistics:")
    print(f"  'a':   {article_counts['a']:5d} occurrences ({article_counts['a']/total_words*100:.2f}% of words)")
    print(f"  'the': {article_counts['the']:5d} occurrences ({article_counts['the']/total_words*100:.2f}% of words)")
    print(f"  'an':  {article_counts['an']:5d} occurrences ({article_counts['an']/total_words*100:.2f}% of words)")
    print(f"  Total: {sum(article_counts.values()):5d} articles ({sum(article_counts.values())/total_words*100:.2f}% of all words)")

    if sum(article_counts.values()) / total_words < 0.05:
        print("\n❌ WARNING: Articles are less than 5% of words - seems low!")
    else:
        print("\n✅ Article frequency looks normal")


def test_2_tokenized_data_preserves_articles(tokenizer):
    """Test 2: Check if tokenization preserves articles."""
    print("\n" + "="*80)
    print("TEST 2: Tokenization Preserves Articles")
    print("="*80)

    print("\nLoading and tokenizing TinyStories samples...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Track article token IDs (with leading space, as used in context)
    article_token_ids = {
        262: ' a',    # ' a' with leading space
        264: ' the',  # ' the' with leading space
        389: ' an',   # ' an' with leading space
    }

    article_token_counts = {262: 0, 264: 0, 389: 0}
    total_tokens = 0
    samples_checked = 0

    for i, sample in enumerate(dataset):
        if i >= 100:
            break

        text = sample['text']
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)

        for token_id in tokens:
            if token_id in article_token_ids:
                article_token_counts[token_id] += 1

        samples_checked += 1

    print(f"\n✅ Tokenized {samples_checked} samples, {total_tokens} tokens")
    print(f"\nArticle token statistics:")
    for token_id, token_text in article_token_ids.items():
        count = article_token_counts[token_id]
        pct = count / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  Token {token_id:5d} ('{token_text}'):  {count:5d} occurrences ({pct:.2f}% of tokens)")

    total_article_tokens = sum(article_token_counts.values())
    print(f"  Total article tokens: {total_article_tokens:5d} ({total_article_tokens/total_tokens*100:.2f}% of all tokens)")

    if total_article_tokens / total_tokens < 0.03:
        print("\n❌ WARNING: Article tokens are less than 3% - seems low!")
    else:
        print("\n✅ Article token frequency looks normal")

    return article_token_ids


def test_3_dataset_class_processing(tokenizer):
    """Test 3: Check what TinyStoriesDataset class does to the data."""
    print("\n" + "="*80)
    print("TEST 3: TinyStoriesDataset Class Processing")
    print("="*80)

    print("\nCreating TinyStoriesDataset...")

    try:
        # Create dataset (will use cache if available, or process full split)
        dataset = TinyStoriesDataset(
            tokenizer=tokenizer,
            split='train',
            max_seq_len=512,
        )

        print(f"✅ Dataset created: {len(dataset)} sequences")
        print(f"   (analyzing first 1000 for speed...)")

        # Sample some sequences and check for article tokens
        article_token_ids = {262, 264, 389}  # ' a', ' the', ' an'

        print("\nChecking first 10 sequences for article tokens...")
        print("-" * 80)

        for i in range(min(10, len(dataset))):
            sequence = dataset[i]

            # Count article tokens in this sequence
            if isinstance(sequence, dict):
                tokens = sequence['input_ids']
            else:
                tokens = sequence

            # Convert to list to avoid tensor/Python type mismatch
            if hasattr(tokens, 'tolist'):
                token_list = tokens.tolist()
            else:
                token_list = list(tokens)

            article_count = sum(1 for token_id in token_list if token_id in article_token_ids)
            total_tokens = len(tokens)
            pct = article_count / total_tokens * 100

            print(f"Sequence {i:2d}: {article_count:3d} article tokens / {total_tokens:3d} total ({pct:.1f}%)")

            # Show first few tokens
            print(f"  First 20 tokens: {tokens[:20].tolist() if hasattr(tokens, 'tolist') else tokens[:20]}")

            # Decode and show
            if hasattr(tokens, 'tolist'):
                decoded = tokenizer.decode(tokens.tolist())
            else:
                decoded = tokenizer.decode(tokens)
            print(f"  Decoded text: '{decoded[:100]}...'")
            print()

        # Statistics across first 1000 sequences (for speed)
        print("\nStatistics across first 1000 sequences:")
        print("-" * 80)

        total_article_tokens = 0
        total_tokens = 0
        sequences_with_articles = 0
        num_to_check = min(1000, len(dataset))

        for i in range(num_to_check):
            sequence = dataset[i]

            if isinstance(sequence, dict):
                tokens = sequence['input_ids']
            else:
                tokens = sequence

            # Convert to list to avoid tensor/Python type mismatch
            if hasattr(tokens, 'tolist'):
                token_list = tokens.tolist()
            else:
                token_list = list(tokens)

            article_count = sum(1 for token_id in token_list if token_id in article_token_ids)
            total_article_tokens += article_count
            total_tokens += len(tokens)

            if article_count > 0:
                sequences_with_articles += 1

        print(f"Total sequences checked: {num_to_check}")
        print(f"Sequences with articles: {sequences_with_articles} ({sequences_with_articles/num_to_check*100:.1f}%)")
        print(f"Total article tokens: {total_article_tokens} ({total_article_tokens/total_tokens*100:.2f}% of all tokens)")

        if total_article_tokens / total_tokens < 0.03:
            print("\n❌ PROBLEM: Article tokens are less than 3% of all tokens!")
            print("   This is too low - articles should be ~4-6% of tokens")
            print("   Issue is in dataset.py processing!")
        else:
            print("\n✅ Article token frequency looks normal in dataset")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


def test_4_token_distribution_comparison(tokenizer):
    """Test 4: Compare token distributions between raw data and dataset class."""
    print("\n" + "="*80)
    print("TEST 4: Token Distribution Comparison")
    print("="*80)

    print("\nComparing raw tokenization vs dataset class output...")

    # Get tokens from raw data
    print("\n1. Raw tokenization (direct from TinyStories):")
    print("-" * 80)

    dataset_raw = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    raw_token_counts = Counter()
    raw_total = 0

    for i, sample in enumerate(dataset_raw):
        if i >= 100:
            break
        tokens = tokenizer.encode(sample['text'])
        raw_token_counts.update(tokens)
        raw_total += len(tokens)

    # Show top tokens
    print(f"Total tokens: {raw_total}")
    print(f"\nTop 20 most frequent tokens:")
    for token_id, count in raw_token_counts.most_common(20):
        token_text = tokenizer.decode([token_id])
        pct = count / raw_total * 100
        print(f"  Token {token_id:5d} ('{token_text[:20]:20s}'): {count:6d} ({pct:.2f}%)")

    # Check specific article tokens
    print(f"\nArticle token frequencies:")
    for token_id in [262, 264, 389]:  # ' a', ' the', ' an'
        count = raw_token_counts[token_id]
        token_text = tokenizer.decode([token_id])
        pct = count / raw_total * 100 if raw_total > 0 else 0
        print(f"  Token {token_id:5d} ('{token_text}'): {count:6d} ({pct:.2f}%)")

    # Get tokens from dataset class
    print("\n2. Dataset class output:")
    print("-" * 80)

    try:
        dataset_processed = TinyStoriesDataset(
            tokenizer=tokenizer,
            split='train',
            max_seq_len=512,
        )

        print(f"Dataset loaded: {len(dataset_processed)} sequences")
        print(f"Analyzing first 100 sequences...")

        processed_token_counts = Counter()
        processed_total = 0
        num_to_check = min(100, len(dataset_processed))

        for i in range(num_to_check):
            sequence = dataset_processed[i]
            if isinstance(sequence, dict):
                tokens = sequence['input_ids']
            else:
                tokens = sequence

            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()

            processed_token_counts.update(tokens)
            processed_total += len(tokens)

        print(f"Total tokens: {processed_total}")
        print(f"\nTop 20 most frequent tokens:")
        for token_id, count in processed_token_counts.most_common(20):
            token_text = tokenizer.decode([token_id])
            pct = count / processed_total * 100
            print(f"  Token {token_id:5d} ('{token_text[:20]:20s}'): {count:6d} ({pct:.2f}%)")

        print(f"\nArticle token frequencies:")
        for token_id in [262, 264, 389]:  # ' a', ' the', ' an'
            count = processed_token_counts[token_id]
            token_text = tokenizer.decode([token_id])
            pct = count / processed_total * 100 if processed_total > 0 else 0
            print(f"  Token {token_id:5d} ('{token_text}'): {count:6d} ({pct:.2f}%)")

        # Compare
        print("\n3. Comparison:")
        print("-" * 80)
        for token_id in [262, 264, 389]:
            raw_count = raw_token_counts[token_id]
            processed_count = processed_token_counts[token_id]
            raw_pct = raw_count / raw_total * 100 if raw_total > 0 else 0
            processed_pct = processed_count / processed_total * 100 if processed_total > 0 else 0
            token_text = tokenizer.decode([token_id])

            match = "✅" if abs(raw_pct - processed_pct) < 1.0 else "❌"
            print(f"{match} Token {token_id} ('{token_text}'):")
            print(f"     Raw:       {raw_pct:.2f}%")
            print(f"     Processed: {processed_pct:.2f}%")
            print(f"     Difference: {processed_pct - raw_pct:+.2f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_5_check_sequence_packing(tokenizer):
    """Test 5: Check if sequence packing affects article tokens."""
    print("\n" + "="*80)
    print("TEST 5: Sequence Packing Analysis")
    print("="*80)

    print("\nChecking how sequence packing handles articles...")

    # Load a few samples and check how they're packed
    dataset_raw = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print("\nOriginal samples before packing:")
    print("-" * 80)

    samples = []
    for i, sample in enumerate(dataset_raw):
        if i >= 5:
            break
        text = sample['text']
        tokens = tokenizer.encode(text)

        article_count = sum(1 for t in tokens if t in [262, 264, 389])
        samples.append((text, tokens, article_count))

        print(f"\nSample {i+1}:")
        print(f"  Text: '{text[:100]}...'")
        print(f"  Tokens: {len(tokens)} total, {article_count} articles ({article_count/len(tokens)*100:.1f}%)")

    # Now check the dataset class
    print("\n\nAfter dataset processing:")
    print("-" * 80)

    try:
        dataset_processed = TinyStoriesDataset(
            tokenizer=tokenizer,
            split='train',
            max_seq_len=512,
        )

        print(f"Dataset loaded: {len(dataset_processed)} sequences")
        print(f"Showing first 5 sequences...")

        for i in range(min(5, len(dataset_processed))):
            sequence = dataset_processed[i]
            if isinstance(sequence, dict):
                tokens = sequence['input_ids']
            else:
                tokens = sequence

            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()

            article_count = sum(1 for t in tokens if t in [262, 264, 389])

            print(f"\nSequence {i+1}:")
            print(f"  Tokens: {len(tokens)} total, {article_count} articles ({article_count/len(tokens)*100:.1f}%)")
            print(f"  First 30 tokens: {tokens[:30]}")

            # Check for padding or special handling
            if 0 in tokens:  # Padding token
                padding_count = tokens.count(0)
                print(f"  Padding tokens: {padding_count}")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_6_validate_training_batch(tokenizer):
    """Test 6: Check what a training batch actually looks like."""
    print("\n" + "="*80)
    print("TEST 6: Training Batch Validation")
    print("="*80)

    print("\nCreating a training batch to see what model actually receives...")

    try:
        from torch.utils.data import DataLoader

        dataset = TinyStoriesDataset(
            tokenizer=tokenizer,
            split='train',
            max_seq_len=512,
        )

        print(f"Dataset loaded: {len(dataset)} sequences")
        print(f"Creating DataLoader with batch_size=4...")

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Get first batch
        batch = next(iter(dataloader))

        print(f"\n✅ Batch created")
        print(f"Batch shape: {batch.shape if not isinstance(batch, dict) else batch['input_ids'].shape}")

        # Extract tokens
        if isinstance(batch, dict):
            batch_tokens = batch['input_ids']
        else:
            batch_tokens = batch

        print(f"\nAnalyzing batch content:")
        print("-" * 80)

        for i in range(batch_tokens.shape[0]):
            sequence = batch_tokens[i]

            if hasattr(sequence, 'tolist'):
                tokens = sequence.tolist()
            else:
                tokens = list(sequence)

            article_count = sum(1 for t in tokens if t in [262, 264, 389])
            non_pad_count = sum(1 for t in tokens if t != 0)

            print(f"\nBatch item {i}:")
            print(f"  Total length: {len(tokens)}")
            print(f"  Non-padding tokens: {non_pad_count}")
            print(f"  Article tokens: {article_count} ({article_count/non_pad_count*100:.1f}% of non-padding)")
            print(f"  First 30 tokens: {tokens[:30]}")

            # Decode
            decoded = tokenizer.decode([t for t in tokens if t != 0])
            print(f"  Decoded: '{decoded[:100]}...'")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostic tests."""

    print("\nThis script will test:")
    print("  1. Raw TinyStories data quality")
    print("  2. Tokenization preserves articles")
    print("  3. Dataset class processing")
    print("  4. Token distribution comparison")
    print("  5. Sequence packing analysis")
    print("  6. Training batch validation")
    print()
    input("Press Enter to start diagnostics...")

    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = load_tokenizer('./tokenizer/wikimini_32k')
        print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")

        # Run all tests
        test_1_raw_data_has_articles()
        test_2_tokenized_data_preserves_articles(tokenizer)
        test_3_dataset_class_processing(tokenizer)
        test_4_token_distribution_comparison(tokenizer)
        test_5_check_sequence_packing(tokenizer)
        test_6_validate_training_batch(tokenizer)

        # Final summary
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)
        print("\n📋 Summary:")
        print("  Check the output above for any ❌ marks")
        print("  Key findings to look for:")
        print("    1. Are articles present in raw data? (TEST 1)")
        print("    2. Do articles survive tokenization? (TEST 2)")
        print("    3. Does dataset class preserve articles? (TEST 3)")
        print("    4. Do token distributions match? (TEST 4)")
        print("    5. Does packing affect articles? (TEST 5)")
        print("    6. Are articles in training batches? (TEST 6)")
        print("\n  If articles disappear in any test, that's the root cause!")

    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
