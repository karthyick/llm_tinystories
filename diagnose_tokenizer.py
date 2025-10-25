#!/usr/bin/env python3
"""
Tokenizer Root Cause Analysis

This script performs deep investigation into why articles are missing from generation.
Tests every aspect of the tokenizer to find where articles get lost.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.tokenizer import load_tokenizer
from datasets import load_dataset
import json

print("="*80)
print("TOKENIZER ROOT CAUSE ANALYSIS")
print("="*80)


def test_1_basic_article_encoding():
    """Test 1: Can tokenizer encode/decode articles correctly?"""
    print("\n" + "="*80)
    print("TEST 1: Basic Article Encoding")
    print("="*80)

    tokenizer = load_tokenizer('./tokenizer/wikimini_32k')

    test_cases = [
        "a",
        "the",
        "an",
        "a little",
        "the boy",
        "an apple",
        "there was a",
        "Once upon a time there was a little girl",
    ]

    print("\nEncoding/Decoding Test:")
    print("-" * 80)

    for text in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        match = "✅" if decoded.strip() == text.strip() else "❌"

        print(f"{match} Input:   '{text}'")
        print(f"   Tokens:  {tokens}")
        print(f"   Decoded: '{decoded}'")

        # Show each token
        print(f"   Token breakdown:")
        for i, tok_id in enumerate(tokens):
            tok_text = tokenizer.decode([tok_id])
            print(f"      [{i}] id={tok_id:5d} → '{tok_text}'")
        print()

    return tokenizer


def test_2_vocabulary_search(tokenizer):
    """Test 2: Search vocabulary for article-related tokens."""
    print("\n" + "="*80)
    print("TEST 2: Vocabulary Search for Articles")
    print("="*80)

    # Try to access vocabulary
    print("\nSearching for article tokens in vocabulary...")
    print("-" * 80)

    # Test different ways articles might be encoded
    search_patterns = [
        "a",
        " a",
        "a ",
        " a ",
        "the",
        " the",
        "the ",
        " the ",
        "an",
        " an",
        "an ",
        " an ",
    ]

    print("\nDirect token ID lookup:")
    for pattern in search_patterns:
        tokens = tokenizer.encode(pattern)
        if tokens:
            print(f"  '{pattern}' → token_ids={tokens} → decodes to '{tokenizer.decode(tokens)}'")

    # Check specific token IDs we saw in validation
    print("\n\nChecking specific token IDs from validation output:")
    print("-" * 80)

    known_ids = {
        68: "a",
        13131: "the",
        31065: "little",
        16298: "girl",
        7798: "boy",
    }

    for tok_id, expected in known_ids.items():
        try:
            decoded = tokenizer.decode([tok_id])
            print(f"  Token ID {tok_id:5d} (expected '{expected}') → '{decoded}'")
        except Exception as e:
            print(f"  Token ID {tok_id:5d} → ERROR: {e}")


def test_3_tinystories_data_tokenization(tokenizer):
    """Test 3: How is actual TinyStories training data tokenized?"""
    print("\n" + "="*80)
    print("TEST 3: TinyStories Training Data Tokenization")
    print("="*80)

    print("\nLoading TinyStories dataset (first 10 samples)...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        print("\nAnalyzing how articles appear in training data:")
        print("-" * 80)

        samples_checked = 0
        article_token_stats = {
            'a': [],
            'the': [],
            'an': []
        }

        for i, sample in enumerate(dataset):
            if i >= 10:
                break

            text = sample['text']
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)

            samples_checked += 1

            # Check if articles are in original text
            has_a = ' a ' in text.lower()
            has_the = ' the ' in text.lower()
            has_an = ' an ' in text.lower()

            # Check if articles are in decoded text
            decoded_has_a = ' a ' in decoded.lower()
            decoded_has_the = ' the ' in decoded.lower()
            decoded_has_an = ' an ' in decoded.lower()

            print(f"\nSample {i+1}:")
            print(f"  Original has articles: a={has_a}, the={has_the}, an={has_an}")
            print(f"  Decoded has articles:  a={decoded_has_a}, the={decoded_has_the}, an={decoded_has_an}")

            if has_a != decoded_has_a or has_the != decoded_has_the or has_an != decoded_has_an:
                print(f"  ⚠️  MISMATCH DETECTED!")
                print(f"  Original: '{text[:100]}...'")
                print(f"  Decoded:  '{decoded[:100]}...'")

            # Show tokenization of first sentence
            first_sentence = text.split('.')[0] + '.'
            tokens_first = tokenizer.encode(first_sentence)
            print(f"  First sentence tokens ({len(tokens_first)} tokens):")
            print(f"    Original: '{first_sentence}'")
            print(f"    Tokens: {tokens_first[:20]}...")  # Show first 20
            print(f"    Token breakdown:")
            for j, tok_id in enumerate(tokens_first[:15]):  # Show first 15 in detail
                tok_text = tokenizer.decode([tok_id])
                print(f"      [{j}] id={tok_id:5d} → '{tok_text}'")

        print(f"\n✅ Checked {samples_checked} samples")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Skipping this test...")


def test_4_article_context_patterns(tokenizer):
    """Test 4: How does tokenizer handle common article patterns?"""
    print("\n" + "="*80)
    print("TEST 4: Common Article Patterns")
    print("="*80)

    # These are patterns from TinyStories that SHOULD have articles
    patterns = [
        "Once upon a time",
        "there was a little",
        "there was a girl",
        "there was a boy",
        "Once upon a time there was a little girl named",
        "She was a good",
        "He was a brave",
        "They saw a big",
        "It was a beautiful",
    ]

    print("\nAnalyzing common story patterns:")
    print("-" * 80)

    for pattern in patterns:
        tokens = tokenizer.encode(pattern)
        decoded = tokenizer.decode(tokens)

        # Count articles in original vs decoded
        orig_a_count = pattern.lower().count(' a ')
        decoded_a_count = decoded.lower().count(' a ')

        match = "✅" if orig_a_count == decoded_a_count else "❌"

        print(f"\n{match} Pattern: '{pattern}'")
        print(f"   Articles in original: {orig_a_count}")
        print(f"   Articles in decoded:  {decoded_a_count}")
        print(f"   Tokens ({len(tokens)}): {tokens}")
        print(f"   Token breakdown:")
        for i, tok_id in enumerate(tokens):
            tok_text = tokenizer.decode([tok_id])
            is_article = tok_text.strip().lower() in ['a', 'an', 'the']
            marker = "👉" if is_article else "  "
            print(f"     {marker} [{i}] id={tok_id:5d} → '{tok_text}'")


def test_5_whitespace_handling(tokenizer):
    """Test 5: How does tokenizer handle whitespace around articles?"""
    print("\n" + "="*80)
    print("TEST 5: Whitespace Handling")
    print("="*80)

    test_cases = [
        ("a", "just 'a'"),
        (" a", "space + a"),
        ("a ", "a + space"),
        (" a ", "space + a + space"),
        ("was a little", "was a little"),
        ("was  a  little", "was  a  little (double space)"),
        ("wasa little", "wasa little (no space)"),
    ]

    print("\nWhitespace sensitivity test:")
    print("-" * 80)

    for text, description in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        match = "✅" if decoded == text else "❌"

        print(f"\n{match} {description}")
        print(f"   Input:   '{text}' (len={len(text)})")
        print(f"   Tokens:  {tokens}")
        print(f"   Decoded: '{decoded}' (len={len(decoded)})")


def test_6_tokenizer_config(tokenizer):
    """Test 6: Check tokenizer configuration."""
    print("\n" + "="*80)
    print("TEST 6: Tokenizer Configuration")
    print("="*80)

    print("\nTokenizer properties:")
    print("-" * 80)

    # Get tokenizer info
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Type: {type(tokenizer)}")

    # Check for special tokens
    try:
        print(f"  BOS token: {tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else 'N/A'}")
        print(f"  EOS token: {tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else 'N/A'}")
        print(f"  PAD token: {tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else 'N/A'}")
        print(f"  UNK token: {tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else 'N/A'}")
    except:
        print("  (Special tokens not accessible)")

    # Try to access tokenizer internals
    print("\n  Tokenizer attributes:")
    for attr in dir(tokenizer):
        if not attr.startswith('_') and not callable(getattr(tokenizer, attr)):
            try:
                value = getattr(tokenizer, attr)
                if not isinstance(value, (list, dict)) or len(str(value)) < 100:
                    print(f"    {attr}: {value}")
            except:
                pass


def test_7_compare_with_expected(tokenizer):
    """Test 7: Compare with what we EXPECT tokenizer to do."""
    print("\n" + "="*80)
    print("TEST 7: Expected vs Actual Behavior")
    print("="*80)

    print("\nWhat we EXPECT for 'there was a little girl':")
    print("-" * 80)
    print("  Expected tokens (roughly):")
    print("    'there' (or ' there')")
    print("    ' was' (or 'was')")
    print("    ' a' ← SHOULD BE ITS OWN TOKEN")
    print("    ' little'")
    print("    ' girl'")

    print("\nWhat we ACTUALLY get:")
    print("-" * 80)
    text = "there was a little girl"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"  Input: '{text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: '{decoded}'")
    print(f"\n  Token-by-token breakdown:")
    for i, tok_id in enumerate(tokens):
        tok_text = tokenizer.decode([tok_id])
        is_article = tok_text.strip().lower() in ['a', 'an', 'the']

        if is_article:
            print(f"    ✅ [{i}] id={tok_id:5d} → '{tok_text}' ← ARTICLE TOKEN")
        else:
            print(f"       [{i}] id={tok_id:5d} → '{tok_text}'")

    # Check if 'a' appears as its own token
    has_article_token = any(
        tokenizer.decode([tok_id]).strip().lower() in ['a', 'an', 'the']
        for tok_id in tokens
    )

    print(f"\n  Has article as separate token: {'✅ YES' if has_article_token else '❌ NO'}")

    if not has_article_token:
        print("\n  ⚠️  PROBLEM IDENTIFIED:")
        print("     Article 'a' is NOT appearing as its own token!")
        print("     It may be merged with adjacent words during tokenization.")


def main():
    """Run all diagnostic tests."""

    print("\nThis script will test:")
    print("  1. Basic article encoding/decoding")
    print("  2. Vocabulary search for articles")
    print("  3. TinyStories data tokenization")
    print("  4. Common article patterns")
    print("  5. Whitespace handling")
    print("  6. Tokenizer configuration")
    print("  7. Expected vs actual behavior")
    print()
    input("Press Enter to start diagnostics...")

    try:
        # Run all tests
        tokenizer = test_1_basic_article_encoding()
        test_2_vocabulary_search(tokenizer)
        test_3_tinystories_data_tokenization(tokenizer)
        test_4_article_context_patterns(tokenizer)
        test_5_whitespace_handling(tokenizer)
        test_6_tokenizer_config(tokenizer)
        test_7_compare_with_expected(tokenizer)

        # Final summary
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)
        print("\n📋 Summary:")
        print("  Check the output above for any ❌ marks or warnings")
        print("  Key things to look for:")
        print("    1. Does 'a' decode correctly in TEST 1?")
        print("    2. Is there a token ID for ' a' in TEST 2?")
        print("    3. Are articles lost in TinyStories samples (TEST 3)?")
        print("    4. Are article patterns tokenized correctly (TEST 4)?")
        print("    5. Does TEST 7 show 'a' as separate token?")
        print("\n  The test with the most ❌ marks is likely the root cause!")

    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
