#!/usr/bin/env python3
"""
Model Root Cause Analysis

Tests what the model actually learned about articles.
Since we know the training data has articles, this will reveal
if the model learned to predict them correctly in context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer
import numpy as np

print("="*80)
print("MODEL ARTICLE LEARNING DIAGNOSTIC")
print("="*80)


def load_model(checkpoint_path: str, tokenizer_path: str):
    """Load trained model and tokenizer."""
    print("\nLoading model and tokenizer...")

    tokenizer = load_tokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']['model']
    else:
        config = {
            'vocab_size': len(tokenizer),
            'n_positions': 512,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'd_ff': 3072,
            'dropout': 0.1,
        }

    config['vocab_size'] = len(tokenizer)
    model = WikiMiniModel(config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    print(f"✅ Device: {device}")

    return model, tokenizer, device


def test_1_article_prediction_in_context(model, tokenizer, device):
    """Test 1: Check if model predicts articles in contexts where they're required."""
    print("\n" + "="*80)
    print("TEST 1: Article Prediction in Context")
    print("="*80)

    # Article token IDs
    article_tokens = {
        262: ' a',
        264: ' the',
        389: ' an',
    }

    # Test prompts where articles are grammatically required
    test_prompts = [
        "Once upon a time there was",
        "There was",
        "She saw",
        "He found",
        "They met",
        "I have",
        "once upon",
        "there was little",
    ]

    print("\nTesting article predictions in contexts where articles are required:")
    print("-" * 80)

    total_tests = len(test_prompts)
    articles_in_top5 = 0
    articles_in_top20 = 0

    for prompt in test_prompts:
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        # Get predictions
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]  # Last token predictions
            probs = F.softmax(logits.float(), dim=0)

        print(f"\nPrompt: '{prompt}'")

        # Check article probabilities and ranks
        any_article_in_top5 = False
        any_article_in_top20 = False

        for token_id, token_text in article_tokens.items():
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1

            status = "✅" if rank <= 5 else "⚠️" if rank <= 20 else "❌"
            print(f"  {status} '{token_text}' (token {token_id}): {prob:.4%} (rank #{rank})")

            if rank <= 5:
                any_article_in_top5 = True
                any_article_in_top20 = True
            elif rank <= 20:
                any_article_in_top20 = True

        if any_article_in_top5:
            articles_in_top5 += 1
        if any_article_in_top20:
            articles_in_top20 += 1

        # Show top 5 predictions
        top5 = torch.topk(probs, 5)
        print(f"  Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            idx_item = idx.item()
            token_text = tokenizer.decode([idx_item])
            is_article = "← ARTICLE" if idx_item in article_tokens else ""
            print(f"    {i+1}. '{token_text}' - {prob:.4%} {is_article}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Prompts tested: {total_tests}")
    print(f"  Articles in top-5: {articles_in_top5}/{total_tests} ({articles_in_top5/total_tests*100:.1f}%)")
    print(f"  Articles in top-20: {articles_in_top20}/{total_tests} ({articles_in_top20/total_tests*100:.1f}%)")

    if articles_in_top5 < total_tests * 0.5:
        print("\n❌ PROBLEM: Model rarely predicts articles in contexts where they're needed!")
        print("   This explains why generation is missing articles.")
    elif articles_in_top20 < total_tests * 0.8:
        print("\n⚠️ WARNING: Articles are predicted but with low probability")
        print("   Model learned articles exist but not when to use them")
    else:
        print("\n✅ Model predicts articles in appropriate contexts")


def test_2_article_vs_content_word_probabilities(model, tokenizer, device):
    """Test 2: Compare article probabilities vs content word probabilities."""
    print("\n" + "="*80)
    print("TEST 2: Article vs Content Word Probabilities")
    print("="*80)

    print("\nComparing model's confidence in articles vs content words...")
    print("-" * 80)

    # Test cases: (prompt, expected_article, expected_content_word)
    test_cases = [
        ("Once upon a time there was", " a", " little"),
        ("She saw", " a", " dog"),
        ("He found", " a", " ball"),
    ]

    for prompt, article, content_word in test_cases:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]
            probs = F.softmax(logits.float(), dim=0)

        # Get token IDs
        article_id = tokenizer.encode(article)[0] if isinstance(tokenizer.encode(article), list) else tokenizer.encode(article)
        content_id = tokenizer.encode(content_word)[0] if isinstance(tokenizer.encode(content_word), list) else tokenizer.encode(content_word)

        article_prob = probs[article_id].item()
        content_prob = probs[content_id].item()

        print(f"\nPrompt: '{prompt}'")
        print(f"  Article '{article}': {article_prob:.4%}")
        print(f"  Content word '{content_word}': {content_prob:.4%}")
        print(f"  Ratio (content/article): {content_prob/article_prob if article_prob > 0 else float('inf'):.1f}x")

        if content_prob > article_prob * 10:
            print(f"  ❌ Model strongly prefers content word over article!")
        elif content_prob > article_prob:
            print(f"  ⚠️ Model slightly prefers content word")
        else:
            print(f"  ✅ Model appropriately weighs article")


def test_3_generation_with_different_settings(model, tokenizer, device):
    """Test 3: Try generation with different sampling settings."""
    print("\n" + "="*80)
    print("TEST 3: Generation with Different Sampling Settings")
    print("="*80)

    prompt = "Once upon a time there was"
    print(f"\nPrompt: '{prompt}'")
    print(f"Testing different generation settings...")
    print("-" * 80)

    # Settings to try
    settings = [
        {"name": "Greedy (argmax)", "temp": None, "top_k": None},
        {"name": "Temperature 1.0", "temp": 1.0, "top_k": None},
        {"name": "Temperature 0.8", "temp": 0.8, "top_k": None},
        {"name": "Temp 0.8 + top_k 50", "temp": 0.8, "top_k": 50},
        {"name": "Temp 1.0 + top_k 10", "temp": 1.0, "top_k": 10},
    ]

    for setting in settings:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        # Generate 20 tokens
        generated = tokens.copy()
        for _ in range(20):
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids)
                logits = outputs['logits'][0, -1, :]

            # Apply temperature
            if setting['temp'] is not None:
                logits = logits / setting['temp']

            probs = F.softmax(logits.float(), dim=0)

            # Apply top-k
            if setting['top_k'] is not None:
                top_k = min(setting['top_k'], probs.size(0))
                top_probs, top_indices = torch.topk(probs, top_k)
                # Renormalize
                top_probs = top_probs / top_probs.sum()
                # Sample from top-k
                next_token = top_indices[torch.multinomial(top_probs, 1)]
            elif setting['temp'] is None:
                # Greedy
                next_token = torch.argmax(probs)
            else:
                # Sample from full distribution
                next_token = torch.multinomial(probs, 1)

            generated.append(next_token.item())
            input_ids = torch.tensor([generated], device=device)

        text = tokenizer.decode(generated)

        # Count articles
        article_count = sum(1 for token in generated if token in [262, 264, 389])

        status = "✅" if article_count >= 2 else "⚠️" if article_count == 1 else "❌"
        print(f"\n{status} {setting['name']}:")
        print(f"  Generated: '{text}'")
        print(f"  Articles in output: {article_count}")


def test_4_check_article_token_embeddings(model, tokenizer, device):
    """Test 4: Check if article token embeddings are learned properly."""
    print("\n" + "="*80)
    print("TEST 4: Article Token Embeddings Analysis")
    print("="*80)

    print("\nAnalyzing learned embeddings for article tokens...")
    print("-" * 80)

    # Get embeddings
    embeddings = model.transformer.wte.weight.data  # Token embeddings

    article_tokens = {262: ' a', 264: ' the', 389: ' an'}

    # Get article embeddings
    article_embeds = []
    for token_id in article_tokens.keys():
        article_embeds.append(embeddings[token_id])

    # Compare similarity between articles
    print("\nSimilarity between article tokens:")
    for i, (id1, text1) in enumerate(article_tokens.items()):
        for id2, text2 in list(article_tokens.items())[i+1:]:
            emb1 = embeddings[id1]
            emb2 = embeddings[id2]
            similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
            print(f"  '{text1}' <-> '{text2}': {similarity:.4f}")

    # Find nearest neighbors for article tokens
    print("\nNearest neighbors for article tokens:")
    for token_id, token_text in article_tokens.items():
        emb = embeddings[token_id]

        # Compute similarities to all tokens
        similarities = F.cosine_similarity(emb.unsqueeze(0), embeddings, dim=1)

        # Get top 10 (excluding self)
        top10 = torch.topk(similarities, 11)  # +1 to account for self

        print(f"\n  '{token_text}' (token {token_id}):")
        count = 0
        for sim, idx in zip(top10.values, top10.indices):
            idx_item = idx.item()
            if idx_item == token_id:
                continue  # Skip self

            neighbor_text = tokenizer.decode([idx_item])
            print(f"    {neighbor_text:20s} similarity: {sim:.4f}")
            count += 1
            if count >= 5:
                break

    # Check if articles cluster together
    print("\n" + "-" * 80)
    avg_article_similarity = 0
    count = 0
    for i, id1 in enumerate(article_tokens.keys()):
        for id2 in list(article_tokens.keys())[i+1:]:
            avg_article_similarity += F.cosine_similarity(
                embeddings[id1], embeddings[id2], dim=0
            ).item()
            count += 1
    avg_article_similarity /= count

    print(f"\nAverage similarity between articles: {avg_article_similarity:.4f}")
    if avg_article_similarity > 0.7:
        print("✅ Articles cluster together (similar embeddings)")
    else:
        print("⚠️ Articles don't cluster strongly")


def test_5_loss_analysis_on_articles(model, tokenizer, device):
    """Test 5: Check model's loss specifically on article tokens."""
    print("\n" + "="*80)
    print("TEST 5: Loss Analysis on Article Tokens")
    print("="*80)

    print("\nAnalyzing model's prediction loss on sentences with articles...")
    print("-" * 80)

    # Test sentences with articles
    test_sentences = [
        "Once upon a time there was a little girl",
        "She saw a big dog in the park",
        "He found a ball under the tree",
        "There was an old man by the river",
    ]

    total_loss = 0
    article_loss = 0
    other_loss = 0
    article_count = 0
    other_count = 0

    article_tokens = {262, 264, 389}

    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)

        # Create input/target pairs
        input_ids = torch.tensor([tokens[:-1]], device=device)
        target_ids = torch.tensor([tokens[1:]], device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids)
            logits = outputs['logits'].float()

        # Compute loss per token
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='none'
        )

        # Separate article vs other token losses
        for i, (loss, target) in enumerate(zip(losses, target_ids[0])):
            target_item = target.item()
            if target_item in article_tokens:
                article_loss += loss.item()
                article_count += 1
            else:
                other_loss += loss.item()
                other_count += 1
            total_loss += loss.item()

    avg_article_loss = article_loss / article_count if article_count > 0 else 0
    avg_other_loss = other_loss / other_count if other_count > 0 else 0
    avg_total_loss = total_loss / (article_count + other_count)

    print(f"\nLoss analysis:")
    print(f"  Average loss (all tokens): {avg_total_loss:.4f}")
    print(f"  Average loss (articles):   {avg_article_loss:.4f}")
    print(f"  Average loss (other):      {avg_other_loss:.4f}")
    print(f"  Ratio (article/other):     {avg_article_loss/avg_other_loss if avg_other_loss > 0 else float('inf'):.2f}x")

    if avg_article_loss > avg_other_loss * 2:
        print("\n❌ PROBLEM: Model has much higher loss on articles!")
        print("   Model struggles to predict articles compared to other tokens")
    elif avg_article_loss > avg_other_loss * 1.2:
        print("\n⚠️ WARNING: Model has somewhat higher loss on articles")
    else:
        print("\n✅ Model's article prediction loss is reasonable")


def main():
    """Run all diagnostic tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose model article learning')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer directory')
    args = parser.parse_args()

    try:
        # Load model
        model, tokenizer, device = load_model(args.checkpoint, args.tokenizer)

        # Run all tests
        test_1_article_prediction_in_context(model, tokenizer, device)
        test_2_article_vs_content_word_probabilities(model, tokenizer, device)
        test_3_generation_with_different_settings(model, tokenizer, device)
        test_4_check_article_token_embeddings(model, tokenizer, device)
        test_5_loss_analysis_on_articles(model, tokenizer, device)

        # Final summary
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)
        print("\n📋 Key Questions Answered:")
        print("  1. Does model predict articles in context? (TEST 1)")
        print("  2. Does model prefer content words over articles? (TEST 2)")
        print("  3. Do different sampling settings help? (TEST 3)")
        print("  4. Are article embeddings learned properly? (TEST 4)")
        print("  5. Is article prediction loss high? (TEST 5)")
        print("\n💡 Check for ❌ marks above to identify the root cause!")

    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
