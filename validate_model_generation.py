#!/usr/bin/env python3
"""
Comprehensive Model Generation Validator

This script identifies WHERE generation breaks:
1. Model's raw probability distributions (what it SHOULD predict)
2. Generation code behavior (what it ACTUALLY selects)
3. Train vs Inference mismatch

Run after training to diagnose generation issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: str, tokenizer_path: str, device='cuda'):
    """Load trained model and tokenizer."""
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']['model']
    else:
        # Default config if not in checkpoint
        logger.warning("No config in checkpoint, using defaults")
        config = {
            'vocab_size': 32000,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'd_ffn': 2048,
            'max_seq_len': 2048,
            'dropout': 0.0,
            'rope_percentage': 0.5,
            'rope_base': 10000,
            'rms_norm_eps': 1e-6,
            'tie_embeddings': True,
            'use_flash_attention': True,
        }

    # Override vocab size with tokenizer size
    config['vocab_size'] = len(tokenizer)

    # Create model
    model = WikiMiniModel(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {params/1e6:.2f}M parameters")

    return model, tokenizer, config


def check_model_probabilities(model, tokenizer, prompt: str, device='cuda'):
    """
    Test 1: Check what the model ACTUALLY predicts (raw probabilities).
    This reveals if model learned correctly, independent of generation code.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Model's Raw Probability Distribution")
    logger.info("="*80)
    logger.info(f"Prompt: '{prompt}'")

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Get model predictions
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids)
        logits = outputs['logits']  # [1, seq_len, vocab_size]

        # Get probabilities for next token after prompt
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        probs = torch.softmax(next_token_logits, dim=-1)

        # Get top 20 predictions
        top_probs, top_indices = torch.topk(probs, k=20)

        logger.info("\nTop 20 Most Likely Next Tokens:")
        logger.info("-" * 60)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token = tokenizer.decode([idx.item()])
            logger.info(f"{i:2d}. '{token}' → {prob.item()*100:.2f}%")

        # Check specific tokens of interest
        logger.info("\nSpecific Token Probabilities:")
        logger.info("-" * 60)
        check_words = ["a", "the", "little", "girl", "boy", "todlers", "toddler"]
        for word in check_words:
            word_tokens = tokenizer.encode(word)
            if word_tokens:
                word_id = word_tokens[0]
                word_prob = probs[word_id].item() * 100
                # Find rank
                rank = (probs > probs[word_id]).sum().item() + 1
                logger.info(f"  '{word}' (id={word_id}) → {word_prob:.4f}% (rank #{rank})")

    return probs, top_indices


def check_greedy_generation(model, tokenizer, prompt: str, max_tokens: int = 50, device='cuda'):
    """
    Test 2: Pure greedy generation (always pick highest probability).
    This should produce the "best" output according to model.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Greedy Generation (Always Pick Top Token)")
    logger.info("="*80)

    tokens = tokenizer.encode(prompt)
    generated = tokens.copy()

    logger.info(f"Starting with: '{prompt}'")
    logger.info("\nGeneration Process:")
    logger.info("-" * 60)

    for step in range(max_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids)
            logits = outputs['logits']
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)

            # Greedy: always pick highest probability
            next_token = torch.argmax(probs).item()
            next_token_prob = probs[next_token].item()

            generated.append(next_token)
            decoded_token = tokenizer.decode([next_token])

            # Show first 10 steps in detail
            if step < 10:
                logger.info(f"Step {step+1:2d}: '{decoded_token}' (p={next_token_prob*100:.2f}%)")

            # Stop at end of sentence
            if decoded_token in ['.', '!', '?'] and step > 20:
                break

    result = tokenizer.decode(generated)
    logger.info("\nFull Greedy Output:")
    logger.info("-" * 60)
    logger.info(f"'{result}'")

    # Check for issues
    logger.info("\nQuality Checks:")
    logger.info("-" * 60)
    has_articles = any(word in result.lower() for word in [' a ', ' an ', ' the '])
    has_todlers = 'todlers' in result.lower()
    logger.info(f"  Contains articles (a/an/the): {'✅ YES' if has_articles else '❌ NO'}")
    logger.info(f"  Contains 'todlers' typo: {'❌ YES (BAD!)' if has_todlers else '✅ NO'}")

    return result


def check_temperature_sampling(model, tokenizer, prompt: str, temperatures=[0.8, 1.0], device='cuda'):
    """
    Test 3: Temperature sampling (like actual generation code).
    Compare different temperatures.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Temperature Sampling")
    logger.info("="*80)

    for temp in temperatures:
        logger.info(f"\nTemperature: {temp}")
        logger.info("-" * 60)

        tokens = tokenizer.encode(prompt)
        generated = tokens.copy()

        for _ in range(50):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids)
                logits = outputs['logits']
                next_token_logits = logits[0, -1, :] / temp
                probs = torch.softmax(next_token_logits, dim=-1)

                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                decoded_token = tokenizer.decode([next_token])
                if decoded_token in ['.', '!', '?'] and len(generated) > 30:
                    break

        result = tokenizer.decode(generated)
        logger.info(f"Output: '{result[:200]}...'")

        # Quick check
        has_articles = any(word in result.lower() for word in [' a ', ' an ', ' the '])
        logger.info(f"Has articles: {'✅' if has_articles else '❌'}")


def check_training_context(model, tokenizer, device='cuda'):
    """
    Test 4: Check model's behavior on KNOWN good training sequences.
    If model works on training-like data but fails on new prompts,
    it suggests overfitting or prompt sensitivity.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Training-like Context Test")
    logger.info("="*80)

    # Test with complete, natural story opening
    training_like_prompt = "Once upon a time, there was a little girl named Lily. She lived in"

    logger.info(f"Using training-like prompt: '{training_like_prompt}'")

    tokens = tokenizer.encode(training_like_prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids)
        logits = outputs['logits']
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k=10)

        logger.info("\nTop 10 predictions after training-like context:")
        logger.info("-" * 60)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token = tokenizer.decode([idx.item()])
            logger.info(f"{i:2d}. '{token}' → {prob.item()*100:.2f}%")


def compare_with_generate_function(model, tokenizer, prompt: str, config, device='cuda'):
    """
    Test 5: Test generation with actual generate.py parameters.
    This replicates the exact settings used in generate.py.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 5: generate.py Style Generation (with repetition penalty)")
    logger.info("="*80)

    logger.info(f"Prompt: '{prompt}'")
    logger.info("\nUsing settings from generate.py:")
    logger.info("  temperature: 0.8")
    logger.info("  top_k: 50")
    logger.info("  top_p: 0.9")
    logger.info("  repetition_penalty: 1.1")

    tokens = tokenizer.encode(prompt)
    generated = tokens.copy()

    # Track generated for repetition penalty
    for step in range(50):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :].clone()

            # Apply repetition penalty (like generate.py)
            for token_id in set(generated):
                logits[token_id] /= 1.1

            # Apply temperature
            logits = logits / 0.8

            # Apply top-k
            indices_to_remove = logits < torch.topk(logits, 50)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')

            # Apply top-p
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            decoded_token = tokenizer.decode([next_token])
            if decoded_token in ['.', '!', '?'] and step > 20:
                break

    result = tokenizer.decode(generated)
    logger.info(f"\nOutput: '{result[:200]}...'")

    # Check quality
    has_articles = any(word in result.lower() for word in [' a ', ' an ', ' the '])
    logger.info(f"\nHas articles: {'✅' if has_articles else '❌'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate model generation")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer/wikimini_32k',
                       help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='Once upon a time there was',
                       help='Test prompt')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    logger.info("="*80)
    logger.info("MODEL GENERATION VALIDATOR")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Test Prompt: '{args.prompt}'")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("\nLoading model...")
    model, tokenizer, config = load_trained_model(args.checkpoint, args.tokenizer, device)
    logger.info(f"Model loaded on {device}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Run all tests
    try:
        # Test 1: Raw probabilities
        check_model_probabilities(model, tokenizer, args.prompt, device=device)

        # Test 2: Greedy generation
        check_greedy_generation(model, tokenizer, args.prompt, device=device)

        # Test 3: Temperature sampling
        check_temperature_sampling(model, tokenizer, args.prompt, device=device)

        # Test 4: Training-like context
        check_training_context(model, tokenizer, device=device)

        # Test 5: generate.py style
        compare_with_generate_function(model, tokenizer, args.prompt, config, device=device)

        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)
        logger.info("\n📊 Summary:")
        logger.info("  - Check TEST 1 to see if model predicts articles with high probability")
        logger.info("  - Check TEST 2 to see if greedy decoding produces good text")
        logger.info("  - If TEST 1 is good but TEST 2 is bad → generation code issue")
        logger.info("  - If TEST 1 is bad → model didn't learn properly despite good perplexity")
        logger.info("  - Check for 'todlers' appearing anywhere → indicates model issue")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
