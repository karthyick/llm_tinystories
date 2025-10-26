#!/usr/bin/env python3
"""
TinyStories Model Evaluation Script
Tests the trained model with various prompts and evaluates quality

Key checks:
- Article presence ("a", "the", "an") - THE CRITICAL TEST
- Grammar quality
- Perplexity
- Generation coherence
"""

import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# TinyStories test prompts (appropriate for children's stories)
TEST_PROMPTS = [
    "Once upon a time there was",
    "One day, a little girl named Lily",
    "In the garden, a small bird",
    "A big dog wanted to play with",
    "The happy cat found",
    "Two friends decided to go to",
    "Mom said to",
    "In the park there was",
    "A brave boy saw",
    "The little mouse was looking for",
]

def load_model_and_tokenizer(checkpoint_path: str, device: str = 'cuda'):
    """
    Load the trained TinyStories model and custom tokenizer from checkpoint
    """
    print(f"{Colors.OKCYAN}Loading model from: {checkpoint_path}{Colors.ENDC}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract components
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model', None))
        config = checkpoint.get('config', {})

        print(f"{Colors.OKGREEN}✓ Checkpoint loaded successfully{Colors.ENDC}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Step: {checkpoint.get('step', checkpoint.get('global_step', 'N/A'))}")
        if 'val_loss' in checkpoint:
            print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
        if 'val_ppl' in checkpoint:
            print(f"  - Validation PPL: {checkpoint['val_ppl']:.2f}")

        # Load tokenizer
        tokenizer_path = config.get('data', {}).get('tokenizer_path', './tokenizer/tinystories_10k')
        print(f"\n{Colors.OKCYAN}Loading tokenizer from: {tokenizer_path}{Colors.ENDC}")
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"{Colors.OKGREEN}✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,}){Colors.ENDC}")

        # Create model
        model_config = config.get('model', {})
        print(f"\n{Colors.OKCYAN}Creating model...{Colors.ENDC}")
        model = WikiMiniModel(model_config)

        # Load trained weights
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{Colors.OKGREEN}✓ Model loaded ({total_params:,} parameters){Colors.ENDC}\n")

        return model, tokenizer, config

    except Exception as e:
        print(f"{Colors.FAIL}Failed to load checkpoint: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        raise

def generate_story(model, tokenizer, prompt: str, max_length: int = 200,
                   temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                   device: str = 'cuda') -> Dict:
    """
    Generate a story from a prompt using nucleus sampling
    """
    start_time = time.time()

    # Tokenize input
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids]).to(device)

    generated_ids = input_ids[0].tolist()

    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits']

            # Get next token logits
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token.item())

            # Update input_ids for next iteration
            input_ids = torch.tensor([generated_ids]).to(device)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    generation_time = time.time() - start_time

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)

    # Calculate stats
    num_tokens = len(generated_ids)
    tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'num_tokens': num_tokens,
        'generation_time': generation_time,
        'tokens_per_sec': tokens_per_sec,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p
    }

def check_articles(text: str) -> Dict:
    """
    Check for presence of articles in generated text
    THE CRITICAL TEST - this was the original problem!
    """
    # Find all articles
    articles_a = re.findall(r'\ba\b', text.lower())
    articles_the = re.findall(r'\bthe\b', text.lower())
    articles_an = re.findall(r'\ban\b', text.lower())

    total_articles = len(articles_a) + len(articles_the) + len(articles_an)

    # Count words for ratio
    words = text.split()
    num_words = len(words)
    article_ratio = (total_articles / num_words * 100) if num_words > 0 else 0

    return {
        'has_articles': total_articles > 0,
        'count_a': len(articles_a),
        'count_the': len(articles_the),
        'count_an': len(articles_an),
        'total_articles': total_articles,
        'num_words': num_words,
        'article_ratio': article_ratio
    }

def evaluate_grammar(text: str) -> Dict:
    """
    Simple grammar evaluation
    """
    issues = []

    # Check for basic sentence structure
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Check capitalization
    if sentences:
        for sent in sentences[:5]:  # Check first 5 sentences
            if sent and not sent[0].isupper():
                issues.append("Missing capitalization")
                break

    # Check for double spaces
    if '  ' in text:
        issues.append("Double spaces")

    # Check for proper punctuation
    if text and not text.strip()[-1] in '.!?':
        issues.append("Missing end punctuation")

    # Grammar score (simple heuristic)
    score = 10 - min(len(issues) * 2, 8)

    return {
        'score': score,
        'issues': issues,
        'num_sentences': len(sentences)
    }

def calculate_perplexity(model, tokenizer, text: str, device: str = 'cuda') -> float:
    """
    Calculate perplexity for generated text
    """
    try:
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss']
            perplexity = torch.exp(loss).item()

        return perplexity
    except Exception as e:
        print(f"{Colors.WARNING}Could not calculate perplexity: {e}{Colors.ENDC}")
        return None

def print_story_result(result: Dict, index: int):
    """
    Pretty print a story generation result with article analysis
    """
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Story #{index + 1}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

    print(f"\n{Colors.OKCYAN}Prompt:{Colors.ENDC}")
    print(f"  {result['prompt']}")

    print(f"\n{Colors.OKGREEN}Generated Story:{Colors.ENDC}")
    print(f"  {result['generated_text']}")

    # Article analysis (THE KEY CHECK!)
    articles = result.get('articles', {})
    print(f"\n{Colors.BOLD}📝 Article Analysis (CRITICAL TEST):{Colors.ENDC}")

    if articles.get('has_articles', False):
        print(f"  {Colors.OKGREEN}✅ Articles present!{Colors.ENDC}")
        print(f"    • 'a': {articles['count_a']} times")
        print(f"    • 'the': {articles['count_the']} times")
        print(f"    • 'an': {articles['count_an']} times")
        print(f"    • Total: {articles['total_articles']} articles in {articles['num_words']} words")
        print(f"    • Article ratio: {articles['article_ratio']:.1f}%")
    else:
        print(f"  {Colors.FAIL}❌ NO ARTICLES FOUND!{Colors.ENDC}")
        print(f"  {Colors.WARNING}   This indicates the model did not learn articles properly.{Colors.ENDC}")

    # Grammar analysis
    grammar = result.get('grammar', {})
    print(f"\n{Colors.BOLD}📚 Grammar Analysis:{Colors.ENDC}")
    score = grammar.get('score', 0)
    if score >= 8:
        print(f"  {Colors.OKGREEN}✅ Score: {score}/10 (Excellent){Colors.ENDC}")
    elif score >= 6:
        print(f"  {Colors.OKGREEN}🟡 Score: {score}/10 (Good){Colors.ENDC}")
    else:
        print(f"  {Colors.WARNING}🟠 Score: {score}/10 (Needs improvement){Colors.ENDC}")

    if grammar.get('issues'):
        print(f"  Issues: {', '.join(grammar['issues'])}")

    # Generation stats
    print(f"\n{Colors.OKBLUE}⚡ Generation Stats:{Colors.ENDC}")
    print(f"  • Tokens: {result['num_tokens']}")
    print(f"  • Time: {result['generation_time']:.2f}s")
    print(f"  • Speed: {result['tokens_per_sec']:.1f} tokens/sec")

    if result.get('perplexity'):
        ppl = result['perplexity']
        print(f"  • Perplexity: {ppl:.2f}", end='')
        if ppl < 10:
            print(f" {Colors.OKGREEN}(Excellent){Colors.ENDC}")
        elif ppl < 20:
            print(f" {Colors.OKGREEN}(Good){Colors.ENDC}")
        else:
            print(f" {Colors.WARNING}(Fair){Colors.ENDC}")

def run_evaluation(model, tokenizer, prompts: List[str], device: str = 'cuda',
                   max_length: int = 200, temperature: float = 0.8,
                   top_k: int = 50, top_p: float = 0.95) -> List[Dict]:
    """
    Run comprehensive evaluation on multiple prompts
    """
    results = []

    print(f"\n{Colors.BOLD}{Colors.HEADER}Starting TinyStories Model Evaluation{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    for i, prompt in enumerate(prompts):
        print(f"{Colors.OKCYAN}Generating story {i+1}/{len(prompts)}...{Colors.ENDC}")

        # Generate story
        result = generate_story(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )

        # Check for articles (THE CRITICAL TEST!)
        result['articles'] = check_articles(result['generated_text'])

        # Evaluate grammar
        result['grammar'] = evaluate_grammar(result['generated_text'])

        # Calculate perplexity
        result['perplexity'] = calculate_perplexity(model, tokenizer, result['generated_text'], device)

        results.append(result)
        print_story_result(result, i)

    return results

def print_summary(results: List[Dict]):
    """
    Print comprehensive summary statistics
    """
    print(f"\n{Colors.BOLD}{Colors.HEADER}Evaluation Summary{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    # Article statistics (THE MOST IMPORTANT!)
    stories_with_articles = sum(1 for r in results if r.get('articles', {}).get('has_articles', False))
    article_success_rate = (stories_with_articles / len(results) * 100) if results else 0

    print(f"{Colors.BOLD}🎯 ARTICLE TEST (PRIMARY OBJECTIVE):{Colors.ENDC}")
    if article_success_rate >= 90:
        print(f"  {Colors.OKGREEN}✅ SUCCESS: {stories_with_articles}/{len(results)} stories have articles ({article_success_rate:.0f}%){Colors.ENDC}")
        print(f"  {Colors.OKGREEN}   Model learned to use articles correctly!{Colors.ENDC}")
    elif article_success_rate >= 50:
        print(f"  {Colors.WARNING}🟡 PARTIAL: {stories_with_articles}/{len(results)} stories have articles ({article_success_rate:.0f}%){Colors.ENDC}")
        print(f"  {Colors.WARNING}   Model sometimes uses articles but not consistently.{Colors.ENDC}")
    else:
        print(f"  {Colors.FAIL}❌ FAILURE: {stories_with_articles}/{len(results)} stories have articles ({article_success_rate:.0f}%){Colors.ENDC}")
        print(f"  {Colors.FAIL}   Model did not learn articles properly. Needs retraining.{Colors.ENDC}")

    # Average article counts
    avg_a = sum(r.get('articles', {}).get('count_a', 0) for r in results) / len(results)
    avg_the = sum(r.get('articles', {}).get('count_the', 0) for r in results) / len(results)
    avg_an = sum(r.get('articles', {}).get('count_an', 0) for r in results) / len(results)

    print(f"\n  Average article usage per story:")
    print(f"    • 'a': {avg_a:.1f}")
    print(f"    • 'the': {avg_the:.1f}")
    print(f"    • 'an': {avg_an:.1f}")

    # Grammar statistics
    print(f"\n{Colors.BOLD}📚 Grammar Quality:{Colors.ENDC}")
    avg_grammar = sum(r.get('grammar', {}).get('score', 0) for r in results) / len(results)

    if avg_grammar >= 8:
        quality = f"{Colors.OKGREEN}Excellent (8-9/10){Colors.ENDC}"
    elif avg_grammar >= 6:
        quality = f"{Colors.OKGREEN}Good (6-7/10){Colors.ENDC}"
    else:
        quality = f"{Colors.WARNING}Needs improvement (<6/10){Colors.ENDC}"

    print(f"  • Average score: {avg_grammar:.1f}/10 - {quality}")

    # Performance metrics
    print(f"\n{Colors.BOLD}⚡ Performance:{Colors.ENDC}")
    avg_tokens = sum(r['num_tokens'] for r in results) / len(results)
    avg_time = sum(r['generation_time'] for r in results) / len(results)
    avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)

    print(f"  • Total stories: {len(results)}")
    print(f"  • Avg tokens/story: {avg_tokens:.0f}")
    print(f"  • Avg time/story: {avg_time:.2f}s")
    print(f"  • Avg speed: {avg_speed:.1f} tokens/sec")

    # Perplexity
    perplexities = [r['perplexity'] for r in results if r.get('perplexity') is not None]
    if perplexities:
        avg_ppl = sum(perplexities) / len(perplexities)
        print(f"  • Avg perplexity: {avg_ppl:.2f}", end='')

        if avg_ppl < 10:
            print(f" {Colors.OKGREEN}(Excellent){Colors.ENDC}")
        elif avg_ppl < 20:
            print(f" {Colors.OKGREEN}(Good){Colors.ENDC}")
        else:
            print(f" {Colors.WARNING}(Fair){Colors.ENDC}")

    # Overall verdict
    print(f"\n{Colors.BOLD}🏆 Overall Assessment:{Colors.ENDC}")

    if article_success_rate >= 90 and avg_grammar >= 7:
        print(f"  {Colors.OKGREEN}{Colors.BOLD}✅ MODEL PASSED - Ready for use!{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}   Articles learned correctly, good grammar quality.{Colors.ENDC}")
    elif article_success_rate >= 50:
        print(f"  {Colors.WARNING}🟡 MODEL PARTIAL - May need more training{Colors.ENDC}")
        print(f"  {Colors.WARNING}   Articles sometimes present but not consistent.{Colors.ENDC}")
    else:
        print(f"  {Colors.FAIL}❌ MODEL FAILED - Needs retraining{Colors.ENDC}")
        print(f"  {Colors.FAIL}   Articles not learned. Check vocabulary size and training.{Colors.ENDC}")

    print()

def save_results(results: List[Dict], output_path: str):
    """
    Save detailed results to JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'num_stories': len(results),
        'article_success_rate': sum(1 for r in results if r.get('articles', {}).get('has_articles', False)) / len(results) * 100,
        'avg_grammar_score': sum(r.get('grammar', {}).get('score', 0) for r in results) / len(results),
        'stories': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"{Colors.OKGREEN}✓ Results saved to: {output_path}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='Test TinyStories Model Quality')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Custom prompts (default: use standard TinyStories prompts)')
    parser.add_argument('--max-length', type=int, default=150,
                        help='Maximum generation length (default: 150)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-K sampling (default: 50)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-P (nucleus) sampling (default: 0.95)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results (default: evaluation_results.json)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"{Colors.WARNING}CUDA not available, using CPU{Colors.ENDC}")
        args.device = 'cpu'

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.device)

    # Use custom prompts or default TinyStories prompts
    prompts = args.prompts if args.prompts else TEST_PROMPTS

    # Run evaluation
    results = run_evaluation(
        model, tokenizer, prompts,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save:
        save_results(results, args.output)

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ Evaluation completed!{Colors.ENDC}\n")

if __name__ == '__main__':
    main()
