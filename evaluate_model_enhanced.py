#!/usr/bin/env python3
"""
Enhanced TinyStories Model Evaluation Script
Tests with multiple generation strategies and post-processes output

Enhancements:
1. Multiple temperature/top-p configurations
2. Repetition penalty to reduce repeated words
3. Automatic capitalization and punctuation fixes
4. Comparative analysis across different settings
"""

import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer

# Color codes
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

# Test prompts
TEST_PROMPTS = [
    "Once upon a time there was",
    "One day, a little girl named Lily",
    "In the garden, a small bird",
    "A big dog wanted to play with",
    "The happy cat found",
]

# Generation configurations to test
GENERATION_CONFIGS = [
    {
        'name': 'Balanced',
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'repetition_penalty': 1.2,
        'description': 'Good balance between creativity and coherence'
    },
    {
        'name': 'Conservative',
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.9,
        'repetition_penalty': 1.3,
        'description': 'More coherent, less repetition'
    },
    {
        'name': 'Creative',
        'temperature': 0.9,
        'top_k': 60,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'description': 'More creative and varied'
    },
]

def load_model_and_tokenizer(checkpoint_path: str, device: str = 'cuda'):
    """Load model and tokenizer"""
    print(f"{Colors.OKCYAN}Loading model from: {checkpoint_path}{Colors.ENDC}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model', None))
    config = checkpoint.get('config', {})

    print(f"{Colors.OKGREEN}✓ Checkpoint loaded (Epoch: {checkpoint.get('epoch', 'N/A')}, Step: {checkpoint.get('step', checkpoint.get('global_step', 'N/A'))}){Colors.ENDC}")

    tokenizer_path = config.get('data', {}).get('tokenizer_path', './tokenizer/tinystories_10k')
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"{Colors.OKGREEN}✓ Tokenizer loaded (vocab: {tokenizer.vocab_size:,}){Colors.ENDC}")

    model = WikiMiniModel(config.get('model', {}))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{Colors.OKGREEN}✓ Model loaded ({total_params:,} params){Colors.ENDC}\n")

    return model, tokenizer, config

def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.2) -> torch.Tensor:
    """
    Apply repetition penalty to logits
    Reduces probability of tokens that already appeared
    """
    if penalty == 1.0:
        return logits

    # Get unique tokens in input
    for token_id in torch.unique(input_ids):
        # Reduce logit for tokens that already appeared
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits

def generate_story_enhanced(
    model, tokenizer, prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    device: str = 'cuda'
) -> Dict:
    """
    Enhanced generation with repetition penalty
    """
    start_time = time.time()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids]).to(device)
    generated_ids = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            logits = outputs['logits']
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                input_ids[0],
                repetition_penalty
            )

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token.item())
            input_ids = torch.tensor([generated_ids]).to(device)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generation_time = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids)

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'num_tokens': len(generated_ids),
        'generation_time': generation_time,
        'tokens_per_sec': len(generated_ids) / generation_time if generation_time > 0 else 0,
    }

def post_process_text(text: str) -> str:
    """
    Post-process generated text to fix common issues:
    1. Fix capitalization
    2. Fix punctuation
    3. Remove excessive whitespace
    4. Fix common grammar issues
    """
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences (approximate)
    # Look for sentence endings or line breaks
    sentences = re.split(r'([.!?]\s+|\n)', text)

    fixed_sentences = []
    current_sentence = ""

    for part in sentences:
        if part.strip():
            if re.match(r'[.!?]\s*', part):
                # This is punctuation
                current_sentence += part
                if current_sentence.strip():
                    fixed_sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part

    # Add last sentence if exists
    if current_sentence.strip():
        # Add period if missing
        if not current_sentence.strip()[-1] in '.!?':
            current_sentence += '.'
        fixed_sentences.append(current_sentence.strip())

    # Capitalize first letter of each sentence
    fixed_sentences = [s[0].upper() + s[1:] if s else s for s in fixed_sentences]

    # Join sentences
    result = ' '.join(fixed_sentences)

    # Fix common patterns
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)  # Remove space before punctuation
    result = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)  # Capitalize after period
    result = re.sub(r'\s+', ' ', result)  # Remove double spaces

    return result

def calculate_repetition_score(text: str) -> Dict:
    """
    Calculate how repetitive the text is
    """
    words = text.lower().split()
    if not words:
        return {'score': 0, 'unique_ratio': 0, 'repeated_words': []}

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Find words repeated more than normal
    repeated_words = [(word, count) for word, count in word_counts.items() if count > 3]
    repeated_words.sort(key=lambda x: x[1], reverse=True)

    unique_ratio = len(word_counts) / len(words) * 100

    # Score: 10 = no repetition, 0 = very repetitive
    score = min(10, int(unique_ratio / 10))

    return {
        'score': score,
        'unique_ratio': unique_ratio,
        'repeated_words': repeated_words[:5],  # Top 5 most repeated
        'total_words': len(words),
        'unique_words': len(word_counts)
    }

def check_articles(text: str) -> Dict:
    """Check for article presence"""
    articles_a = re.findall(r'\ba\b', text.lower())
    articles_the = re.findall(r'\bthe\b', text.lower())
    articles_an = re.findall(r'\ban\b', text.lower())

    total_articles = len(articles_a) + len(articles_the) + len(articles_an)
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
    """Evaluate grammar quality"""
    issues = []
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Check capitalization
    if sentences:
        for sent in sentences[:5]:
            if sent and not sent[0].isupper():
                issues.append("Missing capitalization")
                break

    # Check for double spaces
    if '  ' in text:
        issues.append("Double spaces")

    # Check for proper ending
    if text and not text.strip()[-1] in '.!?':
        issues.append("Missing end punctuation")

    score = 10 - min(len(issues) * 2, 8)

    return {
        'score': score,
        'issues': issues,
        'num_sentences': len(sentences)
    }

def calculate_perplexity(model, tokenizer, text: str, device: str = 'cuda') -> float:
    """Calculate perplexity"""
    try:
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss']
            perplexity = torch.exp(loss).item()

        return perplexity
    except:
        return None

def print_comparison_result(results_by_config: Dict, prompt: str, index: int):
    """Print comparison across different generation configs"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Prompt #{index + 1}: {prompt}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    for config_name, result in results_by_config.items():
        print(f"{Colors.OKCYAN}{Colors.BOLD}[{config_name} Setting]{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Original:{Colors.ENDC}")
        print(f"  {result['generated_text'][:200]}...")
        print(f"\n{Colors.OKGREEN}Post-Processed:{Colors.ENDC}")
        print(f"  {result['post_processed'][:200]}...")

        # Quick stats
        articles = result['articles']
        repetition = result['repetition']
        grammar = result['grammar_processed']

        print(f"\n{Colors.OKBLUE}Quality Metrics:{Colors.ENDC}")
        print(f"  • Articles: {articles['total_articles']} ({'✅' if articles['has_articles'] else '❌'})")
        print(f"  • Grammar: {grammar['score']}/10")
        print(f"  • Repetition: {repetition['score']}/10 ({repetition['unique_ratio']:.1f}% unique)")
        if repetition['repeated_words']:
            top_repeated = repetition['repeated_words'][0]
            print(f"  • Most repeated: '{top_repeated[0]}' ({top_repeated[1]}× times)")
        print()

def run_enhanced_evaluation(
    model, tokenizer, prompts: List[str],
    device: str = 'cuda',
    max_length: int = 150
) -> Dict:
    """
    Run evaluation with multiple generation configurations
    """
    print(f"\n{Colors.BOLD}{Colors.HEADER}Enhanced TinyStories Evaluation{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Testing {len(GENERATION_CONFIGS)} generation configurations:{Colors.ENDC}")
    for config in GENERATION_CONFIGS:
        print(f"  • {config['name']}: {config['description']}")
    print()

    all_results = {config['name']: [] for config in GENERATION_CONFIGS}

    for i, prompt in enumerate(prompts):
        print(f"{Colors.OKCYAN}Evaluating prompt {i+1}/{len(prompts)}{Colors.ENDC}")

        results_by_config = {}

        for config in GENERATION_CONFIGS:
            # Generate story
            result = generate_story_enhanced(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=config['temperature'],
                top_k=config['top_k'],
                top_p=config['top_p'],
                repetition_penalty=config['repetition_penalty'],
                device=device
            )

            # Post-process
            result['post_processed'] = post_process_text(result['generated_text'])

            # Analyze both original and post-processed
            result['articles'] = check_articles(result['generated_text'])
            result['repetition'] = calculate_repetition_score(result['generated_text'])
            result['grammar_original'] = evaluate_grammar(result['generated_text'])
            result['grammar_processed'] = evaluate_grammar(result['post_processed'])
            result['perplexity'] = calculate_perplexity(model, tokenizer, result['generated_text'], device)
            result['config'] = config['name']

            results_by_config[config['name']] = result
            all_results[config['name']].append(result)

        # Print comparison
        print_comparison_result(results_by_config, prompt, i)

    return all_results

def print_enhanced_summary(all_results: Dict):
    """Print comprehensive summary comparing all configurations"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}Comparative Summary{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    for config_name, results in all_results.items():
        print(f"{Colors.BOLD}{Colors.OKCYAN}[{config_name} Configuration]{Colors.ENDC}")

        # Article stats
        article_success = sum(1 for r in results if r['articles']['has_articles'])
        article_rate = (article_success / len(results) * 100) if results else 0

        # Average metrics
        avg_grammar_orig = sum(r['grammar_original']['score'] for r in results) / len(results)
        avg_grammar_proc = sum(r['grammar_processed']['score'] for r in results) / len(results)
        avg_repetition = sum(r['repetition']['score'] for r in results) / len(results)
        avg_unique = sum(r['repetition']['unique_ratio'] for r in results) / len(results)

        perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
        avg_ppl = sum(perplexities) / len(perplexities) if perplexities else 0

        print(f"  🎯 Articles: {article_success}/{len(results)} ({article_rate:.0f}%) ", end='')
        if article_rate >= 90:
            print(f"{Colors.OKGREEN}✅{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}🟡{Colors.ENDC}")

        print(f"  📚 Grammar: {avg_grammar_orig:.1f}/10 → {avg_grammar_proc:.1f}/10 (post-processed)")
        print(f"  🔄 Repetition: {avg_repetition:.1f}/10 ({avg_unique:.1f}% unique words)")
        print(f"  📊 Perplexity: {avg_ppl:.2f}")
        print()

    # Recommendation
    print(f"{Colors.BOLD}📌 Recommendation:{Colors.ENDC}")

    # Find best config by combined score
    best_config = None
    best_score = 0

    for config_name, results in all_results.items():
        article_rate = sum(1 for r in results if r['articles']['has_articles']) / len(results) * 100
        avg_grammar = sum(r['grammar_processed']['score'] for r in results) / len(results)
        avg_repetition = sum(r['repetition']['score'] for r in results) / len(results)

        # Combined score
        score = (article_rate / 10) + avg_grammar + avg_repetition

        if score > best_score:
            best_score = score
            best_config = config_name

    print(f"  {Colors.OKGREEN}✨ Best setting: {Colors.BOLD}{best_config}{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}   Use this for final generation{Colors.ENDC}\n")

def save_enhanced_results(all_results: Dict, output_path: str):
    """Save detailed results"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    json_results = {
        'timestamp': datetime.now().isoformat(),
        'configurations': {name: results for name, results in all_results.items()}
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"{Colors.OKGREEN}✓ Results saved to: {output_path}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced TinyStories Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prompts', type=str, nargs='+', default=None, help='Custom prompts')
    parser.add_argument('--max-length', type=int, default=150, help='Max generation length')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='enhanced_evaluation.json', help='Output file')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"{Colors.WARNING}CUDA not available, using CPU{Colors.ENDC}")
        args.device = 'cpu'

    # Load model
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.device)

    # Use custom prompts or defaults
    prompts = args.prompts if args.prompts else TEST_PROMPTS

    # Run enhanced evaluation
    all_results = run_enhanced_evaluation(
        model, tokenizer, prompts,
        device=args.device,
        max_length=args.max_length
    )

    # Print summary
    print_enhanced_summary(all_results)

    # Save results
    if not args.no_save:
        save_enhanced_results(all_results, args.output)

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ Enhanced evaluation completed!{Colors.ENDC}\n")

if __name__ == '__main__':
    main()
