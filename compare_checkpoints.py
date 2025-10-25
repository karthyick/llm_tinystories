"""
Checkpoint Comparison Script
Calls your existing generate.py to test multiple checkpoints
"""

import subprocess
import os
from pathlib import Path
import argparse
from datetime import datetime

# Test prompts
TEST_PROMPTS = [
    "Once upon a time there was",
    "One day, a little girl named",
    "There were two friends who",
    "The cat and the dog",
    "In the garden, there was",
]

def find_checkpoints(checkpoint_dir='checkpoints', ppl_range='7.0-10.0', max_count=5):
    """Find checkpoint files in PPL range"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all PPL checkpoints
    ppl_checkpoints = list(checkpoint_dir.glob('checkpoint_best_ppl_*.pth'))
    
    # Sort by PPL
    def get_ppl(path):
        try:
            ppl_str = path.stem.split('_')[-1]
            return float(ppl_str)
        except:
            return 999.0
    
    ppl_checkpoints.sort(key=get_ppl)
    
    # Filter by range
    ppl_min, ppl_max = map(float, ppl_range.split('-'))
    selected = [ckpt for ckpt in ppl_checkpoints 
                if ppl_min <= get_ppl(ckpt) <= ppl_max]
    
    # Limit count
    if len(selected) > max_count:
        step = len(selected) // max_count
        selected = selected[::step][:max_count]
    
    return selected

def run_generation(checkpoint_path, prompt, tokenizer_path='./tokenizer/wikimini_32k'):
    """Run generate.py with given checkpoint and prompt"""
    
    cmd = [
        'python', 'generate.py',
        '--checkpoint', str(checkpoint_path),
        '--tokenizer', tokenizer_path,
        '--prompt', prompt,
        '--max_tokens', '200',
        '--temperature', '0.85',
        '--top_k', '50',
        '--top_p', '0.9',
        '--repetition_penalty', '1.1',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8'
        )
        
        output = result.stdout
        
        # Simple parsing: split on the "====" separator (60 chars)
        separator = '=' * 60
        
        if separator in output:
            parts = output.split(separator)
            # parts[0]: everything before first ====
            # parts[1]: the generated text
            # parts[2]: everything after second ====
            
            if len(parts) >= 3:
                generated_text = parts[1].strip()
                if generated_text:
                    return generated_text, None
            
            # Fallback
            return None, f"Found separator but couldn't extract text. Parts: {len(parts)}"
        
        return None, "'Generated text:' separator not found in output"
        
    except subprocess.TimeoutExpired:
        return None, "Generation timed out (>60s)"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Compare checkpoint quality')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--tokenizer-path', type=str, default='./tokenizer/wikimini_32k')
    parser.add_argument('--output', type=str, default='checkpoint_comparison.txt')
    parser.add_argument('--test-range', type=str, default='7.0-10.0')
    parser.add_argument('--max-checkpoints', type=int, default=5)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CHECKPOINT QUALITY COMPARISON (Using generate.py)")
    print("=" * 80)
    
    # Find checkpoints
    print(f"\nSearching for checkpoints in {args.checkpoint_dir}...")
    checkpoints = find_checkpoints(args.checkpoint_dir, args.test_range, args.max_checkpoints)
    
    print(f"\n✅ Testing {len(checkpoints)} checkpoints")
    print("\nCheckpoints to test:")
    for ckpt in checkpoints:
        ppl = ckpt.stem.split('_')[-1]
        print(f"  - PPL {ppl}: {ckpt.name}")
    
    # Open output file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = open(args.output, 'w', encoding='utf-8')
    output_file.write("=" * 80 + "\n")
    output_file.write(f"CHECKPOINT COMPARISON RESULTS\n")
    output_file.write(f"Generated: {timestamp}\n")
    output_file.write("=" * 80 + "\n\n")
    
    # Test each checkpoint
    results = {}
    
    for checkpoint_path in checkpoints:
        ppl = checkpoint_path.stem.split('_')[-1]
        
        print("\n" + "=" * 80)
        print(f"TESTING: PPL {ppl} (calling generate.py)")
        print("=" * 80)
        
        output_file.write("\n" + "=" * 80 + "\n")
        output_file.write(f"CHECKPOINT: {checkpoint_path.name} (PPL {ppl})\n")
        output_file.write("=" * 80 + "\n\n")
        
        checkpoint_results = []
        
        # Test each prompt
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
            output_file.write(f"\n{'─' * 80}\n")
            output_file.write(f"Prompt {i}: {prompt}\n")
            output_file.write(f"{'─' * 80}\n")
            
            # Generate using your generate.py
            print("  Calling generate.py...", end="", flush=True)
            generated, error = run_generation(
                checkpoint_path, 
                prompt, 
                args.tokenizer_path
            )
            
            if error:
                print(f" ❌ Error!")
                print(f"  {error}")
                output_file.write(f"❌ ERROR: {error}\n")
                continue
            
            print(f" ✅ Done!")
            print(f"  Generated ({len(generated)} chars):")
            print(f"    {generated[:150]}{'...' if len(generated) > 150 else ''}")
            
            output_file.write(f"{generated}\n")
            
            checkpoint_results.append({
                'prompt': prompt,
                'generated': generated
            })
        
        results[ppl] = checkpoint_results
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    output_file.write("\n\n" + "=" * 80 + "\n")
    output_file.write("COMPARISON SUMMARY\n")
    output_file.write("=" * 80 + "\n\n")
    
    for ppl, checkpoint_results in results.items():
        if not checkpoint_results:
            continue
            
        avg_length = sum(len(r['generated']) for r in checkpoint_results) / len(checkpoint_results)
        print(f"\nPPL {ppl}:")
        print(f"  Average generation length: {avg_length:.0f} chars")
        
        output_file.write(f"\nPPL {ppl}:\n")
        output_file.write(f"  Average length: {avg_length:.0f} chars\n")
        
        # Quality indicators
        has_names = sum(1 for r in checkpoint_results if any(name in r['generated'].lower() 
                       for name in ['tim', 'lily', 'emma', 'max', 'lucy', 'sam', 'anna', 'ben'])) / len(checkpoint_results)
        has_dialogue = sum(1 for r in checkpoint_results if '"' in r['generated'] or "'" in r['generated']) / len(checkpoint_results)
        
        print(f"  Named characters: {has_names*100:.0f}%")
        print(f"  Has dialogue: {has_dialogue*100:.0f}%")
        
        output_file.write(f"  Named characters: {has_names*100:.0f}%\n")
        output_file.write(f"  Has dialogue: {has_dialogue*100:.0f}%\n")
    
    output_file.close()
    
    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {args.output}")
    print("=" * 80)
    
    # Recommendation
    print("\n📊 RECOMMENDATION:")
    print("Review the output file and pick the checkpoint with:")
    print("  ✅ Named characters")
    print("  ✅ Coherent narrative")
    print("  ✅ Proper grammar")
    print("  ✅ Natural dialogue")
    print("  ✅ Complete story arcs")

if __name__ == '__main__':
    main()