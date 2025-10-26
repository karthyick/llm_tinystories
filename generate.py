"""Text generation script for WikiMini 95M model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import Optional, List
import time

from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer


class TextGenerator:
    """Text generator for WikiMini model."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, checkpoint_path: str) -> WikiMiniModel:
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']['model']
        else:
            # Default config if not in checkpoint
            config = {
                'vocab_size': 32000,
                'd_model': 768,
                'n_layers': 12,
                'n_heads': 12,
                'd_ffn': 2048,
                'max_seq_len': 2048,
                'dropout': 0.0,  # No dropout for inference
                'rope_percentage': 0.5,
                'rope_base': 10000,
                'rms_norm_eps': 1e-6,
                'tie_embeddings': True,
                'use_flash_attention': True,
            }
        
        # Override vocab size with tokenizer size
        config['vocab_size'] = len(self.tokenizer)
        
        # Create model
        model = WikiMiniModel(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {params/1e6:.2f}M parameters")
        
        return model
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        
        # Track generated tokens for repetition penalty
        generated_ids = input_ids[0].tolist()
        
        # Generate tokens
        start_time = time.time()
        
        for _ in range(max_tokens):
            # Get model predictions
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids)
            
            logits = outputs['logits'][0, -1, :]  # Get last token logits
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    logits[token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('inf')
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token.item())
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Calculate statistics
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_ids) / elapsed
        
        return generated_text, tokens_per_sec
    
    def interactive_generate(self):
        """Interactive text generation loop."""
        print("\n" + "="*60)
        print("Interactive Text Generation")
        print("="*60)
        print("Type 'quit' to exit, 'help' for options\n")
        
        # Default settings
        settings = {
            'max_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'do_sample': True,
        }
        
        while True:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            
            if prompt.lower() == 'help':
                print("\nSettings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print("\nCommands:")
                print("  set <param> <value> - Change generation parameter")
                print("  quit - Exit")
                print()
                continue
            
            if prompt.startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param in settings:
                        try:
                            if param == 'do_sample':
                                settings[param] = value.lower() == 'true'
                            elif param in ['max_tokens', 'top_k']:
                                settings[param] = int(value)
                            else:
                                settings[param] = float(value)
                            print(f"Set {param} = {settings[param]}")
                        except ValueError:
                            print(f"Invalid value for {param}")
                    else:
                        print(f"Unknown parameter: {param}")
                continue
            
            if not prompt:
                continue
            
            # Generate text
            print("Generating...", end="", flush=True)
            text, tokens_per_sec = self.generate(prompt, **settings)
            print(f" Done! ({tokens_per_sec:.1f} tokens/sec)")
            
            print("\nGenerated text:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            print()


def main():
    parser = argparse.ArgumentParser(description="Generate text with WikiMini model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./tokenizer/wikimini_32k",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = TextGenerator(
        model_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )
    
    if args.prompt:
        # Single generation
        text, tokens_per_sec = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
        )
        
        print("\nGenerated text:")
        print("="*60)
        print(text)
        print("="*60)
        print(f"\nGeneration speed: {tokens_per_sec:.1f} tokens/sec")
    else:
        # Interactive mode
        generator.interactive_generate()


if __name__ == "__main__":
    main()