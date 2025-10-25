#!/usr/bin/env python
"""Test the data pipeline to ensure everything works."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from src.data.tokenizer import load_tokenizer
from src.data.dataset import TinyStoriesDataset, create_dataloaders

def test_data_pipeline():
    print("="*60)
    print("Testing Data Pipeline")
    print("="*60)
    
    # Check if tokenizer exists
    tokenizer_path = "./tokenizer/wikimini_32k"
    if not Path(tokenizer_path).exists():
        print(f"\n⚠️ Tokenizer not found at {tokenizer_path}")
        print("Please run: python scripts/train_tokenizer.py first")
        return False
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"   ✓ Loaded tokenizer with {len(tokenizer)} tokens")
    
    # Test tokenization
    print("\n2. Testing tokenization...")
    test_text = "Once upon a time, there was a little girl who loved to read stories."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   Original: {test_text}")
    print(f"   Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print(f"   Decoded: {decoded}")

    # Create small dataset for testing
    print("\n3. Creating test dataset (validation split)...")
    dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        split="validation",
        max_seq_len=512,  # Small for testing
        cache_dir="./data/cache_test",
    )
    print(f"   ✓ Dataset created with {len(dataset)} sequences")
    
    # Test single sample
    print("\n4. Testing single sample...")
    sample = dataset[0]
    print(f"   Input shape: {sample['input_ids'].shape}")
    print(f"   Labels shape: {sample['labels'].shape}")
    print(f"   First 10 tokens: {sample['input_ids'][:10].tolist()}")
    
    # Test dataloader
    print("\n5. Testing DataLoader...")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=4,
        max_seq_len=512,
        cache_dir="./data/cache_test",
        dataset_name="tinystories",
        num_workers=0,  # Windows requirement
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"   Batch input shape: {batch['input_ids'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")
    
    # Test GPU transfer
    if torch.cuda.is_available():
        print("\n6. Testing GPU transfer...")
        device = torch.device('cuda')
        input_ids = batch['input_ids'].to(device)
        print(f"   ✓ Successfully moved batch to {device}")
    
    print("\n" + "="*60)
    print("✅ Data pipeline test complete!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_data_pipeline()