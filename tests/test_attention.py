"""Test attention module independently."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention

def test_attention():
    """Test multi-head attention with various configurations."""
    print("="*60)
    print("Testing Multi-Head Attention Implementation")
    print("="*60)
    
    # Test configuration
    batch_size = 2
    seq_len = 128
    d_model = 768
    n_heads = 12
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Head dimension: {d_model // n_heads}")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create attention module
    print("\nCreating attention module...")
    attention = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        max_seq_len=2048,
        rope_percentage=0.5,
        use_flash_attention=False,  # Test manual computation first
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in attention.parameters())
    print(f"Attention parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Move to device
    attention = attention.to(device)
    attention.eval()  # Set to eval mode for testing
    
    # Create dummy input
    print("\nCreating test input...")
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output, _ = attention(x)
    
    # Check output shape
    print(f"\nOutput shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check for NaN
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    print("✓ No NaN values in output")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    attention.train()
    x.requires_grad = True
    output, _ = attention(x)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    print("\nGradient norms:")
    for name, param in attention.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {grad_norm:.6f}")
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN!"
    
    print("\n" + "="*60)
    print("✓ All attention tests passed!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_attention()