"""Quick test to verify model components work."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

print("Testing imports...")

try:
    from src.model.rmsnorm import RMSNorm
    print("✓ RMSNorm imported")
except ImportError as e:
    print(f"✗ RMSNorm import failed: {e}")
    sys.exit(1)

try:
    from src.model.rope import RotaryPositionEmbeddings
    print("✓ RoPE imported")
except ImportError as e:
    print(f"✗ RoPE import failed: {e}")
    sys.exit(1)

try:
    from src.model.swiglu import SwiGLU
    print("✓ SwiGLU imported")
except ImportError as e:
    print(f"✗ SwiGLU import failed: {e}")
    sys.exit(1)

try:
    from src.model.attention import MultiHeadAttention
    print("✓ MultiHeadAttention imported")
except ImportError as e:
    print(f"✗ MultiHeadAttention import failed: {e}")
    sys.exit(1)

try:
    from src.model.transformer_block import TransformerBlock, WikiMiniModel
    print("✓ TransformerBlock and WikiMiniModel imported")
except ImportError as e:
    print(f"✗ TransformerBlock import failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All imports successful!")
print("="*50)

# Quick model test
print("\nCreating minimal model...")

config = {
    'vocab_size': 1000,  # Small vocab for testing
    'd_model': 128,      # Small model
    'n_layers': 2,       # Few layers
    'n_heads': 4,        # Few heads
    'max_seq_len': 128,
    'dropout': 0.0,
    'use_flash_attention': False,
}

try:
    model = WikiMiniModel(config)
    print("✓ Model created successfully")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    input_ids = torch.randint(0, 1000, (1, 10), device=device)
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"  Forward pass successful!")
    print(f"  Output shape: {outputs['logits'].shape}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests passed! Ready for full testing.")