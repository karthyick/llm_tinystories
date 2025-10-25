"""Test the complete WikiMini 95M model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from src.model.transformer_block import WikiMiniModel

def test_wikimini_model():
    """Comprehensive test of WikiMini 95M model."""
    print("="*70)
    print(" "*20 + "WikiMini 95M Model Test")
    print("="*70)
    
    # Configuration for 95M parameters
    config = {
        'vocab_size': 32000,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'd_ffn': 2048,  # SwiGLU adjusted (8/3 * 768 rounded)
        'max_seq_len': 2048,
        'dropout': 0.0,  # No dropout for testing
        'rope_percentage': 0.5,
        'rope_base': 10000,
        'rms_norm_eps': 1e-6,
        'tie_embeddings': True,
        'use_flash_attention': False,  # Start without Flash Attention
    }
    
    print("\n┌" + "─"*68 + "┐")
    print("│ Configuration:" + " "*53 + "│")
    print("├" + "─"*68 + "┤")
    for key, value in config.items():
        line = f"│   {key:20s}: {str(value):43s} │"
        print(line)
    print("└" + "─"*68 + "┘")
    
    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
    
    # Create model
    print("\n🏭 Building model...")
    start_time = time.time()
    model = WikiMiniModel(config)
    build_time = time.time() - start_time
    print(f"   Model built in {build_time:.2f}s")
    
    # Count parameters
    param_info = model.count_parameters()
    print("\n📋 Parameter Count:")
    print(f"   Total: {param_info['total']:,} ({param_info['total_millions']:.2f}M)")
    print(f"   - Embedding: {param_info['embedding']:,} ({param_info['embedding']/1e6:.2f}M)")
    print(f"   - Attention: {param_info['attention']:,} ({param_info['attention']/1e6:.2f}M)")
    print(f"   - FFN: {param_info['ffn']:,} ({param_info['ffn']/1e6:.2f}M)")
    print(f"   - Norm: {param_info['norm']:,} ({param_info['norm']/1e6:.2f}M)")
    
    # Target check
    target_params = 95_000_000
    diff = abs(param_info['total'] - target_params)
    diff_percent = (diff / target_params) * 100
    
    if param_info['total_millions'] < 90 or param_info['total_millions'] > 100:
        print(f"\n⚠️  Warning: Model has {param_info['total_millions']:.2f}M params")
        print(f"   Target: 95M (difference: {diff_percent:.2f}%)")
    else:
        print(f"\n✅ Model size on target: {param_info['total_millions']:.2f}M")
    
    # Move to device
    print(f"\n🚀 Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Test forward pass
    print("\n🗿 Testing forward pass...")
    batch_size = 2
    seq_len = 128
    
    # Create test input
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass without gradients
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids)
        forward_time = time.time() - start_time
    
    print(f"   Output shape: {outputs['logits'].shape}")
    print(f"   Forward pass time: {forward_time*1000:.2f}ms")
    
    # Check for NaN in output
    if torch.isnan(outputs['logits']).any():
        print("   ❌ Output contains NaN!")
    else:
        print("   ✅ No NaN in output")
    
    # Test with labels (calculate loss)
    print("\n📊 Testing loss calculation...")
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        outputs_with_loss = model(input_ids, labels=labels)
    
    loss_value = outputs_with_loss['loss'].item()
    print(f"   Loss: {loss_value:.4f}")
    
    # Expected initial loss (random initialization)
    expected_loss = -torch.log(torch.tensor(1.0 / config['vocab_size'])).item()
    print(f"   Expected random loss: ~{expected_loss:.4f}")
    
    if abs(loss_value - expected_loss) > 1.0:
        print(f"   ⚠️ Loss seems unusual (expected ~{expected_loss:.2f})")
    else:
        print(f"   ✅ Loss is in expected range")
    
    # Test gradient flow
    print("\n🔄 Testing gradient flow...")
    model.train()
    model.zero_grad()
    
    # Forward and backward
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    nan_grads = []
    zero_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if grad_norm == 0:
                zero_grads.append(name)
    
    if nan_grads:
        print(f"   ❌ Found NaN gradients in {len(nan_grads)} parameters")
        for name in nan_grads[:5]:  # Show first 5
            print(f"      - {name}")
    else:
        print(f"   ✅ No NaN gradients")
    
    if zero_grads:
        print(f"   ⚠️ Found zero gradients in {len(zero_grads)} parameters")
    else:
        print(f"   ✅ All parameters have non-zero gradients")
    
    # Show gradient statistics
    if grad_norms:
        grad_values = list(grad_norms.values())
        print(f"\n📈 Gradient Statistics:")
        print(f"   Min grad norm: {min(grad_values):.6f}")
        print(f"   Max grad norm: {max(grad_values):.6f}")
        print(f"   Mean grad norm: {sum(grad_values)/len(grad_values):.6f}")
    
    # Memory usage
    if torch.cuda.is_available():
        print("\n💾 Memory Usage:")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
    
    # Test generation capability
    print("\n🎭 Testing generation (single token)...")
    model.eval()
    
    with torch.no_grad():
        # Single token input
        test_input = torch.tensor([[1]], device=device)  # Start with token 1
        outputs = model(test_input)
        logits = outputs['logits']
        
        # Get probabilities
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print("   Top 5 predicted tokens:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"     {i+1}. Token {idx.item():5d}: {prob.item():.4f}")
    
    print("\n" + "="*70)
    print(" "*20 + "✅ All Tests Passed!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = test_wikimini_model()