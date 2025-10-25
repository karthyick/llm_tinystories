#!/usr/bin/env python3
"""
Test training setup before starting actual training.
This catches common issues early to avoid wasting time.
"""

import sys
import torch
from pathlib import Path
from src.data.tokenizer import load_tokenizer

def test_tokenizer():
    """Test that tokenizer loads correctly with all special tokens."""
    print("=" * 70)
    print("TEST 1: Tokenizer Loading")
    print("=" * 70)

    tokenizer_path = "./tokenizer/tinystories_10k"

    try:
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"✅ Tokenizer loaded from {tokenizer_path}")
        print(f"   Vocabulary size: {tokenizer.vocab_size:,}")

        # Check special tokens
        print("\nSpecial Tokens:")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        print(f"   eos_token_id: {tokenizer.eos_token_id}")
        print(f"   unk_token_id: {tokenizer.unk_token_id}")

        # Verify critical tokens exist
        assert tokenizer.pad_token_id is not None, "pad_token_id is None!"
        assert tokenizer.eos_token_id is not None, "eos_token_id is None!"
        assert tokenizer.unk_token_id is not None, "unk_token_id is None!"

        print("   ✅ All special tokens found")

        # Test encoding/decoding
        test_text = "Once upon a time there was a little girl."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        print(f"\nTest Encoding:")
        print(f"   Original: {test_text}")
        print(f"   Tokens: {len(encoded)} tokens")
        print(f"   Decoded: {decoded}")
        print(f"   Match: {'✅' if test_text.strip() == decoded.strip() else '❌'}")

        # Check for article tokens
        print(f"\nArticle Tokens:")
        for article in [" a", " the", " an"]:
            tokens = tokenizer.encode(article)
            if tokens:
                decoded = tokenizer.decode(tokens)
                print(f"   '{article}' → {tokens[0]:4d} → '{decoded}' ✅")
            else:
                print(f"   '{article}' → NOT FOUND ❌")

        return True

    except Exception as e:
        print(f"❌ Tokenizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test that config file is valid."""
    print("\n" + "=" * 70)
    print("TEST 2: Configuration File")
    print("=" * 70)

    try:
        import yaml
        config_path = "config/train_config_tinystories_33M_TOP10K.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"✅ Config loaded from {config_path}")

        # Check critical parameters
        critical_params = {
            'vocab_size': config['model']['vocab_size'],
            'learning_rate': config['optimizer']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'tokenizer_path': config['data']['tokenizer_path'],
        }

        print("\nCritical Parameters:")
        for key, value in critical_params.items():
            print(f"   {key}: {value}")

        # Verify expected values
        assert critical_params['vocab_size'] == 10000, f"vocab_size should be 10000, got {critical_params['vocab_size']}"
        # Use approximate comparison for float (YAML may parse as 0.0005)
        assert abs(critical_params['learning_rate'] - 5e-4) < 1e-10, f"learning_rate should be 5e-4, got {critical_params['learning_rate']}"
        assert critical_params['tokenizer_path'] == "./tokenizer/tinystories_10k", f"Wrong tokenizer path"

        print("   ✅ All critical parameters correct")

        return True

    except Exception as e:
        print(f"❌ Config test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test that model can be created with correct size."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Creation")
    print("=" * 70)

    try:
        import yaml
        from src.model.transformer_block import WikiMiniModel

        with open("config/train_config_tinystories_33M_TOP10K.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config['model']

        print(f"Creating model with config:")
        print(f"   vocab_size: {model_config['vocab_size']}")
        print(f"   d_model: {model_config['d_model']}")
        print(f"   n_layers: {model_config['n_layers']}")

        model = WikiMiniModel(model_config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Check expected size (should be around 23.5M with 10K vocab)
        expected_min = 23_000_000
        expected_max = 25_000_000

        if expected_min <= total_params <= expected_max:
            print(f"   ✅ Parameter count in expected range (23-25M)")
        else:
            print(f"   ⚠️  Parameter count outside expected range (23-25M)")

        # Test forward pass with dummy data
        print(f"\nTesting forward pass...")
        batch_size = 2
        seq_len = 64

        input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        labels = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output logits shape: {outputs['logits'].shape}")
        print(f"   Loss: {outputs['loss'].item():.4f}")
        print(f"   ✅ Forward pass successful")

        return True

    except Exception as e:
        print(f"❌ Model creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset loading and processing."""
    print("\n" + "=" * 70)
    print("TEST 4: Dataset Loading (Small Sample)")
    print("=" * 70)

    try:
        from datasets import load_dataset

        print("Loading small sample of TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        # Take first 3 examples
        samples = []
        for i, example in enumerate(dataset):
            if i >= 3:
                break
            samples.append(example['text'])

        print(f"✅ Loaded {len(samples)} sample stories")

        print(f"\nSample story:")
        print(f"   Length: {len(samples[0])} chars")
        print(f"   Preview: {samples[0][:100]}...")

        # Test tokenization
        tokenizer = load_tokenizer("./tokenizer/tinystories_10k")

        encoded = tokenizer.encode(samples[0])
        print(f"\nTokenization:")
        print(f"   Tokens: {len(encoded)}")
        print(f"   First 10 token IDs: {encoded[:10]}")

        return True

    except Exception as e:
        print(f"❌ Dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\n" + "=" * 70)
    print("TEST 5: CUDA/GPU Check")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")

        # Check memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total memory: {total_memory:.2f} GB")

        # Test tensor creation on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"   ✅ GPU tensor operations work")
        except Exception as e:
            print(f"   ❌ GPU tensor operations failed: {e}")
            return False

        return True
    else:
        print(f"❌ CUDA is NOT available")
        print(f"   Training will be very slow on CPU!")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING SETUP TESTS")
    print("=" * 70)
    print()

    results = {
        "Tokenizer": test_tokenizer(),
        "Config": test_config(),
        "Model": test_model_creation(),
        "Dataset": test_dataset(),
        "CUDA": test_cuda(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:15s} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED! Ready to start training!")
        print("\nRun: python train.py --config config/train_config_tinystories_33M_TOP10K.yaml")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED! Fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
