"""Diagnose environment and imports."""

import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current directory:", os.getcwd())
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\n" + "="*60)
print("Testing imports...")
print("="*60)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

print("\n" + "="*60)
print("Testing local module imports...")
print("="*60)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.model.rmsnorm import RMSNorm
    print("✓ RMSNorm imported successfully")
except ImportError as e:
    print(f"✗ RMSNorm import failed: {e}")

try:
    from src.model.rope import RotaryPositionEmbeddings
    print("✓ RoPE imported successfully")
except ImportError as e:
    print(f"✗ RoPE import failed: {e}")

try:
    from src.model.swiglu import SwiGLU
    print("✓ SwiGLU imported successfully")
except ImportError as e:
    print(f"✗ SwiGLU import failed: {e}")

print("\n" + "="*60)
print("Environment diagnosis complete.")
print("="*60)