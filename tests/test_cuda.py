"""Quick CUDA verification test."""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Test if we can actually create tensors on GPU
    print("\n" + "="*50)
    print("Testing GPU tensor creation and operations...")
    print("="*50)

    # Create a test tensor on GPU
    x = torch.randn(1000, 1000, device='cuda')
    print(f"Created tensor on GPU: device={x.device}, dtype={x.dtype}")

    # Test BF16
    if torch.cuda.is_bf16_supported():
        print(f"BFloat16 supported: YES")
        x_bf16 = x.to(torch.bfloat16)
        print(f"BF16 tensor: device={x_bf16.device}, dtype={x_bf16.dtype}")

        # Test matmul with BF16
        y_bf16 = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
        z = torch.matmul(x_bf16, y_bf16)
        print(f"BF16 matmul successful: {z.shape}")
    else:
        print(f"BFloat16 supported: NO")

    # Test a simple operation
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"Matrix multiplication result: shape={z.shape}, device={z.device}")

    print("\n✅ CUDA is working correctly!")
else:
    print("\n❌ CUDA is NOT available!")
