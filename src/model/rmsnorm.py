"""Root Mean Square Layer Normalization (RMSNorm) implementation.

Critical implementation details:
1. Use multiplication with rsqrt, NOT division
2. No mean subtraction (unlike LayerNorm)
3. Compute in FP32 for numerical stability even when using BF16/FP16
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that removes the mean subtraction
    and only performs re-scaling via root mean square.
    
    Based on the paper: 'Root Mean Square Layer Normalization'
    https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability (1e-6 for BF16, 1e-5 for FP16)
        """
        super().__init__()
        self.hidden_size = hidden_size
        # CRITICAL FIX: Ensure eps is stored as float, not string
        self.eps = float(eps) if isinstance(eps, str) else eps
        
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor.
        
        CRITICAL BUG TO AVOID:
        The most common bug is using division with torch.rsqrt:
        WRONG: x / torch.rsqrt(variance + eps)  # This is x * sqrt(variance)
        RIGHT: x * torch.rsqrt(variance + eps)  # This is x / sqrt(variance)
        
        Args:
            x: Input tensor of shape [..., hidden_size]
        
        Returns:
            Normalized tensor of same shape as input
        """
        # Store original dtype (for mixed precision training)
        input_dtype = x.dtype
        
        # CRITICAL: Compute in float32 for numerical stability
        x_float32 = x.float()
        
        # Compute RMS (root mean square)
        # RMS = sqrt(mean(x^2))
        variance = x_float32.pow(2).mean(dim=-1, keepdim=True)

        # CRITICAL: Use rsqrt (reciprocal square root) with multiplication
        # rsqrt(x) = 1/sqrt(x), so x * rsqrt(variance) = x / sqrt(variance)
        # PERFORMANCE FIX: PyTorch automatically broadcasts scalars, no need for tensor()
        x_normalized = x_float32 * torch.rsqrt(variance + self.eps)
        
        # Apply learned scale and cast back to original dtype
        return self.weight * x_normalized.to(input_dtype)
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, eps={self.eps}'


class RMSNormOptimized(nn.Module):
    """Optimized RMSNorm with optional fused operations.
    
    This version includes optimizations for better performance:
    1. Option for in-place operations
    2. Support for sequence parallelism
    3. Optional residual connection fusion
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # CRITICAL FIX: Ensure eps is stored as float, not string
        self.eps = float(eps) if isinstance(eps, str) else eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
    
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RMSNorm with optional residual connection.
        
        Args:
            x: Input tensor
            residual: Optional residual to add before normalization
        
        Returns:
            Normalized tensor (and residual if provided)
        """
        # Add residual if provided (pre-norm architecture)
        if residual is not None:
            x = x + residual
            residual = x  # Save for skip connection
        
        # Original dtype for mixed precision
        input_dtype = x.dtype
        
        # Compute in FP32
        if self.memory_efficient:
            # In-place operations to save memory
            x = x.float()
            variance = x.pow_(2).mean(dim=-1, keepdim=True)
            # PERFORMANCE FIX: Use scalar directly
            x.mul_(torch.rsqrt(variance + self.eps))
        else:
            # Standard computation
            x_float32 = x.float()
            variance = x_float32.pow(2).mean(dim=-1, keepdim=True)
            # PERFORMANCE FIX: Use scalar directly
            x = x_float32 * torch.rsqrt(variance + self.eps)
        
        # Apply weight and cast back
        if self.elementwise_affine:
            x = self.weight * x
        
        x = x.to(input_dtype)
        
        if residual is not None:
            return x, residual
        return x


def rmsnorm_func(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional version of RMSNorm for use in torch.compile or custom kernels.
    
    This can be used with torch.compile for better optimization.
    """
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    # Ensure eps is properly handled
    eps_val = float(eps) if isinstance(eps, str) else eps
    x = x * torch.rsqrt(variance + eps_val)
    return (weight * x).to(input_dtype)


# Comparison with LayerNorm for reference
def compare_normalization():
    """Compare RMSNorm with LayerNorm to understand the differences."""
    import torch.nn as nn
    
    batch_size, seq_len, hidden = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden)
    
    # LayerNorm: normalizes by mean and variance
    layer_norm = nn.LayerNorm(hidden)
    ln_out = layer_norm(x)
    
    # RMSNorm: normalizes by RMS only (no mean subtraction)
    rms_norm = RMSNorm(hidden)
    rms_out = rms_norm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"LayerNorm output shape: {ln_out.shape}")
    print(f"RMSNorm output shape: {rms_out.shape}")
    print(f"Mean difference: {(ln_out - rms_out).abs().mean().item():.6f}")
    
    # RMSNorm is 15-20% faster due to simpler computation
    return ln_out, rms_out