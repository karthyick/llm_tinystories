"""SwiGLU (Swish-Gated Linear Unit) activation function implementation.

Critical implementation details:
1. Requires THREE weight matrices (gate, value, down-projection)
2. Hidden dimension should be adjusted to ~8/3 * d_model for parameter parity
3. No bias terms in modern implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation function.
    
    SwiGLU combines the Swish activation (SiLU) with a gating mechanism
    for improved gradient flow in deep networks.
    
    Based on the paper: 'GLU Variants Improve Transformer'
    https://arxiv.org/abs/2002.05202
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension (d_model)
            hidden_dim: Hidden dimension for FFN. If None, uses 8/3 * input_dim
            output_dim: Output dimension. If None, uses input_dim
            multiple_of: Round hidden_dim to nearest multiple for hardware efficiency
            bias: Whether to use bias terms (modern LLMs use False)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        # CRITICAL: Adjust hidden dimension for parameter parity
        # Standard FFN with ReLU/GELU uses 4 * d_model
        # SwiGLU needs 3 matrices, so use (8/3) * d_model for same param count
        if hidden_dim is None:
            hidden_dim = int(8 * input_dim / 3)
        
        # Round to nearest multiple for better hardware utilization
        if multiple_of > 1:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.hidden_dim = hidden_dim
        
        # Three linear projections required for SwiGLU
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=bias)  # Gate projection
        self.w_up = nn.Linear(input_dim, hidden_dim, bias=bias)    # Value/up projection
        self.w_down = nn.Linear(hidden_dim, self.output_dim, bias=bias)  # Down projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Formula: SwiGLU(x) = (Swish(xW_gate) ⊗ xW_up) W_down
        where Swish(x) = x * sigmoid(x) = SiLU(x)
        
        Args:
            x: Input tensor of shape [..., input_dim]
        
        Returns:
            Output tensor of shape [..., output_dim]
        """
        # Gate path with Swish/SiLU activation
        gate = F.silu(self.w_gate(x))
        
        # Value path (no activation)
        value = self.w_up(x)
        
        # Element-wise multiplication (gating)
        hidden = gate * value
        
        # Down projection to output dimension
        output = self.w_down(hidden)
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f'input_dim={self.input_dim}, '
            f'hidden_dim={self.hidden_dim}, '
            f'output_dim={self.output_dim}'
        )


class SwiGLUParallel(nn.Module):
    """Parallel version of SwiGLU that combines gate and up projections.
    
    This is more efficient as it reduces the number of separate matmuls.
    Used in models like LLaMA and Mistral.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        if hidden_dim is None:
            hidden_dim = int(8 * input_dim / 3)
        
        if multiple_of > 1:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.hidden_dim = hidden_dim
        
        # Combined gate and up projection for efficiency
        # Output shape: [batch, seq, 2 * hidden_dim]
        self.w_gate_up = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, self.output_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU with parallel projections."""
        # Single matmul for both gate and up projections
        gate_up = self.w_gate_up(x)
        
        # Split into gate and up components
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Apply SwiGLU
        hidden = F.silu(gate) * up
        output = self.w_down(hidden)
        
        return output


class GeGLU(nn.Module):
    """GELU-Gated Linear Unit - alternative to SwiGLU.
    
    Some models use GeGLU instead of SwiGLU. The difference is using
    GELU instead of SiLU for the gating activation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        if hidden_dim is None:
            hidden_dim = int(8 * input_dim / 3)
        
        self.hidden_dim = hidden_dim
        
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, self.output_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation."""
        gate = F.gelu(self.w_gate(x))
        value = self.w_up(x)
        hidden = gate * value
        output = self.w_down(hidden)
        return output


def calculate_ffn_params(d_model: int, activation: str = "swiglu") -> dict:
    """Calculate FFN parameters for different activation functions.
    
    This helper ensures parameter parity across different activation types.
    """
    if activation == "relu" or activation == "gelu":
        # Standard FFN: 2 matrices
        hidden_dim = 4 * d_model
        num_params = 2 * d_model * hidden_dim
    elif activation in ["swiglu", "geglu"]:
        # Gated FFN: 3 matrices, adjust hidden dimension
        hidden_dim = int(8 * d_model / 3)
        # Round to multiple of 256 for hardware efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        num_params = d_model * hidden_dim * 2 + hidden_dim * d_model
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    return {
        "activation": activation,
        "d_model": d_model,
        "hidden_dim": hidden_dim,
        "num_params": num_params,
        "params_millions": num_params / 1e6,
    }


# Example usage and parameter comparison
if __name__ == "__main__":
    d_model = 768
    
    # Compare parameter counts
    print("FFN Parameter Comparison:")
    for act in ["relu", "gelu", "swiglu"]:
        params = calculate_ffn_params(d_model, act)
        print(f"{act.upper()}:")
        print(f"  Hidden dim: {params['hidden_dim']}")
        print(f"  Parameters: {params['params_millions']:.2f}M")
    
    # Test SwiGLU
    batch_size, seq_len = 2, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    swiglu = SwiGLU(d_model)
    output = swiglu(x)
    
    print(f"\nSwiGLU Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"SwiGLU parameters: {sum(p.numel() for p in swiglu.parameters()) / 1e6:.2f}M")