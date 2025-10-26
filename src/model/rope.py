"""Rotary Position Embeddings (RoPE) implementation.

Critical implementation details:
1. Apply RoPE only to Q and K, never to V
2. Use head_dim, not full model dimension
3. Ensure proper dimension pairing for rotation
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionEmbeddings(nn.Module):
    """Rotary Position Embeddings (RoPE) for transformer models.
    
    Based on the paper: 'RoFormer: Enhanced Transformer with Rotary Position Embedding'
    https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # CRITICAL: head_dim must be even for proper pairing
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        
        # Precompute frequencies
        self._precompute_freqs(device)
    
    def _precompute_freqs(self, device: Optional[torch.device] = None):
        """Precompute the frequency tensor for RoPE."""
        # Calculate theta frequencies
        # theta_i = base^(-2i/d) for i in [0, 1, ..., d/2-1]
        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        
        # Create position indices
        positions = torch.arange(self.max_seq_len).float()
        
        # Compute outer product: [seq_len, head_dim/2]
        freqs = torch.einsum('i,j->ij', positions, theta)
        
        # Convert to cos and sin for rotation
        freqs_cos = torch.cos(freqs)  # [seq_len, head_dim/2]
        freqs_sin = torch.sin(freqs)  # [seq_len, head_dim/2]
        
        # Duplicate for full dimension coverage
        # [seq_len, head_dim/2] -> [seq_len, head_dim]
        freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
        freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
        
        # Register as buffers (not trainable, moves with model to device)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.
        
        CRITICAL: This is the most common bug - incorrect dimension pairing.
        For input [1, 2, 3, 4], output should be [-3, -4, 1, 2].
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
            position_ids: Optional custom position IDs
        
        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.shape[1]
        
        # Get the frequency tensors for current sequence length
        if position_ids is not None:
            freqs_cos = self.freqs_cos[position_ids]
            freqs_sin = self.freqs_sin[position_ids]
        else:
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        
        # Reshape for broadcasting
        # [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
        freqs_cos = freqs_cos[None, :, None, :]
        freqs_sin = freqs_sin[None, :, None, :]
        
        # Apply rotation using the formula:
        # x_rotated = x * cos + rotate_half(x) * sin
        q_rotated = q * freqs_cos + self.rotate_half(q) * freqs_sin
        k_rotated = k * freqs_cos + self.rotate_half(k) * freqs_sin
        
        return q_rotated, k_rotated
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - apply RoPE to Q and K only.
        
        CRITICAL: Never apply RoPE to V (value) tensor!
        """
        return self.apply_rotary_pos_emb(q, k, position_ids)


# Alternative implementation using complex numbers directly
class RotaryPositionEmbeddingsComplex(nn.Module):
    """Alternative RoPE implementation using complex number operations.
    
    This can be more efficient on some hardware but requires careful handling.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        
        # Precompute complex exponentials
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # Store as cos/sin values
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using cached cos/sin values."""
        if seq_len is None:
            seq_len = q.shape[1]
        
        # Apply rotation
        q_embed = (q * self.cos_cached[:, :seq_len]) + \
                  (self.rotate_half(q) * self.sin_cached[:, :seq_len])
        k_embed = (k * self.cos_cached[:, :seq_len]) + \
                  (self.rotate_half(k) * self.sin_cached[:, :seq_len])
        
        return q_embed, k_embed
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


# Test function for RoPE
def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE implementation...")
    
    batch_size = 2
    seq_len = 128
    n_heads = 12
    head_dim = 64
    
    # Create RoPE module
    rope = RotaryPositionEmbeddings(head_dim=head_dim, max_seq_len=2048)
    
    # Create dummy Q and K tensors
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Apply RoPE
    q_rot, k_rot = rope(q, k)
    
    # Check shapes
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} != {k.shape}"
    
    # Check for NaN
    assert not torch.isnan(q_rot).any(), "Q contains NaN after RoPE"
    assert not torch.isnan(k_rot).any(), "K contains NaN after RoPE"
    
    print("✓ RoPE test passed!")
    print(f"  Input shape: {q.shape}")
    print(f"  Output shape: {q_rot.shape}")
    
    return True


if __name__ == "__main__":
    test_rope()