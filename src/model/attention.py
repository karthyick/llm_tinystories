"""Multi-Head Attention with RoPE integration and memory optimizations.

Critical implementation details:
1. Apply RoPE only to Q and K, never to V
2. Use SDPA for Flash Attention 2 support
3. Pre-normalization architecture
4. Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .rope import RotaryPositionEmbeddings


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE and Flash Attention support.
    
    This implementation:
    - Uses Rotary Position Embeddings (RoPE) on Q and K only
    - Supports Flash Attention 2 via torch.nn.functional.scaled_dot_product_attention
    - Uses no bias terms (modern approach)
    - Includes proper causal masking
    - Memory-efficient implementation
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
        rope_percentage: float = 0.5,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Windows Flash Attention: Test with PyTorch 2.10+ nightly
        # Older versions had freezing issues, but newer versions may work
        import sys
        import logging
        logger = logging.getLogger(__name__)

        if sys.platform == 'win32' and use_flash_attention:
            # Allow Flash Attention on Windows with PyTorch 2.10+
            # If freezing occurs, set use_flash_attention: false in config
            self.use_flash_attention = use_flash_attention
            logger.info("[Windows] Attempting Flash Attention with PyTorch 2.10+ - if freezing occurs, disable in config")
        elif sys.platform == 'win32':
            self.use_flash_attention = False
            logger.info("[Windows] Flash Attention disabled - using manual attention")
        else:
            self.use_flash_attention = use_flash_attention

        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projections (no bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE for positional encoding
        # Apply to only part of head dimensions (typically 50%)
        rope_dim = int(self.head_dim * rope_percentage)
        self.rope_dim = rope_dim
        self.rope = RotaryPositionEmbeddings(
            head_dim=rope_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Pre-allocate causal mask more efficiently
        # We'll create it on-demand based on sequence length
        self.register_buffer('cached_mask', None, persistent=False)
        self.register_buffer('cached_mask_size', torch.tensor(0), persistent=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal mask for the given sequence length.

        CRITICAL: Always returns mask on the specified device to prevent CPU OOM errors.
        """
        if self.cached_mask is None or self.cached_mask_size < seq_len:
            # Create a new mask directly on the target device
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.cached_mask = mask
            self.cached_mask_size = torch.tensor(seq_len)

        # CRITICAL: Ensure the returned mask is on the correct device
        # This prevents CPU OOM when broadcasting during attn_scores + causal_mask
        return self.cached_mask[:seq_len, :seq_len].to(device)
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to partial dimensions of Q and K.
        
        Args:
            q: Query tensor [batch, seq_len, n_heads, head_dim]
            k: Key tensor [batch, seq_len, n_heads, head_dim]
            position_ids: Optional custom position IDs
        
        Returns:
            Rotated Q and K tensors
        """
        # Split into RoPE and pass-through dimensions
        if self.rope_dim > 0:
            q_rope, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_rope, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
            
            # Apply RoPE to the first part
            q_rope, k_rope = self.rope(q_rope, k_rope, position_ids)
            
            # Concatenate back
            q = torch.cat([q_rope, q_pass], dim=-1)
            k = torch.cat([k_rope, k_pass], dim=-1)
        
        return q, k
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            use_cache: Whether to return KV cache for inference
            past_kv: Past key-value cache for inference

        Returns:
            Output tensor and optional KV cache
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)  # [batch, seq_len, d_model]
        v = self.v_proj(x)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K only (not V!)
        q, k = self._apply_rope(q, k, position_ids)
        
        # Handle KV cache for inference
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        kv_cache = (k, v) if use_cache else None
        
        # Transpose for attention computation
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Use Flash Attention 2 via SDPA when available
        # This is MUCH more memory efficient than manual attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention 2 is automatically used when available
            # It handles the causal mask internally when is_causal=True
            # NOTE: Windows compatibility - skip context manager to avoid freezing
            import sys
            if sys.platform == 'win32':
                # On Windows, use SDPA without explicit kernel selection
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True if attention_mask is None else False,
                    scale=self.scale,
                )
            else:
                # On Linux, use explicit kernel selection for best performance
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,  # Use Flash Attention when possible
                    enable_math=True,   # Fallback to math implementation
                    enable_mem_efficient=True  # Use memory-efficient attention
                ):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True if attention_mask is None else False,
                        scale=self.scale,
                    )
        else:
            # Manual attention computation (fallback)
            # This is memory-intensive and should only be used for small sequences
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if attention_mask is None:
                causal_mask = self._get_causal_mask(seq_len, x.device)
                # Expand mask for batch and heads
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_scores = attn_scores + causal_mask
            else:
                attn_scores = attn_scores + attention_mask

            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Compute output
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output, kv_cache


# Test the attention implementation
def test_attention():
    """Test multi-head attention with various configurations."""
    print("Testing Multi-Head Attention...")
    
    # Test configuration
    batch_size = 2
    seq_len = 128
    d_model = 768
    n_heads = 12
    
    # Create attention module
    attention = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        max_seq_len=2048,
        rope_percentage=0.5,
        use_flash_attention=True,  # Enable Flash Attention
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention = attention.to(device)
    attention.eval()  # Set to eval mode for testing
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
    
    # Forward pass
    with torch.no_grad():
        output, _ = attention(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check for NaN
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    
    print("✓ Multi-Head Attention test passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Device: {device}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    return True


if __name__ == "__main__":
    test_attention()