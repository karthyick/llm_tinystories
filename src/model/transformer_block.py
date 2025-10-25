"""Transformer block with pre-normalization architecture and memory optimizations.

Critical implementation details:
1. Pre-normalization: RMSNorm BEFORE attention and FFN
2. Residual connections after each sub-layer
3. Modern component stack: RoPE + RMSNorm + SwiGLU
4. Gradient checkpointing support for memory efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from torch.utils.checkpoint import checkpoint

from .rmsnorm import RMSNorm
from .attention import MultiHeadAttention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization.
    
    This follows the modern architecture used in LLaMA, Mistral, etc:
    - Pre-normalization with RMSNorm
    - Multi-head attention with RoPE
    - SwiGLU activation in FFN
    - Residual connections
    - Gradient checkpointing support
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
        rope_percentage: float = 0.5,
        rms_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Pre-normalization layers
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        
        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_percentage=rope_percentage,
            use_flash_attention=use_flash_attention,
        )
        
        # SwiGLU FFN
        # Default hidden dimension: 8/3 * d_model for parameter parity
        if d_ffn is None:
            d_ffn = int(8 * d_model / 3)
            # Round to multiple of 256 for hardware efficiency
            d_ffn = 256 * ((d_ffn + 255) // 256)
        
        self.ffn = SwiGLU(
            input_dim=d_model,
            hidden_dim=d_ffn,
            output_dim=d_model,
            bias=False,
        )
    
    def _attention_block(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Attention sub-block with pre-norm."""
        # Pre-normalization
        x_norm = self.attn_norm(x)
        
        # Multi-head attention
        attn_output, kv_cache = self.attention(
            x_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_kv=past_kv,
        )
        
        # Residual connection
        return attn_output, kv_cache
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block with pre-norm."""
        # Pre-normalization
        x_norm = self.ffn_norm(x)
        
        # Feed-forward
        ffn_output = self.ffn(x_norm)
        
        return ffn_output
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            use_cache: Whether to return KV cache
            past_kv: Past key-value cache
        
        Returns:
            Output tensor and optional KV cache
        """
        # Attention block with residual
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            def attention_fn(x_in):
                attn_out, _ = self._attention_block(
                    x_in,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,  # Can't use cache with checkpointing
                    past_kv=None,
                )
                return attn_out
            
            attn_output = checkpoint(attention_fn, x, use_reentrant=False)
            kv_cache = None
        else:
            attn_output, kv_cache = self._attention_block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                past_kv=past_kv,
            )
        
        # Add residual for attention
        x = x + attn_output
        
        # FFN block with residual
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for FFN as well
            ffn_output = checkpoint(self._ffn_block, x, use_reentrant=False)
        else:
            ffn_output = self._ffn_block(x)
        
        # Add residual for FFN
        x = x + ffn_output
        
        return x, kv_cache


class WikiMiniModel(nn.Module):
    """Complete WikiMini 95M language model.
    
    Architecture:
    - Token embeddings with weight tying
    - Stack of transformer blocks
    - Final RMSNorm
    - LM head (tied with embeddings)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract config values with defaults
        self.vocab_size = config.get('vocab_size', 32000)
        self.d_model = config.get('d_model', 768)
        self.n_layers = config.get('n_layers', 12)
        self.n_heads = config.get('n_heads', 12)
        self.d_ffn = config.get('d_ffn', None)
        self.max_seq_len = config.get('max_seq_len', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.rope_percentage = config.get('rope_percentage', 0.5)
        self.rope_base = config.get('rope_base', 10000)
        self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)
        self.tie_embeddings = config.get('tie_embeddings', True)
        self.use_flash_attention = config.get('use_flash_attention', True)
        self.use_gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ffn=self.d_ffn,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len,
                rope_base=self.rope_base,
                rope_percentage=self.rope_percentage,
                rms_norm_eps=self.rms_norm_eps,
                use_flash_attention=self.use_flash_attention,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )
            for _ in range(self.n_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(self.d_model, eps=self.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Weight tying
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all transformer blocks."""
        self.use_gradient_checkpointing = True
        for block in self.blocks:
            block.use_gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for all transformer blocks."""
        self.use_gradient_checkpointing = False
        for block in self.blocks:
            block.use_gradient_checkpointing = False

    def count_parameters(self) -> dict:
        """Count model parameters by component.

        Returns:
            Dictionary with parameter counts for each component
        """
        # Count by component type
        embedding_params = sum(p.numel() for p in self.token_embedding.parameters())

        attention_params = 0
        ffn_params = 0
        norm_params = 0

        for block in self.blocks:
            # Attention parameters
            attention_params += sum(p.numel() for p in block.attention.parameters())
            # FFN parameters
            ffn_params += sum(p.numel() for p in block.ffn.parameters())
            # Norm parameters (attention + ffn norms)
            norm_params += sum(p.numel() for p in block.attn_norm.parameters())
            norm_params += sum(p.numel() for p in block.ffn_norm.parameters())

        # Final norm
        norm_params += sum(p.numel() for p in self.final_norm.parameters())

        # LM head (only if not tied)
        if not self.tie_embeddings:
            lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        else:
            lm_head_params = 0  # Shared with embeddings

        total_params = sum(p.numel() for p in self.parameters())

        return {
            'total': total_params,
            'total_millions': total_params / 1e6,
            'embedding': embedding_params,
            'attention': attention_params,
            'ffn': ffn_params,
            'norm': norm_params,
            'lm_head': lm_head_params,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            labels: Optional labels for language modeling loss
            use_cache: Whether to return KV cache
            past_key_values: Past KV cache for inference

        Returns:
            Dictionary with 'logits' and optionally 'loss' and 'past_key_values'
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Apply dropout to embeddings
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Process through transformer blocks
        past_key_values_out = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            # Get past KV for this layer if available
            past_kv = past_key_values[i] if past_key_values is not None else None

            # Process through block
            x, kv_cache = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                past_kv=past_kv,
            )

            # Store KV cache if needed
            if use_cache:
                past_key_values_out.append(kv_cache)

        # Final normalization
        x = self.final_norm(x)

        # Language modeling head
        logits = self.lm_head(x)
        
        # Prepare output
        output = {'logits': logits}

        # Calculate loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Calculate cross-entropy loss
            loss = nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,  # Standard ignore index
            )

            output['loss'] = loss

        # Add KV cache to output if requested
        if use_cache:
            output['past_key_values'] = past_key_values_out

        return output


def create_model(config: Dict[str, Any]) -> WikiMiniModel:
    """Create a WikiMini model from configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        WikiMiniModel instance
    """
    return WikiMiniModel(config)


# Test the complete model
if __name__ == "__main__":
    # Test configuration for ~95M parameters
    config = {
        'vocab_size': 32000,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'd_ffn': 2048,  # Adjusted for SwiGLU
        'max_seq_len': 2048,
        'dropout': 0.1,
        'rope_percentage': 0.5,
        'rope_base': 10000,
        'rms_norm_eps': 1e-6,
        'tie_embeddings': True,
        'use_flash_attention': True,
        'gradient_checkpointing': True,  # Enable for memory efficiency
    }
    
    # Create model
    model = WikiMiniModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"WikiMini Model:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Layers: {model.n_layers}")
    print(f"  Hidden size: {model.d_model}")
    print(f"  Attention heads: {model.n_heads}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Small test batch
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    # Enable gradient checkpointing
    model.enable_gradient_checkpointing()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    print("\n✓ Model test passed!")