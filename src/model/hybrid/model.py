"""M/A/F hybrid language model with shared embedding and a DeepSeek-style MTP head."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import MultiHeadAttention
from ..rmsnorm import RMSNorm
from ..swiglu import SwiGLU
from .mamba2 import Mamba2Mixer
from .moe import MoESwiGLU


class HybridLM(nn.Module):
    """Independent M/A/F residual layers; F is routed MoE or dense SwiGLU."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.pattern = config.get("layer_pattern", "MFMFMFAF").upper()
        if not self.pattern or any(kind not in "MAF" for kind in self.pattern):
            raise ValueError("layer_pattern must contain only M, A, and F")
        d = config["d_model"]
        self.mtp_weight = config.get("mtp_weight", 0.3)
        self.aux_weight = config.get("aux_weight", 1e-4)
        self.use_moe = config.get("use_moe", True)
        self.token_embedding = nn.Embedding(self.vocab_size, d)
        self.layers = nn.ModuleList()
        for kind in self.pattern:
            norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
            if kind == "M":
                block = Mamba2Mixer(d, config.get("mamba_heads", config["n_heads"]),
                                    config.get("state_size", 8), config.get("conv_kernel", 4))
            elif kind == "A":
                block = MultiHeadAttention(d, config["n_heads"], config.get("dropout", 0.0),
                    config.get("max_seq_len", 256), config.get("rope_base", 10000),
                    config.get("rope_percentage", 0.5), config.get("use_flash_attention", False))
            elif self.use_moe:
                block = MoESwiGLU(d, config.get("expert_ffn", config.get("d_ffn", 4 * d)),
                                  config.get("n_experts", 8), config.get("top_k", 2))
            else:
                block = SwiGLU(d, config.get("dense_ffn", config.get("d_ffn", 4 * d)), multiple_of=1)
            self.layers.append(nn.ModuleDict({"norm": norm, "block": block}))
        self.final_norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
        self.mtp_hidden_norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
        self.mtp_embed_norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
        self.mtp_projection = nn.Linear(2 * d, d, bias=False)
        self.mtp_block = nn.ModuleList([
            Mamba2Mixer(d, config.get("mamba_heads", config["n_heads"]),
                        config.get("state_size", 8), config.get("conv_kernel", 4))
            if config.get("mtp_block", "M") == "M" else SwiGLU(d, config.get("d_ffn", 4*d), multiple_of=1)
        ])
        self.lm_head = nn.Linear(d, self.vocab_size, bias=False)
        self.apply(self._init_weights)
        residual_scale = len(self.layers) ** -0.5
        for layer in self.layers:
            block = layer["block"]
            residual_projection = getattr(block, "out_proj", None) or getattr(block, "o_proj", None) or getattr(block, "w_down", None)
            if residual_projection is not None:
                with torch.no_grad():
                    residual_projection.weight.mul_(residual_scale)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        aux_losses = []
        for layer in self.layers:
            block = layer["block"]
            output = block(layer["norm"](x))
            if isinstance(output, tuple):
                output, aux = output
                if aux is not None:
                    aux_losses.append(aux)
            x = x + output
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        mtp_loss = logits.new_zeros(())
        loss = None
        main_loss = logits.new_zeros(())
        if labels is not None:
            main_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-100)
            if input_ids.shape[1] > 2:
                mtp_input = torch.cat([
                    self.mtp_hidden_norm(hidden[:, :-2]),
                    self.mtp_embed_norm(self.token_embedding(input_ids[:, 1:-1]))
                ], dim=-1)
                mtp_hidden = self.mtp_projection(mtp_input)
                mtp_hidden = mtp_hidden + self.mtp_block[0](mtp_hidden)
                mtp_logits = self.lm_head(self.final_norm(mtp_hidden))
                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, self.vocab_size), labels[:, 1:-1].reshape(-1), ignore_index=-100)
            loss = main_loss + self.mtp_weight * mtp_loss + self.aux_weight * aux_loss
        return {"logits": logits, "loss": loss, "main_loss": main_loss,
                "mtp_loss": mtp_loss, "aux_loss": aux_loss}
