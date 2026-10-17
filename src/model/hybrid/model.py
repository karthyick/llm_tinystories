"""Hybrid Mamba-2, attention, MoE, and multi-token prediction language model."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import MultiHeadAttention
from ..rmsnorm import RMSNorm
from .mamba2 import Mamba2Mixer
from .moe import MoESwiGLU


class HybridBlock(nn.Module):
    def __init__(self, config: dict[str, Any], kind: str):
        super().__init__()
        d = config["d_model"]
        self.mix_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        if kind == "A":
            self.mixer = MultiHeadAttention(d, config["n_heads"], config.get("dropout", 0.0),
                config.get("max_seq_len", 256), config.get("rope_base", 10000),
                config.get("rope_percentage", 0.5), config.get("use_flash_attention", False))
        else:
            self.mixer = Mamba2Mixer(d, config.get("mamba_heads", config["n_heads"]),
                config.get("state_size", 8), config.get("conv_kernel", 4))
        self.moe = MoESwiGLU(d, config.get("d_ffn", 4 * d), config.get("n_experts", 4), config.get("top_k", 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = self.mixer(self.mix_norm(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        ff, aux = self.moe(self.ffn_norm(x))
        return x + ff, aux


class HybridLM(nn.Module):
    """HybridLM(config) returns aligned main CE, t+2 MTP CE, and MoE aux loss."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.pattern = config.get("layer_pattern", "MMMAMMMA").upper()
        if not self.pattern or any(kind not in "MA" for kind in self.pattern):
            raise ValueError("layer_pattern must contain only M and A")
        d = config["d_model"]
        self.mtp_weight = config.get("mtp_weight", 0.1)
        self.aux_weight = config.get("aux_weight", 0.01)
        self.token_embedding = nn.Embedding(self.vocab_size, d)
        self.layers = nn.ModuleList([HybridBlock(config, kind) for kind in self.pattern])
        self.final_norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
        self.mtp_norm = RMSNorm(d, config.get("rms_norm_eps", 1e-6))
        self.mtp_block = nn.Sequential(nn.Linear(2*d, d, bias=False), nn.SiLU(), RMSNorm(d))
        self.lm_head = nn.Linear(d, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        aux_losses = []
        for layer in self.layers:
            x, aux = layer(x)
            aux_losses.append(aux)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        aux_loss = torch.stack(aux_losses).mean()
        mtp_loss = logits.new_zeros(())
        loss = None
        if labels is not None:
            main_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-100)
            # labels are pre-shifted; index 1 is the t+2 target for hidden state t.
            mtp_hidden = self.mtp_block(torch.cat([self.mtp_norm(hidden[:, :-1]), self.mtp_norm(self.token_embedding(input_ids[:, 1:]))], dim=-1))
            mtp_logits = self.lm_head(mtp_hidden)
            mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = main_loss + self.mtp_weight * mtp_loss + self.aux_weight * aux_loss
        else:
            main_loss = logits.new_zeros(())
        return {"logits": logits, "loss": loss, "main_loss": main_loss,
                "mtp_loss": mtp_loss, "aux_loss": aux_loss}
