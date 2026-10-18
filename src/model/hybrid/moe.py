"""Top-k sigmoid-routed SwiGLU experts with aux-loss-free bias balancing."""

import torch
import torch.nn as nn

from ..swiglu import SwiGLU


class MoESwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_experts: int = 8, top_k: int = 2,
                 balance_bias_step: float = 1e-3):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts, self.top_k = n_experts, top_k
        self.balance_bias_step = balance_bias_step
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ffn, multiple_of=1) for _ in range(n_experts)])
        self.shared_expert = SwiGLU(d_model, d_ffn, multiple_of=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = self.router(flat).sigmoid()
        indices = (scores + self.expert_bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routed = torch.zeros_like(flat)
        counts = torch.bincount(indices.reshape(-1), minlength=self.n_experts)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel():
                routed.index_add_(0, token_idx, expert(flat.index_select(0, token_idx)) * weights[token_idx, slot, None])
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                mean_count = counts.float().mean()
                self.expert_bias.add_(self.balance_bias_step * torch.sign(mean_count - counts.float()))
        importance = scores.mean(dim=0)
        load = counts.to(scores.dtype) / max(1, flat.shape[0] * self.top_k)
        aux_loss = self.n_experts * torch.sum(importance * load)
        return (routed + self.shared_expert(flat)).view(shape), aux_loss
