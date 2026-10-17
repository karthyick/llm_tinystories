"""Top-k routed SwiGLU experts with one always-active shared expert."""

import torch
import torch.nn as nn

from ..swiglu import SwiGLU


class MoESwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ffn, multiple_of=1) for _ in range(n_experts)])
        self.shared_expert = SwiGLU(d_model, d_ffn, multiple_of=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat)
        probs = logits.softmax(dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel():
                contribution = expert(flat.index_select(0, token_idx))
                routed.index_add_(0, token_idx, contribution * weights[token_idx, slot, None])
        importance = probs.mean(dim=0)
        load = torch.bincount(indices.reshape(-1), minlength=self.n_experts).to(probs.dtype)
        load = load / (flat.shape[0] * self.top_k)
        aux_loss = self.n_experts * torch.sum(importance * load)
        return (routed + self.shared_expert(flat)).view(shape), aux_loss
