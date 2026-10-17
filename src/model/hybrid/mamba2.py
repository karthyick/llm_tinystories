"""Small, causal PyTorch implementation of a Mamba-2 style SSD mixer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rmsnorm import RMSNorm


class Mamba2Mixer(nn.Module):
    """Selective state-space mixer using a simple sequential SSD recurrence."""

    def __init__(self, d_model: int, n_heads: int = 4, state_size: int = 8,
                 conv_kernel: int = 4, rms_norm_eps: float = 1e-6):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.inner_dim = 2 * d_model
        self.head_dim, self.state_size = self.inner_dim // n_heads, state_size
        self.conv_kernel = conv_kernel
        inner = self.inner_dim
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, bias=True)
        self.x_proj = nn.Linear(inner, n_heads * (state_size * 2 + 1), bias=False)
        self.dt_proj = nn.Linear(n_heads, n_heads, bias=True)
        self.log_a = nn.Parameter(torch.zeros(n_heads))
        self.norm = RMSNorm(inner, eps=rms_norm_eps)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        # Left padding plus valid convolution ensures position t only sees <= t.
        u = F.silu(self.conv(F.pad(u.transpose(1, 2), (self.conv_kernel - 1, 0))).transpose(1, 2))
        params = self.x_proj(u).view(batch, length, self.n_heads, 2 * self.state_size + 1)
        b, c, raw_dt = params[..., :self.state_size], params[..., self.state_size:2*self.state_size], params[..., -1]
        dt = F.softplus(raw_dt + self.dt_proj(raw_dt)) + 1e-4
        a = -torch.exp(self.log_a).view(1, self.n_heads, 1, 1)
        state = x.new_zeros(batch, self.n_heads, self.head_dim, self.state_size)
        values = u.view(batch, length, self.n_heads, self.head_dim)
        outputs = []
        for t in range(length):
            dtt = dt[:, t, :, None, None]
            bt = b[:, t, :, None, :]
            ct = c[:, t, :, None, :]
            vt = values[:, t, :, :, None]
            state = torch.exp(a * dtt) * state + dtt * bt * vt
            outputs.append((state * ct).sum(dim=-1))
        y = torch.stack(outputs, dim=1).reshape(batch, length, -1)
        return self.out_proj(self.norm(y) * F.silu(gate))
