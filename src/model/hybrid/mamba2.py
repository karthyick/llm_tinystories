"""Causal Mamba-2 style mixer with sequential and quadratic SSD scans."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rmsnorm import RMSNorm


def sequential_scan(x, b, c, dt, a, d):
    """Reference recurrence; x [B,L,H,P], b/c [B,L,H,N], dt [B,L,H]."""
    state = x.new_zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1])
    outputs = []
    for t in range(x.shape[1]):
        delta = dt[:, t, :, None, None]
        state = torch.exp(a * delta) * state + delta * b[:, t, :, None, :] * x[:, t, :, :, None]
        outputs.append((state * c[:, t, :, None, :]).sum(dim=-1) + d * x[:, t])
    return torch.stack(outputs, dim=1)


def segsum_scan(x, b, c, dt, a, d):
    """Vectorized segmented-sum form of the same recurrence (O(L^2))."""
    # Inclusive cumulative log-decay; pair (s,t) carries decay s+1..t.
    log_decay = dt * a.reshape(1, 1, -1)
    cumulative = log_decay.cumsum(dim=1)
    length = x.shape[1]
    t = torch.arange(length, device=x.device)
    # [B,T,S,H], mask source positions after target positions.
    exponent = cumulative[:, :, None, :] - cumulative[:, None, :, :]
    valid = t[None, :] <= t[:, None]
    exponent = exponent.masked_fill(~valid[None, :, :, None], float("-inf"))
    decay = torch.exp(exponent)
    # Keep input channels distinct while summing source positions.
    state_terms = (decay[:, :, :, :, None, None] *
                   (dt[:, None, :, :, None, None] * b[:, None, :, :, None, :] *
                    x[:, None, :, :, :, None])).sum(dim=2)
    y = (state_terms * c[:, :, :, None, :]).sum(dim=-1) + d * x
    return y


class Mamba2Mixer(nn.Module):
    """Small single-group SSD mixer, faithful to Mamba-2 data flow."""

    def __init__(self, d_model: int, n_heads: int = 4, state_size: int = 8,
                 conv_kernel: int = 4, rms_norm_eps: float = 1e-6,
                 scan_impl: str = "segsum"):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.inner_dim = 2 * d_model
        self.head_dim, self.state_size = self.inner_dim // n_heads, state_size
        self.conv_kernel, self.scan_impl = conv_kernel, scan_impl
        inner = self.inner_dim
        self.in_proj = nn.Linear(d_model, inner + inner + 2 * state_size + n_heads, bias=False)
        conv_channels = inner + 2 * state_size
        self.conv = nn.Conv1d(conv_channels, conv_channels, conv_kernel, groups=conv_channels, bias=True)
        self.dt_bias = nn.Parameter(torch.empty(n_heads))
        self.a_log = nn.Parameter(torch.empty(n_heads))
        self.d_skip = nn.Parameter(torch.ones(n_heads))
        self.norm = RMSNorm(inner, eps=rms_norm_eps)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        nn.init.uniform_(self.a_log, math.log(1.0), math.log(16.0))
        # Initialize softplus(dt_bias) log-uniformly between 1e-3 and 0.1.
        dt0 = torch.exp(torch.empty(n_heads).uniform_(math.log(1e-3), math.log(0.1)))
        with torch.no_grad():
            self.dt_bias.copy_(torch.log(torch.expm1(dt0)))

    def scan(self, x, b, c, dt):
        a_vec = -torch.exp(self.a_log)
        d = self.d_skip.reshape(1, 1, self.n_heads, 1)
        if self.scan_impl == "sequential":
            return sequential_scan(x, b, c, dt, a_vec.reshape(1, self.n_heads, 1, 1), d.squeeze(1))
        return segsum_scan(x, b, c, dt, a_vec.reshape(1, 1, self.n_heads), d)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        raw = self.in_proj(hidden)
        i = self.inner_dim
        z, x, b, c, raw_dt = torch.split(raw, [i, i, self.state_size, self.state_size, self.n_heads], -1)
        conv_input = torch.cat([x, b, c], dim=-1)
        conv_input = F.pad(conv_input.transpose(1, 2), (self.conv_kernel - 1, 0))
        conv_output = F.silu(self.conv(conv_input).transpose(1, 2))
        x, b, c = torch.split(conv_output, [i, self.state_size, self.state_size], -1)
        x = x.view(batch, length, self.n_heads, self.head_dim)
        b = b.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        c = c.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        dt = F.softplus(raw_dt + self.dt_bias)
        y = self.scan(x, b, c, dt).reshape(batch, length, i)
        return self.out_proj(self.norm(y) * F.silu(z))
