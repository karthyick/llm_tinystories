# HybridLM design

```text
token IDs ──> tied embedding ──> [Mamba-2 | attention] × layer_pattern
                                      │ each layer:
                                      ├─ pre-norm mixer + residual
                                      └─ pre-norm top-k MoE + shared expert + residual
                         ┌────────────┴────────────┐
                         v                         v
                  final norm + LM head     MTP(t+2) light block
                         └────────────┬────────────┘
                         main CE + weighted MTP CE + weighted MoE aux
```

| HybridLM piece | NVIDIA Nemotron idea borrowed | Local implementation |
|---|---|---|
| `M` layers | Mostly Mamba-2 / SSD mixers | Input-dependent delta, scalar per-head decay, B/C state projections, causal depthwise convolution, gated output |
| `A` layers | Sparse attention in a hybrid stack | Existing `MultiHeadAttention` and RoPE |
| MoE FFN | Top-k routed experts plus shared expert | SwiGLU expert pool; one shared expert always runs; balancing auxiliary loss |
| MTP head | Multi-token prediction | One light head uses current and next input states to predict the t+2 label |
| Shared embedding/head | Parameter-efficient language model | Tied token embedding and vocabulary projection |

## Limits

- The SSD recurrence is a simple sequential PyTorch scan. It is strictly causal but slow and intended for short sequences and correctness experiments.
- This is an educational scale model, not a Nemotron checkpoint or faithful reproduction of NVIDIA's kernels, training recipe, or distributed expert routing.
- GPU deployment should replace the scan with optimized Mamba-2/SSD kernels and remeasure numerical behavior and throughput on KR-ULTRA's 5090.
- The MTP objective is auxiliary and its validation is excluded from dense-versus-hybrid comparison; only main next-token CE is reported.
