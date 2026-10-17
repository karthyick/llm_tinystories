# HybridLM design

```text
token IDs -> tied embedding
                 |
        +--------v---------+
        | M: Mamba-2 mixer |  norm -> mixer -> residual
        +--------+---------+
                 v
        +------------------+
        | F: MoE / SwiGLU  |  norm -> shared + selected experts -> residual
        +--------+---------+
                 v
        +------------------+
        | M: Mamba-2 mixer |  norm -> mixer -> residual
        +--------+---------+
                 v
        +------------------+       +-------------------------------+
        | A: attention     |       | MTP: norm(h[t]) || norm(e[t+1])|
        +--------+---------+       | -> projection -> one block     |
                 |                 | -> shared LM head -> t+2 CE    |
                 v                 +---------------+---------------+
        final RMSNorm + shared LM head               |
                 +--------------------+--------------+
                                      v
                         main CE + 0.3 * MTP CE + 1e-4 * balance loss
```

Pattern strings contain independent residual layers: `M` mixer, `A` attention, `F` feed-forward. Example `MFMFMFAF` begins with Mamba, interleaves FFNs, uses sparse attention, and ends with FFN.

| Piece | NVIDIA / DeepSeek source | What we did at tiny scale |
|---|---|---|
| M/A/F ordering | Nemotron-H, arXiv:2504.03624; Nemotron 3, arXiv:2512.20856 | Separate residual mixer and FFN layers; Mamba-first pattern with occasional attention and final FFN. |
| Mamba-2 | `state-spaces/mamba` `mamba2_simple.py`; Nemotron-H, arXiv:2504.03624 | Single projection to z/x/B/C/dt; causal depthwise width-4 convolution over x/B/C; one group; selective decay, D skip, and `norm(y * silu(z))` gated RMSNorm. Sequential reference plus quadratic segsum scan for training, with target/source transitions contracted to `[B,T,T,H]`. |
| MoE | DeepSeek-V3, arXiv:2412.19437; Nemotron 3, arXiv:2512.20856 | Sigmoid scores, top-2 among 8 SwiGLU experts, one shared SwiGLU expert; non-gradient bias adjusts by +/-1e-3 from batch load; balance loss weight 1e-4. |
| MTP | DeepSeek-V3, arXiv:2412.19437 | Concatenated normalized hidden and next-token embedding, projection, one Mamba block, shared embedding/LM head; predicts t+2 with 0.3 weight. |

## Limits

- The quadratic segmented-sum implementation is intended for short sequences; it uses `[B,T,T,H]` transitions and is checked against the sequential recurrence.
- No distributed expert routing or optimized CUDA kernels are included. Parameter counts and throughput describe this single-device experiment only.
- MTP and auxiliary losses do not enter held-out main next-token CE.
