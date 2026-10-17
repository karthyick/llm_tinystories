# TinyStories: dense vs hybrid (Mac run)

Device `mps`; seed 17; 3 optimizer steps/model; seq_len 128; batch 2; AdamW with 20-step linear warmup and cosine decay.
Same first 5 MB TinyStories train slice (4,948,281 chars), shared 4,000-token BPE and 90/10 token split. Equal token budget: 768 train tokens/model.
Active params count all dense/shared weights plus top-k routed expert weights; total params count all experts. Init/train/validation CE is main next-token CE; optimizer includes MTP and balance auxiliary objectives.

| Model | Total params | Active params | Init main CE | Train main CE (step:value) | Held-out main CE | Tokens/s | Train seconds |
|---|---:|---:|---:|---|---:|---:|---:|
| Dense WikiMini | 1,860,192 | 1,860,192 | 8.3429 | 1:8.255, 2:8.317, 3:8.269 | 8.2439 | 6817.9 | 0.1 |
| HybridLM MoE | 1,669,392 | 1,005,840 | 8.3143 | 1:8.316, 2:8.321, 3:8.297 | 8.2928 | 1349.5 | 0.6 |
| HybridLM dense FFN | 1,002,736 | 1,002,736 | 8.3299 | 1:8.306, 2:8.295, 3:8.290 | 8.2784 | 3559.2 | 0.2 |

## Dense WikiMini samples

1. One day, a little girl named Lily................................
2.  found a needle in her room. She second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second

## HybridLM MoE samples

1. One day, a little girl named Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily
2.  found a needle in her room. She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She

## HybridLM dense FFN samples

1. One day, a little girl named Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily Lily
2.  found a needle in her room. She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She She

## Reading

Dense WikiMini has the lowest held-out CE (8.244); HybridLM MoE is +0.049 above it after 3 updates. This scale favors dense optimization: routed experts receive little data each, while recurrent mixing and routing add compute overhead. The hybrid shows no quality or speed advantage in this run.
