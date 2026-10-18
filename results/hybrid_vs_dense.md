# TinyStories: dense vs hybrid

Device `cuda`; seed 17; 1200 optimizer steps/model; seq_len 128; batch 32; AdamW with 60-step linear warmup and cosine decay; MTP weight decays 0.3 to 0.1.
Same first 5 MB TinyStories train slice (4,948,280 chars), shared 4,000-token BPE and 90/10 token split. Equal token budget: 4,915,200 train tokens/model.
Active params count all dense/shared weights plus top-k routed expert weights; total params count all experts. Init/train/validation CE is main next-token CE; optimizer includes MTP and balance auxiliary objectives.

| Model | Total params | Active params | Init main CE | Train main CE (step:value) | Held-out main CE | Tokens/s | Train seconds |
|---|---:|---:|---:|---|---:|---:|---:|
| Dense WikiMini | 2,611,328 | 2,611,328 | 8.3200 | 1:8.319, 120:4.726, 300:3.794, 600:3.167, 900:2.932, 1200:2.818 | 3.0616 | 160111.1 | 30.7 |
| HybridLM MoE | 2,348,624 | 1,463,888 | 8.3220 | 1:8.322, 120:4.327, 300:3.639, 600:3.129, 900:2.896, 1200:2.851 | 3.0927 | 68846.9 | 71.4 |
| HybridLM dense FFN | 1,459,792 | 1,459,792 | 8.3079 | 1:8.315, 120:4.331, 300:3.670, 600:3.164, 900:2.967, 1200:2.889 | 3.1313 | 216527.5 | 22.7 |

## Dense WikiMini samples

1. One day, a little girl named Lily went to the park with her mom. They saw a big, red ball. Lily wanted to play with it, but she didn't want to play with it
2.  found a needle in her room. She was so excited to see the most beautiful flower. She ran to the door and saw a big, red ball. She was so happy and she said, "

## HybridLM MoE samples

1. One day, a little girl named Lily went to the park with her mom. She saw a big, red ball and wanted to play with it. Lily was scared and wanted to play with the ball
2.  found a needle in her room. She was so excited to see it. She ran to the door and saw a big, shiny ball. She was so excited to see it. She looked around and

## HybridLM dense FFN samples

1. One day, a little girl named Lily went to the park. She saw a big, shiny bird. She was so excited to see the bird. She was so excited. She asked her mom,
2.  found a needle in her room. She was so happy and excited. She was so happy and thanked the man for her help.<\|endoftext\|>Once upon a time, there was a little girl named

## Reading

Dense WikiMini has the lowest held-out CE (3.062); HybridLM MoE is +0.031 above it after 1200 updates. This scale favors dense optimization: routed experts receive little data each, while recurrent mixing and routing add compute overhead. The hybrid shows no quality or speed advantage in this run.
