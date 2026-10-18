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

## S4/S5 full-checkpoint comparison

The same evaluator path used the complete `data/cache/tinystories_validation_512.pkl` (8,961 sequences, 4,578,810 non-padding tokens), batch size 64, and CUDA bf16 autocast. Both checkpoint configs use `tokenizer/tinystories_10k` and vocab 10,000. CE is main next-token CE only. Peak memory is allocated GiB.

| Model | Total params | Active params | Val CE | Val PPL | Tokens/s | Peak mem | Article test | Train steps / epochs seen |
|---|---:|---:|---:|---:|---:|---:|---|---|
| HybridLM | 26,095,380 | 15,773,460 | 1.382765 | 3.9859 | 505,891 | 5.076 GiB | 5/5 | 10,000 / ~2.9 (early stop at 10,600) |
| Dense WikiMini | 24,557,120 | 24,557,120 | 1.384790 | 3.9940 | 1,191,558.6 | 3.537 GiB | 5/5 | 6,948 / 2 |

Both checkpoints use the `tokenizer/tinystories_10k` vocabulary of 10,000 tokens. The old ml-workspace dense checkpoint was rejected because it scores CE 7.7 on today's cache with an older tokenizer, so dense was retrained on the same pipeline. The hybrid checkpoint is best at step 10,000, about 2.9 epochs seen, and the dense checkpoint is best at step 6,948, 2 epochs seen. Both runs early-stopped at step 10,600, and hybrid resumed once after the ULTRA hard reset. The train.py hybrid validation loss of 1.90 includes 0.3 x MTP, so only this table's main-head CE is comparable.

### Five enhanced samples per checkpoint

Samples below are the Balanced setting, truncated to 60 generated tokens, from the five required prompts. The article test is the evaluator's article-presence check, 5/5 for both checkpoints.

#### HybridLM

1. `Once upon a time there was` → Once upon a time there was a little girl named Molly. She was three years old and loved to explore the world around her.
2. `One day, a little girl named Lily` → One day, a little girl named Lily found a big red button on the ground. She picked it up and said, "Look, Tom! I can reverse things!"
3. `In the garden, a small bird` → In the garden, a small bird was happily playing with some of her friends. They were all having so much fun together in the sunshine and flying around in circles!
4. `A big dog wanted to play with` → A big dog wanted to play with the ball. The dog was very excited. He jumped and tried to catch the ball in his mouth.
5. `The happy cat found` → The happy cat found a new friend. They played together all day and became best friends. The end.

#### Dense WikiMini

1. `Once upon a time there was` → Once upon a time there was a little girl named Lucy. She liked to play outside in the garden and watch the birds fly by. One day, when she went out on her bike, she saw something strange. There were t...
2. `One day, a little girl named Lily` → One day, a little girl named Lily went to the park with her mommy. They brought some bread for lunch and cookies to eat. When they arrived at the park, they saw that it was gloomy outside. Lily said...
3. `In the garden, a small bird` → In the garden, a small bird flew down to Jack and said, "Hi Jack! What's your name?" Jack replied, "My name is Jack. Nice to meet you!" The two of them started talking together for hours about all sor...
4. `A big dog wanted to play with` → A big dog wanted to play with the little cat. The big dog was not nice. He told the little cat, "You are too small and slow." But the little cat did not want to listen. She wanted to be friends with t...
5. `The happy cat found` → The happy cat found a new home and was very grateful. From then on, the cat would always look out for other cats who had come to visit and have lots of fun together!Once upon a time there were two fri...

### Honest reading against section 4

Quality is a tie: the CE gap is 0.002, inside run noise. Hybrid uses 36% fewer active parameters, but is about 2.3x slower in tokens/s and uses about 1.4x more peak memory because it uses a pure-Torch Mamba-2 scan without a fused kernel. Against section 4, this is the "hybrid PPL within +5% of dense" tie outcome, not a design win. Keep it for long-context and MTP follow-up, where the design has a reason to earn back its current speed and memory cost.
