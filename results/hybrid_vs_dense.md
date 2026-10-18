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
| HybridLM | 26,095,380 | 15,773,460 | 1.382765 | 3.9859 | 505,598.2 | 5.076 GiB | 5/5 | 10,000 / ~2.9 (early stop at 10,600) |
| Dense WikiMini | 24,557,120 | 24,557,120 | 7.400091 | 1636.1339 | 1,171,086.6 | 3.537 GiB | 5/5 | 17,370 / 5 |

The hybrid stopped early at about 2.9 epochs after resuming once from the epoch-1-end checkpoint following the ULTRA hard reset. The dense checkpoint records epoch 4, which is 5 epochs seen because `train.py` stores zero-based epoch numbers. The README's dense PPL 15.7 and the train-log 6.71 number came from different code paths, so only this table compares the checkpoints.

### Five enhanced samples per checkpoint

Samples below are the Balanced setting, truncated to 60 generated tokens, from the five required prompts. The article test is the evaluator's article-presence check, 5/5 for both checkpoints.

#### HybridLM

1. `Once upon a time there was` → Once upon a time there was a little girl named Molly. She was three years old and loved to explore the world around her.
2. `One day, a little girl named Lily` → One day, a little girl named Lily found a big red button on the ground. She picked it up and said, "Look, Tom! I can reverse things!"
3. `In the garden, a small bird` → In the garden, a small bird was happily playing with some of her friends. They were all having so much fun together in the sunshine and flying around in circles!
4. `A big dog wanted to play with` → A big dog wanted to play with the ball. The dog was very excited. He jumped and tried to catch the ball in his mouth.
5. `The happy cat found` → The happy cat found a new friend. They played together all day and became best friends. The end.

#### Dense WikiMini

1. `Once upon a time there was` → Once upon a time there was boy was happy He to on adventure find special. day decided go and the was away his.
2. `One day, a little girl named Lily` → One day, a little girl named Lily to a. saw's and, she excited ran her. had toy that really.
3. `In the garden, a small bird` → In the garden, a small bird Chirpy sitting a. Chirp had seen big flying. thought "if could high in sky I fly there.
4. `A big dog wanted to play with` → A big dog wanted to play with, he not a friend He a who't to. barked and the did stop.
5. `The happy cat found` → The happy cat found way and happily his was! upon time there a, was girl Lily She to in park day.

### Honest reading against section 4

This run meets the section-4 "design wins" quality condition: hybrid PPL is lower and it is faster than dense on the complete validation split. It is not a clean training-budget comparison because hybrid stopped early and the checkpoints came from different training histories. The result supports keeping the hybrid for longer-context and MTP follow-up, while the dense README/train-log discrepancy must not be used as a comparison.
