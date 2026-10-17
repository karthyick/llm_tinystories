# TinyStories: dense vs hybrid (Mac run)

Device: `mps`; seed 17; 60 steps per model; 64 tokens/sequence; 2 sequences/step.
Data: first 5 MB of TinyStories train text (4,948,281 characters), one shared 4,000-token BPE; first 90% train, final 10% held out.
Validation reports main next-token CE only. Runtime and tokens/s are wall-clock for training.

| Model | Parameters | Train loss (step:loss) | Held-out main CE | Tokens/s | Train seconds |
|---|---:|---|---:|---:|---:|
| Dense WikiMini | 185,504 | 1:8.295, 15:7.695, 30:6.917, 45:6.519, 60:6.153 | 6.1171 | 18065.3 | 0.4 |
| HybridLM | 198,648 | 1:33.051, 15:25.355, 30:18.006, 45:13.747, 60:13.203 | 11.4875 | 4624.4 | 1.7 |

## Dense WikiMini samples

1. One day, a little girl named Lily........................
2.  found a needle in her room. She........................

## HybridLM samples

1. One day, a little girl named Lily. nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse
2.  found a needle in her room. She nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse nurse
