# Model Generation Validation Guide

## Problem
Your training metrics look good (PPL 7.37), but generation is broken:
- Missing articles ("a", "the")
- Producing "todlers" typo that's NOT in training data

## Solution: validate_model_generation.py

This single script runs 5 comprehensive tests to find WHERE generation breaks.

## Usage

```bash
python validate_model_generation.py \
    --checkpoint checkpoints/your_model.pt \
    --tokenizer ./tokenizer/wikimini_32k \
    --prompt "Once upon a time there was"
```

## What It Tests

### Test 1: Model's Raw Probability Distribution
**Purpose:** Check what the model ACTUALLY predicts (independent of generation code)

**What to look for:**
- Are articles ("a", "the", "little") in top 20 predictions?
- What probability does model assign to "a" vs "todlers"?
- If "a" has high probability here but doesn't appear in generation → generation code bug
- If "a" has low probability here → model didn't learn properly

### Test 2: Greedy Generation (Always Pick Highest Probability)
**Purpose:** See the model's "best" output by always choosing most likely token

**What to look for:**
- Does greedy generation produce articles?
- Does "todlers" appear?
- If greedy is good but temperature sampling is bad → temperature/sampling bug
- If greedy is bad → model issue

### Test 3: Temperature Sampling
**Purpose:** Test different sampling strategies (what your generate.py uses)

**What to look for:**
- Compare temperature 0.8 vs 1.0
- Are articles disappearing with higher temperature?
- This tests if your actual generation parameters are the issue

### Test 4: Training-like Context Test
**Purpose:** See if model works on familiar patterns from training

**What to look for:**
- Does model predict well on complete story openings?
- If this works but short prompts don't → prompt sensitivity issue
- If this also fails → fundamental model problem

### Test 5: Built-in generate() Function
**Purpose:** Test the model's own generate() method with different configs

**What to look for:**
- Does temperature=0 (greedy) work?
- Does top-k or top-p sampling work?
- Compare with your generate.py to find differences

## Interpreting Results

### Scenario A: Model learned correctly, generation code broken
```
TEST 1: "a" has 15% probability (rank #2) ✅
TEST 2: Output has articles ✅
TEST 5: Built-in generate works ✅
Your generate.py: No articles ❌
→ FIX: Check your generate.py implementation
```

### Scenario B: Model didn't learn articles
```
TEST 1: "a" has 0.01% probability (rank #450) ❌
TEST 2: Output missing articles ❌
→ FIX: Training issue (despite good PPL)
  - Check if articles were accidentally filtered from data
  - Check tokenizer's handling of articles
  - May need to retrain
```

### Scenario C: Sampling hyperparameters too aggressive
```
TEST 1: "a" has good probability ✅
TEST 2: Greedy output has articles ✅
TEST 3: temp=0.8 no articles ❌
→ FIX: Lower temperature or adjust top-k/top-p in generate.py
```

### Scenario D: "todlers" appearing
```
Any test shows "todlers" ❌
→ CRITICAL: Model learned a typo that's not in training data
  - Check if model was trained on different data than you think
  - Check if checkpoint is from correct training run
  - May indicate data contamination
```

## Example Output

```
TEST 1: Model's Raw Probability Distribution
Top 20 Most Likely Next Tokens:
 1. 'a' → 18.45%        ← Should be high!
 2. 'little' → 12.30%
 3. 'girl' → 8.20%
...
20. 'todlers' → 0.01%   ← Should be very low or absent!

Specific Token Probabilities:
  'a' (id=10) → 18.4500% (rank #1)      ← GOOD
  'the' (id=15) → 6.2100% (rank #5)     ← GOOD
  'todlers' (id=2456) → 0.0001% (rank #8934) ← GOOD
```

## Quick Diagnosis

Run the script and check in order:

1. **First, look at TEST 1 specific token probabilities**
   - If articles have high probability → model is fine, generation code is broken
   - If articles have low probability → model didn't learn despite good perplexity

2. **Then check TEST 2 output**
   - If greedy works → issue is with sampling parameters
   - If greedy fails but TEST 1 is good → rare edge case, debug model.generate()

3. **Check for "todlers" anywhere**
   - If present → verify you're using correct checkpoint and data

## Notes

- Good perplexity (7.37) doesn't guarantee good generation
- Perplexity averages over all tokens; model might predict common words wrong
- Articles are very common, so even if model always predicts them wrong, impact on perplexity is small if it gets other words right

## After Running

Based on results:
- **If model is good:** Fix generate.py sampling code
- **If model is bad:** Check training data, retrain if needed
- **If uncertain:** Share TEST 1 output for diagnosis
