# 🔍 Model Generation Validation

## The Problem

Your training looks successful:
- ✅ Loss decreasing
- ✅ Perplexity good (7.37)
- ✅ Training data is clean
- ✅ Tokenizer works perfectly

But generation is broken:
- ❌ Missing articles ("a", "the")
- ❌ Producing "todlers" typo that's NOT in training data

## The Solution

**One comprehensive validation script** that finds exactly where generation breaks.

## Quick Start

### Step 1: Copy your trained checkpoint
```bash
# From your Windows machine, copy checkpoint to this repo
# e.g., copy to: checkpoints/best_model.pt
```

### Step 2: Run validation
```bash
python validate_model_generation.py \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer ./tokenizer/wikimini_32k \
    --prompt "Once upon a time there was"
```

Or use the example script:
```bash
./run_validation_example.sh
```

## What It Does

The script runs **5 critical tests** in sequence:

### 1️⃣ Model's Raw Probabilities
Shows what the model ACTUALLY predicts (before any generation code)
- **Key question:** Does model assign high probability to articles?
- **If YES:** Model is fine, generation code is broken
- **If NO:** Model didn't learn despite good perplexity

### 2️⃣ Greedy Generation
Always picks the highest probability token
- **Key question:** Does greedy produce good text?
- **If YES:** Issue is with sampling parameters (temperature, top-k)
- **If NO:** Model has learned wrong patterns

### 3️⃣ Temperature Sampling
Tests different sampling strategies
- **Key question:** Do higher temperatures break generation?
- Helps tune generation hyperparameters

### 4️⃣ Training-like Context
Tests model on familiar patterns
- **Key question:** Does model work on complete story openings?
- **If YES but short prompts fail:** Prompt sensitivity issue

### 5️⃣ Built-in generate() Method
Uses model's own generation function
- **Key question:** Does built-in method work better?
- Helps compare with custom generate.py

## Expected Output

```
TEST 1: Model's Raw Probability Distribution
Top 20 Most Likely Next Tokens:
 1. 'a' → 18.45%        ← Should be high!
 2. 'little' → 12.30%
 3. 'girl' → 8.20%

Specific Token Probabilities:
  'a' → 18.45% (rank #1)      ✅ GOOD
  'the' → 6.21% (rank #5)     ✅ GOOD
  'todlers' → 0.0001% (rank #8934) ✅ GOOD
```

## Diagnosis Guide

### Scenario A: Model Good, Generation Bad
```
TEST 1: Articles have high probability ✅
TEST 2: Greedy output good ✅
Your generate.py: Bad output ❌
→ Fix your generate.py implementation
```

### Scenario B: Model Didn't Learn
```
TEST 1: Articles have low probability ❌
TEST 2: Greedy output bad ❌
→ Training issue - model didn't learn articles
   Check: tokenizer, data preprocessing, retrain
```

### Scenario C: Temperature Too High
```
TEST 1: Articles high probability ✅
TEST 2: Greedy good ✅
TEST 3: Temperature 0.8 bad ❌
→ Lower temperature in generate.py
```

### Scenario D: Model Has "todlers" Typo
```
Any test shows "todlers" ❌
→ CRITICAL: Wrong checkpoint or data contamination
   Verify: correct checkpoint, correct training data
```

## Files

- **validate_model_generation.py** - Main validation script (comprehensive)
- **VALIDATION_GUIDE.md** - Detailed guide with interpretation
- **run_validation_example.sh** - Example runner script
- **This file** - Quick reference

## Why Validation After Training?

**Good perplexity ≠ Good generation**

Perplexity averages over all tokens. Your model could:
- Predict rare words perfectly (99% of vocab)
- Predict common words (articles) terribly (1% of vocab)
- Still get good overall perplexity!

But generation uses EVERY token, so one broken token type ruins output.

This script reveals:
1. What model ACTUALLY learned (probability distributions)
2. What generation code ACTUALLY does (sampling behavior)
3. Where the disconnect happens

## Next Steps

1. Run validation script on your trained model
2. Check TEST 1 first - are articles predicted correctly?
3. Based on results, either:
   - Fix generation code (if model is good)
   - Debug training/data (if model is bad)
   - Tune hyperparameters (if sampling is issue)

## Questions?

The validation script output is self-explanatory, but:
- See **VALIDATION_GUIDE.md** for detailed interpretation
- Each test logs exactly what to look for
- Final summary explains what to fix

---

**Bottom line:** This single script will tell you definitively whether your model learned correctly or if generation code is broken. No more guessing! 🎯
