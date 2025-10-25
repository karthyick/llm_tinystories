# Validation Script - Quick Start

## Your Checkpoint Location

```
checkpoints_wikimini\best_model.pth
```

## Run Validation (Windows)

```powershell
cd C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2

python validate_model_generation.py `
    --checkpoint checkpoints_wikimini\best_model.pth `
    --tokenizer .\tokenizer\wikimini_32k `
    --prompt "Once upon a time there was"
```

## What to Look For

### ✅ Good Model (Should Look Like This)
```
TEST 1: Model's Raw Probability Distribution
Specific Token Probabilities:
  'a' → 15-20% (rank #1-3)     ← HIGH = GOOD
  'the' → 5-10% (rank #3-10)   ← HIGH = GOOD
  'little' → 3-8% (rank #5-15) ← GOOD
  'todlers' → <0.01% (rank >1000) ← LOW = GOOD

TEST 2: Greedy Generation
Output: 'Once upon a time there was a little girl named...'
Contains articles: ✅ YES
```

### ❌ Bad Model (Problem Indicators)
```
TEST 1: Model's Raw Probability Distribution
Specific Token Probabilities:
  'a' → 0.5% (rank #234)       ← LOW = BAD!
  'todlers' → 2.5% (rank #15)  ← HIGH = BAD!

TEST 2: Greedy Generation
Output: 'Once upon a time there was little named...'
Contains articles: ❌ NO
```

## Diagnosis Guide

| TEST 1: 'a' probability | TEST 2: Greedy | Diagnosis | Fix |
|------------------------|----------------|-----------|-----|
| **High (>10%)** ✅ | **Has articles** ✅ | Model good, generate.py broken | Fix generate.py sampling |
| **High (>10%)** ✅ | **No articles** ❌ | Generation code issue | Debug TEST 3 temperature |
| **Low (<1%)** ❌ | **No articles** ❌ | Model didn't learn | Retrain or check tokenizer |
| Any | **Has "todlers"** ❌ | Wrong checkpoint | Verify checkpoint file |

## Expected Runtime

- Total: ~1 minute
- Model loading: 10 seconds
- Each test: 5-10 seconds

## If You Get Errors

### ModuleNotFoundError
```powershell
# Check you're in the right directory
pwd
# Should show: C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2

# Check virtual environment is active
# Prompt should show: (shared_venv)
```

### Checkpoint not found
```powershell
# Verify file exists
dir checkpoints_wikimini\best_model.pth

# If file is elsewhere, update path
python validate_model_generation.py `
    --checkpoint path\to\your\checkpoint.pth `
    --tokenizer .\tokenizer\wikimini_32k
```

### Out of memory
```powershell
# Use CPU instead
python validate_model_generation.py `
    --checkpoint checkpoints_wikimini\best_model.pth `
    --tokenizer .\tokenizer\wikimini_32k `
    --device cpu
```

## What Each Test Shows

1. **TEST 1: Raw Probabilities** - What model ACTUALLY learned
2. **TEST 2: Greedy (always pick best)** - Model's "best" output
3. **TEST 3: Temperature Sampling** - Effect of randomness
4. **TEST 4: Training Context** - Works on familiar patterns?
5. **TEST 5: generate.py Style** - Replicates your actual code

## Most Important

**Focus on TEST 1 "Specific Token Probabilities" section first!**

This single section tells you if your model learned to predict articles correctly, independent of any generation code bugs.

---

Run it now and share the TEST 1 output! 🚀
