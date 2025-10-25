# Running Validation on Windows

The validation script has been fixed and is now ready to use on your Windows machine!

## What Was Fixed

The script now properly:
- ✅ Imports `WikiMiniModel` (not generic GPT class)
- ✅ Uses your `load_tokenizer` function
- ✅ Handles model output format: `outputs['logits']`
- ✅ Matches your `generate.py` implementation
- ✅ Works with Windows paths

## Quick Start

### 1. Find Your Checkpoint

Your trained model checkpoint should be in:
```
C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2\checkpoints\
```

Look for files like:
- `best_model.pt`
- `checkpoint_epoch_10.pt`
- `final_model.pt`

### 2. Run Validation

```powershell
cd C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2

python validate_model_generation.py `
    --checkpoint checkpoints\best_model.pt `
    --tokenizer .\tokenizer\wikimini_32k `
    --prompt "Once upon a time there was"
```

Note: On Windows PowerShell, use backtick (`) for line continuation, not backslash (\\)

### 3. Read Output

The script will run 5 tests and show you:

**TEST 1: Raw Probabilities**
```
Top 20 Most Likely Next Tokens:
 1. 'a' → 18.45%        ← This should be HIGH
 2. 'little' → 12.30%
...

Specific Token Probabilities:
  'a' → 18.45% (rank #1)      ← Check this!
  'todlers' → 0.0001% (rank #8934) ← Should be LOW or absent
```

**Key Questions:**
- Does `'a'` have high probability (>10%)? ✅ Model learned correctly
- Does `'a'` have low probability (<1%)? ❌ Model didn't learn
- Does `'todlers'` appear anywhere? ❌ Wrong checkpoint or data issue

**TEST 2: Greedy Generation**
```
Step  1: 'a' (p=18.45%)
Step  2: 'little' (p=12.30%)
...
Full Greedy Output:
'Once upon a time there was a little girl named...'

Quality Checks:
  Contains articles (a/an/the): ✅ YES
```

If TEST 1 shows high probability for 'a' but TEST 2 doesn't generate it → generation code bug

## Common Scenarios

### Scenario A: Model Good, Generation Bad
```
TEST 1: 'a' has 18% probability ✅
TEST 2: Greedy generates articles ✅
Your generate.py: No articles ❌

→ FIX: Bug in your generate.py sampling code
```

### Scenario B: Model Didn't Learn
```
TEST 1: 'a' has 0.01% probability ❌
TEST 2: Greedy missing articles ❌

→ FIX: Training issue despite good PPL
   - Check tokenizer handling of articles
   - May need to retrain
```

### Scenario C: Temperature Too High
```
TEST 1: 'a' high probability ✅
TEST 2: Greedy works ✅
TEST 3: temp=0.8 fails ❌

→ FIX: Lower temperature in generate.py
```

## Expected Runtime

- Model loading: ~10 seconds
- Each test: ~5-10 seconds
- Total: ~1 minute

## Troubleshooting

### Error: "No module named 'src.model'"
```powershell
# Make sure you're in the right directory
cd C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2

# Check you're in the right virtual environment
(shared_venv) PS C:\...  ← Should see this
```

### Error: "Checkpoint not found"
```powershell
# List your checkpoints
dir checkpoints

# Use the correct path (Windows uses backslash)
--checkpoint checkpoints\your_model.pt
```

### CUDA Out of Memory
```powershell
# Use CPU if GPU memory is full
python validate_model_generation.py `
    --checkpoint checkpoints\best_model.pt `
    --tokenizer .\tokenizer\wikimini_32k `
    --device cpu
```

## Next Steps

1. **Run the validation** with your trained checkpoint
2. **Check TEST 1 output first** - this is the most important
3. **Based on results:**
   - If model is good → Fix your generate.py
   - If model is bad → Check training/data
   - If uncertain → Share TEST 1 output

## Files Created

- ✅ `validate_model_generation.py` - Main script (FIXED)
- ✅ `VALIDATION_README.md` - Quick reference
- ✅ `VALIDATION_GUIDE.md` - Detailed interpretation
- ✅ `run_validation_example.sh` - Linux example (not for Windows)
- ✅ This file - Windows-specific guide

All files are now committed and pushed to your repo!
