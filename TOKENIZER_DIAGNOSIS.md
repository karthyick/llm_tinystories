# Tokenizer Root Cause Diagnosis

## The Problem

From validation results, we know:
- ❌ Model assigns 0.0005% probability to "a" (should be 10-20%)
- ✅ Model assigns 32% probability to " little" (but should be "a little")
- ❌ Output: "there was little named" (missing "a")

**Hypothesis:** The tokenizer is encoding "a little" as a single token " little", bypassing the article entirely.

## Run Diagnosis

```powershell
cd C:\Users\KR-ultra\Source\code_base\repos\python\llm\wikimini_2

python diagnose_tokenizer.py
```

## What Each Test Shows

### TEST 1: Basic Article Encoding
**What to look for:**
```
✅ Input:   'a'
   Tokens:  [68]
   Decoded: 'a'
```
- Does 'a' encode and decode correctly?
- What token ID does 'a' get?

**Red flag:**
```
❌ Input:   'a little'
   Tokens:  [12345]        ← Only ONE token!
   Decoded: ' little'      ← Article disappeared!
```

### TEST 2: Vocabulary Search
**What to look for:**
```
Token ID 68 (expected 'a') → 'a'          ← GOOD
Token ID 68 (expected 'a') → ' little'   ← BAD - wrong token!
```
- Does token ID 68 really contain "a"?
- Are there separate tokens for " a", "a", "a "?

### TEST 3: TinyStories Data Tokenization
**What to look for:**
```
Sample 1:
  Original has articles: a=True, the=True
  Decoded has articles:  a=False, the=False  ← PROBLEM!
  ⚠️  MISMATCH DETECTED!
```
- Are articles present in original but missing in decoded?
- This means tokenizer is DROPPING articles during encode/decode cycle

### TEST 4: Common Article Patterns
**What to look for:**
```
❌ Pattern: 'there was a little girl'
   Articles in original: 1
   Articles in decoded:  0     ← LOST!
   Token breakdown:
      [0] id=12345 → 'there'
      [1] id=23456 → ' was'
      [2] id=34567 → ' little' ← Should be ' a' + ' little'!
      [3] id=45678 → ' girl'
```
- Are articles appearing as separate tokens?
- Or are they merged into the next word?

### TEST 5: Whitespace Handling
**What to look for:**
```
❌ space + a + space
   Input:   ' a ' (len=3)
   Decoded: '  ' (len=2)  ← Article vanished!
```
- Does the tokenizer preserve whitespace correctly?
- This could explain why articles disappear

### TEST 6: Tokenizer Configuration
**What to look for:**
```
Vocab size: 32000
Type: <class 'tokenizers.Tokenizer'>
```
- What type of tokenizer is being used?
- Is it BPE, WordPiece, or something else?

### TEST 7: Expected vs Actual
**Summary test:**
```
What we EXPECT:
  'there' + ' was' + ' a' + ' little' + ' girl'

What we ACTUALLY get:
  'there' + ' was' + ' little' + ' girl'
                     ↑
                Missing ' a' token!

Has article as separate token: ❌ NO

⚠️  PROBLEM IDENTIFIED:
   Article 'a' is NOT appearing as its own token!
   It may be merged with adjacent words during tokenization.
```

## Common Root Causes

### Cause 1: BPE Merge Rules
**Symptom:** Articles merged with following words
```
Tokenizer learned: "a little" → [token_for_" little"]
Instead of:        "a little" → [token_for_"a", token_for_" little"]
```

**Why this happened:**
- During tokenizer training, "a little" appeared together so frequently
- BPE algorithm merged them into a single token
- Model never sees "a" as a separate concept

**Fix:** Retrain tokenizer with article protection

### Cause 2: Whitespace Stripping
**Symptom:** Articles disappear during encode/decode
```
Input:  "there was a little"
Encode: [123, 456, 789]  ← Only 3 tokens
Decode: "there was little"  ← Article gone!
```

**Why this happened:**
- Tokenizer strips leading/trailing whitespace
- "a" surrounded by spaces gets normalized away

**Fix:** Use different tokenizer or add special handling

### Cause 3: Vocabulary Missing Articles
**Symptom:** No token ID for articles
```
Searching for 'a' in vocabulary... NOT FOUND
```

**Why this happened:**
- Tokenizer trained on pre-processed data that had articles removed
- Vocabulary never learned article tokens

**Fix:** Retrain tokenizer on clean data

## Quick Check

**Fastest way to confirm the issue:**

```python
from src.data.tokenizer import load_tokenizer

tokenizer = load_tokenizer('./tokenizer/wikimini_32k')

# This should give us 2 tokens: "a" + " little"
tokens = tokenizer.encode("a little")
print(f"Tokens: {tokens}")
print(f"Count: {len(tokens)}")  # Should be 2

for tok_id in tokens:
    print(f"  {tok_id} → '{tokenizer.decode([tok_id])}'")
```

**Expected (good):**
```
Tokens: [68, 31065]
Count: 2
  68 → 'a'
  31065 → ' little'
```

**If you see (bad):**
```
Tokens: [31065]
Count: 1
  31065 → ' little'  ← Article was merged!
```

## Next Steps

1. **Run the diagnostic:** `python diagnose_tokenizer.py`
2. **Look for ❌ marks** - those indicate problems
3. **Check TEST 7** - it summarizes the root cause
4. **Based on findings:**
   - If articles are merged → Need new tokenizer
   - If articles are dropped → Check tokenizer training data
   - If whitespace issue → Need different tokenization strategy

## Solution

Once root cause is identified, you'll need to:
1. Retrain the tokenizer with proper article handling
2. Retrain the model with the new tokenizer
3. Verify with validation script that articles now work

The good news: Your model training pipeline works! It's just the tokenizer that's broken.
