# 🔬 Model Diagnosis Quick Start

## What We Found So Far

✅ **DATA PIPELINE IS PERFECT** - Articles are preserved at every stage:
- Raw data: 9.93% articles
- Tokenized: 6.23% articles
- Training batches: 5.9-8.8% articles

❌ **MODEL GENERATES WITHOUT ARTICLES** - Despite good training data:
- Output: "there was little named" (missing "a")
- Good perplexity (7.37) but broken generation
- Produces "todlers" (not in training data)

## 🎯 Next Step: Diagnose the Model

Now we need to check **what the model actually learned** about articles.

### Run Model Diagnostics

```powershell
python diagnose_model.py `
    --checkpoint checkpoints_wikimini\best_model.pth `
    --tokenizer .\tokenizer\wikimini_32k
```

This will run 5 comprehensive tests:

### Test 1: Article Prediction in Context
**What it does:** Checks if model predicts articles when grammatically required

**Example:**
```
Prompt: "Once upon a time there was"
Expected: Model should predict ' a' or ' the' next
Check: Are articles in top-5 predictions?
```

**Good result:**
```
✅ ' a' (token 262): 15.2% (rank #2)
✅ ' the' (token 264): 8.5% (rank #4)
```

**Bad result:**
```
❌ ' a' (token 262): 0.05% (rank #1261)  ← This is what we saw before!
❌ ' the' (token 264): 0.00% (rank #8543)
```

### Test 2: Article vs Content Word Probabilities
**What it does:** Compares model's confidence in articles vs content words

**Example:**
```
Prompt: "Once upon a time there was"
Article ' a': 0.05%
Content word ' little': 32.03%
Ratio: 640x  ← Model STRONGLY prefers content word!
```

**This explains:** Why model outputs "there was little" instead of "there was a little"

### Test 3: Generation with Different Sampling
**What it does:** Tests if different sampling settings produce articles

**Settings tested:**
- Greedy (argmax)
- Temperature 1.0
- Temperature 0.8
- Temperature 0.8 + top_k 50
- Temperature 1.0 + top_k 10

**Check:** Do ANY settings produce articles?
- If NO → Model didn't learn articles at all
- If YES → Generation settings are too restrictive

### Test 4: Article Token Embeddings
**What it does:** Checks if article embeddings are learned properly

**Good result:**
```
✅ Articles cluster together (similarity > 0.7)
✅ Nearest neighbors are other function words (determiners, prepositions)
```

**Bad result:**
```
❌ Articles don't cluster (similarity < 0.5)
❌ Nearest neighbors are random unrelated words
```

### Test 5: Loss Analysis on Articles
**What it does:** Checks if model has high loss specifically on articles

**Example:**
```
Average loss (all tokens): 2.05
Average loss (articles):   4.82  ← 2.35x higher!
Average loss (other):      2.05
```

**This means:** Model struggles to predict articles even during training

## 📊 Interpreting Results

After running the diagnostic, look for these patterns:

### Pattern A: Model Never Learned Articles
**Symptoms:**
- ❌ Articles rank #1000+ in predictions
- ❌ Articles have much higher loss than other tokens
- ❌ No generation setting produces articles

**Root cause:** Model capacity or training signal too weak

**Fix:**
- Increase model size
- Weight article loss higher during training
- Train longer

### Pattern B: Model Learned Articles But Doesn't Use Them
**Symptoms:**
- ⚠️ Articles in top-20 but not top-5
- ⚠️ Slightly higher loss on articles
- ⚠️ Some generation settings produce articles

**Root cause:** Model learned articles exist but not when to use them

**Fix:**
- Improve context modeling (more layers/heads)
- Add attention to grammatical patterns
- Use better positional encodings

### Pattern C: Generation Settings Too Restrictive
**Symptoms:**
- ✅ Articles in top-5 predictions
- ✅ Normal loss on articles
- ❌ But generated text still missing articles

**Root cause:** Sampling filters out articles

**Fix:**
- Use temperature 1.0 instead of 0.8
- Remove or increase top-k
- Try nucleus (top-p) sampling

### Pattern D: Embeddings Not Learned
**Symptoms:**
- ❌ Article embeddings don't cluster
- ❌ Nearest neighbors are random words
- ❌ Low similarity between ' a' and ' the'

**Root cause:** Model treating articles as separate unrelated tokens

**Fix:**
- Pre-train on grammatical patterns
- Use syntactic embeddings
- Curriculum learning (start with grammar)

## 🛠️ What to Do After Diagnosis

1. **Run the diagnostic** and save the output
2. **Identify the pattern** (A, B, C, or D above)
3. **Apply the suggested fix**
4. **Re-train or fine-tune** the model
5. **Re-run validation** to check if fixed

## 📝 Example Output to Look For

**If you see this:**
```
TEST 1: Article Prediction in Context
Prompt: 'Once upon a time there was'
  ❌ ' a' (token 262): 0.0512% (rank #1261)
  ❌ ' the' (token 264): 0.0000% (rank #13131)
  Top 5 predictions:
    1. ' little' - 32.03%
    2. ' girl' - 15.22%
    3. ' boy' - 8.45%
    4. ' dog' - 3.21%
    5. ' cat' - 2.87%
```

**Then:**
- ❌ Model never learned to predict articles in context
- Root cause: Pattern A (never learned articles)
- Fix: Increase model capacity or weight article loss

**Expected good output:**
```
TEST 1: Article Prediction in Context
Prompt: 'Once upon a time there was'
  ✅ ' a' (token 262): 15.32% (rank #2)
  ✅ ' the' (token 264): 8.45% (rank #4)
  Top 5 predictions:
    1. ' little' - 18.22%
    2. ' a' - 15.32% ← ARTICLE
    3. ' girl' - 12.45%
    4. ' the' - 8.45% ← ARTICLE
    5. ' boy' - 5.87%
```

## 💡 Quick Actions

**If diagnostic takes too long:**
Just run TEST 1 manually in Python:

```python
import torch
from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer

tokenizer = load_tokenizer('./tokenizer/wikimini_32k')
checkpoint = torch.load('checkpoints_wikimini/best_model.pth', weights_only=False)
model = WikiMiniModel(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = "Once upon a time there was"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits'][0, -1, :]
    probs = torch.softmax(logits, dim=0)

# Check articles
print(f"' a' (262): {probs[262]:.4%} (rank #{(probs > probs[262]).sum().item()})")
print(f"' the' (264): {probs[264]:.4%} (rank #{(probs > probs[264]).sum().item()})")

# Top 5
top5 = torch.topk(probs, 5)
for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
    print(f"{i+1}. '{tokenizer.decode([idx.item()])}' - {prob:.4%}")
```

This single test will tell you if the model learned articles or not!
