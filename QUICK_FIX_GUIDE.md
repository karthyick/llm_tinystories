# 🚀 Quick Fix Guide - Get Articles Working NOW

## 🎯 Root Cause Found!

Your model has a **3,485x bias against articles**. It learned content words but not articles because:
- Articles are only 6% of tokens
- Loss gets averaged, so article errors are drowned out
- Model optimized for content words (easy) and ignored articles (hard)

---

## ⚡ Two Ways to Fix

### Option A: Quick Test (Use Temperature 1.0)

**TIME: 1 minute**

Your diagnostic showed temperature helps:
- Temp 0.8: 2 articles / 20 tokens (10%)
- **Temp 1.0: 3 articles / 20 tokens (15%)** ← 50% better!

**Try it now:**

```powershell
# In your generation script, change:
temperature = 0.8

# To:
temperature = 1.0

# And remove top-k:
# top_k = 50  # Comment out
```

This won't fix the root cause but will improve generation immediately.

---

### Option B: Real Fix (Re-train with Weighted Loss)

**TIME: ~6 hours (10 epochs)**

**BEST SOLUTION:** Make the model care about articles by weighting them 10x more in the loss.

#### Step 1: Find your training script loss computation

Look in your `train.py` or training script for something like:

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
```

#### Step 2: Replace with weighted loss

```python
# Add at top of file:
from fix_article_training import compute_weighted_loss

# Replace loss computation:
# OLD:
# loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

# NEW:
loss, article_loss, other_loss = compute_weighted_loss(
    logits,
    labels,
    article_weight=10.0  # Weight articles 10x more
)

# Add logging (every 100 steps):
if step % 100 == 0:
    print(f"  Article loss: {article_loss.item():.4f}")
    print(f"  Other loss: {other_loss.item():.4f}")
    print(f"  Ratio: {article_loss.item()/other_loss.item():.2f}x")
```

#### Step 3: Re-train for 10 epochs

```powershell
# Continue training from your checkpoint
python train.py --num_epochs 10
```

#### Step 4: Watch the metrics

**What you want to see:**

```
Epoch 1:
  Article loss: 8.54  ← High (model can't predict articles)
  Other loss: 2.99
  Ratio: 2.86x

Epoch 5:
  Article loss: 2.88  ← Improving!
  Other loss: 2.02
  Ratio: 1.42x

Epoch 10:
  Article loss: 2.12  ← Close to other loss!
  Other loss: 1.99
  Ratio: 1.07x  ← SUCCESS!
```

When ratio gets close to 1.0x, the model learned articles!

#### Step 5: Test generation

```powershell
python scripts/generate.py --prompt "Once upon a time there was"
```

**Expected output:**
```
"Once upon a time there was a little girl named Lily. She lived in a small house..."
                              ↑                    ↑  ↑
                              Articles present!
```

---

## 📊 How to Know It's Working

### During Training (Check Every Epoch):

Run this quick test:

```python
python -c "
from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer
import torch
import torch.nn.functional as F

tok = load_tokenizer('./tokenizer/wikimini_32k')
ckpt = torch.load('checkpoints_wikimini/best_model.pth', weights_only=False)
model = WikiMiniModel(ckpt['config']['model'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tokens = tok.encode('Once upon a time there was')
input_ids = torch.tensor([tokens])

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits'][0, -1, :]
    probs = F.softmax(logits, dim=0)

rank = (probs > probs[262]).sum().item() + 1
print(f'Article rank: {rank}')
"
```

**Progress tracking:**
- Epoch 0: rank ~470 ❌
- Epoch 5: rank ~100 ⚠️
- Epoch 10: rank ~10 ✅
- Epoch 15: rank ~3 🎯

When rank gets below 10, you're done!

---

## 🎓 Expected Timeline

### With Weighted Loss (Recommended):

| Epoch | Article Rank | Status |
|-------|--------------|--------|
| 0 (current) | #470 | ❌ Broken |
| 5 | #100 | ⚠️ Learning |
| 10 | #10 | ✅ Good |
| 15 | #3 | 🎯 Excellent |

**Total time:** ~6-9 hours of training

**Final result:** Model generates proper articles!

---

## 🔍 Troubleshooting

### Problem: Article loss not decreasing

```
Epoch 5:
  Article loss: 8.20  ← Still high!
  Other loss: 2.01
  Ratio: 4.08x  ← Not improving
```

**Solution:** Increase weight to 20.0

```python
loss, article_loss, other_loss = compute_weighted_loss(
    logits, labels, article_weight=20.0  # Increase from 10 to 20
)
```

### Problem: Perplexity getting too high

```
Epoch 10:
  Perplexity: 15.2  ← Was 7.37, now doubled!
```

**Solution:** Reduce weight to 5.0

```python
loss, article_loss, other_loss = compute_weighted_loss(
    logits, labels, article_weight=5.0  # Reduce from 10 to 5
)
```

### Problem: Model generates TOO MANY articles

```
Generated: "Once upon a time a there was a a little a girl..."
                          ↑                ↑  ↑       ↑
                          Too many articles!
```

**Solution:** Reduce weight to 5.0 or train a few more epochs to balance

---

## 📝 Summary

### What we found:
- ✅ Data has articles (6% of tokens)
- ❌ Model learned content words but ignored articles (3,485x bias)
- ⚠️ Higher temperature helps but not enough

### Best fix:
1. Add weighted loss (10x weight on articles)
2. Re-train for 10 epochs
3. Watch article loss decrease
4. Test generation

### Quick fix:
- Use temperature 1.0 instead of 0.8 for generation

### Success metric:
- Article rank should be <10 (currently #470)

---

## 🚀 Ready to Fix It?

**Run this to see the implementation:**

```powershell
python fix_article_training.py
```

This shows:
- ✅ Complete code for weighted loss
- ✅ How to integrate into training loop
- ✅ Monitoring code
- ✅ Expected training output
- ✅ Hyperparameter recommendations

**Then modify your training script and re-train!**

Good luck! The fix is straightforward and should work within 10 epochs. 🎯
