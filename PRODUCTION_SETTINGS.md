# Production Settings - Quick Reference

Based on Epoch 4 enhanced evaluation results.

## 🏆 Recommended: Balanced Configuration

**Overall Score**: 26.2/30 (87.3%)

```python
# Generation parameters
temperature = 0.8
top_k = 50
top_p = 0.95
repetition_penalty = 1.2

# Length
max_length = 200  # Adjust based on needs

# Post-processing
enable_post_processing = True  # Highly recommended (+3 points grammar)
```

### Expected Quality:
- ✅ Articles: 100% presence
- ✅ Grammar: 9.2/10 (with post-processing)
- ✅ Repetition: 7.0/10 (75% unique words)
- ✅ Perplexity: 17.4 (good coherence)

---

## Alternative Configurations

### Conservative (Least Repetition)

Use when repetition is a major concern.

```python
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.3
```

**Quality**:
- Articles: 100% ✅
- Grammar: 8.8/10
- Repetition: 7.2/10 (79.5% unique - best)
- Perplexity: 16.3 (most coherent)

### Creative (Most Variety)

Use when you need more variety and creativity.

```python
temperature = 0.9
top_k = 60
top_p = 0.95
repetition_penalty = 1.1
```

**Quality**:
- Articles: 100% ✅
- Grammar: 9.6/10 (highest)
- Repetition: 6.2/10 (67.5% unique - more repetitive)
- Perplexity: 17.9

---

## Post-Processing Function

Always enable post-processing for production use. Improves grammar by 3+ points.

```python
def post_process_text(text: str) -> str:
    """
    Fix common issues:
    - Capitalization at sentence start
    - Missing punctuation
    - Spacing issues
    """
    import re

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences
    sentences = re.split(r'([.!?]\s+|\n)', text)

    fixed_sentences = []
    current_sentence = ""

    for part in sentences:
        if part.strip():
            if re.match(r'[.!?]\s*', part):
                current_sentence += part
                if current_sentence.strip():
                    fixed_sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part

    if current_sentence.strip():
        if not current_sentence.strip()[-1] in '.!?':
            current_sentence += '.'
        fixed_sentences.append(current_sentence.strip())

    # Capitalize first letter
    fixed_sentences = [s[0].upper() + s[1:] if s else s for s in fixed_sentences]

    result = ' '.join(fixed_sentences)

    # Fix patterns
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)
    result = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)
    result = re.sub(r'\s+', ' ', result)

    return result
```

---

## Generation Example

```python
from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer
import torch

# Load model (use best checkpoint for production)
checkpoint = torch.load('checkpoints/checkpoint_best_ppl_8.65.pth', weights_only=False)
model = WikiMiniModel(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = load_tokenizer('./tokenizer/tinystories_10k')

# Generate with Balanced settings
prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids]).to('cuda')

generated_ids = []
with torch.no_grad():
    for _ in range(200):  # max_length
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]

        # Apply repetition penalty (1.2)
        for token_id in torch.unique(input_ids[0]):
            if logits[token_id] > 0:
                logits[token_id] /= 1.2
            else:
                logits[token_id] *= 1.2

        # Temperature (0.8)
        logits = logits / 0.8

        # Top-k (50)
        indices_to_remove = logits < torch.topk(logits, 50)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

        # Top-p (0.95)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > 0.95
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

# Decode
text = tokenizer.decode(generated_ids)

# Post-process
text = post_process_text(text)

print(text)
```

---

## Quality Checklist

Before using in production, verify:

- [ ] Articles present in generated stories (should be ~100%)
- [ ] Grammar score > 8/10 (with post-processing)
- [ ] Repetition score > 6/10
- [ ] Text makes sense (coherent narrative)
- [ ] No excessive word repetition
- [ ] Proper capitalization and punctuation

---

## Troubleshooting

### Too Much Repetition
→ Increase `repetition_penalty` to 1.3 or use Conservative config

### Too Random/Incoherent
→ Decrease `temperature` to 0.7 or use Conservative config

### Too Boring/Predictable
→ Increase `temperature` to 0.9 or use Creative config

### Missing Articles
→ This should not happen! If it does, check:
- Using correct checkpoint (10K vocab, not 32K)
- Model loaded properly
- Tokenizer path is './tokenizer/tinystories_10k'

### Bad Grammar
→ Ensure post-processing is enabled
→ If still poor, wait for Epoch 5 to complete

---

## Performance Notes

**Speed**: ~80-100 tokens/second (on GPU)

**Memory**:
- Model: ~100MB (24.5M parameters)
- Generation: ~200MB peak

**Optimal Batch Size**: 1 for interactive generation

---

## When to Retrain

Consider retraining if:
- Article success rate drops below 95%
- Perplexity increases above 25
- Grammar score (post-processed) below 7/10
- User feedback indicates poor quality

Otherwise, current Epoch 4 checkpoint is production-ready for:
- ✅ Article generation (main goal)
- ✅ Basic children's stories
- ✅ Educational content
- ✅ Story prompts/starters

---

**Last Updated**: Based on Epoch 4 evaluation (2025-10-26)
**Next Review**: After Epoch 5 completes

**Status**: 🟢 Ready for production use with Balanced configuration
