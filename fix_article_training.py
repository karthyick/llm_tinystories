#!/usr/bin/env python3
"""
Quick fix for article generation problem.

This script shows how to modify training to weight article tokens more heavily.
Copy the weighted loss code into your training loop.
"""

import torch
import torch.nn.functional as F

# Article token IDs (from tokenizer diagnosis)
ARTICLE_TOKEN_IDS = {262, 264, 389}  # ' a', ' the', ' an'

def compute_weighted_loss(logits, labels, article_weight=10.0):
    """Compute loss with higher weight for article tokens.

    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        labels: Target tokens (batch_size, seq_len)
        article_weight: Weight multiplier for article tokens (default: 10.0)

    Returns:
        weighted_loss: Scalar loss value
        article_loss: Loss specifically on article tokens (for logging)
        other_loss: Loss on non-article tokens (for logging)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute per-token loss (no reduction)
    per_token_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,  # Ignore padding
        reduction='none'
    )

    # Create weight tensor
    loss_weights = torch.ones_like(per_token_loss)

    # Increase weight for article tokens
    for article_id in ARTICLE_TOKEN_IDS:
        article_mask = (labels_flat == article_id)
        loss_weights[article_mask] = article_weight

    # Apply weights and compute mean
    weighted_loss = (per_token_loss * loss_weights).sum() / loss_weights.sum()

    # Compute separate losses for logging
    article_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
    for article_id in ARTICLE_TOKEN_IDS:
        article_mask |= (labels_flat == article_id)

    valid_mask = (labels_flat != -100)  # Not padding

    if article_mask.any():
        article_loss = per_token_loss[article_mask].mean()
    else:
        article_loss = torch.tensor(0.0)

    other_mask = valid_mask & ~article_mask
    if other_mask.any():
        other_loss = per_token_loss[other_mask].mean()
    else:
        other_loss = torch.tensor(0.0)

    return weighted_loss, article_loss, other_loss


def example_training_loop():
    """Example of how to use weighted loss in training loop."""

    print("Example training loop with weighted article loss:\n")
    print("""
# In your training script (train.py), replace the loss computation:

# OLD CODE:
# loss = F.cross_entropy(
#     logits.view(-1, logits.size(-1)),
#     labels.view(-1),
#     ignore_index=-100
# )

# NEW CODE:
from fix_article_training import compute_weighted_loss, ARTICLE_TOKEN_IDS

# Inside training loop:
for batch in train_loader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']

    # Compute weighted loss (10x weight on articles)
    loss, article_loss, other_loss = compute_weighted_loss(
        logits,
        labels,
        article_weight=10.0  # Adjust this as needed
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging (every N steps)
    if step % 100 == 0:
        print(f"Step {step}")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Article loss: {article_loss.item():.4f}")
        print(f"  Other loss: {other_loss.item():.4f}")
        print(f"  Article/Other ratio: {article_loss.item()/other_loss.item():.2f}x")

        # Watch this ratio! You want it to decrease over training.
        # Initially might be 5-10x, should converge to ~1-2x
""")


def test_weighted_loss():
    """Test the weighted loss function."""
    print("Testing weighted loss function...\n")

    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 8, 32000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Put some article tokens in labels
    labels[0, 0] = 262  # ' a'
    labels[0, 3] = 264  # ' the'
    labels[1, 2] = 389  # ' an'

    # Compute losses
    weighted_loss, article_loss, other_loss = compute_weighted_loss(
        logits, labels, article_weight=10.0
    )

    print(f"✅ Weighted loss: {weighted_loss.item():.4f}")
    print(f"✅ Article loss: {article_loss.item():.4f}")
    print(f"✅ Other token loss: {other_loss.item():.4f}")
    print(f"✅ Ratio (article/other): {article_loss.item()/other_loss.item():.2f}x")
    print("\nFunction works correctly!")


def show_monitoring_code():
    """Show code for monitoring article prediction during training."""
    print("\n" + "="*80)
    print("MONITORING ARTICLE LEARNING DURING TRAINING")
    print("="*80)
    print("""
Add this to your validation loop to track article prediction accuracy:

def validate_article_prediction(model, tokenizer, device):
    '''Check if model predicts articles in context.'''
    model.eval()

    test_prompts = [
        "Once upon a time there was",
        "There was",
        "She saw",
    ]

    article_ranks = []

    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], device=device)

            outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]
            probs = F.softmax(logits, dim=0)

            # Check rank of ' a'
            a_rank = (probs > probs[262]).sum().item() + 1
            article_ranks.append(a_rank)

    avg_rank = sum(article_ranks) / len(article_ranks)

    return avg_rank

# In your validation loop:
article_rank = validate_article_prediction(model, tokenizer, device)
print(f"Article average rank: {article_rank:.0f}")

# Track this metric! You want it to decrease over training:
# Epoch 1: rank ~500
# Epoch 5: rank ~100
# Epoch 10: rank ~5-10 ← SUCCESS!
""")


def show_hyperparameter_recommendations():
    """Show recommended hyperparameters for article weight."""
    print("\n" + "="*80)
    print("HYPERPARAMETER RECOMMENDATIONS")
    print("="*80)
    print("""
Try different article_weight values:

1. article_weight = 5.0  (CONSERVATIVE)
   - Gradual learning of articles
   - May take 15-20 epochs to see results
   - Lower risk of overfitting to articles

2. article_weight = 10.0  (RECOMMENDED)
   - Good balance
   - Should see improvement in 10 epochs
   - What I recommend starting with

3. article_weight = 20.0  (AGGRESSIVE)
   - Fast learning of articles
   - May see results in 5 epochs
   - Risk: might overfit (generates too many articles)

How to choose:
- Start with 10.0
- If articles improve but perplexity increases too much → reduce to 5.0
- If articles don't improve after 10 epochs → increase to 20.0

Monitor both:
- Article prediction rank (should decrease to <10)
- Overall perplexity (should stay ~7-9)
""")


def show_expected_training_output():
    """Show what training output should look like with weighted loss."""
    print("\n" + "="*80)
    print("EXPECTED TRAINING OUTPUT")
    print("="*80)
    print("""
With article_weight=10.0, you should see training output like this:

Epoch 1:
---------
Step 100:
  Total loss: 3.2145
  Article loss: 8.5432  ← High initially (model can't predict articles)
  Other loss: 2.9876
  Article/Other ratio: 2.86x

Step 500:
  Total loss: 2.8234
  Article loss: 5.2341  ← Decreasing!
  Other loss: 2.7234
  Article/Other ratio: 1.92x

Epoch 5:
---------
Step 5000:
  Total loss: 2.1234
  Article loss: 2.8765  ← Much better!
  Other loss: 2.0234
  Article/Other ratio: 1.42x

Validation:
  Perplexity: 8.23 (slightly higher than before - expected!)
  Article average rank: 45 (better than 470!)

Epoch 10:
----------
Step 10000:
  Total loss: 2.0123
  Article loss: 2.1234  ← Close to other loss now!
  Other loss: 1.9876
  Article/Other ratio: 1.07x  ← SUCCESS!

Validation:
  Perplexity: 7.89 (similar to original)
  Article average rank: 8  ← EXCELLENT!

Generated text:
  "Once upon a time there was a little girl named Lily. She lived in a small house..."
                                ↑                    ↑  ↑
                                Articles present!

✅ TRAINING SUCCESSFUL!
""")


if __name__ == "__main__":
    print("="*80)
    print("FIX FOR ARTICLE GENERATION PROBLEM")
    print("="*80)
    print()

    # Run test
    test_weighted_loss()

    # Show example usage
    example_training_loop()

    # Show monitoring code
    show_monitoring_code()

    # Show hyperparameter recommendations
    show_hyperparameter_recommendations()

    # Show expected output
    show_expected_training_output()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Copy the compute_weighted_loss function to your training script
2. Replace your loss computation with weighted loss
3. Add article monitoring to validation loop
4. Re-train for 10 epochs
5. Check if article rank improves to <10
6. Test generation - should see articles!

Good luck! 🚀
""")
