#!/bin/bash
# Example: How to run the validation script after training

# Replace with your actual checkpoint path
CHECKPOINT="checkpoints/best_model.pt"  # or checkpoint_epoch_10.pt, etc.

echo "=================================="
echo "Model Generation Validator"
echo "=================================="
echo ""
echo "This script will run 5 comprehensive tests on your trained model"
echo "to diagnose WHY generation is broken despite good training metrics."
echo ""
echo "Using checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Please specify your checkpoint path:"
    echo "  1. Find your checkpoint file (likely in checkpoints/ directory)"
    echo "  2. Update CHECKPOINT variable in this script, or run directly:"
    echo ""
    echo "     python validate_model_generation.py \\"
    echo "         --checkpoint YOUR_CHECKPOINT_PATH.pt \\"
    echo "         --tokenizer ./tokenizer/wikimini_32k \\"
    echo "         --prompt 'Once upon a time there was'"
    echo ""
    exit 1
fi

# Run validation with different prompts
echo "Running Test 1/3: Standard prompt"
echo "----------------------------------"
python validate_model_generation.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer ./tokenizer/wikimini_32k \
    --prompt "Once upon a time there was"

echo ""
echo ""
echo "Running Test 2/3: Complete sentence prompt"
echo "----------------------------------"
python validate_model_generation.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer ./tokenizer/wikimini_32k \
    --prompt "Once upon a time, there was a little girl named Lily. She loved to"

echo ""
echo ""
echo "Running Test 3/3: Short prompt"
echo "----------------------------------"
python validate_model_generation.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer ./tokenizer/wikimini_32k \
    --prompt "There was"

echo ""
echo "=================================="
echo "VALIDATION COMPLETE"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Check TEST 1 results - do articles have high probability?"
echo "  2. Check TEST 2 results - does greedy generation work?"
echo "  3. Look for 'todlers' typo appearing anywhere"
echo ""
echo "See VALIDATION_GUIDE.md for detailed interpretation"
