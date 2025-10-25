# TinyStories Training Startup Script
# Cleans old cache and starts training with 10K tokenizer

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "TinyStories 23.5M Model Training" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Delete old cache
Write-Host "[1/2] Deleting old cache (32K tokenization)..." -ForegroundColor Yellow
if (Test-Path "./data/cache") {
    Remove-Item -Path "./data/cache" -Recurse -Force
    Write-Host "✅ Cache deleted successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  No cache found (this is fine for first run)" -ForegroundColor Yellow
}
Write-Host ""

# Step 2: Verify tokenizer exists
Write-Host "[2/2] Verifying tokenizer..." -ForegroundColor Yellow
if (Test-Path "./tokenizer/tinystories_10k/tokenizer.json") {
    Write-Host "✅ 10K tokenizer found" -ForegroundColor Green

    # Get tokenizer info
    $tokenizerContent = Get-Content "./tokenizer/tinystories_10k/tokenizer.json" -Raw | ConvertFrom-Json
    Write-Host "   Vocabulary size: 10,000" -ForegroundColor Gray
} else {
    Write-Host "❌ ERROR: Tokenizer not found!" -ForegroundColor Red
    Write-Host "   Please run: python train_custom_tokenizer.py --vocab_size 10000 --output_dir ./tokenizer/tinystories_10k" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Start training
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Starting Training" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Config: config/train_config_tinystories_33M_TOP10K.yaml" -ForegroundColor Gray
Write-Host "Model: 23.5M parameters (10K vocab)" -ForegroundColor Gray
Write-Host "Expected time: 30-40 hours on RTX 5090" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop training" -ForegroundColor Yellow
Write-Host ""

# Run training
python train.py --config config/train_config_tinystories_33M_TOP10K.yaml
