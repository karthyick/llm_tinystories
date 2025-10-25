# Windows PowerShell Setup Script for TinyStories with Karpathy's Tokenizer
# Run this in PowerShell (not bash!)

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "TinyStories Setup - Karpathy's Tokenizer (Windows)" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Step 1: Create tokenizer directory
Write-Host "[1/4] Creating tokenizer directory..." -ForegroundColor Yellow
$tokenizerDir = "./tokenizer/llama2c_tinystories"
if (!(Test-Path $tokenizerDir)) {
    New-Item -ItemType Directory -Path $tokenizerDir -Force | Out-Null
    Write-Host "  ✓ Created: $tokenizerDir" -ForegroundColor Green
} else {
    Write-Host "  ✓ Directory already exists: $tokenizerDir" -ForegroundColor Green
}

# Step 2: Download Karpathy's tokenizer
Write-Host "[2/4] Downloading Karpathy's tokenizer (4096 vocab)..." -ForegroundColor Yellow
$tokenizerUrl = "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"
$tokenizerPath = "$tokenizerDir/tokenizer.model"

try {
    Invoke-WebRequest -Uri $tokenizerUrl -OutFile $tokenizerPath -ErrorAction Stop
    $fileSize = (Get-Item $tokenizerPath).Length
    Write-Host "  ✓ Downloaded: tokenizer.model ($([math]::Round($fileSize/1KB, 2)) KB)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Download manually from:" -ForegroundColor Yellow
    Write-Host "  $tokenizerUrl" -ForegroundColor Cyan
    Write-Host "  Save to: $tokenizerPath" -ForegroundColor Cyan
    exit 1
}

# Step 3: Delete old cache
Write-Host "[3/4] Cleaning old cache (32K tokenization)..." -ForegroundColor Yellow
$cacheDir = "./data/cache"
if (Test-Path $cacheDir) {
    Remove-Item -Path $cacheDir -Recurse -Force
    Write-Host "  ✓ Deleted old cache: $cacheDir" -ForegroundColor Green
} else {
    Write-Host "  ✓ No cache to delete (fresh start)" -ForegroundColor Green
}

# Step 4: Verify setup
Write-Host "[4/4] Verifying setup..." -ForegroundColor Yellow

$configPath = "./config/train_config_33M_KARPATHY_TOKENIZER.yaml"
if (Test-Path $configPath) {
    Write-Host "  ✓ Config found: $configPath" -ForegroundColor Green
} else {
    Write-Host "  ✗ Config missing: $configPath" -ForegroundColor Red
    Write-Host "    Make sure you pulled the latest code!" -ForegroundColor Yellow
}

if (Test-Path $tokenizerPath) {
    Write-Host "  ✓ Tokenizer ready: $tokenizerPath" -ForegroundColor Green
} else {
    Write-Host "  ✗ Tokenizer missing!" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start training:" -ForegroundColor Cyan
Write-Host "     python train.py --config config/train_config_33M_KARPATHY_TOKENIZER.yaml" -ForegroundColor White
Write-Host ""
Write-Host "  2. Expected duration: 30-40 hours on RTX 5090" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Expected result:" -ForegroundColor Cyan
Write-Host "     - Validation loss: <2.0" -ForegroundColor White
Write-Host "     - Grammar: 8-9/10" -ForegroundColor White
Write-Host "     - Articles: Always present ✓" -ForegroundColor Green
Write-Host ""
Write-Host "To test generation after training:" -ForegroundColor Yellow
Write-Host "  python generate.py --checkpoint checkpoints/checkpoint_latest.pth" -ForegroundColor White
Write-Host ""
