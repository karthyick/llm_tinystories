# Windows PowerShell Commands for TinyStories Setup

**Important:** Windows PowerShell uses different syntax than bash!

---

## Quick Setup (Copy-Paste These Commands)

### Method 1: Automated Script (Recommended)

```powershell
# Download and run setup script
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karthyick/llm_tinystories/main/WINDOWS_SETUP.ps1" -OutFile "setup.ps1"
.\setup.ps1
```

### Method 2: Manual Commands

```powershell
# 1. Create tokenizer directory
New-Item -ItemType Directory -Path "./tokenizer/llama2c_tinystories" -Force

# 2. Download Karpathy's tokenizer
Invoke-WebRequest -Uri "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model" -OutFile "./tokenizer/llama2c_tinystories/tokenizer.model"

# 3. Delete old cache
Remove-Item -Path "./data/cache" -Recurse -Force -ErrorAction SilentlyContinue

# 4. Verify download
Get-Item "./tokenizer/llama2c_tinystories/tokenizer.model"
```

---

## Common PowerShell vs Bash Differences

| Task | Bash | PowerShell |
|------|------|------------|
| **Create directory** | `mkdir -p path/` | `New-Item -ItemType Directory -Path "path/" -Force` |
| **Download file** | `wget URL -O file` | `Invoke-WebRequest -Uri "URL" -OutFile "file"` |
| **Delete directory** | `rm -rf path/` | `Remove-Item -Path "path/" -Recurse -Force` |
| **List files** | `ls -lah` | `Get-ChildItem` or `ls` (alias) |
| **Check file** | `ls file` | `Get-Item "file"` or `Test-Path "file"` |

---

## Step-by-Step Instructions

### Step 1: Create Tokenizer Directory

```powershell
# PowerShell syntax
New-Item -ItemType Directory -Path "./tokenizer/llama2c_tinystories" -Force
```

**Expected output:**
```
Directory: C:\Users\...\tinystories\tokenizer

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        26-10-2025  03:31 AM                llama2c_tinystories
```

### Step 2: Download Tokenizer

**Option A: Full command (recommended)**
```powershell
Invoke-WebRequest -Uri "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model" -OutFile "./tokenizer/llama2c_tinystories/tokenizer.model"
```

**Option B: Short alias**
```powershell
iwr "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model" -OutFile "./tokenizer/llama2c_tinystories/tokenizer.model"
```

**Expected output:**
```
(Downloads file silently - no output means success)
```

### Step 3: Verify Download

```powershell
Get-Item "./tokenizer/llama2c_tinystories/tokenizer.model"
```

**Expected output:**
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        26-10-2025  03:32 AM         488000 tokenizer.model
```

**Check file size:**
```powershell
(Get-Item "./tokenizer/llama2c_tinystories/tokenizer.model").Length / 1KB
```

**Expected:** ~476 KB (around 488000 bytes)

### Step 4: Delete Old Cache

```powershell
Remove-Item -Path "./data/cache" -Recurse -Force -ErrorAction SilentlyContinue
```

**Verify deletion:**
```powershell
Test-Path "./data/cache"
```

**Expected output:** `False` (directory doesn't exist)

---

## Troubleshooting

### Issue: "Invoke-WebRequest" download fails

**Error:**
```
Invoke-WebRequest : The remote server returned an error: (403) Forbidden.
```

**Solution 1: Add User-Agent**
```powershell
$headers = @{'User-Agent' = 'Mozilla/5.0'}
Invoke-WebRequest -Uri "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model" -OutFile "./tokenizer/llama2c_tinystories/tokenizer.model" -Headers $headers
```

**Solution 2: Manual Download**
1. Open in browser: https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
2. Save as: `tokenizer.model`
3. Move to: `./tokenizer/llama2c_tinystories/tokenizer.model`

### Issue: "wget" not recognized

**Error:**
```
wget : The term 'wget' is not recognized...
```

**Solution:** Don't use `wget`! Use PowerShell commands above.

### Issue: Path contains spaces

**Error:**
```
The term 'C:\Users\My' is not recognized...
```

**Solution:** Always quote paths with spaces
```powershell
# Good
New-Item -ItemType Directory -Path "C:\Users\My Documents\project" -Force

# Bad (will fail with spaces)
New-Item -ItemType Directory -Path C:\Users\My Documents\project -Force
```

### Issue: Permission denied

**Error:**
```
Remove-Item : Access to the path '...' is denied.
```

**Solution 1: Run PowerShell as Administrator**
- Right-click PowerShell
- Select "Run as Administrator"

**Solution 2: Close programs using the files**
- Close any Python processes
- Close file explorer windows in that directory

---

## After Setup: Start Training

```powershell
# Activate your virtual environment first (if using one)
# .\venv\Scripts\Activate.ps1

# Start training
python train.py --config config/train_config_33M_KARPATHY_TOKENIZER.yaml
```

**Expected initial output:**
```
Loading tokenizer from ./tokenizer/llama2c_tinystories/tokenizer.model
Creating model with 4096 vocabulary size
Model parameters: 21.35M
Training for 5 epochs
...
```

---

## Complete Setup Script

**Save as `setup.ps1` and run with `.\setup.ps1`:**

```powershell
# See WINDOWS_SETUP.ps1 file for complete script
```

---

## Quick Reference Card

**Copy these to a text file for quick access:**

```powershell
# Download tokenizer
iwr "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model" -OutFile "./tokenizer/llama2c_tinystories/tokenizer.model"

# Delete cache
Remove-Item -Path "./data/cache" -Recurse -Force -ErrorAction SilentlyContinue

# Verify tokenizer
Get-Item "./tokenizer/llama2c_tinystories/tokenizer.model"

# Start training
python train.py --config config/train_config_33M_KARPATHY_TOKENIZER.yaml

# Test generation
python generate.py --checkpoint checkpoints/checkpoint_latest.pth
```

---

## Expected Timeline (Windows)

| Step | Time | Command |
|------|------|---------|
| Download tokenizer | 5-10 sec | `iwr ... -OutFile ...` |
| Delete cache | 1 sec | `Remove-Item ...` |
| Start training | 30-40 hrs | `python train.py ...` |
| Test generation | 1 min | `python generate.py ...` |

---

## Success Indicators

**After download:**
```powershell
PS> Get-Item "./tokenizer/llama2c_tinystories/tokenizer.model"

    Directory: ...\tokenizer\llama2c_tinystories

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        26-10-2025  03:32 AM         488000 tokenizer.model  ✓
```

**During training (epoch 3+):**
```
Epoch 3, Step 10000: Loss: 2.0 | Grammar emerging
Articles appearing in test generation ✓
```

**After training (epoch 5):**
```
Final validation loss: 1.45 ✓
Grammar score: 8-9/10 ✓
Articles always present ✓
```

---

**For more details, see QUICK_START_PRETRAINED_TOKENIZER.md**

