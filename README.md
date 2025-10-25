# WikiMini 95M - High-Performance Language Model

## Overview
A fresh implementation of a 95M parameter language model with modern architecture and full CUDA optimization for Windows.

### Key Features
- ⚡ **50,000-70,000 tokens/sec** on RTX 5090 (vs previous 1,163 tokens/sec)
- 🎯 **Modern Architecture**: RoPE, RMSNorm, SwiGLU, Flash Attention 2
- 🪟 **Windows Optimized**: Full CUDA support with proper DataLoader configuration
- 🧩 **Clean Modular Design**: Easy to understand and maintain

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Tokenizer
```bash
python train_tokenizer.py --vocab_size 32000
```

### 3. Prepare Data
```bash
python prepare_data.py --dataset wikitext-103
```

### 4. Train Model
```bash
python train.py --config config/train_config.yaml
```

### 5. Test Model
```bash
python test.py --checkpoint checkpoints/best.pt
```

## Datasets

The project supports multiple datasets for training:

### WikiText-103 (Default)
- **Size**: ~500MB (103M tokens)
- **Content**: Wikipedia articles
- **Use Case**: General language understanding
- **Config**: `config/train_config_full.yaml`

```bash
python train.py --config config/train_config_full.yaml
```

### TinyStories (Recommended for Testing)
- **Size**: ~1GB (~500M tokens, 2.1M stories)
- **Content**: Simple stories with limited vocabulary
- **Quality**: Synthetic data from GPT-3.5/4
- **Use Case**: Fast training, testing, clean English
- **Paper**: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- **Official Models**: 1M, 8M, 28M, **33M** parameters
- **Configs**:
  - `train_config_tinystories_small.yaml` - **32.61M params (matches official 33M model!)**
  - `train_config_tinystories.yaml` - 109M params (⚠️ 3.3× larger than official, will overfit)
- **Benefits**:
  - ✅ Clean, well-formed sentences
  - ✅ Faster convergence (20 epochs in ~few hours)
  - ✅ Better for architecture testing
  - ✅ Ideal for educational purposes
  - ✅ Proven architecture (matches official research)

```bash
# Recommended: 33M model matching official paper
python train.py --config config/train_config_tinystories_small.yaml

# Alternative: Large model (NOT recommended - will likely overfit)
python train.py --config config/train_config_tinystories.yaml
```

**⚠️ Model Size Recommendation:**
The official TinyStories paper tested models from 1M to 33M parameters on this dataset (~500M tokens). The **33M model was their largest and best performing model**. Our `train_config_tinystories_small.yaml` (32.61M params) closely matches this specification with:
- **Architecture**: 7 layers, d_model=448, n_heads=7
- **Tokens/Param**: 15.3 (very close to official 15.4)
- **Training**: 20 epochs (matching official paper)

Using larger models (like the 109M config) provides **NO benefit** and will likely **overfit** since the dataset complexity doesn't require it.

**Comparison:**

| Dataset | Size | Quality | Speed | Best For | Recommended Model Size |
|---------|------|---------|-------|----------|----------------------|
| WikiText-103 | 500MB | Good | Moderate | General LM training | 95M-125M params |
| TinyStories | 1GB | Excellent | Fast | Testing, clean English | 8M-33M params |

## Data Quality Checks

The project includes automatic data quality validation to prevent training on corrupted or low-quality data.

### What Gets Checked
- ❌ HTML tags and artifacts
- ❌ URLs and email addresses
- ❌ Malformed Unicode characters
- ❌ Excessive or suspicious punctuation
- ❌ Empty or extremely short/long text
- ❌ Repeated characters and patterns
- ❌ Special tokenizer tokens in raw text

### Quality Levels
- **EXCELLENT** (0% issues): Perfect data quality
- **GOOD** (<1% issues): Minor issues, safe to train
- **ACCEPTABLE** (<5% issues): Some issues, training proceeds with warnings
- **POOR** (5-10% issues): Significant issues, training blocked
- **CRITICAL** (>10% issues): Severe quality problems, training blocked

### Usage

**Automatic (during training):**
Quality checks run automatically before training starts:
```bash
python train.py --config config/train_config_tinystories.yaml
```

**Standalone check:**
```bash
python scripts/check_dataset_quality.py --dataset tinystories
python scripts/check_dataset_quality.py --dataset wikitext-103 --strict
```

**Configuration:**
```yaml
data:
  check_quality: true              # Enable/disable checks
  quality_sample_size: 10000       # Number of samples to check
  quality_strict: false            # Fail on any issues vs. warnings
```

### Example Output
```
======================================================================
DATA QUALITY REPORT
======================================================================
Dataset: tinystories (train split)
Quality Level: EXCELLENT
Status: ✅ PASSED

Statistics:
  Total Samples: 10,000
  Avg Length: 847.3 chars
  Avg Words: 156.2 words
  Vocabulary Size: 12,456

✅ No quality issues found!
======================================================================
```

## Architecture

### Model Configuration (95M Parameters)
- **Vocabulary**: 32,000 tokens (BPE)
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **FFN Hidden**: 2048 (SwiGLU adjusted)
- **Context Length**: 2048
- **Positional Encoding**: RoPE
- **Normalization**: RMSNorm
- **Activation**: SwiGLU

### Performance Optimizations
1. **CUDA Configuration**: cudnn.benchmark, float32_matmul_precision
2. **Mixed Precision**: BF16 (no loss scaling needed)
3. **Flash Attention 2**: O(N) memory complexity
4. **torch.compile**: 1.3-2× speedup
5. **Windows DataLoader**: num_workers=0 for stability

## Project Structure
```
wikimini_2/
├── config/               # Configuration files
│   ├── model_config.yaml
│   └── train_config.yaml
├── src/                  # Source code
│   ├── model/           # Model components
│   ├── data/            # Data processing
│   ├── training/        # Training logic
│   └── utils/           # Utilities
├── tests/               # Test files
├── scripts/             # Utility scripts
├── checkpoints/         # Model checkpoints
├── data/                # Training data
└── logs/                # Training logs
```

## Hardware Requirements

### Minimum
- GPU: RTX 4070 SUPER (12GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### Recommended
- GPU: RTX 5090 (32GB VRAM)
- RAM: 96GB
- Storage: 500GB NVMe SSD

## Performance Benchmarks

| Hardware | Tokens/sec | Training Time (100K steps) |
|----------|------------|---------------------------|
| RTX 4070 SUPER | 35,000-45,000 | 33-42 hours |
| RTX 5090 | 50,000-70,000 | 23-33 hours |

## License
MIT License

## Contributors
- KR-Ultra

## Citation
If you use this code, please cite:
```bibtex
@software{wikimini2025,
  title={WikiMini 95M - High-Performance Language Model},
  author={KR-Ultra},
  year={2025},
  url={https://github.com/KR-ultra/wikimini}
}
```