#!/usr/bin/env python
"""Training script for WikiMini 95M Language Model.

Windows optimizations:
1. num_workers=0 for DataLoader (critical!)
2. BF16 mixed precision
3. torch.compile disabled for Windows (Triton not supported)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Use newer torch.amp API (PyTorch 2.0+) instead of deprecated torch.cuda.amp
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, Optional, Tuple
import yaml
import json
import argparse
from tqdm import tqdm
from datetime import datetime

from src.model.transformer_block import WikiMiniModel
from src.data.dataset import TinyStoriesDataset, create_dataloaders
from src.data.tokenizer import load_tokenizer
from src.data.quality_checker import check_dataset_quality

# Setup logging (console only - file handlers added in main())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create separate logger for evaluation metrics
eval_logger = logging.getLogger('evaluation')

# WikiText-appropriate test prompts (Wikipedia-style articles)
EVAL_PROMPTS = [
   "Once upon a time there was",
    "One day, a little girl",
    "The cat and the dog",
    "Mom said to",
    "In the garden there was",
]

def setup_cuda_optimizations():
    """Setup CUDA optimizations for better performance."""
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere/Ada/Blackwell
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set high precision for FP32 matmul
        torch.set_float32_matmul_precision('high')

        logger.info("CUDA optimizations enabled")

        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Memory: {gpu_memory:.2f} GB")


def setup_file_logging(log_dir: str = "./logs") -> Tuple[str, str]:
    """Setup file logging for training and evaluation.

    Args:
        log_dir: Directory to save log files

    Returns:
        Tuple of (training_log_path, evaluation_log_path)
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_file = log_path / f"training_{timestamp}.log"
    eval_log_file = log_path / f"evaluation_{timestamp}.log"

    # Setup training log file handler (detailed logs)
    training_handler = logging.FileHandler(training_log_file, mode='w', encoding='utf-8')
    training_handler.setLevel(logging.INFO)
    training_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    training_handler.setFormatter(training_formatter)

    # Add handler to root logger (captures all training logs)
    logging.getLogger().addHandler(training_handler)

    # Setup evaluation log file handler (metrics only)
    eval_handler = logging.FileHandler(eval_log_file, mode='w', encoding='utf-8')
    eval_handler.setLevel(logging.INFO)
    eval_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    eval_handler.setFormatter(eval_formatter)

    # Add handler to evaluation logger only
    eval_logger.addHandler(eval_handler)
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False  # Don't send to root logger

    # Write headers to evaluation log
    eval_logger.info("="*80)
    eval_logger.info("EVALUATION METRICS LOG")
    eval_logger.info("="*80)
    eval_logger.info("Format: [Timestamp] - Epoch/Step - Loss - Perplexity - Status")
    eval_logger.info("="*80)

    logger.info(f"Training logs will be saved to: {training_log_file}")
    logger.info(f"Evaluation logs will be saved to: {eval_log_file}")

    return str(training_log_file), str(eval_log_file)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix scientific notation strings from YAML
    def fix_scientific_notation(d):
        """Recursively convert scientific notation strings to floats."""
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and ('e' in v.lower() or 'e-' in v):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
                elif isinstance(v, dict):
                    fix_scientific_notation(v)
        return d
    
    config = fix_scientific_notation(config)
    return config


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in a readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class Trainer:
    """Trainer class for WikiMini model."""
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Moving model to device: {self.device}")
        self.model = self.model.to(self.device)

        # Verify model is on GPU
        first_param_device = next(self.model.parameters()).device
        logger.info(f"Model device confirmed: {first_param_device}")
        
        # Windows-specific: Check if we're on Windows
        is_windows = sys.platform == 'win32'
        
        # torch.compile handling (disabled on Windows due to Triton issues)
        if self.config['training'].get('use_compile', True) and not is_windows:
            compile_mode = self.config['training'].get('compile_mode', 'reduce-overhead')
            compile_fullgraph = self.config['training'].get('compile_fullgraph', True)
            
            logger.info(f"Compiling model with torch.compile(mode='{compile_mode}', fullgraph={compile_fullgraph})...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    fullgraph=compile_fullgraph,
                )
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                logger.info("Continuing without compilation")
        elif is_windows:
            logger.info("Windows detected: Skipping torch.compile (Triton not supported on Windows)")
        
        # Optimizer setup
        self.setup_optimizer()
        
        # Mixed precision setup
        # CRITICAL FIX: BF16 does NOT need GradScaler (only FP16 does)
        self.use_amp = self.config['training'].get('use_bf16', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = None  # BF16 doesn't need gradient scaling
            self.amp_dtype = torch.bfloat16
            logger.info("Using BFloat16 mixed precision training (no gradient scaling)")
        else:
            self.scaler = None
            self.amp_dtype = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0  # Track current epoch for logging
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')
        self.best_checkpoint_step = 0  # Track when best model was saved
        self.training_start_time = None
        self.total_tokens_processed = 0  # For accurate tokens/sec calculation
        self.current_train_loss = 0.0  # Track recent train loss for gap calculation

        # Performance tracking
        self.recent_losses = []  # Track recent losses for trend analysis
        self.recent_grad_norms = []  # Track gradient health
        self.max_grad_norm_seen = 0.0  # Track max gradient norm

        # GPU memory tracking
        self.peak_gpu_memory = 0.0

        # Load tokenizer for text generation
        self.tokenizer = None  # Will be set from outside
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        opt_config = self.config['optimizer']
        
        # Ensure learning rate is float
        lr = float(opt_config['learning_rate']) if isinstance(opt_config['learning_rate'], str) else opt_config['learning_rate']
        adam_epsilon = float(opt_config['adam_epsilon']) if isinstance(opt_config['adam_epsilon'], str) else opt_config['adam_epsilon']
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(opt_config['adam_beta1'], opt_config['adam_beta2']),
            eps=adam_epsilon,
            weight_decay=opt_config['weight_decay'],
            fused=torch.cuda.is_available(),  # Use fused optimizer if available
        )
        
        logger.info(f"Created AdamW optimizer (fused={torch.cuda.is_available()})")
        
        # Create scheduler
        scheduler_config = self.config['scheduler']
        scheduler_type = scheduler_config.get('type', 'cosine').lower()
        total_steps = len(self.train_loader) * self.config['training'].get('num_epochs', 10)

        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, LambdaLR

        warmup_steps = scheduler_config.get('warmup_steps', 0)

        if scheduler_type == 'constant':
            # Constant learning rate (TinyStories official setup)
            if warmup_steps > 0:
                # Warmup to constant LR
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                constant_scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0)
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, constant_scheduler],
                    milestones=[warmup_steps]
                )
                logger.info(f"Scheduler: constant LR with {warmup_steps} warmup steps")
            else:
                # Pure constant LR (no warmup)
                self.scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0)
                logger.info(f"Scheduler: constant LR (no warmup)")

        elif scheduler_type == 'cosine':
            # Cosine schedule with warmup (WikiText setup)
            min_lr = float(scheduler_config.get('min_lr', lr * 0.1)) if 'min_lr' in scheduler_config else lr * 0.1

            # Warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('cosine_cycles', 10000),
                T_mult=1,
                eta_min=min_lr
            )

            # Combine schedulers
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"Scheduler: cosine annealing with {warmup_steps} warmup steps, min_lr={min_lr}")

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: 'constant', 'cosine'")

        logger.info(f"Learning rate: {lr}")

    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int) -> Dict[str, float]:
        """Single training step with gradient accumulation.

        Args:
            batch: Input batch
            accumulation_step: Current accumulation step (0 to grad_accum_steps-1)

        Returns:
            Dictionary with metrics
        """
        self.model.train()

        # Get gradient accumulation steps
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        # Move data to device
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)

        # Forward pass with standard cross-entropy loss
        # Using standard loss (matches ALL 30+ successful TinyStories implementations)
        # Research shows: NO weighted loss needed! Grammar emerges naturally with correct vocab size.
        if self.use_amp:
            # Use new torch.amp API: autocast('cuda', ...) instead of autocast(device_type='cuda', ...)
            with autocast('cuda', dtype=self.amp_dtype, enabled=True):
                # Standard loss computation (model computes cross-entropy internally)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']

                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
        else:
            # Standard loss computation (model computes cross-entropy internally)
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            logger.error(f"Loss is {loss.item()}, skipping this batch")
            return {
                'loss': float('nan'),
                'perplexity': float('nan'),
                'grad_norm': 0.0,
                'lr': self.scheduler.get_last_lr()[0],
            }

        # Backward pass (no GradScaler for BF16)
        loss.backward()

        # Track actual tokens processed (exclude padding)
        actual_tokens = (labels != -100).sum().item()
        self.total_tokens_processed += actual_tokens

        # Only step optimizer every N accumulation steps
        is_accumulation_step = (accumulation_step + 1) % grad_accum_steps == 0

        if is_accumulation_step:
            # Gradient clipping
            if self.config['optimizer'].get('max_grad_norm', 1.0) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['optimizer']['max_grad_norm']
                )
            else:
                grad_norm = 0.0

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Scheduler step (only when optimizer steps)
            self.scheduler.step()
        else:
            grad_norm = 0.0

        # Calculate perplexity (scale back the loss)
        actual_loss = loss.item() * grad_accum_steps
        perplexity = torch.exp(torch.tensor(actual_loss)).item()

        # Return standard training metrics
        return {
            'loss': actual_loss,
            'perplexity': perplexity,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'is_step': is_accumulation_step,
        }
    
    def get_lr_phase_info(self) -> str:
        """Get information about current learning rate phase."""
        current_lr = self.scheduler.get_last_lr()[0]
        initial_lr = self.config['optimizer']['learning_rate']
        warmup_steps = self.config['scheduler']['warmup_steps']

        if self.global_step < warmup_steps:
            progress = (self.global_step / warmup_steps) * 100
            return f"Warmup ({progress:.0f}%)"
        else:
            lr_ratio = current_lr / initial_lr
            if lr_ratio > 0.5:
                return "Cosine Decay (High LR)"
            elif lr_ratio > 0.1:
                return "Cosine Decay (Mid LR)"
            else:
                return "Cosine Decay (Low LR)"

    def check_gradient_health(self, grad_norm: float) -> str:
        """Check if gradients are healthy.

        Args:
            grad_norm: Current gradient norm

        Returns:
            Health status string
        """
        if grad_norm > 10.0:
            return "⚠️  High gradients - may need gradient clipping adjustment"
        elif grad_norm < 0.001:
            return "⚠️  Very small gradients - possible vanishing gradient"
        elif grad_norm == 0.0:
            return ""  # Accumulation step, no gradient
        else:
            return "✓ Healthy gradients"

    def get_quality_insights(self, val_ppl: float) -> str:
        """Get actionable insights based on validation perplexity.

        Args:
            val_ppl: Validation perplexity

        Returns:
            Quality assessment string with recommendations
        """
        if val_ppl >= 1000:
            return "🔴 Early Training - Model mostly gibberish, continue training"
        elif val_ppl >= 300:
            return "🟡 Making Progress - Words forming, needs more training"
        elif val_ppl >= 100:
            return "🟢 Good Progress - Coherent phrases, light testing possible"
        elif val_ppl >= 50:
            return "🟢 Strong Model - Good sentences, ready for serious testing"
        elif val_ppl >= 30:
            return "🟢 Very Good Model - High quality output, production-ready candidate"
        else:
            return "🟢 Excellent Model - State-of-art quality!"

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """Generate text sample from prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text string
        """
        if self.tokenizer is None:
            return "[No tokenizer available]"

        self.model.eval()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate tokens
        generated_ids = input_ids.copy()

        for _ in range(max_length):
            # Get logits for next token
            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype, enabled=True):
                    outputs = self.model(input_ids=input_tensor)
                    logits = outputs['logits']
            else:
                outputs = self.model(input_ids=input_tensor)
                logits = outputs['logits']

            # Get last token logits and apply temperature
            next_token_logits = logits[0, -1, :] / temperature

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append to sequence
            generated_ids.append(next_token)

            # Update input tensor (keep last max_seq_len tokens)
            max_seq_len = self.config['data']['max_seq_len']
            if len(generated_ids) > max_seq_len:
                generated_ids = generated_ids[-max_seq_len:]

            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

            # Stop at end of sequence token if present
            if hasattr(self.tokenizer, 'eos_token_id') and next_token == self.tokenizer.eos_token_id:
                break

        # Decode generated text
        try:
            generated_text = self.tokenizer.decode(generated_ids)
        except:
            generated_text = "[Decode error]"

        return generated_text

    @torch.no_grad()
    def validate(self, generate_samples: bool = False, epoch: Optional[int] = None) -> Dict[str, float]:
        """Run validation with optional sample generation.

        Args:
            generate_samples: Whether to generate text samples
            epoch: Current epoch number (for logging)

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        losses = []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            if self.use_amp:
                # Use new torch.amp API: autocast('cuda', ...) instead of autocast(device_type='cuda', ...)
                with autocast('cuda', dtype=self.amp_dtype, enabled=True):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)

        results = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
        }

        # Log to evaluation file
        status = ""
        if avg_loss < self.best_val_loss:
            status = "🌟 NEW BEST LOSS"
        if perplexity < self.best_val_ppl:
            status += " 🌟 NEW BEST PPL" if status else "🌟 NEW BEST PPL"

        if epoch is not None:
            eval_logger.info(
                f"Epoch {epoch+1:3d} | Step {self.global_step:6d} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:7.2f} | {status if status else 'OK'}"
            )
        else:
            eval_logger.info(
                f"Step {self.global_step:6d} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:7.2f} | {status if status else 'OK'}"
            )

        # Generate text samples if requested
        if generate_samples and self.tokenizer is not None:
            logger.info("\n" + "="*70)
            logger.info("Sample Generations:")
            logger.info("="*70)

            for i, prompt in enumerate(EVAL_PROMPTS[:3], 1):  # Generate from first 3 prompts
                sample = self.generate_sample(prompt, max_length=40, temperature=0.8)
                logger.info(f"\nPrompt {i}: {prompt}")
                logger.info(f"Generated: {sample}")

                # Also log to evaluation file
                eval_logger.info(f"  Sample {i}: {prompt} → {sample}")

            logger.info("="*70 + "\n")

        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint (with torch.compile unwrapping)."""
        # CRITICAL FIX: Unwrap compiled model before saving
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'best_checkpoint_step': self.best_checkpoint_step,
            'metrics': metrics,
            'config': self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint (by loss)
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            self.best_checkpoint_step = self.global_step
            best_path = self.checkpoint_dir / 'checkpoint_best_loss.pth'
            torch.save(checkpoint, best_path)
            elapsed = time.time() - self.training_start_time
            logger.info(
                f"✨ New best loss! Val loss: {metrics['val_loss']:.4f} "
                f"(saved at step {self.global_step}, {format_time(elapsed)} elapsed)"
            )

        # Save best checkpoint (by perplexity)
        if metrics['val_perplexity'] < self.best_val_ppl:
            self.best_val_ppl = metrics['val_perplexity']
            best_ppl_path = self.checkpoint_dir / f'checkpoint_best_ppl_{metrics["val_perplexity"]:.2f}.pth'
            torch.save(checkpoint, best_ppl_path)
            logger.info(
                f"✨ New best perplexity! Val PPL: {metrics['val_perplexity']:.2f} "
                f"(saved at step {self.global_step})"
            )

        # Save epoch checkpoint
        if (epoch + 1) % self.config['training'].get('save_steps', 5) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()

        # Update current epoch for logging
        self.current_epoch = epoch

        epoch_loss = 0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            # Train step with accumulation tracking
            metrics = self.train_step(batch, accumulation_step=step)

            epoch_loss += metrics['loss']

            # Only increment global step when optimizer actually stepped
            if metrics.get('is_step', False):
                self.global_step += 1
                num_batches += 1

            # Update progress bar with training metrics
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{metrics['lr']:.2e}",
                'grad': f"{metrics['grad_norm']:.2f}",
            })

            # Track recent training loss (exponential moving average)
            if metrics.get('is_step', False):
                alpha = 0.1  # EMA smoothing factor
                self.current_train_loss = alpha * metrics['loss'] + (1 - alpha) * self.current_train_loss

                # Track loss trend (last 10 steps)
                self.recent_losses.append(metrics['loss'])
                if len(self.recent_losses) > 10:
                    self.recent_losses.pop(0)

                # Track gradient norms
                if metrics['grad_norm'] > 0:
                    self.recent_grad_norms.append(metrics['grad_norm'])
                    if len(self.recent_grad_norms) > 10:
                        self.recent_grad_norms.pop(0)
                    self.max_grad_norm_seen = max(self.max_grad_norm_seen, metrics['grad_norm'])

            # Logging (only when we actually step)
            if metrics.get('is_step', False) and self.global_step % self.config['training'].get('logging_steps', 10) == 0:
                elapsed = time.time() - self.training_start_time
                tokens_per_sec = self.total_tokens_processed / elapsed if elapsed > 0 else 0

                # Calculate ETAs
                total_steps = len(self.train_loader) * self.config['training'].get('num_epochs', 10)
                steps_remaining = total_steps - self.global_step
                seconds_per_step = elapsed / self.global_step if self.global_step > 0 else 0
                eta_seconds = steps_remaining * seconds_per_step

                # Steps until next validation
                eval_steps = self.config['training'].get('eval_steps', 500)
                steps_to_next_val = eval_steps - (self.global_step % eval_steps)

                # GPU memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.max_memory_allocated() / 1024**3
                    self.peak_gpu_memory = max(self.peak_gpu_memory, current_memory)
                    memory_str = f"{current_memory:.2f}GB (peak: {self.peak_gpu_memory:.2f}GB)"
                else:
                    memory_str = "N/A"

                # LR phase
                lr_phase = self.get_lr_phase_info()

                # Loss trend (improving or getting worse)
                if len(self.recent_losses) >= 2:
                    recent_avg = np.mean(self.recent_losses[-5:]) if len(self.recent_losses) >= 5 else np.mean(self.recent_losses)
                    older_avg = np.mean(self.recent_losses[:5]) if len(self.recent_losses) >= 5 else self.recent_losses[0]
                    loss_trend = "↓ Improving" if recent_avg < older_avg else "↑ Increasing"
                else:
                    loss_trend = "..."

                logger.info(
                    f"\nStep {self.global_step} | "
                    f"Loss: {metrics['loss']:.4f} ({loss_trend}) | "
                    f"PPL: {metrics['perplexity']:.2f} | "
                    f"Grad: {metrics['grad_norm']:.2f}\n"
                    f"LR: {metrics['lr']:.2e} ({lr_phase}) | "
                    f"Throughput: {tokens_per_sec:.1f} tok/s | "
                    f"GPU Mem: {memory_str}\n"
                    f"ETA: {format_time(eta_seconds)} | "
                    f"Next Val: {steps_to_next_val} steps"
                )

                # Gradient health check
                if metrics['grad_norm'] > 0:
                    grad_health = self.check_gradient_health(metrics['grad_norm'])
                    if grad_health:
                        logger.info(f"{grad_health}")

                # ARTICLE FIX: Article learning progress check
                if article_ratio > 3.0:
                    logger.info("❌ Article loss >> Other loss - Model struggling with articles")
                elif article_ratio > 2.0:
                    logger.info("⚠️  Article loss high - Still learning articles")
                elif article_ratio > 1.5:
                    logger.info("🟡 Article loss improving - Getting better")
                elif article_ratio > 0.8 and article_ratio < 1.2:
                    logger.info("✅ Article loss balanced - Model learned articles!")
                elif article_ratio < 0.8:
                    logger.info("⚠️  Article loss too low - May be overfitting to articles")

            # Validation (only when we actually step)
            if metrics.get('is_step', False) and self.global_step % self.config['training'].get('eval_steps', 500) == 0:
                # Run validation with sample generation
                val_metrics = self.validate(generate_samples=True, epoch=self.current_epoch)

                # Calculate train-val gap
                train_val_gap = self.current_train_loss - val_metrics['val_loss']

                # Calculate progress percentage
                total_steps = len(self.train_loader) * self.config['training'].get('num_epochs', 10)
                progress_pct = (self.global_step / total_steps) * 100

                # Get quality insights
                quality_insight = self.get_quality_insights(val_metrics['val_perplexity'])

                logger.info(
                    f"\n{'='*70}\n"
                    f"Validation at Step {self.global_step} (Epoch {self.current_epoch + 1}, {progress_pct:.1f}% complete)\n"
                    f"{'='*70}\n"
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val PPL: {val_metrics['val_perplexity']:.2f}\n"
                    f"Train Loss: {self.current_train_loss:.4f} | "
                    f"Train-Val Gap: {train_val_gap:+.4f}\n"
                    f"\n{quality_insight}\n"
                    f"{'='*70}"
                )

                # Overfitting warning
                if train_val_gap < -0.5:  # Train loss much lower than val loss
                    logger.warning("⚠️  Possible overfitting detected! Train loss << Val loss")
                elif train_val_gap > 0.5:  # Val loss much lower than train loss
                    logger.info("📈 Model generalizing well! Val loss < Train loss")

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics)

                # Back to training mode
                self.model.train()

        return epoch_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch+1})")
        logger.info(f"Total training steps: {(num_epochs - start_epoch) * len(self.train_loader)}")

        self.training_start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Validation at end of epoch with sample generation
            val_metrics = self.validate(generate_samples=True, epoch=epoch)

            # Calculate train-val gap
            train_val_gap = train_loss - val_metrics['val_loss']

            # Get quality insights
            quality_insight = self.get_quality_insights(val_metrics['val_perplexity'])

            # Log epoch summary
            elapsed = time.time() - self.training_start_time
            logger.info(
                f"\n{'='*70}\n"
                f"Epoch {epoch+1}/{num_epochs} Summary\n"
                f"{'='*70}\n"
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}\n"
                f"Val Perplexity: {val_metrics['val_perplexity']:.2f} | "
                f"Train-Val Gap: {train_val_gap:+.4f}\n"
                f"Time Elapsed: {format_time(elapsed)}\n"
                f"\n{quality_insight}\n"
                f"{'='*70}"
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)
        
        total_time = time.time() - self.training_start_time

        # Get final quality assessment
        final_quality = self.get_quality_insights(self.best_val_ppl)

        logger.info(
            f"\n{'='*70}\n"
            f"Training Completed!\n"
            f"{'='*70}\n"
            f"Total Time: {format_time(total_time)}\n"
            f"Best Validation Loss: {self.best_val_loss:.4f}\n"
            f"Best Validation Perplexity: {self.best_val_ppl:.2f}\n"
            f"\n{final_quality}\n"
            f"{'='*70}"
        )


def main():
    parser = argparse.ArgumentParser(description='Train WikiMini 95M Language Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (overrides config value)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    try:
        # Setup CUDA optimizations
        setup_cuda_optimizations()

        # Setup file logging
        training_log, eval_log = setup_file_logging(log_dir="./logs")

        # Load configuration
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

        # Get num_epochs from config or command line (command line overrides)
        num_epochs = args.num_epochs if args.num_epochs is not None else config['training'].get('num_epochs', 10)
        logger.info(f"Training for {num_epochs} epochs")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {config['data']['tokenizer_path']}")
        tokenizer = load_tokenizer(config['data']['tokenizer_path'])

        # Override vocab_size in config
        config['model']['vocab_size'] = len(tokenizer)

        # Create model
        logger.info("Creating model...")
        model = WikiMiniModel(config['model'])

        # Count parameters
        num_params = count_parameters(model)
        logger.info(f"Model parameters: {num_params/1e6:.2f}M")

        # Run data quality checks (if enabled)
        if config['data'].get('check_quality', True):
            logger.info("\n" + "=" * 70)
            logger.info("Running data quality checks...")
            logger.info("=" * 70)

            dataset_name = config['data'].get('dataset', 'tinystories')
            quality_sample_size = config['data'].get('quality_sample_size', 10000)
            quality_strict = config['data'].get('quality_strict', False)

            quality_passed = check_dataset_quality(
                dataset_name=dataset_name,
                split="train",
                sample_size=quality_sample_size,
                strict=quality_strict,
            )

            if not quality_passed:
                error_msg = (
                    "\n⚠️  DATA QUALITY CHECK FAILED!\n"
                    "The dataset has quality issues that may affect training.\n"
                    "Options:\n"
                    "  1. Fix the dataset issues\n"
                    "  2. Set 'check_quality: false' in config to skip checks (not recommended)\n"
                    "  3. Set 'quality_strict: false' to allow training with warnings\n"
                )
                logger.error(error_msg)
                raise ValueError("Data quality check failed. See logs above for details.")

            logger.info("✅ Data quality check passed!\n")
        else:
            logger.warning("⚠️  Data quality checks are DISABLED in config")
            logger.warning("This may result in training on corrupted data!\n")

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=config['training']['batch_size'],
            max_seq_len=config['data']['max_seq_len'],
            cache_dir=config['data']['cache_dir'],
            dataset_name=config['data'].get('dataset', 'tinystories'),  # TinyStories by default
            num_workers=0,  # CRITICAL for Windows
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=config['training'].get('output_dir', './checkpoints'),
        )

        # Pass tokenizer to trainer for text generation
        trainer.tokenizer = tokenizer

        # CRITICAL FIX: Load checkpoint AFTER trainer creation to restore full state
        start_epoch = 0
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

            # Load model state (unwrap if necessary)
            model_to_load = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer and scheduler state
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore training state
            trainer.global_step = checkpoint.get('global_step', 0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))
            trainer.best_checkpoint_step = checkpoint.get('best_checkpoint_step', 0)
            start_epoch = checkpoint.get('epoch', 0) + 1

            logger.info(f"Resumed from epoch {start_epoch}, step {trainer.global_step}")
            logger.info(f"Best val loss: {trainer.best_val_loss:.4f}, Best val PPL: {trainer.best_val_ppl:.2f}")
            logger.info(f"Best checkpoint was at step {trainer.best_checkpoint_step}")

        # Train model
        logger.info("Starting training...")
        trainer.train(num_epochs, start_epoch=start_epoch)

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n" + "="*70)
        logger.info("Training interrupted by user (Ctrl+C)")
        logger.info("="*70)

        # Try to save emergency checkpoint
        try:
            if 'trainer' in locals():
                logger.info("Saving emergency checkpoint...")
                emergency_metrics = {'val_loss': trainer.best_val_loss}
                trainer.save_checkpoint(epoch=-1, metrics=emergency_metrics)
                logger.info("Emergency checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

    except Exception as e:
        logger.error("="*70)
        logger.error("Training failed with error:")
        logger.error("="*70)
        logger.error(str(e), exc_info=True)
        raise

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


if __name__ == "__main__":
    main()
