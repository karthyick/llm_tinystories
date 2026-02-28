"""Evaluation utilities for TinyStories training.

Provides:
- evaluate(): Compute validation loss and perplexity on up to N held-out examples
- EvalLogger: Append step, train_loss, val_loss, perplexity rows to CSV
- EarlyStopping: Track val_loss history and trigger stop after patience consecutive increases
"""

import math
import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int = 1000,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Compute validation loss and perplexity on held-out examples.

    Runs model in eval mode with torch.no_grad(). Restores the original
    training/eval mode when done. Stops after processing max_samples examples.

    Args:
        model: Language model that returns dict with 'loss' key when labels are provided.
        val_loader: DataLoader yielding batches with 'input_ids' and 'labels' keys.
        device: Device to run evaluation on.
        max_samples: Maximum number of examples to evaluate (default: 1000).
            Evaluation stops as soon as this many samples have been processed.
        use_amp: Whether to use automatic mixed precision (default: False).
        amp_dtype: AMP dtype (e.g. torch.bfloat16) when use_amp=True (default: None).

    Returns:
        Dict with keys:
            - val_loss (float): Average cross-entropy loss over evaluated batches.
            - val_perplexity (float): exp(val_loss), capped at exp(100) to avoid overflow.
            - samples_evaluated (int): Actual number of examples processed.
            - batches_evaluated (int): Number of batches processed.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0
    samples_seen = 0

    with torch.no_grad():
        for batch in val_loader:
            if samples_seen >= max_samples:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_size = input_ids.size(0)

            if use_amp and amp_dtype is not None:
                with autocast("cuda", dtype=amp_dtype, enabled=True):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1
            samples_seen += batch_size

    avg_loss = total_loss / max(num_batches, 1)
    # Cap the exponent to avoid overflow on very high loss values
    perplexity = math.exp(min(avg_loss, 100.0))

    if was_training:
        model.train()

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "samples_evaluated": samples_seen,
        "batches_evaluated": num_batches,
    }


class EvalLogger:
    """Logs evaluation metrics to a CSV file.

    Appends one row per evaluation call with columns:
        step, train_loss, val_loss, perplexity

    Creates the file with a header row if it does not already exist.
    Parent directories are created automatically.

    Example CSV output:
        step,train_loss,val_loss,perplexity
        200,3.142100,3.215400,24.9100
        400,2.891200,3.012300,20.3200
    """

    CSV_COLUMNS = ["step", "train_loss", "val_loss", "perplexity"]

    def __init__(self, output_path: str = "./results/eval_results.csv") -> None:
        """Initialize EvalLogger.

        Args:
            output_path: Path to the CSV file. Parent directories are created
                automatically. The file is created with a header row if it does
                not already exist.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Write CSV header if the file does not already exist."""
        if not self.output_path.exists():
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def log(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
    ) -> None:
        """Append one evaluation row to the CSV file.

        Args:
            step: Global training step number.
            train_loss: Training loss (or EMA of recent losses) at this step.
            val_loss: Validation loss at this step.
            perplexity: Validation perplexity (exp(val_loss)) at this step.
        """
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(
                {
                    "step": int(step),
                    "train_loss": round(float(train_loss), 6),
                    "val_loss": round(float(val_loss), 6),
                    "perplexity": round(float(perplexity), 4),
                }
            )


class EarlyStopping:
    """Stops training when validation loss stops improving.

    Triggers after `patience` consecutive evaluations where val_loss does not
    strictly improve (decrease). Resets the counter whenever an improvement
    is observed.

    Example:
        stopper = EarlyStopping(patience=3)
        for step in training_loop:
            val_loss = evaluate(...)
            if stopper.step(val_loss):
                logger.info("Early stopping triggered")
                break

    Attributes:
        patience (int): Number of non-improving evals before stopping.
        best_val_loss (float): Lowest val_loss seen so far.
        consecutive_increases (int): Count of evals without improvement.
        should_stop (bool): True once the patience limit is reached.
    """

    def __init__(self, patience: int = 3) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of consecutive non-improving evaluations before
                training should stop. Must be >= 1.

        Raises:
            ValueError: If patience < 1.
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.best_val_loss: float = float("inf")
        self.consecutive_increases: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        An improvement is defined as val_loss strictly less than the current best.
        Any non-improvement (equal or worse) increments the counter.

        Args:
            val_loss: Validation loss from the most recent evaluation.

        Returns:
            True if training should stop (patience exceeded), False otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.consecutive_increases = 0
            self.should_stop = False
        else:
            self.consecutive_increases += 1
            if self.consecutive_increases >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset all state (e.g., at the start of a new training run)."""
        self.best_val_loss = float("inf")
        self.consecutive_increases = 0
        self.should_stop = False
