"""Unit tests for the evaluation module.

Covers:
1. test_perplexity_computation  — evaluate() returns exp(loss) within tolerance
2. test_eval_logger_csv         — EvalLogger creates CSV with correct headers and appends rows
3. test_early_stopping          — EarlyStopping triggers after patience consecutive increases,
                                   does not trigger on decreasing losses, resets on improvement
"""

import csv
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EarlyStopping, EvalLogger, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstantLossModel(nn.Module):
    """Minimal model that always returns a fixed loss value for testing."""

    def __init__(self, fixed_loss: float) -> None:
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))  # needs at least one param
        self.fixed_loss = fixed_loss

    def forward(self, input_ids=None, labels=None):
        # Return the fixed loss as a scalar tensor on the same device as param
        loss = torch.tensor(self.fixed_loss, dtype=torch.float32, device=self._param.device)
        return {"loss": loss, "logits": torch.zeros(1)}


def _make_val_loader(n_batches: int = 5, batch_size: int = 4, seq_len: int = 8) -> DataLoader:
    """Build a tiny DataLoader with dummy token ids and labels."""
    input_ids = torch.randint(0, 100, (n_batches * batch_size, seq_len))
    labels = torch.randint(0, 100, (n_batches * batch_size, seq_len))
    dataset = TensorDataset(input_ids, labels)

    def collate(batch):
        ids, lbl = zip(*batch)
        return {"input_ids": torch.stack(ids), "labels": torch.stack(lbl)}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# 1. Perplexity computation tests
# ---------------------------------------------------------------------------


class TestPerplexityComputation:
    """Tests that evaluate() correctly computes val_loss and perplexity."""

    def test_perplexity_equals_exp_loss_for_known_loss(self):
        """evaluate() should return perplexity = exp(val_loss) within 1e-4 tolerance."""
        known_loss = 2.5
        model = ConstantLossModel(fixed_loss=known_loss)
        loader = _make_val_loader(n_batches=3, batch_size=4)
        device = torch.device("cpu")

        result = evaluate(model, loader, device, max_samples=100)

        expected_perplexity = math.exp(known_loss)
        assert abs(result["val_loss"] - known_loss) < 1e-4, (
            f"val_loss {result['val_loss']:.6f} != expected {known_loss}"
        )
        assert abs(result["val_perplexity"] - expected_perplexity) < 1e-3, (
            f"val_perplexity {result['val_perplexity']:.6f} != exp({known_loss})={expected_perplexity:.6f}"
        )

    def test_perplexity_for_zero_loss(self):
        """When loss is 0, perplexity should be exp(0) = 1.0."""
        model = ConstantLossModel(fixed_loss=0.0)
        loader = _make_val_loader(n_batches=2, batch_size=4)
        device = torch.device("cpu")

        result = evaluate(model, loader, device, max_samples=100)

        assert abs(result["val_loss"] - 0.0) < 1e-6
        assert abs(result["val_perplexity"] - 1.0) < 1e-4

    def test_high_loss_does_not_overflow(self):
        """Loss values above 100 should be capped to prevent exp() overflow."""
        model = ConstantLossModel(fixed_loss=200.0)
        loader = _make_val_loader(n_batches=2, batch_size=4)
        device = torch.device("cpu")

        result = evaluate(model, loader, device, max_samples=100)

        assert math.isfinite(result["val_perplexity"]), "perplexity should be finite even for very high loss"

    def test_max_samples_limits_batches_processed(self):
        """evaluate() should stop after processing max_samples examples."""
        model = ConstantLossModel(fixed_loss=1.0)
        n_batches = 10
        batch_size = 4
        loader = _make_val_loader(n_batches=n_batches, batch_size=batch_size)
        device = torch.device("cpu")

        max_samples = 8  # only 2 batches of 4
        result = evaluate(model, loader, device, max_samples=max_samples)

        # samples_evaluated should not exceed max_samples significantly
        assert result["samples_evaluated"] <= max_samples + batch_size, (
            f"Too many samples evaluated: {result['samples_evaluated']} > {max_samples}"
        )
        assert result["batches_evaluated"] < n_batches, "Should have stopped before all batches"

    def test_model_restored_to_train_mode_after_eval(self):
        """evaluate() must restore the model to train() mode if it was training before."""
        model = ConstantLossModel(fixed_loss=1.0)
        model.train()
        assert model.training, "precondition: model should be in train mode"

        loader = _make_val_loader(n_batches=2, batch_size=4)
        evaluate(model, loader, torch.device("cpu"), max_samples=100)

        assert model.training, "model should be restored to train mode after evaluate()"

    def test_model_stays_in_eval_mode_if_started_eval(self):
        """evaluate() must leave model in eval() mode if it was in eval mode before."""
        model = ConstantLossModel(fixed_loss=1.0)
        model.eval()
        assert not model.training, "precondition: model should be in eval mode"

        loader = _make_val_loader(n_batches=2, batch_size=4)
        evaluate(model, loader, torch.device("cpu"), max_samples=100)

        assert not model.training, "model should remain in eval mode"

    def test_returns_required_keys(self):
        """evaluate() must return all required keys."""
        model = ConstantLossModel(fixed_loss=1.0)
        loader = _make_val_loader()
        result = evaluate(model, loader, torch.device("cpu"), max_samples=100)

        for key in ("val_loss", "val_perplexity", "samples_evaluated", "batches_evaluated"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 2. EvalLogger CSV tests
# ---------------------------------------------------------------------------


class TestEvalLoggerCsv:
    """Tests that EvalLogger creates the CSV with correct headers and appends rows."""

    def test_csv_created_with_correct_headers(self, tmp_path):
        """EvalLogger should create the CSV file with the correct column headers."""
        csv_path = tmp_path / "eval_results.csv"
        logger = EvalLogger(output_path=str(csv_path))

        assert csv_path.exists(), "CSV file should be created on init"

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["step", "train_loss", "val_loss", "perplexity"], (
                f"Unexpected headers: {reader.fieldnames}"
            )

    def test_log_appends_row_with_correct_values(self, tmp_path):
        """log() should append exactly one row with the provided values."""
        csv_path = tmp_path / "eval_results.csv"
        logger = EvalLogger(output_path=str(csv_path))

        logger.log(step=200, train_loss=3.1421, val_loss=3.2154, perplexity=24.91)

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        row = rows[0]
        assert int(row["step"]) == 200
        assert abs(float(row["train_loss"]) - 3.1421) < 1e-4
        assert abs(float(row["val_loss"]) - 3.2154) < 1e-4
        assert abs(float(row["perplexity"]) - 24.91) < 1e-2

    def test_multiple_log_calls_append_multiple_rows(self, tmp_path):
        """Multiple log() calls should append multiple rows in order."""
        csv_path = tmp_path / "eval_results.csv"
        logger = EvalLogger(output_path=str(csv_path))

        records = [
            (200, 3.5, 3.6, 36.6),
            (400, 3.1, 3.2, 24.5),
            (600, 2.8, 2.9, 18.2),
        ]
        for step, tl, vl, ppl in records:
            logger.log(step=step, train_loss=tl, val_loss=vl, perplexity=ppl)

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        for i, (step, tl, vl, ppl) in enumerate(records):
            assert int(rows[i]["step"]) == step
            assert abs(float(rows[i]["val_loss"]) - vl) < 1e-4

    def test_does_not_overwrite_existing_csv(self, tmp_path):
        """A second EvalLogger on the same path should not erase existing rows."""
        csv_path = tmp_path / "eval_results.csv"

        logger1 = EvalLogger(output_path=str(csv_path))
        logger1.log(step=100, train_loss=4.0, val_loss=4.1, perplexity=60.3)

        # Create a second logger pointing to the same file
        logger2 = EvalLogger(output_path=str(csv_path))
        logger2.log(step=200, train_loss=3.5, val_loss=3.6, perplexity=36.6)

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    def test_parent_directory_created_automatically(self, tmp_path):
        """EvalLogger should create missing parent directories."""
        csv_path = tmp_path / "deep" / "nested" / "eval_results.csv"
        logger = EvalLogger(output_path=str(csv_path))

        assert csv_path.exists(), "CSV should be created even if parent dirs did not exist"


# ---------------------------------------------------------------------------
# 3. EarlyStopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Tests that EarlyStopping triggers, skips, and resets correctly."""

    def test_does_not_trigger_on_strictly_decreasing_losses(self):
        """Early stopping should not trigger when val_loss keeps decreasing."""
        stopper = EarlyStopping(patience=3)
        for loss in [4.0, 3.5, 3.0, 2.5, 2.0]:
            triggered = stopper.step(loss)
            assert not triggered, f"Should not trigger on decreasing losses (loss={loss})"

    def test_triggers_after_exactly_patience_consecutive_increases(self):
        """Should trigger exactly after `patience` consecutive non-improving evals."""
        patience = 3
        stopper = EarlyStopping(patience=patience)

        # Establish a good baseline
        stopper.step(3.0)

        # Feed patience-1 non-improving values → should NOT trigger yet
        for i in range(patience - 1):
            triggered = stopper.step(3.5)
            assert not triggered, f"Should not trigger before {patience} consecutive increases (i={i})"

        # The patience-th non-improving value → SHOULD trigger
        triggered = stopper.step(3.5)
        assert triggered, f"Should trigger after {patience} consecutive non-improving evals"
        assert stopper.should_stop is True

    def test_resets_counter_on_improvement(self):
        """Counter should reset to 0 when val_loss improves, preventing premature stop."""
        stopper = EarlyStopping(patience=3)

        stopper.step(3.0)   # baseline
        stopper.step(3.5)   # non-improving (count=1)
        stopper.step(3.5)   # non-improving (count=2)
        stopper.step(2.5)   # improvement → counter resets

        # Now two more non-improving — should NOT trigger (counter started fresh)
        assert not stopper.step(2.8), "Should not trigger after reset + 1 non-improving"
        assert not stopper.step(2.8), "Should not trigger after reset + 2 non-improving"
        # Third non-improving after reset → triggers
        assert stopper.step(2.8), "Should trigger after reset + 3 non-improving"

    def test_updates_best_val_loss_on_improvement(self):
        """best_val_loss should be updated when a lower val_loss is seen."""
        stopper = EarlyStopping(patience=3)
        stopper.step(3.0)
        stopper.step(2.5)
        stopper.step(2.0)

        assert stopper.best_val_loss == 2.0

    def test_consecutive_increases_count(self):
        """consecutive_increases should accurately count non-improving evals."""
        stopper = EarlyStopping(patience=5)
        stopper.step(3.0)  # baseline, consecutive_increases stays 0

        for expected_count in range(1, 4):
            stopper.step(3.5)
            assert stopper.consecutive_increases == expected_count

    def test_reset_clears_all_state(self):
        """reset() should restore initial state so the stopper can be reused."""
        stopper = EarlyStopping(patience=2)
        stopper.step(3.0)
        stopper.step(3.5)
        stopper.step(3.5)  # triggers
        assert stopper.should_stop

        stopper.reset()

        assert stopper.best_val_loss == float("inf")
        assert stopper.consecutive_increases == 0
        assert stopper.should_stop is False
        # Should not trigger immediately after reset
        assert not stopper.step(5.0)

    def test_patience_one_triggers_on_first_non_improvement(self):
        """With patience=1, any non-improving eval should trigger stopping."""
        stopper = EarlyStopping(patience=1)
        stopper.step(3.0)  # baseline
        triggered = stopper.step(3.0)  # equal → non-improving
        assert triggered

    def test_invalid_patience_raises_value_error(self):
        """EarlyStopping should raise ValueError for patience < 1."""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStopping(patience=0)

        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStopping(patience=-5)

    def test_equal_loss_counts_as_non_improving(self):
        """A val_loss equal to best (not strictly less) should increment the counter."""
        stopper = EarlyStopping(patience=2)
        stopper.step(3.0)  # best = 3.0
        stopper.step(3.0)  # equal, not strictly less → count=1
        triggered = stopper.step(3.0)  # count=2, patience=2 → triggers
        assert triggered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
