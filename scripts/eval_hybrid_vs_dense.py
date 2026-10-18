#!/usr/bin/env python3
"""Benchmark dense and hybrid checkpoints on the complete cached validation split."""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tokenizer import load_tokenizer
from src.model.hybrid import HybridLM, MoESwiGLU
from src.model.transformer_block import WikiMiniModel


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for module in model.modules():
        if isinstance(module, MoESwiGLU):
            per_expert = sum(p.numel() for p in module.experts[0].parameters())
            inactive += (module.n_experts - module.top_k) * per_expert
    return total, total - inactive


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    model_class = HybridLM if model_config.get("type") == "hybrid" else WikiMiniModel
    model = model_class(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, model


@torch.no_grad()
def benchmark(path, validation_path, batch_size, device):
    checkpoint, model = load_checkpoint(path, device)
    with validation_path.open("rb") as handle:
        cached = pickle.load(handle)
    inputs = torch.stack(cached["input_ids"])
    labels = torch.stack(cached["labels"])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss_sum = 0.0
    token_count = 0
    for input_ids, batch_labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(input_ids=input_ids, labels=batch_labels)
            logits = outputs["logits"]
            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch_labels.reshape(-1),
                ignore_index=-100, reduction="sum"
            )
        valid_tokens = (batch_labels != -100).sum().item()
        loss_sum += token_loss.float().item()
        token_count += valid_tokens
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total, active = parameter_counts(model)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    ce = loss_sum / token_count
    return {
        "sequences": len(cached["input_ids"]),
        "model": "HybridLM" if isinstance(model, HybridLM) else "Dense WikiMini",
        "total": total,
        "active": active,
        "ce": ce,
        "ppl": math.exp(ce),
        "tokens_per_sec": token_count / elapsed,
        "peak_memory": peak_memory,
        "step": checkpoint.get("global_step", checkpoint.get("step", "N/A")),
        "epoch": checkpoint.get("epoch", "N/A"),
        "tokenizer": checkpoint["config"]["data"]["tokenizer_path"],
        "vocab": checkpoint["config"]["model"]["vocab_size"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=ROOT / "data/cache/tinystories_validation_512.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results = [
        benchmark(args.hybrid_checkpoint, args.validation, args.batch_size, device),
        benchmark(args.dense_checkpoint, args.validation, args.batch_size, device),
    ]
    print(f"validation={args.validation} sequences={results[0]['sequences']} batch_size={args.batch_size} device={device} autocast=bf16")
    print("model | total_params | active_params | val_ce | val_ppl | tokens/s | peak_mem_GiB | train_step | epoch")
    for result in results:
        print(f"{result['model']} | {result['total']:,} | {result['active']:,} | "
              f"{result['ce']:.6f} | {result['ppl']:.4f} | {result['tokens_per_sec']:.1f} | "
              f"{result['peak_memory']:.3f} | {result['step']} | {result['epoch']}")
        print(f"  tokenizer={result['tokenizer']} vocab={result['vocab']}")


if __name__ == "__main__":
    main()
