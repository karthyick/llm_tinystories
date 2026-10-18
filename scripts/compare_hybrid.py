"""Reproducible three-way TinyStories comparison for dense and hybrid models."""

import os
import argparse
import math
import random
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--d-model', type=int, default=96)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--out', type=Path, default=Path('results/hybrid_vs_dense.md'))
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'auto':
        device_name = os.environ.get('HYBRID_DEVICE') or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device_name = args.device
    device = torch.device(device_name)
    steps = args.steps if args.steps is not None else int(os.environ.get('HYBRID_STEPS', '600'))
    seq_len, batch_size, vocab_target = args.seq_len, args.batch_size, 4000
    print(f"device={device} steps={steps} seq_len={seq_len} batch={batch_size} seed={seed}", flush=True)
    data_path = Path("data/cache/tinystories_sample.txt")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        byte_count = 5_000_000
        request = urllib.request.Request("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
                                         headers={"Range": f"bytes=0-{byte_count - 1}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            data_path.write_bytes(response.read(byte_count))
    raw_text = data_path.read_bytes().decode("utf-8", errors="ignore")
    texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_target, min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]))
    vocab = tokenizer.get_vocab_size()
    ids = torch.tensor([i for text in texts for i in tokenizer.encode(text).ids], dtype=torch.long)
    split = int(ids.numel() * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"stories={len(texts):,} chars={sum(map(len, texts)):,} vocab={vocab:,} train_tokens={len(train_ids):,} val_tokens={len(val_ids):,}", flush=True)

    common = dict(vocab_size=vocab, d_model=args.d_model, n_heads=4, d_ffn=args.d_model * 3, max_seq_len=seq_len,
                  dropout=0.0, use_flash_attention=False, tie_embeddings=True)
    pattern = "MFMFMFAF"
    moe_cfg = dict(common, layer_pattern=pattern, mamba_heads=4, state_size=4,
                   n_experts=8, top_k=2, expert_ffn=96, mtp_weight=0.3, aux_weight=1e-4, use_moe=True)
    dense_ffn_cfg = dict(moe_cfg, use_moe=False, dense_ffn=288)
    dense_model = WikiMiniModel(dict(common, n_layers=8))
    models_to_run = [
        ("Dense WikiMini", dense_model, None),
        ("HybridLM MoE", HybridLM(moe_cfg), moe_cfg),
        ("HybridLM dense FFN", HybridLM(dense_ffn_cfg), dense_ffn_cfg),
    ]
    counts = {}
    for name, model, _ in models_to_run:
        model.to(device)
        counts[name] = parameter_counts(model)
    for name, (total, active) in counts.items():
        print(f"params {name}: total={total:,} active={active:,}", flush=True)

    results = {}
    for name, model, _ in models_to_run:
        # Reset RNG before each run so each architecture receives identical batches.
        random.seed(seed)
        torch.manual_seed(seed)
        model.train()
        init_starts = torch.randint(0, len(train_ids) - seq_len - 1, (batch_size,))
        init_batch = torch.stack([train_ids[i:i + seq_len] for i in init_starts]).to(device)
        init_labels = torch.stack([train_ids[i + 1:i + seq_len + 1] for i in init_starts]).to(device)
        model.eval()
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with torch.no_grad(), amp:
            init_output = model(init_batch, labels=init_labels)
            init_ce = float(init_output.get("main_loss", init_output["loss"]))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        warmup = max(20, steps // 20)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup else
            0.5 * (1 + math.cos((s - warmup) / max(1, steps - warmup) * math.pi)))
        points, start = [], time.perf_counter()
        if device.type == "mps": torch.mps.synchronize()
        for step in range(steps):
            if hasattr(model, "mtp_weight"):
                model.mtp_weight = 0.3 - 0.2 * (step / max(1, steps - 1))
            starts = torch.randint(0, len(train_ids) - seq_len - 1, (batch_size,))
            x = torch.stack([train_ids[i:i + seq_len] for i in starts]).to(device)
            y = torch.stack([train_ids[i + 1:i + seq_len + 1] for i in starts]).to(device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with amp:
                output = model(x, labels=y)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            if (step + 1) in sorted(set([1, steps // 10, steps // 4, steps // 2, 3 * steps // 4, steps])):
                points.append((step + 1, float(output.get("main_loss", output["loss"]).detach())))
            if (step + 1) % 50 == 0: print(f"{name} step {step + 1}/{steps}", flush=True)
        if device.type == "mps": torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        model.eval()
        vals = []
        with torch.no_grad():
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with amp:
                for offset in range(0, min(2048, len(val_ids) - seq_len - 1), seq_len):
                    x = val_ids[offset:offset + seq_len].unsqueeze(0).to(device)
                    y = val_ids[offset + 1:offset + seq_len + 1].unsqueeze(0).to(device)
                    vals.append(float(torch.nn.functional.cross_entropy(model(x)["logits"].reshape(-1, vocab), y.reshape(-1))))
        samples = []
        with torch.no_grad():
            for sample_idx in range(2):
                generated = train_ids[sample_idx * 8:(sample_idx + 1) * 8].unsqueeze(0).to(device)
                for _ in range(32):
                    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                    with amp:
                        logits = model(generated[:, -seq_len:])["logits"]
                    generated = torch.cat([generated, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
                samples.append(tokenizer.decode(generated[0].tolist()))
        results[name] = dict(init_ce=init_ce, losses=points, val=sum(vals)/len(vals), tok_s=steps*batch_size*seq_len/elapsed,
                             seconds=elapsed, samples=samples)
        print(f"finished {name}: {elapsed:.1f}s val={results[name]['val']:.4f}", flush=True)

    lines = ["# TinyStories: dense vs hybrid", "",
        f"Device `{device}`; seed {seed}; {steps} optimizer steps/model; seq_len {seq_len}; batch {batch_size}; AdamW with {warmup}-step linear warmup and cosine decay; MTP weight decays 0.3 to 0.1.",
        f"Same first 5 MB TinyStories train slice ({sum(map(len, texts)):,} chars), shared {vocab:,}-token BPE and 90/10 token split. Equal token budget: {steps*batch_size*seq_len:,} train tokens/model.",
        "Active params count all dense/shared weights plus top-k routed expert weights; total params count all experts. Init/train/validation CE is main next-token CE; optimizer includes MTP and balance auxiliary objectives.", "",
        "| Model | Total params | Active params | Init main CE | Train main CE (step:value) | Held-out main CE | Tokens/s | Train seconds |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for name, result in results.items():
        total, active = counts[name]
        points = ", ".join(f"{step}:{loss:.3f}" for step, loss in result["losses"])
        lines.append(f"| {name} | {total:,} | {active:,} | {result['init_ce']:.4f} | {points} | {result['val']:.4f} | {result['tok_s']:.1f} | {result['seconds']:.1f} |")
    for name, result in results.items():
        lines.extend(["", f"## {name} samples", ""])
        lines.extend(f"{idx}. {sample.replace('|', chr(92) + '|')}" for idx, sample in enumerate(result["samples"], 1))
    winner = min(results, key=lambda key: results[key]["val"])
    dense_ce, moe_ce = results["Dense WikiMini"]["val"], results["HybridLM MoE"]["val"]
    if winner == "Dense WikiMini":
        reading = f"Dense WikiMini has the lowest held-out CE ({dense_ce:.3f}); HybridLM MoE is {moe_ce-dense_ce:+.3f} above it after {steps} updates. This scale favors dense optimization: routed experts receive little data each, while recurrent mixing and routing add compute overhead. The hybrid shows no quality or speed advantage in this run."
    else:
        reading = f"{winner} has the lowest held-out CE ({results[winner]['val']:.3f}); Dense WikiMini is {dense_ce-results[winner]['val']:+.3f} above it. This single-seed, small-slice run is limited evidence; it does not establish a scaling advantage for hybrid routing or recurrent mixing."
    lines.extend(["", "## Reading", "", reading])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


if __name__ == "__main__": main()
