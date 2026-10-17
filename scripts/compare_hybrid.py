"""Small, reproducible dense-versus-hybrid TinyStories training run."""

import os
import random
import time
import urllib.request
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.hybrid import HybridLM
from src.model.transformer_block import WikiMiniModel


def main():
    seed = 17
    random.seed(seed)
    torch.manual_seed(seed)
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(os.environ.get("HYBRID_DEVICE", default_device))
    steps = int(os.environ.get("HYBRID_STEPS", "60"))
    seq_len, batch_size, vocab_target = 64, 2, 4000
    print(f"device={device} steps={steps} seq_len={seq_len} batch={batch_size} seed={seed}", flush=True)
    data_path = Path("data/cache/tinystories_sample.txt")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        byte_count = 5_000_000
        request = urllib.request.Request(
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
            headers={"Range": f"bytes=0-{byte_count - 1}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            data_path.write_bytes(response.read(byte_count))
    raw_text = data_path.read_bytes().decode("utf-8", errors="ignore")
    texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    total_chars = sum(map(len, texts))
    print(f"loaded {len(texts)} stories ({total_chars:,} chars, {data_path.stat().st_size:,} bytes)", flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_target, min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]))
    vocab = tokenizer.get_vocab_size()
    ids = torch.tensor([i for text in texts for i in tokenizer.encode(text).ids], dtype=torch.long)
    split = int(ids.numel() * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    if len(val_ids) < seq_len + 1:
        raise RuntimeError("TinyStories slice did not yield enough validation tokens")
    print(f"vocab={vocab} train_tokens={len(train_ids):,} val_tokens={len(val_ids):,}", flush=True)

    common = dict(vocab_size=vocab, d_model=32, n_heads=4, d_ffn=32,
                  max_seq_len=seq_len, dropout=0.0, use_flash_attention=False)
    dense = WikiMiniModel(dict(common, n_layers=2)).to(device)
    hybrid_cfg = dict(common, n_layers=2, layer_pattern="MA", mamba_heads=4,
                      state_size=4, n_experts=2, top_k=1, d_ffn=96,
                      mtp_weight=0.1, aux_weight=0.01)
    hybrid = HybridLM(hybrid_cfg).to(device)
    model_runs = [("Dense WikiMini", dense), ("HybridLM", hybrid)]
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in model_runs}
    print(f"params dense={counts['Dense WikiMini']:,} hybrid={counts['HybridLM']:,}", flush=True)

    results = {}
    for name, model in model_runs:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        losses = []
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        model.train()
        for step in range(steps):
            starts = torch.randint(0, len(train_ids) - seq_len - 1, (batch_size,))
            x = torch.stack([train_ids[i:i + seq_len] for i in starts]).to(device)
            y = torch.stack([train_ids[i + 1:i + seq_len + 1] for i in starts]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, labels=y)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % 10 == 0:
                print(f"{name} step {step + 1}/{steps}", flush=True)
            if (step + 1) in {1, steps // 4, steps // 2, 3 * steps // 4, steps}:
                losses.append((step + 1, float(loss.detach())))
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        model.eval()
        vals = []
        with torch.no_grad():
            for start_idx in range(0, min(2048, len(val_ids) - seq_len - 1), seq_len):
                x = val_ids[start_idx:start_idx + seq_len].unsqueeze(0).to(device)
                y = val_ids[start_idx + 1:start_idx + seq_len + 1].unsqueeze(0).to(device)
                vals.append(float(torch.nn.functional.cross_entropy(
                    model(x)["logits"].reshape(-1, vocab), y.reshape(-1))))
        val_loss = sum(vals) / len(vals)
        samples = []
        with torch.no_grad():
            for sample_idx in range(2):
                generated = train_ids[sample_idx * 8:(sample_idx + 1) * 8].unsqueeze(0).to(device)
                for _ in range(24):
                    logits = model(generated[:, -seq_len:])["logits"]
                    generated = torch.cat([generated, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
                samples.append(tokenizer.decode(generated[0].tolist()))
        results[name] = dict(params=counts[name], losses=losses, val=val_loss,
                             tok_s=steps * batch_size * seq_len / elapsed,
                             samples=samples, seconds=elapsed)
        print(f"finished {name}: {elapsed:.1f}s val={val_loss:.4f} tok/s={results[name]['tok_s']:.1f}", flush=True)

    lines = ["# TinyStories: dense vs hybrid (Mac run)", "",
             f"Device: `{device}`; seed {seed}; {steps} steps per model; {seq_len} tokens/sequence; {batch_size} sequences/step.",
             f"Data: first 5 MB of TinyStories train text ({total_chars:,} characters), one shared {vocab:,}-token BPE; first 90% train, final 10% held out.",
             "Validation reports main next-token CE only. Runtime and tokens/s are wall-clock for training.", "",
             "| Model | Parameters | Train loss (step:loss) | Held-out main CE | Tokens/s | Train seconds |",
             "|---|---:|---|---:|---:|---:|"]
    for name, result in results.items():
        points = ", ".join(f"{step}:{loss:.3f}" for step, loss in result["losses"])
        lines.append(f"| {name} | {result['params']:,} | {points} | {result['val']:.4f} | {result['tok_s']:.1f} | {result['seconds']:.1f} |")
    for name, result in results.items():
        lines.extend(["", f"## {name} samples", ""])
        for i, sample in enumerate(result["samples"], 1):
            lines.append(f"{i}. {sample.replace('|', '\\|')}")
    Path("results").mkdir(exist_ok=True)
    Path("results/hybrid_vs_dense.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
