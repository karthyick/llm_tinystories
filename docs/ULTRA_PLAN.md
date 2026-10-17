# HybridLM: finish plan for KR-ULTRA (RTX 5090)

Goal: an NVIDIA-style hybrid small LM (Mamba-2 + MoE + MTP) trained next to the existing
dense TinyStories model, compared honestly, and merged to `main`.

```
 DONE on the Mac (2026-10-18)                 TODO on ULTRA
 ----------------------------                 -------------
 HybridLM code + 7 tests       ──────────▶   S0 setup  ─▶ S1 tests on CUDA
 init bug fixed (CE 33 → 8.31)                S2 small A/B (3 rows, 1-3M params)
 NVIDIA corrections in code                   S3 train.py switch + hybrid yaml data block
 branch pushed                                S4 full run: hybrid ~26M vs dense ~24.6M
                                              S5 results + honest reading ─▶ S6 PR, review, merge
```

## 1. Where it stands (each line re-checked 2026-10-18 03:5x IST on the Mac)

| Claim | Proof (command → result) |
|---|---|
| Branch pushed | `git ls-remote --heads origin orch/20261018-025111-01b6` → `4bf6051` |
| 2 commits, author KR | `git log main..HEAD` → `4bf6051 wip(model): …`, `1a06f91 feat(model): …`, both "Karthick Raja M" |
| Tree clean | `git status --short \| wc -l` → `0` |
| Hybrid tests pass | `pytest tests/test_hybrid.py` → 7 passed (shape + causality, sequential == segsum scan, init CE near ln(vocab) ×2, MoE top-k + shared + aux, full model causality + MTP backward, tiny overfit) |
| No regression | branch: `pytest tests` → 13 failed / 101 passed; clean `main` 30cb207: 13 failed / 94 passed. The same 13 `tests/test_server.py` failures exist on `main` (e.g. `assert 400 in [200, 405]` at test_server.py:325), so none are ours; +7 = the new hybrid tests |
| Round-1 bug fixed | round 1: step-1 loss 33.05 vs ln(4000)=8.29, samples "nurse nurse…". Now init main CE 8.31 (`results/hybrid_vs_dense.md`, `test_initial_main_ce_near_uniform*`) |
| Sizes matched | dense 33M yaml → 24,557,120 params; hybrid yaml → 26,095,412 total (fewer active, top-2 of 8 experts) |
| A/B result | **NOT DONE.** `results/hybrid_vs_dense.md` is a 3-step smoke, numbers mean nothing |

NVIDIA / source corrections, checked in code:

| Idea | Source | In code? | Where |
|---|---|---|---|
| Separate Mamba / FFN residual layers, few attention, first M last F | Nemotron-H arXiv 2504.03624 | yes | `model.py` pattern `MFMFMFAF` |
| Sigmoid router + aux-loss-free expert bias (step 1e-3) + balance loss 1e-4 | Nemotron 3 arXiv 2512.20856 | yes | `moe.py:18,25,38`, yaml `aux_weight: 0.0001` |
| Shared expert always on | Nemotron 3 | yes | `moe.py` `shared_expert` |
| MTP: norm(h) + norm(emb) → proj → block → shared head, t+2, weight 0.3 | DeepSeek-V3 arXiv 2412.19437 | yes | `model.py:95-102` |
| A = -exp(A_log) init 1..16, dt_bias log-uniform 1e-3..0.1, D skip | mamba2_simple.py | yes | `mamba2.py:60-90` |
| Exact scan + equality test | Mamba-2 arXiv 2405.21060 | yes | `segsum_scan` default, `test_sequential_scan_equals_segsum_scan` |
| Squared-ReLU experts | Nemotron 3 | **no** (SwiGLU kept on purpose) | decide in S2 only if time |
| LatentMoE | Nemotron 3.5 | **no** (multi-GPU fix, not needed at this size) | none |

## 2. Gaps found while writing this plan

| Gap | Why it matters | Fixed in step |
|---|---|---|
| `train.py:1020` always builds `WikiMiniModel` | the full run can't use HybridLM | S3 |
| hybrid yaml has no `data:` block (tokenizer path, dataset) and few training keys | `train.py:1013` reads `config['data']['tokenizer_path']` | S3 |
| `scripts/compare_hybrid.py` has no CLI args (steps/size hard-coded) | need different budgets on CUDA | S2 |
| segsum scan memory grows with seq² (B·H·T·T) | seq 512, batch 64 must fit 32 GB | S1 memory probe |
| ULTRA design chat (2026-10-17/18) never read | KR's own ideas may change the pattern | S0 |

## 3. Steps on ULTRA (each has a pass check; stop and fix on FAIL)

| Step | Do | Pass check (proof to paste) |
|---|---|---|
| S0 | Read the ULTRA design chat (recent sessions, grep "nvidia\|llm design"); `git fetch && git switch orch/20261018-025111-01b6` in the ULTRA checkout; `python -m venv .venv`; install CUDA torch + datasets tokenizers pyyaml pytest fastapi httpx | `python -c "import torch;print(torch.cuda.get_device_name(0))"` → RTX 5090; any design change from the chat written into this file |
| S1 | `pytest tests -q`; then a forward+backward of the hybrid yaml at batch 64, seq 512, bf16 | 13 failed / 101 passed (same 13 server tests); `torch.cuda.max_memory_allocated()` < 28 GB. If over: chunked SSD scan (chunk 64) + equality test vs sequential |
| S2 | add `--steps --d-model --device` args to `compare_hybrid.py`; run 3 rows (dense / hybrid MoE / hybrid dense-FFN), 1-3M params, seq 128, ≥ 1,000 steps, same seed/data/tokens | init CE ≈ ln(vocab) for all 3; held-out CE well below ln(vocab); samples are real sentences; table + reading in `results/hybrid_vs_dense.md` |
| S3 | `train.py`: `model.type: dense\|hybrid` switch (default dense, so nothing changes for old configs); copy the `data:`/`training:` blocks from the 33M yaml into the hybrid yaml; log `main_loss/mtp_loss/aux_loss` separately | old dense config still starts and logs the same; `python train.py --config config/train_config_tinystories_hybrid.yaml` runs 200 steps with falling loss |
| S4 | full runs, same tokenizer (10K), same token budget: dense 33M yaml and hybrid yaml; hidden, logged to file | both finish; val loss curves saved; tokens/s and GPU memory logged |
| S5 | `evaluate_model_enhanced.py` on both checkpoints: val perplexity (dense README target 15.7), article test 30/30, 5 samples each | results table with PPL, tokens/s, memory, params total/active, samples; plain reading: who wins and why, no spin |
| S6 | push, PR → `main`, independent review (code-reviewer), fix blocking findings, squash-merge; delete branch + Mac/ULTRA worktrees; update handoff memory | PR state MERGED, `git status --porcelain` empty on every checkout, `git ls-remote --heads origin orch/*` empty |

## 4. What "good" looks like (decide up front so the result can't be spun)

| Outcome at S5 | Meaning | Next |
|---|---|---|
| hybrid PPL ≤ dense PPL and faster per token | design wins at this size | keep, try longer context (1K-2K) where Mamba should pull ahead |
| hybrid PPL within +5% of dense | tie at tiny scale (expected) | keep, test long context + MTP speculative decoding speed |
| hybrid PPL > +5% worse | MoE or Mamba hurts at this size | compare with the dense-FFN hybrid row from S2 to see which part hurts; ship whichever row wins |

Expectation, stated before the run: dense likely wins or ties at ~25M params. The NVIDIA
gains (long context, MoE capacity, MTP drafting) show at bigger sizes and longer sequences.

## 5. Time on the 5090 (estimates, check against logs)

| Step | Estimate | Basis |
|---|---|---|
| S0-S1 | 30 min | install + test |
| S2 | 30-45 min | 3 small runs |
| S3 | 1 h | code + 200-step smoke |
| S4 | dense ~6 h (README); hybrid 1.5-3× that with the pure-PyTorch scan | no fused Mamba kernels on Windows |
| S5-S6 | 1 h | eval + PR |

Sources: Nemotron-H https://arxiv.org/abs/2504.03624 · Nemotron 3 https://arxiv.org/abs/2512.20856 ·
DeepSeek-V3 https://arxiv.org/abs/2412.19437 · Mamba-2 https://arxiv.org/abs/2405.21060 ·
reference code https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
