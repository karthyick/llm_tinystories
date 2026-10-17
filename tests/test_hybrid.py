import torch

from src.model.hybrid import HybridLM, Mamba2Mixer, MoESwiGLU


def _config():
    return dict(vocab_size=31, d_model=16, n_layers=2, n_heads=2, mamba_heads=2,
                d_ffn=24, n_experts=2, top_k=1, layer_pattern="MA",
                max_seq_len=12, dropout=0.0, use_flash_attention=False)


def test_mamba_shape_and_causality():
    torch.manual_seed(1)
    mixer = Mamba2Mixer(16, n_heads=2).eval()
    x = torch.randn(2, 8, 16)
    changed = x.clone()
    changed[:, 5:] += 10
    assert mixer(x).shape == x.shape
    assert torch.allclose(mixer(x)[:, :5], mixer(changed)[:, :5], atol=1e-6)


def test_moe_top_k_shared_expert_and_aux_loss():
    moe = MoESwiGLU(16, 24, n_experts=3, top_k=2)
    x = torch.randn(2, 4, 16)
    out, aux = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(aux)
    (out.sum() + aux).backward()
    assert moe.shared_expert.w_gate.weight.grad is not None
    assert moe.shared_expert.w_gate.weight.grad.abs().sum() > 0
    assert torch.all(moe.router(x).topk(2, dim=-1).indices < 3)


def test_full_model_shape_causality_and_mtp_backward():
    torch.manual_seed(2)
    model = HybridLM(_config()).eval()
    ids = torch.randint(0, 31, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = torch.randint(0, 31, changed[:, 5:].shape)
    with torch.no_grad():
        logits = model(ids)["logits"]
        changed_logits = model(changed)["logits"]
    assert logits.shape == (2, 8, 31)
    assert torch.allclose(logits[:, :5], changed_logits[:, :5], atol=1e-6)
    model.train()
    labels = torch.randint(0, 31, ids.shape)
    out = model(ids, labels)
    assert all(torch.isfinite(out[k]) for k in ("loss", "main_loss", "mtp_loss", "aux_loss"))
    expected_main = torch.nn.functional.cross_entropy(
        out["logits"].reshape(-1, 31), labels.reshape(-1)
    )
    assert torch.allclose(out["main_loss"], expected_main)
    out["loss"].backward()
    assert model.token_embedding.weight.grad is not None
    assert model.mtp_block[0].weight.grad is not None


def test_tiny_batch_overfit_decreases_loss():
    torch.manual_seed(3)
    model = HybridLM(_config())
    ids = torch.randint(0, 31, (1, 6))
    labels = torch.randint(0, 31, (1, 6))
    opt = torch.optim.AdamW(model.parameters(), lr=0.015)
    with torch.no_grad():
        initial = model(ids, labels)["loss"].item()
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = model(ids, labels)["loss"]
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = model(ids, labels)["loss"].item()
    assert final < initial
