import torch
import subprocess
import sys
from pathlib import Path
import yaml

from src.model.hybrid import HybridLM, Mamba2Mixer, MoESwiGLU
from src.model.hybrid.mamba2 import _segsum_transition, segsum_scan, sequential_scan
from src.model.transformer_block import WikiMiniModel


def test_enhanced_evaluator_uses_checkpoint_model_type(monkeypatch):
    import evaluate_model_enhanced as evaluator

    class FakeTokenizer:
        vocab_size = 31

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    monkeypatch.setattr(evaluator, "HybridLM", FakeModel)
    monkeypatch.setattr(evaluator, "WikiMiniModel", FakeModel)
    monkeypatch.setattr(evaluator, "load_tokenizer", lambda path: FakeTokenizer())
    monkeypatch.setattr(evaluator.torch, "load", lambda *args, **kwargs: {
        "config": {"model": {"type": "hybrid"}, "data": {}},
        "model_state_dict": {},
    })
    model, _, _ = evaluator.load_model_and_tokenizer("checkpoint.pth", "cpu")
    assert model.config["type"] == "hybrid"


def _config():
    return dict(vocab_size=31, d_model=16, n_layers=2, n_heads=2, mamba_heads=2,
                d_ffn=24, expert_ffn=24, dense_ffn=72, n_experts=2, top_k=1, layer_pattern="MF",
                max_seq_len=12, dropout=0.0, use_flash_attention=False)


def test_train_configs_select_expected_model():
    root = Path(__file__).parents[1]
    dense = yaml.safe_load((root / 'config/train_config_tinystories_33M_TOP10K.yaml').read_text())
    hybrid = yaml.safe_load((root / 'config/train_config_tinystories_hybrid.yaml').read_text())
    assert isinstance(WikiMiniModel(dense['model']), WikiMiniModel)
    assert isinstance(HybridLM(hybrid['model']), HybridLM)


def test_compare_hybrid_help():
    result = subprocess.run([sys.executable, 'scripts/compare_hybrid.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    for flag in ('--steps', '--d-model', '--device', '--batch-size', '--seq-len', '--out', '--seed'):
        assert flag in result.stdout


def test_mamba_shape_and_causality():
    torch.manual_seed(1)
    mixer = Mamba2Mixer(16, n_heads=2).eval()
    x = torch.randn(2, 8, 16)
    changed = x.clone()
    changed[:, 5:] += 10
    assert mixer(x).shape == x.shape
    assert torch.allclose(mixer(x)[:, :5], mixer(changed)[:, :5], atol=1e-6)


def test_sequential_scan_equals_segsum_scan():
    torch.manual_seed(7)
    batch, length, heads, head_dim, state_size = 2, 7, 2, 3, 4
    x = torch.randn(batch, length, heads, head_dim)
    b, c = torch.randn(batch, length, heads, state_size), torch.randn(batch, length, heads, state_size)
    dt = torch.rand(batch, length, heads) * 0.1
    a_vec = -torch.rand(heads) * 4 - 1
    a, d = a_vec.view(1, heads, 1, 1), torch.ones(1, heads, 1)
    expected = sequential_scan(x, b, c, dt, a, d)
    actual = segsum_scan(x, b, c, dt, a_vec.view(1, 1, heads), d.view(1, 1, heads, 1))
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_sequential_scan_equals_segsum_at_512():
    torch.manual_seed(71)
    bsz, length, heads, width, states = 1, 512, 2, 3, 4
    x = torch.randn(bsz, length, heads, width)
    b = torch.randn(bsz, length, heads, states)
    c = torch.randn_like(b)
    dt = torch.rand(bsz, length, heads) * 0.01
    a_vec = -torch.rand(heads) * 2 - 1
    d = torch.ones(1, heads, 1)
    expected = sequential_scan(x, b, c, dt, a_vec.view(1, heads, 1, 1), d)
    actual = segsum_scan(x, b, c, dt, a_vec.view(1, 1, heads), d.view(1, 1, heads, 1))
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_segsum_transition_has_no_state_or_channel_expansion():
    bsz, length, heads, width, states = 2, 256, 4, 64, 16
    b = torch.randn(bsz, length, heads, states)
    c = torch.randn_like(b)
    dt = torch.rand(bsz, length, heads)
    decay = torch.ones(bsz, length, length, heads)
    with torch.no_grad():
        transition = _segsum_transition(b, c, dt, decay)
        x = torch.randn(bsz, length, heads, width)
        out = torch.einsum("btsh,bshp->bthp", transition, x)
        scan_out = segsum_scan(x, b, c, dt, -torch.ones(1, 1, heads), torch.ones(1, 1, heads, 1))
    assert transition.shape == (bsz, length, length, heads)
    assert out.shape == (bsz, length, heads, width)
    assert scan_out.shape == x.shape


def test_full_model_attention_pattern_is_causal():
    cfg = _config()
    cfg.update(layer_pattern="MFAFMF", max_seq_len=16)
    model = HybridLM(cfg).eval()
    ids = torch.randint(0, cfg["vocab_size"], (2, 14))
    changed = ids.clone()
    changed[:, 8:] = torch.randint(0, cfg["vocab_size"], changed[:, 8:].shape)
    with torch.no_grad():
        first, second = model(ids)["logits"], model(changed)["logits"]
    assert torch.allclose(first[:, :8], second[:, :8], atol=1e-6)


def test_mtp_logits_use_future_input_from_t_plus_one():
    torch.manual_seed(43)
    model = HybridLM(_config()).eval()
    ids = torch.randint(0, model.vocab_size, (2, 20))
    changed = ids.clone()
    changed[:, 12:] = torch.randint(0, model.vocab_size, changed[:, 12:].shape)
    with torch.no_grad():
        first = model(ids, ids)["mtp_logits"]
        second = model(changed, changed)["mtp_logits"]
    assert torch.allclose(first[:, :10], second[:, :10], atol=1e-6)
    assert not torch.allclose(first[:, 11:], second[:, 11:], atol=1e-6)


def test_expert_bias_buffer_balances_only_training_with_grad():
    moe = MoESwiGLU(16, 24, n_experts=3, top_k=1)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1
    x = torch.zeros(2, 4, 16, requires_grad=True)
    before = moe.expert_bias.clone()
    moe.train()
    moe(x)
    assert moe.expert_bias[1] > before[1] and moe.expert_bias[2] > before[2]
    after_train = moe.expert_bias.clone()
    with torch.no_grad():
        moe.train()
        moe(x)
    assert torch.equal(moe.expert_bias, after_train)
    moe.eval()
    moe(x)
    assert torch.equal(moe.expert_bias, after_train)
    assert "expert_bias" in dict(moe.named_buffers())
    assert "expert_bias" not in dict(moe.named_parameters())


def test_initial_main_ce_near_uniform():
    torch.manual_seed(21)
    model = HybridLM(_config())
    ids = torch.randint(0, model.vocab_size, (4, 16))
    labels = torch.randint(0, model.vocab_size, ids.shape)
    ce = model(ids, labels)["main_loss"]
    assert abs(float(ce.detach()) - torch.log(torch.tensor(float(model.vocab_size))).item()) < 0.5


def test_initial_main_ce_near_uniform_at_4000_vocab():
    torch.manual_seed(17)
    cfg = _config()
    cfg.update(vocab_size=4000, d_model=32, n_heads=4, mamba_heads=4, d_ffn=48,
               expert_ffn=16, dense_ffn=48, state_size=4, layer_pattern="MF")
    model = HybridLM(cfg)
    ids = torch.randint(0, 4000, (2, 32))
    labels = torch.randint(0, 4000, ids.shape)
    ce = model(ids, labels)["main_loss"]
    assert abs(float(ce.detach()) - torch.log(torch.tensor(4000.0)).item()) < 0.5


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
    assert model.mtp_projection.weight.grad is not None


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
