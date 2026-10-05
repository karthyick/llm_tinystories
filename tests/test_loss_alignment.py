"""Regression test: the loss must not shift labels a second time.

Datasets emit input_ids = seq[:-1] and labels = seq[1:] (pre-shifted), so the
model has to score logits[t] against labels[t] with no further shift.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer_block import WikiMiniModel

VOCAB = 50


def _tiny_model() -> WikiMiniModel:
    torch.manual_seed(0)
    cfg = dict(
        vocab_size=VOCAB, d_model=32, n_layers=1, n_heads=2, d_ffn=64,
        max_seq_len=16, dropout=0.0, use_flash_attention=False,
    )
    return WikiMiniModel(cfg).eval()


def test_position_t_scored_against_raw_token_t_plus_1():
    model = _tiny_model()
    seq = torch.randint(1, VOCAB, (2, 9))
    input_ids, labels = seq[:, :-1], seq[:, 1:].clone()  # dataset convention

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    expected = F.cross_entropy(
        out["logits"].reshape(-1, VOCAB), seq[:, 1:].reshape(-1)
    )
    assert torch.allclose(out["loss"], expected, atol=1e-6)

    # Under the old double shift the loss equals the token-after-next score.
    double_shift = F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, VOCAB), seq[:, 2:].reshape(-1)
    )
    assert not torch.allclose(out["loss"], double_shift, atol=1e-4)


def test_changing_only_label_t_changes_only_position_t_loss():
    model = _tiny_model()
    seq = torch.randint(1, VOCAB, (1, 9))
    input_ids, labels = seq[:, :-1], seq[:, 1:].clone()
    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"][0]
    t = 3
    per_pos = F.cross_entropy(logits, labels[0], reduction="none")
    labels2 = labels.clone()
    labels2[0, t] = (labels2[0, t] + 1) % VOCAB
    per_pos2 = F.cross_entropy(logits, labels2[0], reduction="none")
    changed = (per_pos - per_pos2).abs() > 1e-7
    assert changed.nonzero().flatten().tolist() == [t]
    with torch.no_grad():
        l1 = model(input_ids=input_ids, labels=labels)["loss"]
        l2 = model(input_ids=input_ids, labels=labels2)["loss"]
    assert torch.allclose(l1 - l2, (per_pos.mean() - per_pos2.mean()), atol=1e-6)


def test_ignore_index_pad_positions_excluded():
    model = _tiny_model()
    seq = torch.randint(1, VOCAB, (2, 9))
    input_ids, labels = seq[:, :-1], seq[:, 1:].clone()
    labels[:, 5:] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    expected = F.cross_entropy(
        out["logits"][:, :5].reshape(-1, VOCAB), seq[:, 1:6].reshape(-1)
    )
    assert torch.allclose(out["loss"], expected, atol=1e-6)
