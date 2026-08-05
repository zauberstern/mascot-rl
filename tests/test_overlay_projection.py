"""Tests for OverlayProjectionLayer delta modes."""
from __future__ import annotations

import torch

from mascotrl.policy.convex_projection import ConvexProjectionLayer
from mascotrl.policy.overlay_projection import OverlayProjectionLayer


def _rand_batch(k: int, B: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w_raw = torch.randn(B, k, generator=g)
    w_prev = torch.randn(B, k, generator=g) * 0.1
    deltas = torch.randn(B, k, generator=g)
    return w_raw, w_prev, deltas


def test_soft_matches_convex_projection_layer():
    k = 8
    soft = OverlayProjectionLayer(k, delta_mode="soft", turnover_limit=0.2)
    base = ConvexProjectionLayer(k, turnover_limit=0.2)
    w_raw, w_prev, deltas = _rand_batch(k)
    with torch.no_grad():
        a = soft(w_raw, w_prev, deltas)
        b = base(w_raw, w_prev, deltas)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_off_drops_delta_keeps_turnover_and_box():
    k = 6
    layer = OverlayProjectionLayer(k, delta_mode="off", turnover_limit=0.15, max_name_abs_weight=1.0)
    w_prev = torch.zeros(1, k)
    w_raw = torch.ones(1, k) * 3.0  # would violate box and turnover
    deltas = torch.ones(1, k)
    out = layer(w_raw, w_prev, deltas)
    assert out.abs().max().item() <= 1.0 + 1e-5
    assert (out - w_prev).abs().sum().item() <= 0.15 + 0.05  # slack room


def test_option_block_zeros_equity_delta_contribution():
    k = 6
    opt = 3
    layer = OverlayProjectionLayer(
        k, delta_mode="option_block", option_slots=opt, turnover_limit=2.0
    )
    w_prev = torch.zeros(1, k)
    # Push only equity weights; option deltas huge — if equity entered the
    # constraint, projection would fight hard. With option_block, equity Δ=0.
    w_raw = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]], dtype=torch.float32)
    deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
    out = layer(w_raw, w_prev, deltas)
    # Equity block should stay near raw (delta constraint ignores them).
    assert torch.allclose(out[0, opt:].float(), w_raw[0, opt:], atol=1e-4)


def test_joint_satisfies_joint_delta_bound_approximately():
    k = 4
    layer = OverlayProjectionLayer(k, delta_mode="joint", turnover_limit=2.0)
    w_prev = torch.zeros(1, k)
    w_raw = torch.ones(1, k)
    deltas = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    out, s_delta, _ = layer(w_raw, w_prev, deltas, return_slacks=True)
    residual = (out * deltas).sum(dim=-1).abs()
    assert (residual <= s_delta + 1e-3).all()


def test_gradients_flow_soft_and_off():
    for mode in ("soft", "off"):
        layer = OverlayProjectionLayer(4, delta_mode=mode, turnover_limit=0.5)
        w_raw = torch.randn(1, 4, requires_grad=True)
        w_prev = torch.zeros(1, 4)
        deltas = torch.ones(1, 4)
        out = layer(w_raw, w_prev, deltas)
        out.sum().backward()
        assert w_raw.grad is not None
        assert torch.isfinite(w_raw.grad).all()
