"""Objective episode-weight conformance vs paper formulas + autograd."""
from __future__ import annotations

import math

import pytest
import torch

from src.policy.objective_factory import (
    episode_weights,
    mikkila_asym_reward,
    sdr_composite_reward,
)


def test_mean_std_cao_weights_match_cao_eq7():
    g = torch.tensor([-0.2, -0.05, 0.1, 0.25], dtype=torch.float32)
    c = 1.5
    gbar = g.mean()
    sigma = g.std(unbiased=False).clamp_min(1e-8)
    expected = (-1.0 - c * gbar / sigma) * g + (c / (2.0 * sigma)) * g.pow(2)
    assert torch.allclose(episode_weights("mean_std_cao", g, cao_c=c), expected)


def test_meanvar_kolm_weights():
    g = torch.tensor([-0.3, 0.0, 0.2], dtype=torch.float32)
    k = 2.0
    gbar = g.mean()
    expected = (-1.0 - k * gbar) * g + (k / 2.0) * g.pow(2)
    assert torch.allclose(episode_weights("meanvar_kolm", g, kappa=k), expected)


def test_cvar_ru_rockafellar_uryasev_tail_weights():
    g = torch.tensor([0.5, 0.2, -0.1, -0.4, -0.8], dtype=torch.float32)
    alpha = 0.6
    n = g.numel()
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    losses = -g
    topk = torch.topk(losses, k=k, largest=True).indices
    w = episode_weights("cvar_ru", g, alpha=alpha)
    assert (w[topk] > 0).all()
    assert int((w > 0).sum()) == k
    assert torch.allclose(w[topk], torch.full((k,), 1.0 / ((1.0 - alpha) * n)))


def test_entropic_oce_buehler_weights():
    g = torch.tensor([-0.2, 0.0, 0.1, 0.3], dtype=torch.float32)
    lam = 2.0
    n = g.numel()
    log_mean_exp = torch.logsumexp(-lam * g, dim=0) - math.log(n)
    expected = torch.exp(-lam * g - math.log(lam) - log_mean_exp)
    assert torch.allclose(episode_weights("entropic_oce", g, lam=lam), expected)


def test_smse_and_rsqp_neagu():
    g = torch.tensor([-0.5, -0.1, 0.2], dtype=torch.float32)
    smse = episode_weights("smse", g)
    assert torch.allclose(smse, torch.relu(-g).pow(2))
    rsqp_w = episode_weights("rsqp", g)
    pos2 = torch.relu(-g).pow(2)
    rho = torch.sqrt(pos2.mean().clamp_min(1e-12))
    assert torch.allclose(rsqp_w, pos2 / (2.0 * rho))


def test_mikkila_asym_and_sdr_dense():
    assert mikkila_asym_reward(0.1, xi=2.0) == pytest.approx(0.1 - 2.0 * 0.1)
    assert mikkila_asym_reward(-0.1, xi=2.0) == pytest.approx(-0.1 - 2.0 * 0.1)
    r = sdr_composite_reward(0.01, bench_pnl=0.0, beta=1.0)
    assert math.isfinite(r)


@pytest.mark.parametrize(
    "mode",
    ["mean_std_cao", "meanvar_kolm", "cvar_ru", "entropic_oce", "smse", "rsqp"],
)
def test_episode_weights_finite_and_autograd_path(mode: str):
    g = torch.tensor([-0.4, -0.1, 0.05, 0.2], dtype=torch.float32, requires_grad=True)
    # Weights themselves may detach internals; surrogate loss still needs finite w
    w = episode_weights(mode, g.detach())
    assert torch.isfinite(w).all()
    # Autograd through a trivial policy score
    logits = torch.randn(4, requires_grad=True)
    loss = (w.detach() * logits).mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
