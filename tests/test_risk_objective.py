"""Tests for Rockafellar-Uryasev CVaR and entropic risk objectives."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import torch

from src.policy.risk_objective import RiskObjective


def test_coef_zero_is_zero_loss():
    g = torch.Generator().manual_seed(0)
    episode_returns = torch.randn(32, generator=g)
    for mode in ("cvar", "entropic"):
        obj = RiskObjective(mode=mode, coef=0.0)
        loss = obj.loss(episode_returns)
        assert loss.shape == ()
        assert float(loss) == pytest.approx(0.0, **FLOAT_TOL)


def test_mode_none_is_zero():
    g = torch.Generator().manual_seed(1)
    episode_returns = torch.randn(16, generator=g)
    obj = RiskObjective(mode="none", coef=1.0)
    loss = obj.loss(episode_returns)
    assert loss.shape == ()
    assert float(loss) == pytest.approx(0.0, **FLOAT_TOL)


def test_cvar_zeta_converges_toward_quantile():
    """SGD on zeta alone should approach the empirical alpha-quantile of -G."""
    g = torch.Generator().manual_seed(42)
    # Synthetic losses as negative returns: L = -G.
    losses = torch.randn(800, generator=g)
    episode_returns = -losses
    alpha = 0.95
    obj = RiskObjective(mode="cvar", alpha=alpha, coef=1.0, zeta_lr=5e-2)
    opt = torch.optim.SGD([obj.zeta], lr=5e-2)
    for _ in range(3000):
        opt.zero_grad()
        loss = obj.loss(episode_returns)
        loss.backward()
        opt.step()

    target = float(torch.quantile(-episode_returns, alpha))
    assert abs(obj.zeta_value - target) < 0.05


def test_entropic_matches_closed_form():
    g = torch.Generator().manual_seed(7)
    episode_returns = torch.randn(64, generator=g)
    lam = 2.5
    obj = RiskObjective(mode="entropic", lam=lam, coef=1.0)
    got = obj.loss(episode_returns)

    # Overflow-safe Buehler form: (1/λ) log E[e^{-λ G}]
    x = -lam * episode_returns
    x_max = x.max()
    expected = (1.0 / lam) * (
        torch.log(torch.mean(torch.exp(x - x_max))) + x_max
    )
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_cvar_decreases_when_returns_improve():
    obj = RiskObjective(mode="cvar", alpha=0.95, coef=1.0)
    with torch.no_grad():
        obj.zeta.fill_(0.1)
    g_bad = torch.tensor([-0.5, -0.3, -0.1, 0.0, 0.05])
    g_good = g_bad + 0.25
    loss_bad = obj.loss(g_bad).detach()
    loss_good = obj.loss(g_good).detach()
    assert float(loss_good) <= float(loss_bad)
