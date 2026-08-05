"""A9: projection_mode='hard' must actually enforce turnover_limit."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.research_alpha_train import (
    _turnover_cap_project,
    build_research_hist_env,
)


def _toy_panel(t: int = 60, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, fac


def test_turnover_cap_project_respects_budget():
    w = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    w_prev = np.zeros(5)
    out = _turnover_cap_project(w, w_prev=w_prev, tau=0.2)
    assert float(np.abs(out - w_prev).sum()) == pytest.approx(0.2, abs=1e-9)
    # Direction preserved.
    np.testing.assert_allclose(out / np.abs(out).sum(), w / np.abs(w).sum())


def test_turnover_cap_project_passthrough_under_budget():
    w = np.array([0.05, -0.05, 0.0])
    w_prev = np.zeros(3)
    out = _turnover_cap_project(w, w_prev=w_prev, tau=1.0)
    np.testing.assert_allclose(out, w)


def test_turnover_cap_project_rejects_negative_tau():
    with pytest.raises(ValueError, match="turnover_limit"):
        _turnover_cap_project(np.ones(2), w_prev=np.zeros(2), tau=-0.1)


def test_hard_projection_mode_caps_realized_turnover_in_env():
    rets, fac = _toy_panel()
    k = rets.shape[1]
    cfg = {
        "primary_train": "historical_arm_env",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
        "projection_mode": "hard",
        "turnover_limit": 0.1,
        "equity_bps": 5.0,
    }
    env = build_research_hist_env(rets, fac, cfg)
    env.reset()
    rng = np.random.default_rng(1)
    terminated = truncated = False
    max_turnover = 0.0
    while not (terminated or truncated):
        # Aggressive, maximally-changing proposed weights every step.
        raw = rng.normal(size=k)
        raw = raw / np.abs(raw).sum()
        _obs, _reward, terminated, truncated, info = env.step(raw)
        max_turnover = max(max_turnover, float(info["turnover"]))
    assert max_turnover <= 0.1 + 1e-8, f"realized turnover {max_turnover} exceeded cap"


def test_hard_projection_mode_requires_turnover_limit():
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {
        "primary_train": "historical_arm_env",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": 3, "delta_mode": "off"},
        "projection_mode": "hard",
    }
    with pytest.raises(ValueError, match="turnover_limit"):
        build_research_hist_env(rets, fac, cfg)


def test_unknown_projection_mode_rejected():
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {
        "primary_train": "historical_arm_env",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": 3, "delta_mode": "off"},
        "projection_mode": "bogus",
    }
    with pytest.raises(ValueError, match="projection_mode"):
        build_research_hist_env(rets, fac, cfg)
