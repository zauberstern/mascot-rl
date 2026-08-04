"""RC6 reward-to-noise diagnostic: concentrated vs EW gap vs reward std."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL

from src.eval.reward_noise import concentrated_vs_ew_gap, reward_to_noise_diagnostic


def test_concentrated_vs_ew_gap_zero_turnover():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=(50, 10))
    assert concentrated_vs_ew_gap(rets, turnover_limit=0.0) == pytest.approx(0.0, **FLOAT_TOL)


def test_concentrated_vs_ew_gap_scales_with_spread_and_tau():
    # Constant cross-section: best=0.02, worst=-0.02 every day → gap = (tau/2)*0.04
    rets = np.zeros((20, 5), dtype=np.float64)
    rets[:, 0] = 0.02
    rets[:, -1] = -0.02
    tau = 0.10
    gap = concentrated_vs_ew_gap(rets, turnover_limit=tau)
    assert gap == pytest.approx((tau / 2.0) * 0.04)


def test_reward_to_noise_warns_when_gap_below_one_std():
    rets = np.zeros((30, 4), dtype=np.float64)
    rets[:, 0] = 0.001
    rets[:, -1] = -0.001
    # Tiny tilt edge under tau=0.05: gap = 0.025 * 0.002 = 5e-5
    # Large reward noise
    daily = np.full(30, 0.0)
    daily[::2] = 0.01
    daily[1::2] = -0.01
    out = reward_to_noise_diagnostic(rets, daily, turnover_limit=0.05)
    assert out["reward_std"] > 0.0
    assert out["reward_concentrated_vs_ew_gap"] < out["reward_std"]
    assert out["reward_unlearnable"] is True
    assert "reward_noise_warning" in out
    assert out["reward_signal_to_noise"] < 1.0


def test_reward_to_noise_ok_when_gap_dominates_noise():
    rets = np.zeros((40, 5), dtype=np.float64)
    rets[:, 0] = 0.05
    rets[:, -1] = -0.05
    daily = np.full(40, 0.0001)  # near-zero noise
    out = reward_to_noise_diagnostic(rets, daily, turnover_limit=0.2)
    assert out["reward_unlearnable"] is False
    assert out["reward_signal_to_noise"] >= 1.0
    assert "reward_noise_warning" not in out


def test_reward_to_noise_aligns_when_panel_matches_reward_length():
    """Visited-step window: gap uses same T as rewards (not a longer panel)."""
    full = np.zeros((100, 4), dtype=np.float64)
    full[:, 0] = 0.05
    full[:, -1] = -0.05
    # Only first 20 days visited; rewards length 20.
    visited = full[:20]
    daily = np.full(20, 0.0001)
    out_full = reward_to_noise_diagnostic(full, daily, turnover_limit=0.2)
    out_vis = reward_to_noise_diagnostic(visited, daily, turnover_limit=0.2)
    # Gap is mean of per-day edges; constant cross-section → same gap.
    assert out_vis["reward_concentrated_vs_ew_gap"] == pytest.approx(
        out_full["reward_concentrated_vs_ew_gap"]
    )
    assert len(daily) == visited.shape[0]
