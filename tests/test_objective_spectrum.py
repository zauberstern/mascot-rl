"""TDD: objective spectrum + collapse guard."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.eval.collapse_guard import assert_collapse_guard_ok, collapse_guard
from mascotrl.policy.risk_objective import ALLOWED_MODES, RiskObjective
from mascotrl.spectrum.registry import allowed_ids


def test_spectrum_objective_ids_in_allowed_modes() -> None:
    for oid in allowed_ids("objective"):
        assert oid in ALLOWED_MODES or oid in {
            "differential_sharpe",
            "mtm_pnl",
            "mikkila_asym",
            "sdr_composite",
        }


@pytest.mark.parametrize(
    "mode",
    ["smse", "rsqp", "mean_std_cao", "meanvar_kolm", "cvar_ru", "entropic_oce"],
)
def test_new_modes_finite_loss(mode: str) -> None:
    g = torch.randn(32)
    obj = RiskObjective(mode=mode, coef=1.0, objective_primary=True)
    loss = obj.loss(g)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_mean_std_cao_prefers_higher_mean() -> None:
    obj = RiskObjective(mode="mean_std_cao", coef=1.0, cao_c=1.5, objective_primary=True)
    bad = torch.tensor([-0.1, -0.05, 0.0, 0.02])
    good = bad + 0.2
    assert float(obj.loss(good)) < float(obj.loss(bad))


def test_collapse_guard_detects_zero_turnover() -> None:
    rep = collapse_guard([0.0, 0.0, 0.0], action_l1=[0.0, 0.0, 0.0])
    assert rep["collapse_detected"] is True
    assert rep["ok"] is False
    with pytest.raises(ValueError, match="collapse"):
        assert_collapse_guard_ok(rep)


def test_collapse_guard_ok_with_activity() -> None:
    rng = np.random.default_rng(0)
    to = rng.uniform(0.01, 0.2, size=50)
    al = rng.uniform(0.1, 1.0, size=50)
    rep = collapse_guard(to, action_l1=al)
    assert rep["ok"] is True
    assert_collapse_guard_ok(rep)
