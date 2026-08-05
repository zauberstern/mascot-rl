"""Hard turnover cap binding diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.eval.cpcv import CPCVConfig
from mascotrl.eval.research_alpha_cpcv import run_research_alpha_cpcv
from mascotrl.eval.research_alpha_train import _turnover_cap_project


def test_turnover_projection_counts_binding_steps() -> None:
    counter = {"steps": 0, "binding_steps": 0}
    prev = np.asarray([0.5, 0.5])

    _turnover_cap_project(
        np.asarray([0.55, 0.45]), w_prev=prev, tau=0.2, counter=counter
    )
    projected = _turnover_cap_project(
        np.asarray([1.0, 0.0]), w_prev=prev, tau=0.2, counter=counter
    )

    assert counter == {"steps": 2, "binding_steps": 1}
    assert np.abs(projected - prev).sum() == pytest.approx(0.2)


def test_cpcv_stamps_binding_fraction_and_turnover_mean() -> None:
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2020-01-01", periods=90)
    returns = rng.normal(0.001, 0.01, size=(90, 4))
    factors = rng.normal(0.0, 0.005, size=(90, 4))
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "headline_fill": "pct75",
        "n_assets": 4,
        "policy": "single_agent",
        "projection_mode": "hard",
        "turnover_limit": 0.15,
        "train_epochs": 1,
        "lr": 3e-4,
    }

    art = run_research_alpha_cpcv(
        dates,
        returns,
        factors,
        cfg,
        cpcv=CPCVConfig(
            n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0
        ),
        seed=0,
        panel_source="toy",
    )

    assert 0.0 <= art["turnover_cap_binding_fraction"] <= 1.0
    assert np.isfinite(art["turnover_mean"])
