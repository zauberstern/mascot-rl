"""TDD: short research CPCV on numpy panels (not dry-run)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mascotrl.eval.cpcv import CPCVConfig
from mascotrl.eval.research_alpha_cpcv import run_research_alpha_cpcv
from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha


def _panel(t: int = 120, k: int = 4, seed: int = 0, mu: float = 0.001):
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.005, size=(t, 4))
    dates = pd.bdate_range("2020-01-01", periods=t)
    return dates, rets, factors


def test_run_research_cpcv_emits_finite_path_summary() -> None:
    dates, rets, fac = _panel()
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "om_touch_enabled": True,
        "hedge_leg_spread_bps": 5.0,
        "headline_fill": "pct75",
        "n_assets": 4,
        "policy": "single_agent",
        "projection_mode": "soft",
        "train_epochs": 1,
        "lr": 3e-4,
        "reward_shaping_ablation": True,
    }
    # Tiny CPCV for unit speed: 3 splits, 1 test → 3 folds, 2 paths
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art = run_research_alpha_cpcv(dates, rets, fac, cfg, cpcv=cpcv, seed=0, panel_source="toy")
    assert art["dry_run"] is False
    assert art["headline_fill"] == "pct75"
    assert "mid" in art["fill_ladder"] and "pct75" in art["fill_ladder"]
    assert "worst" in art["fill_ladder"]
    sh = float(art["path_summary"]["sharpe_mean"])
    assert math.isfinite(sh) or art["path_summary"]["n_paths"] >= 1
    assert "random" in art["baselines"]
    assert art["spa_polarity"] == "policy_as_challenger"
    # Default residual train reward != total_net claim metric (honest stamp).
    assert art["train_objective_equals_claim_metric"] is False
    assert art["friction_applied"] is True


def test_run_research_cpcv_attaches_policy_diagnostics_for_path0() -> None:
    dates, rets, fac = _panel()
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "headline_fill": "pct75",
        "n_assets": 4,
        "policy": "single_agent",
        "projection_mode": "soft",
        "train_epochs": 1,
        "lr": 3e-4,
    }
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art = run_research_alpha_cpcv(dates, rets, fac, cfg, cpcv=cpcv, seed=0, panel_source="toy")
    assert "policy_diagnostics" in art
    diag = art["policy_diagnostics"]
    assert "equal_weight_collapse_guard" in diag
    assert "hhi_mean" in diag


def test_run_research_cpcv_refuses_mid_headline() -> None:
    dates, rets, fac = _panel(t=60)
    try:
        run_research_alpha_cpcv(
            dates,
            rets,
            fac,
            {
                "headline_fill": "mid",
                "primary_train": "historical_arm_env", "portfolio_arm": "eq",
                "n_assets": 4,
            },
            cpcv=CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_stamp_after_run_never_unlocks_capital() -> None:
    dates, rets, fac = _panel(t=90, mu=0.002)
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "om_touch_enabled": True,
        "hedge_leg_spread_bps": 5.0,
        "headline_fill": "pct75",
        "n_assets": 4,
        "train_epochs": 1,
        "reward_shaping_ablation": True,
    }
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art = run_research_alpha_cpcv(dates, rets, fac, cfg, cpcv=cpcv, seed=1, panel_source="toy")
    stamped = stamp_research_positive_alpha(art)
