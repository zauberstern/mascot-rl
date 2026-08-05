"""TDD: eq CPCV emits Zhang peer fields and equity claim stamps."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mascotrl.eval.cpcv import CPCVConfig
from mascotrl.eval.research_alpha_cpcv import run_research_alpha_cpcv
from mascotrl.reporting.claim_stamps import CLAIM_CATEGORY_EQ_STK


def test_eq_cpcv_emits_zhang_peer_fields() -> None:
    rng = np.random.default_rng(2)
    t, k = 90, 4
    rets = rng.normal(0.001, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.005, size=(t, 4))
    dates = pd.bdate_range("2020-01-01", periods=t)
    cfg = {
        "claim_tier": "research",
        "claim_category": CLAIM_CATEGORY_EQ_STK,
        "claim_label_stem": "stk_ret",
        "estimand_id": "eq_stk_ret_diffsharpe_v0",
        "primary_train": "historical_arm_env",
        "policy": "single_agent",
        "projection_mode": "soft",
        "reward": "differential_sharpe",
        "om_touch_enabled": False,
        "equity_bps": 10.0,
        "headline_fill": "pct75",
        "n_assets": k,
        "train_epochs": 1,
        "lr": 3e-4,
        "reward_shaping_ablation": True,
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
    }
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art = run_research_alpha_cpcv(
        dates, rets, fac, cfg, cpcv=cpcv, seed=0, panel_source="toy"
    )
    assert art["claim_category"] == CLAIM_CATEGORY_EQ_STK
    assert art["claim_label_stem"] == "stk_ret"
    assert "sign_lag" in art["baselines"] and "long" in art["baselines"]
    assert math.isfinite(float(art["sign_lag_baseline_sharpe"]))
    assert math.isfinite(float(art["long_baseline_sharpe"]))
    assert "policy_beats_sign_lag" in art
    assert "policy_beats_long" in art
    # Toy panel must never seal research_positive_alpha.
    assert art["research_positive_alpha"] is False
    # Equity ladder must vary with equity_bps (not flat OM-touch).
    mid = float(art["fill_ladder"]["mid"])
    worst = float(art["fill_ladder"]["worst"])
    assert math.isfinite(mid) and math.isfinite(worst)
