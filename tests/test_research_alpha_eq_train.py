"""TDD: eq stk_ret hist train with differential Sharpe + equity_bps."""
from __future__ import annotations

import numpy as np

from src.eval.research_alpha_train import train_research_hist


def test_eq_diffsharpe_train_smoke() -> None:
    rng = np.random.default_rng(0)
    k = 4
    rets = rng.normal(0.0004, 0.01, size=(50, k))
    fac = rng.normal(0.0, 0.005, size=(50, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "policy": "single_agent",
        "projection_mode": "soft",
        "reward": "differential_sharpe",
        "estimand_id": "eq_stk_ret_diffsharpe_v0",
        "n_assets": k,
        "equity_bps": 10.0,
        "om_touch_enabled": False,
        "reward_shaping_ablation": True,
        "train_epochs": 1,
        "lr": 3e-4,
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out["reward"] == "differential_sharpe"
    assert out["estimand_id"] == "eq_stk_ret_diffsharpe_v0"
    assert out["friction_applied"] is True
    assert out["n_steps"] > 0
    assert np.isfinite(out["mean_reward"])
