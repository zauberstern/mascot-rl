"""Equity nested WFO stamps (W6)."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.run_eq_alloc_campaign import _wfo_enabled
from src.eval.cadence import build_rebalance_mask
from src.eval.equity_nested_wfo import _expanding_folds, run_equity_nested_wfo


def test_expanding_folds_are_causal():
    folds = _expanding_folds(200, n_folds=4, min_train=40, test_frac=0.2)
    assert len(folds) >= 2
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert train_idx.min() == 0


@pytest.mark.parametrize("cadence", ["daily", "monthly"])
def test_run_equity_nested_wfo_stamps_not_cpcv(cadence: str):
    rng = np.random.default_rng(0)
    t, k = 120, 4
    import pandas as pd

    dates = list(pd.date_range("2018-01-01", periods=t, freq="B"))
    rets = rng.normal(0.0003, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "headline_fill": "pct75",
        "primary_train": "historical_arm_env",
        "projection_mode": "soft",
        "train_episodes": 1,
        "train_epochs": 1,
        "train_env_steps": 40,
        "n_minibatches": 1,
        "ppo_hidden": 16,
        "equity_bps": 5.0,
        "impact_c_eq": 0.5,
        "rebalance_cadence": cadence,
        "architecture": "mlp",
        "algo": "ppo",
            "objective": "differential_sharpe",
            "reward": "differential_sharpe",
            "train_world": "historical",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
    }
    if cadence == "monthly":
        cfg["_rebalance_mask"] = build_rebalance_mask(dates, "monthly")
        cfg["_slot_valid_mask"] = np.ones((t, k), dtype=bool)
    art = run_equity_nested_wfo(dates, rets, fac, cfg, seed=0, n_folds=2)
    assert art["is_cpcv"] is False
    assert art["protocol"] == "expanding_window_wfo"
    assert art["n_folds"] >= 1


def test_wfo_is_on_by_default_and_can_be_disabled():
    assert _wfo_enabled(argparse.Namespace(no_wfo=False)) is True
    assert _wfo_enabled(argparse.Namespace(no_wfo=True)) is False
