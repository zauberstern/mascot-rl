"""CPCV stamps monthly rebalance mask from dates (Phase 2 fail-closed regression)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_cpcv_stamps_rebalance_mask_when_monthly_and_missing():
    """Mirror the preamble in run_research_alpha_cpcv without full CPCV."""
    from mascotrl.eval.cadence import build_rebalance_mask
    from mascotrl.eval.yaml_honesty import track_copy

    dates = list(pd.bdate_range("2020-01-01", periods=40))
    cfg = track_copy({"rebalance_cadence": "monthly"})
    cadence = str(cfg.get("rebalance_cadence") or "daily").lower()
    assert cadence not in ("", "daily")
    assert cfg.get("_rebalance_mask") is None
    cfg["_rebalance_mask"] = build_rebalance_mask(dates, cadence)
    m = np.asarray(cfg["_rebalance_mask"], dtype=bool)
    assert m.shape == (40,)
    assert m.any()
    # Fold slice must preserve length of the train window (mask may be all-False
    # on a short window that never hits a month boundary; length is the contract).
    from mascotrl.eval.research_alpha_cpcv import _slice_feature_extras

    train_idx = np.arange(0, 22)
    sliced = _slice_feature_extras(cfg, train_idx)
    sm = np.asarray(sliced["_rebalance_mask"], dtype=bool)
    assert sm.shape == (22,)
    # Full-panel monthly mask must still have true days outside the short slice.
    assert m.sum() >= 1
