"""Long baseline must use CPCV test slices and fold-fitted residualizers."""
from __future__ import annotations

import numpy as np
import pandas as pd

import mascotrl.eval.research_alpha_cpcv as module
from mascotrl.eval.cpcv import CPCVConfig, build_cpcv_folds
from mascotrl.eval.friction import FrictionSpec


def test_long_baseline_scores_concatenated_cpcv_test_indices(monkeypatch) -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=60))
    rets = np.arange(180, dtype=float).reshape(60, 3) / 10000.0
    factors = np.zeros((60, 4), dtype=float)
    config = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    folds = build_cpcv_folds(dates, config)
    calls: list[tuple[int, str]] = []

    def fake_score(returns, *, residualizer, **kwargs):
        del kwargs
        calls.append((len(returns), residualizer.fold_id))
        return {"total_net": np.asarray(returns[:, 0], dtype=float)}

    monkeypatch.setattr(module, "score_equal_weight", fake_score)
    out = module._score_long_baseline_on_cpcv_tests(
        dates=dates,
        returns=rets,
        factors=factors,
        folds=folds,
        friction=FrictionSpec(equity_bps=5.0),
        cadence="daily",
        rebalance_mask=None,
    )

    assert len(calls) == len(folds)
    assert out["n_test_rows"] == sum(size for size, _ in calls)
    assert all(fold_id.startswith("long_baseline_fold_") for _, fold_id in calls)
    assert np.isfinite(out["sharpe"])
