"""RED: equal_weight_tau_matched must obey the same hard L1 turnover cap."""
from __future__ import annotations

import numpy as np


def test_equal_weight_tau_matched_respects_turnover_cap() -> None:
    from mascotrl.eval.parity_harness import score_benchmark_panel
    from mascotrl.eval.friction import FrictionSpec

    rng = np.random.default_rng(0)
    t, k = 60, 4
    rets = rng.normal(0.0, 0.01, size=(t, k))
    fac = np.column_stack([rets.mean(axis=1), np.zeros((t, 3))])
    # Monthly-ish mask so EW target jumps infrequently but still projects.
    mask = np.zeros(t, dtype=bool)
    mask[::5] = True
    scored = score_benchmark_panel(
        ["equal_weight", "equal_weight_tau_matched"],
        rets,
        factors=fac,
        friction=FrictionSpec(equity_bps=5.0),
        rebalance_mask=mask,
        cadence="monthly",
        turnover_cap=0.15,
    )
    assert "equal_weight_tau_matched" in scored
    w = np.asarray(scored["equal_weight_tau_matched"]["weights"], dtype=np.float64)
    assert w.ndim == 2 and w.shape[1] == k
    # Consecutive L1 turnover after first row should never exceed tau (+eps).
    turns = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    assert float(np.nanmax(turns)) <= 0.15 + 1e-6
