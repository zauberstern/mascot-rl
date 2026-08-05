"""Daily policy cadence must not crash lake dyn_hrp panel load (RC6).

``build_rebalance_mask(..., \"daily\")`` returns ``None`` (every-day convention).
``np.asarray(None, dtype=bool)`` has size 1, which previously blew up
``build_dynamic_universe`` with ``rebalance_mask length 1 != len(dates)``.
Universe reselection must use ``universe_cadence`` (default quarterly_63d),
independent of the policy rebalance mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_asarray_none_bool_is_length_one_trap():
    """Document the NumPy trap that caused the Batch canary crash."""
    bad = np.asarray(None, dtype=bool).reshape(-1)
    assert bad.size == 1


def test_load_lake_dyn_hrp_daily_policy_uses_universe_cadence(monkeypatch, tmp_path):
    from mascotrl.eval import equity_substrate as es
    from mascotrl.eval.cadence import build_universe_cadence_mask, quarterly_63d_mask

    n_dates = 200
    n_names = 12
    dates = pd.bdate_range("2014-01-01", periods=n_dates)
    secids = list(range(1000, 1000 + n_names))
    rets = np.random.default_rng(0).normal(0.0, 0.01, size=(n_dates, n_names))

    raw = pd.DataFrame(
        {
            "date": np.repeat(dates, n_names),
            "secid": np.tile(secids, n_dates),
            "ret": rets.reshape(-1),
        }
    )

    monkeypatch.setattr(es, "resolve_lake_root", lambda cfg: tmp_path)
    monkeypatch.setattr(
        "mascotrl.data.equity_panel.load_sp500_security_returns",
        lambda lake, start, end: raw,
    )

    def _fake_wide(raw_df, *, start, end, **kwargs):
        return rets.copy(), list(secids), list(dates), np.ones_like(rets, dtype=bool)

    monkeypatch.setattr(es, "_wide_returns_with_availability", _fake_wide)

    captured: dict = {}

    def _capture_build_dynamic_universe(**kwargs):
        captured["rebalance_mask"] = np.asarray(kwargs["rebalance_mask"], dtype=bool)
        t = len(kwargs["dates"])
        k = int(kwargs["k"])
        slots = [[secids[i % n_names] for i in range(k)] for _ in range(t)]
        valid = np.ones((t, k), dtype=bool)
        return slots, valid, []

    monkeypatch.setattr(
        "mascotrl.data.dynamic_universe.build_dynamic_universe",
        _capture_build_dynamic_universe,
    )
    monkeypatch.setattr(
        "mascotrl.data.dynamic_universe.build_slotted_panel",
        lambda **kwargs: rets[:, : kwargs.get("k", n_names)]
        if False
        else np.asarray(kwargs.get("wide_returns"))[:, :10],
    )
    # build_slotted_panel needs a real callable; stub after import path used inside fn
    import mascotrl.data.dynamic_universe as du

    monkeypatch.setattr(
        du,
        "build_slotted_panel",
        lambda **kw: np.asarray(kw["wide_returns"], dtype=np.float64)[:, :10],
    )
    monkeypatch.setattr(du, "selection_turnover", lambda *a, **k: 0.0)

    cfg = {
        "rebalance_cadence": "daily",
        "selection_start": "2003-01-01",
        "selection_end": "2012-12-31",
        "oos_start": "2014-01-01",
        "oos_end": "2024-12-31",
        "n_assets": 10,
    }
    dates_out, slotted, factors, meta = es.load_lake_dyn_hrp_panel(cfg, k=10)
    assert len(dates_out) == n_dates
    assert cfg.get("_rebalance_mask") is None  # daily policy convention
    u_mask = captured["rebalance_mask"]
    assert u_mask.shape == (n_dates,)
    assert u_mask.size == n_dates
    expected = build_universe_cadence_mask(dates, "quarterly_63d")
    np.testing.assert_array_equal(u_mask, expected)
    assert int(u_mask.sum()) == int(quarterly_63d_mask(dates).sum())
    assert slotted.shape[0] == n_dates
    assert factors.shape[0] == n_dates
    assert isinstance(meta, dict)


def test_load_lake_dyn_hrp_monthly_keeps_legacy_universe_coupling(monkeypatch, tmp_path):
    """When universe_cadence is unset, non-daily policy still drives universe days."""
    from mascotrl.eval import equity_substrate as es
    from mascotrl.eval.cadence import month_end_mask

    n_dates = 80
    n_names = 8
    dates = pd.bdate_range("2014-01-01", periods=n_dates)
    secids = list(range(2000, 2000 + n_names))
    rets = np.random.default_rng(1).normal(0.0, 0.01, size=(n_dates, n_names))
    raw = pd.DataFrame(
        {
            "date": np.repeat(dates, n_names),
            "secid": np.tile(secids, n_dates),
            "ret": rets.reshape(-1),
        }
    )
    monkeypatch.setattr(es, "resolve_lake_root", lambda cfg: tmp_path)
    monkeypatch.setattr(
        "mascotrl.data.equity_panel.load_sp500_security_returns",
        lambda lake, start, end: raw,
    )
    monkeypatch.setattr(
        es,
        "_wide_returns_with_availability",
        lambda raw_df, *, start, end, **kwargs: (
            rets.copy(),
            list(secids),
            list(dates),
            np.ones_like(rets, dtype=bool),
        ),
    )
    captured: dict = {}

    def _capture(**kwargs):
        captured["rebalance_mask"] = np.asarray(kwargs["rebalance_mask"], dtype=bool)
        t, k = len(kwargs["dates"]), int(kwargs["k"])
        slots = [[secids[i % n_names] for i in range(k)] for _ in range(t)]
        return slots, np.ones((t, k), dtype=bool), []

    import mascotrl.data.dynamic_universe as du

    monkeypatch.setattr(du, "build_dynamic_universe", _capture)
    monkeypatch.setattr(
        du,
        "build_slotted_panel",
        lambda **kw: np.asarray(kw["wide_returns"], dtype=np.float64)[
            :, : min(5, n_names)
        ],
    )
    monkeypatch.setattr(du, "selection_turnover", lambda *a, **k: 0.0)

    cfg = {"rebalance_cadence": "monthly", "n_assets": 5}
    es.load_lake_dyn_hrp_panel(cfg, k=5)
    policy = cfg["_rebalance_mask"]
    assert policy is not None
    np.testing.assert_array_equal(policy, month_end_mask(dates))
    np.testing.assert_array_equal(captured["rebalance_mask"], policy)
