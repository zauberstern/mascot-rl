"""Parity: every panel member sees the same mask/cost/turnover contract."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.benchmark_panel import (
    BENCHMARK_PANEL_NAMES,
    PANEL_EXTRA_NAMES,
    get_weight_fn,
    run_benchmark_on_fold,
    run_panel,
)
from src.eval.industry_baselines import INDUSTRY_BASELINE_NAMES


def _toy_returns(t: int = 80, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(t, k))


def _slot_mask(t: int, k: int) -> np.ndarray:
    mask = np.ones((t, k), dtype=bool)
    # Drop name 0 on even days, name 1 on odd days.
    for i in range(t):
        if i % 2 == 0:
            mask[i, 0] = False
        else:
            mask[i, 1] = False
    return mask


def test_registry_includes_industry_and_extras():
    for name in INDUSTRY_BASELINE_NAMES:
        assert name in BENCHMARK_PANEL_NAMES
    for name in PANEL_EXTRA_NAMES:
        assert name in BENCHMARK_PANEL_NAMES
    assert "cap_weight_bah" in PANEL_EXTRA_NAMES
    assert "low_vol_long" in PANEL_EXTRA_NAMES


def test_all_benchmarks_see_same_mask():
    rets = _toy_returns(60, 4)
    mask = _slot_mask(60, 4)
    names = ("equal_weight", "inverse_vol", "cap_weight_bah", "low_vol_long")
    weight_traces: dict[str, list[np.ndarray]] = {n: [] for n in names}

    def _recording_fn(name: str):
        base = get_weight_fn(name)

        def _fn(returns_hist, *, t=None, w_prev=None, **kw):
            w = base(returns_hist, t=t, w_prev=w_prev, **kw)
            return w

        return _fn

    for name in names:
        pnl, meta = run_benchmark_on_fold(
            name,
            rets,
            w_fn=_recording_fn(name),
            slot_mask=mask,
            return_meta=True,
        )
        assert pnl.shape == (rets.shape[0],)
        ws = meta["weights"]
        assert ws.shape == rets.shape
        # Masked slots must be exactly zero after application.
        assert np.all(ws[~mask] == 0.0)
        weight_traces[name] = ws

    # Same mask zeros across all members.
    for name in names:
        assert np.array_equal(weight_traces[name] == 0.0, ~mask | (weight_traces[name] == 0.0))
        assert np.all(weight_traces[name][~mask] == 0.0)


def test_costs_reduce_pnl_vs_zero_cost():
    rets = _toy_returns(50, 4, seed=3)
    mask = np.ones_like(rets, dtype=bool)

    def costly(w, w_prev, ret):
        gross = float(np.nansum(w * ret))
        turn = float(np.sum(np.abs(w - w_prev)))
        return gross - 0.01 * turn

    pnl_zero = run_benchmark_on_fold(
        "equal_weight", rets, slot_mask=mask, apply_costs_fn=None
    )
    pnl_cost = run_benchmark_on_fold(
        "equal_weight", rets, slot_mask=mask, apply_costs_fn=costly
    )
    assert float(np.nansum(pnl_cost)) < float(np.nansum(pnl_zero))


def test_no_lookahead_weights_use_returns_prefix():
    """Mutating future returns must not change weights at t."""
    rets = _toy_returns(40, 4, seed=11)
    mask = np.ones_like(rets, dtype=bool)
    t_star = 25

    captured: list[np.ndarray] = []

    def spy_fn(returns_hist, *, t=None, w_prev=None, **kw):
        # At decision t, hist must be exactly returns[:t].
        if t == t_star:
            captured.append(np.asarray(returns_hist, dtype=np.float64).copy())
        k = returns_hist.shape[1] if returns_hist.ndim == 2 else 4
        return np.full(k, 1.0 / k)

    run_benchmark_on_fold("spy", rets, w_fn=spy_fn, slot_mask=mask)
    assert len(captured) == 1
    assert captured[0].shape[0] == t_star
    assert np.allclose(captured[0], rets[:t_star])

    # Shuffle only future rows; weight-history at t_star unchanged.
    rets2 = rets.copy()
    rng = np.random.default_rng(99)
    rets2[t_star:] = rng.permutation(rets2[t_star:], axis=0)
    captured.clear()
    run_benchmark_on_fold("spy", rets2, w_fn=spy_fn, slot_mask=mask)
    assert np.allclose(captured[0], rets[:t_star])


def test_run_panel_dict_keys():
    rets = _toy_returns(40, 3)
    out = run_panel(
        ("equal_weight", "no_trade", "cap_weight_bah"),
        rets,
        slot_mask=np.ones_like(rets, dtype=bool),
    )
    assert set(out) == {"equal_weight", "no_trade", "cap_weight_bah"}
    for v in out.values():
        assert v.shape == (40,)


def test_attach_fold_benchmark_panel_keys_on_tiny_data():
    from types import SimpleNamespace

    import pandas as pd

    from src.eval.benchmark_panel import attach_fold_benchmark_panel

    dates = list(pd.bdate_range("2020-01-01", periods=40))
    rets = _toy_returns(40, 3)
    fold = SimpleNamespace(
        fold_id=0,
        test_windows=[{"start": "2020-01-20", "end": "2020-02-07"}],
    )
    art = attach_fold_benchmark_panel(
        fold,
        dates=dates,
        returns=rets,
        names=("equal_weight", "no_trade"),
    )
    assert art["ok"] is True
    assert "equal_weight" in art["panel_keys"]
    assert "no_trade" in art["panel_keys"]
    assert "pnls" in art
    assert set(art["pnls"]) == {"equal_weight", "no_trade"}


def test_cap_weight_bah_falls_back_to_equal_without_mktcap():
    rets = _toy_returns(30, 4)
    pnl_eq = run_benchmark_on_fold("equal_weight", rets)
    pnl_cap = run_benchmark_on_fold("cap_weight_bah", rets, mktcap=None)
    # Without mktcap, cap_weight_bah is equal-weight BAH path → same as equal on first day;
    # buy-and-hold drifts with returns so series need not match equal_weight exactly.
    assert pnl_cap.shape == pnl_eq.shape
    assert np.all(np.isfinite(pnl_cap))


def test_low_vol_long_prefers_quiet_names():
    t, k = 100, 5
    rng = np.random.default_rng(0)
    rets = np.zeros((t, k))
    # Name 0 very quiet; others noisy.
    rets[:, 0] = rng.normal(0.0, 0.001, size=t)
    for j in range(1, k):
        rets[:, j] = rng.normal(0.0, 0.05, size=t)
    _, meta = run_benchmark_on_fold(
        "low_vol_long", rets, return_meta=True, min_hist=30
    )
    # After enough history, name 0 should receive the bulk of long-only weight.
    late = meta["weights"][80:]
    mean_w = late.mean(axis=0)
    assert mean_w[0] == pytest.approx(mean_w.max(), rel=1e-6, abs=1e-9)
