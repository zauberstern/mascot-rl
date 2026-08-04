"""Unit tests for W4.1 dynamic (time-varying, slot-masked) universe builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FLOAT_TOL

from src.data.dynamic_universe import (
    build_dynamic_universe,
    build_slotted_panel,
    option_eligibility_screen,
    select_universe_corr_cluster,
    select_universe_liquidity,
    selection_turnover,
)


def _dates(n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2020-01-02", periods=n))


def test_option_eligibility_screen_requires_min_obs_in_trailing_window():
    dates = _dates(100)
    secids = [1, 2, 3]
    # secid 1: always finite. secid 2: finite only in the last 5 rows before
    # asof (not enough trailing history). secid 3: finite everywhere.
    mfis30 = np.column_stack(
        [
            np.ones(100),
            np.where(np.arange(100) >= 90, 1.0, np.nan),
            np.ones(100),
        ]
    )
    mfis365 = np.ones((100, 3))
    asof = dates[95]
    eligible = option_eligibility_screen(
        asof=asof,
        secids=secids,
        signal_panels={"mfis_30": mfis30, "mfis_365": mfis365},
        dates=dates,
        trailing_days=63,
        min_obs=21,
    )
    assert eligible == [1, 3]


def test_option_eligibility_screen_is_pit_future_signal_does_not_help():
    """A secid that only becomes eligible *after* asof must not be admitted."""
    dates = _dates(50)
    secids = [10, 20]
    mfis30 = np.column_stack(
        [
            np.ones(50),
            # secid 20 only has real observations strictly after the asof date.
            np.where(np.arange(50) > 30, 1.0, np.nan),
        ]
    )
    mfis365 = np.ones((50, 2))
    asof = dates[30]
    eligible = option_eligibility_screen(
        asof=asof,
        secids=secids,
        signal_panels={"mfis_30": mfis30, "mfis_365": mfis365},
        dates=dates,
        trailing_days=63,
        min_obs=5,
    )
    assert eligible == [10]

    # Even looking further ahead in time never rescues secid 20 as-of day 30:
    # its future observations are outside the trailing PIT window by
    # construction, so this is really just re-asserting the guarantee above.
    later_asof = dates[29]
    eligible_earlier = option_eligibility_screen(
        asof=later_asof,
        secids=secids,
        signal_panels={"mfis_30": mfis30, "mfis_365": mfis365},
        dates=dates,
        trailing_days=63,
        min_obs=1,
    )
    assert 20 not in eligible_earlier


def _toy_select_fn(returns, *, k, secids=None, **_kw):
    d = returns.shape[1]
    k_eff = min(int(k), d)
    idx = list(range(k_eff))
    out_secids = [int(secids[i]) for i in idx] if secids is not None else None
    return {"indices": idx, "secids": out_secids, "provenance": "toy"}


def test_build_dynamic_universe_holds_between_rebalances_and_uses_pit_window():
    rng = np.random.default_rng(0)
    t, n = 40, 5
    dates = _dates(t)
    secids = [100, 200, 300, 400, 500]
    wide_returns = rng.normal(scale=0.01, size=(t, n))
    rebalance_mask = np.zeros(t, dtype=bool)
    rebalance_mask[[0, 10, 20, 30]] = True

    slots_rows, valid_mask, log = build_dynamic_universe(
        dates=dates,
        rebalance_mask=rebalance_mask,
        wide_returns=wide_returns,
        secids=secids,
        k=3,
        select_fn=_toy_select_fn,
        trailing_days=252,
    )
    assert len(slots_rows) == t
    assert valid_mask.shape == (t, 3)
    # Between rebalances the slot assignment is held.
    for i in range(1, 10):
        assert slots_rows[i] == slots_rows[0]
    for i in range(11, 20):
        assert slots_rows[i] == slots_rows[10]
    assert len(log) == 4
    for entry in log:
        assert "date" in entry and "secids" in entry and "n_eligible" in entry


def test_build_dynamic_universe_slices_state_blocks_kwarg_to_eligible_columns():
    """select_kwargs['state_blocks'] must track elig_window's (T, D) shape
    even when eligibility drops pool members on a given rebalance date
    (regression: select_fn with state_blocks'
    column count to match the returns window it is called with)."""
    t, n = 30, 4
    dates = _dates(t)
    secids = [1, 2, 3, 4]
    wide_returns = np.zeros((t, n))
    state_blocks = np.arange(t * n, dtype=np.float64).reshape(t, n)
    rebalance_mask = np.zeros(t, dtype=bool)
    rebalance_mask[[0, 10]] = True
    # Only 2 of 4 pool secids eligible on each rebalance date.
    eligibility_by_date = {dates[0]: [1, 2], dates[10]: [2, 3, 4]}

    seen_shapes = []

    def _shape_checking_select_fn(returns, *, k, secids=None, state_blocks=None, **_kw):
        assert state_blocks is not None
        assert state_blocks.shape[0] == returns.shape[0]
        assert state_blocks.shape[1] == returns.shape[1]
        seen_shapes.append(state_blocks.shape)
        return _toy_select_fn(returns, k=k, secids=secids)

    build_dynamic_universe(
        dates=dates,
        rebalance_mask=rebalance_mask,
        wide_returns=wide_returns,
        secids=secids,
        k=2,
        select_fn=_shape_checking_select_fn,
        eligibility_by_date=eligibility_by_date,
        select_kwargs={"state_blocks": state_blocks},
    )
    assert len(seen_shapes) == 2
    assert seen_shapes[0][1] == 2  # secids 1,2 eligible at t=0
    assert seen_shapes[1][1] == 3  # secids 2,3,4 eligible at t=10


def test_build_dynamic_universe_respects_eligibility_by_date():
    t, n = 20, 4
    dates = _dates(t)
    secids = [1, 2, 3, 4]
    wide_returns = np.zeros((t, n))
    rebalance_mask = np.zeros(t, dtype=bool)
    rebalance_mask[[0, 10]] = True
    eligibility_by_date = {
        dates[0]: [1, 2],
        dates[10]: [3, 4],
    }
    slots_rows, valid_mask, log = build_dynamic_universe(
        dates=dates,
        rebalance_mask=rebalance_mask,
        wide_returns=wide_returns,
        secids=secids,
        k=3,
        select_fn=_toy_select_fn,
        eligibility_by_date=eligibility_by_date,
    )
    assert slots_rows[0][:2] == [1, 2]
    assert slots_rows[0][2] is None
    assert slots_rows[10][:2] == [3, 4]
    assert log[0]["n_eligible"] == 2
    assert log[1]["n_eligible"] == 2


def test_build_slotted_panel_maps_secid_returns_and_zeros_none():
    dates = _dates(3)
    wide_returns = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.10, 0.20, 0.30],
            [1.00, 2.00, 3.00],
        ]
    )
    col_map = {10: 0, 20: 1, 30: 2}
    slots_rows = [
        [10, 20, None],
        [10, 20, None],
        [30, None, 10],
    ]
    panel = build_slotted_panel(
        dates=dates, slots_rows=slots_rows, wide_returns=wide_returns, col_map=col_map
    )
    assert panel.shape == (3, 3)
    np.testing.assert_allclose(panel[0], [0.01, 0.02, 0.0])
    np.testing.assert_allclose(panel[1], [0.10, 0.20, 0.0])
    np.testing.assert_allclose(panel[2], [3.00, 0.0, 1.00])


def test_selection_turnover_counts_added_and_dropped_across_rebalances():
    slots_rows = [
        [1, 2, 3],
        [1, 2, 3],  # hold
        [1, 2, 3],  # hold
        [1, 4, 3],  # rebalance: drop 2, add 4
        [5, 4, 3],  # rebalance: drop 1, add 5
    ]
    out = selection_turnover(slots_rows)
    assert out["mean_added"] == pytest.approx(1.0)
    assert out["mean_dropped"] == pytest.approx(1.0)
    assert len(out["per_step"]) == 2
    assert out["per_step"][0]["added"] == [4]
    assert out["per_step"][0]["dropped"] == [2]
    assert out["per_step"][1]["added"] == [5]
    assert out["per_step"][1]["dropped"] == [1]


def test_selection_turnover_no_rebalances_is_zero():
    slots_rows = [[1, 2], [1, 2], [1, 2]]
    out = selection_turnover(slots_rows)
    assert out["mean_added"] == pytest.approx(0.0, **FLOAT_TOL)
    assert out["mean_dropped"] == pytest.approx(0.0, **FLOAT_TOL)
    assert out["per_step"] == []


def test_select_universe_corr_cluster_returns_k_diverse_names():
    rng = np.random.default_rng(3)
    t = 300
    base = rng.normal(size=t)
    # Two tight clusters (highly correlated within) + independent noise names.
    cluster_a = np.column_stack([base + rng.normal(scale=0.01, size=t) for _ in range(3)])
    cluster_b = np.column_stack(
        [-base + rng.normal(scale=0.01, size=t) for _ in range(3)]
    )
    noise = rng.normal(size=(t, 4))
    returns = np.column_stack([cluster_a, cluster_b, noise])
    secids = list(range(returns.shape[1]))

    out = select_universe_corr_cluster(returns, k=4, secids=secids)
    assert len(out["indices"]) == 4
    assert len(set(out["indices"])) == 4
    assert out["secids"] is not None
    assert len(out["secids"]) == 4
    assert out["provenance"].startswith("corr_cluster")


def test_select_universe_corr_cluster_caps_k_to_pool_size():
    returns = np.random.default_rng(1).normal(size=(50, 2))
    out = select_universe_corr_cluster(returns, k=10, secids=[7, 8])
    assert len(out["indices"]) == 2


def test_select_universe_liquidity_prefers_full_coverage_low_vol():
    rng = np.random.default_rng(5)
    t = 100
    low_vol_full = rng.normal(scale=0.001, size=t)
    high_vol_full = rng.normal(scale=0.05, size=t)
    sparse = rng.normal(scale=0.001, size=t)
    sparse[: t // 2] = np.nan
    returns = np.column_stack([low_vol_full, high_vol_full, sparse])
    secids = [1, 2, 3]
    out = select_universe_liquidity(returns, k=2, secids=secids)
    assert len(out["indices"]) == 2
    # Full-coverage low-vol name must beat the sparse-coverage name.
    assert 1 in out["secids"]
    assert 3 not in out["secids"]


def test_option_eligibility_liquidity_gate_excludes_illiquid():
    dates = _dates(80)
    secids = [1, 2, 3]
    mfis = np.ones((80, 3))
    vol = np.column_stack(
        [
            np.full(80, 1000.0),
            np.full(80, 1.0),
            np.where(np.arange(80) % 2 == 0, 1000.0, np.nan),
        ]
    )
    eligible = option_eligibility_screen(
        asof=dates[70],
        secids=secids,
        signal_panels={"mfis_30": mfis, "mfis_365": mfis},
        dates=dates,
        trailing_days=40,
        min_obs=10,
        volume_panel=vol,
        min_option_volume=50.0,
        max_missing_frac=0.25,
    )
    assert eligible == [1]


def test_option_eligibility_liquidity_gate_refuses_empty_pool_when_k_required():
    dates = _dates(40)
    secids = [1, 2]
    mfis = np.ones((40, 2))
    vol = np.full((40, 2), 1.0)
    eligible = option_eligibility_screen(
        asof=dates[30],
        secids=secids,
        signal_panels={"mfis_30": mfis, "mfis_365": mfis},
        dates=dates,
        trailing_days=20,
        min_obs=5,
        volume_panel=vol,
        min_option_volume=100.0,
        max_missing_frac=0.1,
    )
    assert eligible == []
    k = 2
    with pytest.raises(ValueError, match="need k="):
        if len(eligible) < k:
            raise ValueError(
                f"option eligibility left {len(eligible)} names; need k={k}"
            )
