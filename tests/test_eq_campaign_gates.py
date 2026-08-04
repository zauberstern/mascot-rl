"""C7: unit tests for scripts.run_eq_alloc_campaign.compute_eq_campaign_gates.

These exercise gate1/gate2/gate3 date-alignment and fallback logic directly,
without running the full CPCV/RL loop that the smoke test in
test_run_eq_alloc_campaign_main_smoke.py exercises end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_eq_alloc_campaign import compute_eq_campaign_gates


def _dates(n: int) -> list[str]:
    return [str(d.date()) for d in pd.bdate_range("2020-01-02", periods=n)]


def test_gate1_extrapolates_break_even_from_fill_ladder() -> None:
    # mid (0.5x) Sharpe = 1.0, pct75 (1.0x) Sharpe = 0.5 -> slope=-0.5 over
    # +0.5x, zero crossing at multiplier = 0.5 + 1.0/(0.5/0.5) = 1.5.
    gates = compute_eq_campaign_gates(
        fill_ladder={"mid": 1.0, "pct75": 0.5},
        policy_sharpe=0.3,
        challenger_sharpes={},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    assert "gate1_error" not in gates
    g1 = gates["gate1"]
    assert g1["break_even_spread_multiplier"] == pytest.approx(1.5)
    assert g1["pass"] is True  # 1.5 >= 0.25 threshold


def test_gate1_nan_when_ladder_missing() -> None:
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.0,
        challenger_sharpes={},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    g1 = gates["gate1"]
    assert g1["break_even_spread_multiplier"] != g1["break_even_spread_multiplier"]  # NaN
    assert g1["pass"] is False


def test_gate2_skipped_when_no_series() -> None:
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.0,
        challenger_sharpes={},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    assert gates["gate2_skipped_reason"] == "no policy path series available"
    assert "gate2" not in gates


def test_gate2_skipped_when_dates_length_mismatch() -> None:
    dates = _dates(40)
    series0 = np.random.default_rng(0).normal(0.0, 0.01, size=40)
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.0,
        challenger_sharpes={},
        series0=series0,
        series0_dates=dates[:35],  # deliberately mismatched
        panel_dates=dates,
        lake_root="/nonexistent",
    )
    assert "length" in gates["gate2_skipped_reason"]
    assert "gate2" not in gates


def test_gate2_runs_and_produces_ff_alpha_when_dates_align() -> None:
    dates = _dates(60)
    rng = np.random.default_rng(1)
    series0 = rng.normal(0.001, 0.01, size=60)
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.0,
        challenger_sharpes={},
        series0=series0,
        series0_dates=dates,
        panel_dates=dates,
        lake_root="/nonexistent_lake_path_so_ff4_falls_back_to_zeros",
    )
    assert "gate2_error" not in gates
    assert "gate2" in gates
    assert "pass" in gates["gate2"]
    assert "t_stat" in gates["gate2"]


def test_gate2_matches_subset_of_panel_dates() -> None:
    """series0_dates can be a strict subset of panel_dates (e.g. a shorter
    OOS test window inside a longer full-panel date axis); gate2 must align
    by date value, not by position."""
    panel_dates = _dates(100)
    series_dates = panel_dates[20:80]  # 60 rows, offset into the panel
    rng = np.random.default_rng(2)
    series0 = rng.normal(0.0005, 0.008, size=len(series_dates))
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.0,
        challenger_sharpes={},
        series0=series0,
        series0_dates=series_dates,
        panel_dates=panel_dates,
        lake_root="/nonexistent",
    )
    assert "gate2" in gates
    assert gates["gate2"]["n"] == len(series_dates)


def test_gate3_beats_best_baseline() -> None:
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=1.5,
        challenger_sharpes={"equal_weight": 1.0, "olps:ons": 0.8},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    g3 = gates["gate3"]
    assert g3["pass"] is True
    assert g3["best_baseline"] == "equal_weight"


def test_gate3_fails_when_policy_worse_than_every_baseline() -> None:
    gates = compute_eq_campaign_gates(
        fill_ladder={},
        policy_sharpe=0.1,
        challenger_sharpes={"equal_weight": 1.0, "olps:ons": 0.8},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    assert gates["gate3"]["pass"] is False


def test_all_three_gates_independently_present_even_if_one_errors() -> None:
    # fill_ladder with non-numeric mid should be caught by gate1's own
    # try/except and not prevent gate2/gate3 from computing.
    gates = compute_eq_campaign_gates(
        fill_ladder={"mid": "not-a-number", "pct75": 0.5},
        policy_sharpe=2.0,
        challenger_sharpes={"equal_weight": 1.0},
        series0=None,
        series0_dates=None,
        panel_dates=_dates(5),
        lake_root="/nonexistent",
    )
    assert "gate1_error" in gates
    assert "gate3" in gates
    assert gates["gate3"]["pass"] is True
