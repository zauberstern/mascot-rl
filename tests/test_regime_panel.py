"""Tests for KPT 10-sector / desk return panels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.regime_return_panel import (
    GICS_SECTOR_TO_KPT10,
    KPT10_NAMES,
    SIC_TO_KPT10,
    _crsp_ret_to_float,
    _normalize_gics_sector,
    load_kpt_gics_sector_returns,
    load_kpt_sector_returns,
    load_style_desk_returns,
)


def test_crsp_minus_codes_become_nan() -> None:
    s = pd.Series(["0.01", "-99", "-66", "-77", "0.02"])
    out = _crsp_ret_to_float(s)
    assert out.iloc[0] == pytest.approx(0.01)
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])
    assert out.iloc[4] == pytest.approx(0.02)


def test_kpt_sector_equal_weight_synthetic(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    rows = []
    # Three energy names (SIC 1311) and three tech (SIC 3674)
    for d in dates:
        for i, (perm, sic, ret) in enumerate(
            [
                (1, 1311, 0.01),
                (2, 1311, 0.03),
                (3, 1311, 0.05),
                (4, 3674, 0.02),
                (5, 3674, 0.04),
                (6, 3674, 0.06),
            ]
        ):
            rows.append(
                {"date": d, "PERMNO": perm, "RET": ret, "SICCD": sic}
            )
    # One day with CRSP -99 that must not enter the mean as -0.99
    rows.append(
        {"date": dates[50], "PERMNO": 99, "RET": -99.0, "SICCD": 1311}
    )
    df = pd.DataFrame(rows)
    path = tmp_path / "macro"
    path.mkdir()
    df.to_parquet(path / "sp500_prices.parquet")
    out = load_kpt_sector_returns(tmp_path, dates)
    assert out is not None
    arr, meta = out
    assert arr.shape == (120, 10)
    assert meta["source"] == "usb_sp500_sic10"
    # Energy EW = (0.01+0.03+0.05)/3 = 0.03 (extra -99 name dropped from finite count? 
    # with 4 names of which one NaN, n_ok=3 still OK; mean of 0.01,0.03,0.05)
    e_idx = KPT10_NAMES.index("energy")
    t_idx = KPT10_NAMES.index("info_tech")
    assert arr[10, e_idx] == pytest.approx(0.03)
    assert arr[10, t_idx] == pytest.approx(0.04)
    assert arr[50, e_idx] == pytest.approx(0.03)


def test_load_style_desk_returns_json(tmp_path: Path) -> None:
    import json

    panel = np.arange(20, dtype=np.float64).reshape(10, 2).tolist()
    p = tmp_path / "desk.json"
    p.write_text(json.dumps({"panel_returns": panel}), encoding="utf-8")
    out = load_style_desk_returns(p)
    assert out is not None
    arr, meta = out
    assert arr.shape == (10, 2)
    assert meta["source"] == "desk"


def test_sic_map_covers_ten_buckets() -> None:
    covered = set(SIC_TO_KPT10.values())
    assert covered == set(KPT10_NAMES)


def test_gics_map_covers_ten_buckets() -> None:
    covered = set(GICS_SECTOR_TO_KPT10.values())
    assert covered == set(KPT10_NAMES)
    assert _normalize_gics_sector("Information Technology") == "info_tech"
    assert _normalize_gics_sector("Real Estate") == "utilities_real_estate"
    assert _normalize_gics_sector("unknown_xyz") is None


def test_gics_sector_equal_weight_synthetic(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    # Five sectors x 3 names => finite_frac >= 0.50 coverage gate.
    sector_specs = [
        ("Energy", 0.01),
        ("Materials", 0.02),
        ("Industrials", 0.03),
        ("Information Technology", 0.04),
        ("Financials", 0.05),
    ]
    rows = []
    link_rows = []
    ric_rows = []
    perm = 1
    for sec_name, base in sector_specs:
        for j in range(3):
            p = perm
            perm += 1
            for d in dates:
                rows.append(
                    {
                        "date": d,
                        "PERMNO": p,
                        "RET": base + 0.01 * j,
                        "SICCD": 1311,
                    }
                )
            link_rows.append(
                {
                    "secid": str(10 * p),
                    "permno": p,
                    "sdate": "2000-01-01",
                    "edate": "2099-01-01",
                    "score": 1,
                }
            )
            ric_rows.append(
                {
                    "secid": str(10 * p),
                    "TR.GICSSector": sec_name,
                    "asof_ts": "2026-08-18",
                }
            )
    (tmp_path / "macro").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "macro" / "sp500_prices.parquet")
    pd.DataFrame(link_rows).to_parquet(tmp_path / "macro" / "crsp_optionm_link.parquet")
    pd.DataFrame(ric_rows).to_parquet(tmp_path / "macro" / "lseg_ric_map.parquet")
    out = load_kpt_gics_sector_returns(tmp_path, dates)
    assert out is not None
    arr, meta = out
    assert meta["source"] == "usb_sp500_gics10"
    assert meta["permno_mapped_frac"] == pytest.approx(1.0)
    assert meta["finite_frac"] >= 0.50
    e_idx = KPT10_NAMES.index("energy")
    t_idx = KPT10_NAMES.index("info_tech")
    # Equal-weight of 0.01, 0.02, 0.03
    assert arr[10, e_idx] == pytest.approx(0.02)
    # Equal-weight of 0.04, 0.05, 0.06
    assert arr[10, t_idx] == pytest.approx(0.05)


def test_gics_below_coverage_gate_returns_none(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    rows = []
    for d in dates:
        for perm in range(1, 11):
            rows.append({"date": d, "PERMNO": perm, "RET": 0.01, "SICCD": 1311})
    (tmp_path / "macro").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "macro" / "sp500_prices.parquet")
    # Only 1 of 10 mapped -> frac 0.1 < 0.80
    pd.DataFrame(
        {
            "secid": ["10"],
            "permno": [1],
            "sdate": ["2000-01-01"],
            "edate": ["2099-01-01"],
            "score": [1],
        }
    ).to_parquet(tmp_path / "macro" / "crsp_optionm_link.parquet")
    pd.DataFrame(
        {
            "secid": ["10"],
            "TR.GICSSector": ["Energy"],
            "asof_ts": ["2026-08-18"],
        }
    ).to_parquet(tmp_path / "macro" / "lseg_ric_map.parquet")
    assert load_kpt_gics_sector_returns(tmp_path, dates) is None


def test_jaccard_sic_vs_gics_operational_report_only() -> None:
    """Report-only helper returns a finite Jaccard in [0, 1]; self-overlap is 1."""
    from src.eval.regime_scorecard import (
        jaccard_sic_vs_gics_operational,
        operational_markov_mask_from_returns,
    )
    from src.eval.walk_forward_hmm import jaccard_turbulent

    rng = np.random.default_rng(0)
    t, k = 400, 10
    r = rng.normal(0, 0.01, size=(t, k))
    r[250:, :] = rng.normal(0, 0.05, size=(t - 250, k))
    mask = operational_markov_mask_from_returns(
        r, turbulence_window=60, hmm_window=120, hmm_step=20
    )
    assert jaccard_turbulent(mask, mask) == pytest.approx(1.0)
    # Second panel slightly perturbed: still a valid report-only Jaccard.
    r2 = r + rng.normal(0, 1e-4, size=r.shape)
    j = jaccard_sic_vs_gics_operational(
        r, r2, turbulence_window=60, hmm_window=120, hmm_step=20
    )
    assert 0.0 <= j <= 1.0
    assert np.isfinite(j)
