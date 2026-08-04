"""Equity label alignment for spectrum arms."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.oos_panel import (
    EQUITY_LABEL_STEM,
    FEATURE_STEMS,
    LABEL_STEM,
    denom_stem_for_label,
    label_matrix,
    pivot_long_marks_to_wide,
)


def test_stk_ret_in_feature_stems_and_never_ffilled():
    assert "stk_ret" in FEATURE_STEMS
    assert EQUITY_LABEL_STEM == "stk_ret"
    # One name, two dates: stk_ret missing on day 0, present on day 1.
    # atm_iv present both days so ffill path exists for non-labels.
    long = pa.table(
        {
            "secid": [1, 1],
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "atm_iv": [0.2, 0.21],
            "dh_ret_lagdelta": [0.01, 0.02],
            "stk_ret": [np.nan, 0.005],
            "mid": [1.0, 1.1],
            "delta": [0.5, 0.5],
            "spot": [100.0, 100.5],
            "strike": [100.0, 100.0],
            "bid_ask_spread": [0.01, 0.01],
            "skew_25d": [0.0, 0.0],
            "dh_denom": [49.0, 49.0],
            "dh_denom_lagdelta": [49.0, 49.0],
            "dh_ret": [0.01, 0.02],
            "fwd_ret": [0.1, 0.1],
            "volume_imbalance": [0.0, 0.0],
            "put_call_oi_ratio": [1.0, 1.0],
        }
    )
    wide = pivot_long_marks_to_wide(long, secids=[1])
    assert np.isnan(wide.loc[pd.Timestamp("2020-01-02"), "stk_ret_0"])
    assert wide.loc[pd.Timestamp("2020-01-03"), "stk_ret_0"] == pytest.approx(0.005, **FLOAT_TOL)
    # Non-label still ffills
    assert wide.loc[pd.Timestamp("2020-01-02"), "atm_iv_0"] == pytest.approx(0.2, **FLOAT_TOL)


def test_post_coverage_ffill_preserves_stk_ret_nan():
    """Second-pass ffill (after coverage filter) must not impute equity labels.

    pivot_long_marks_to_wide already skips label stems; materialize_oos_panel
    then ffills remaining columns. Equity stems must stay in the exclusion set.
    """
    from src.data.oos_panel import no_ffill_label_columns

    idx = pd.date_range("2020-01-02", periods=3, freq="B")
    wide = pd.DataFrame(
        {
            "atm_iv_0": [0.2, np.nan, 0.22],
            "dh_ret_lagdelta_0": [0.01, 0.02, 0.03],
            "dh_ret_0": [0.01, 0.02, 0.03],
            "fwd_ret_0": [0.1, 0.2, 0.3],
            "stk_ret_0": [0.005, np.nan, 0.004],
            "stk_ret_h_days_0": [1.0, np.nan, 1.0],
        },
        index=idx,
    )
    ret_cols = no_ffill_label_columns(wide, n_secids=1)
    assert "stk_ret_0" in ret_cols
    assert "stk_ret_h_days_0" in ret_cols
    feat_cols = [c for c in wide.columns if c not in ret_cols]
    filled = wide.copy()
    filled[feat_cols] = filled[feat_cols].ffill()
    assert np.isnan(filled.loc[idx[1], "stk_ret_0"])
    assert np.isnan(filled.loc[idx[1], "stk_ret_h_days_0"])
    assert filled.loc[idx[1], "atm_iv_0"] == pytest.approx(0.2, **FLOAT_TOL)


def test_label_matrix_accepts_stk_ret():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "stk_ret_0": [0.01, np.nan, -0.02],
            "stk_ret_1": [0.0, 0.01, 0.02],
            "dh_ret_lagdelta_0": [0.0, 0.0, 0.0],
            "dh_ret_lagdelta_1": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    mat = label_matrix(df, 2, stem=EQUITY_LABEL_STEM)
    assert mat.shape == (3, 2)
    assert mat[0, 0] == pytest.approx(0.01, **FLOAT_TOL)
    assert np.isnan(mat[1, 0])


def test_denom_stem_for_stk_ret_is_spot():
    assert denom_stem_for_label(EQUITY_LABEL_STEM) == "spot"
    assert denom_stem_for_label(LABEL_STEM) == "dh_denom_lagdelta"


def test_duckdb_sql_emits_stk_ret_under_label_ok_lag():
    """Static guard: marks SQL must gate stk_ret on label_ok_lag like dh_ret_lagdelta."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "data" / "duckdb_engine.py"
    text = src.read_text()
    assert "END AS stk_ret" in text
    assert "WHEN label_ok_lag AND spot > 0" in text
    # stk_ret appears after dh_ret_lagdelta in the SELECT
    i_dh = text.index("END AS dh_ret_lagdelta")
    i_stk = text.index("END AS stk_ret")
    assert i_stk > i_dh
