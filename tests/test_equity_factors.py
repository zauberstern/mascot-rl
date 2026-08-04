"""Equity factor construction and HAC alpha suite (spectrum Gate 2)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.eval.equity_factors import attach_equity_factor_suite, build_equity_factors


def test_build_equity_factors_aligns_mkt_from_mocked_returns():
    dates = pd.bdate_range("2022-01-03", periods=10)
    mkt_raw = pd.Series(
        np.linspace(-0.01, 0.01, len(dates)),
        index=dates,
        name="vwretd",
    )

    with patch(
        "src.eval.equity_factors.load_equity_daily_returns",
        return_value=mkt_raw,
    ), patch(
        "src.eval.equity_factors.load_cash_daily_returns",
        side_effect=FileNotFoundError("no rf"),
    ), patch(
        "src.eval.equity_factors._try_load_ff_panel",
        return_value=None,
    ):
        df = build_equity_factors(dates, lake_or_path="/unused")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == list(pd.DatetimeIndex(dates))
    assert "mkt" in df.columns
    assert np.isfinite(df["mkt"].to_numpy()).all()
    assert df["mkt"].to_numpy() == pytest.approx(mkt_raw.to_numpy())
    note = df.attrs.get("note") or ""
    assert "smb" in note.lower() or "omitted" in note.lower() or "market" in note.lower()


def test_attach_equity_factor_suite_reports_alpha_and_hlz():
    rng = np.random.default_rng(0)
    n = 80
    dates = pd.bdate_range("2020-01-02", periods=n)
    mkt = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    # Strategy = 0.5*mkt + alpha + noise so OLS recovers positive alpha.
    alpha_daily = 0.001
    strat = 0.5 * mkt + alpha_daily + rng.normal(0.0, 0.002, n)
    factor_df = pd.DataFrame({"mkt": mkt}, index=dates)
    factor_df.attrs["note"] = "mkt only"

    out = attach_equity_factor_suite(strat, factor_df)
    assert out["ok"] is True
    assert "alpha" in out
    assert out["alpha"]["ok"] is True
    assert out["alpha"]["alpha_daily"] == pytest.approx(alpha_daily, abs=5e-4)
    assert "hlz" in out
    assert "clears_conventional_1_96" in out["hlz"]
    assert out["factors_used"] == ["mkt"]
