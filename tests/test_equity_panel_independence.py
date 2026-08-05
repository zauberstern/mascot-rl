"""stk_ret on the standalone equity panel must not require option rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.equity_panel import (
    EVAL_END,
    EVAL_START,
    SELECTION_END,
    SELECTION_START,
    UNIVERSE_MODE_EQUITY_SP500,
    UNIVERSE_MODE_EXPLICIT,
    load_sp500_security_returns,
    materialize_equity_panel,
    reconcile_with_crsp,
)


def _write_tiny_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    macro = lake / "macro"
    macro.mkdir(parents=True)

    # Name 999001 has equity marks only (no options anywhere in this lake).
    dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    )
    closes = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    rets = np.concatenate([[np.nan], closes[1:] / closes[:-1] - 1.0])
    sec = pd.DataFrame(
        {
            "secid": [999001] * len(dates),
            "date": dates,
            "ticker": ["EQONLY"] * len(dates),
            "close": closes,
            "return": rets,
            "shrout": [1000.0] * len(dates),
            "cfadj": [1.0] * len(dates),
            "cfret": [1.0] * len(dates),
            "volume": [1.0e5, 1.1e5, 9.0e4, 1.2e5, 1.0e5],
        }
    )
    # Drop leading NaN return so every persisted row is usable as a label.
    sec = sec.iloc[1:].reset_index(drop=True)
    sec.to_parquet(macro / "sp500_sec.parquet", index=False)

    # Empty CRSP / link: reconcile must fall back to OM return.
    pd.DataFrame(
        columns=["secid", "permno", "sdate", "edate", "score"]
    ).to_parquet(macro / "crsp_optionm_link.parquet", index=False)
    pd.DataFrame(
        columns=["PERMNO", "date", "RET", "DLRET"]
    ).to_parquet(macro / "sp500_prices.parquet", index=False)

    # EQONLY is an index member for the whole panel window.
    pd.DataFrame(
        {
            "ticker": ["EQONLY"],
            "start_date": [pd.Timestamp("2019-01-01")],
            "end_date": [pd.NaT],
        }
    ).to_parquet(macro / "pit_membership.parquet", index=False)
    return lake


def test_calendar_constants_documented():
    assert SELECTION_START == "2003-01-02"
    assert SELECTION_END == "2012-12-31"
    assert EVAL_START == "2014-01-01"
    assert EVAL_END == "2024-12-31"


def test_universe_mode_constants():
    assert UNIVERSE_MODE_EQUITY_SP500 == "equity_sp500"
    assert UNIVERSE_MODE_EXPLICIT == "explicit"


def test_load_sp500_security_returns_reads_lake(tmp_path: Path):
    lake = _write_tiny_lake(tmp_path)
    df = load_sp500_security_returns(lake, "2020-01-01", "2020-01-31")
    assert not df.empty
    assert set(["secid", "date", "ticker", "close", "return"]).issubset(df.columns)
    assert int(df["secid"].iloc[0]) == 999001


def test_reconcile_falls_back_to_om_return(tmp_path: Path):
    lake = _write_tiny_lake(tmp_path)
    raw = load_sp500_security_returns(lake, "2020-01-01", "2020-01-31")
    out = reconcile_with_crsp(
        raw,
        lake / "macro" / "crsp_optionm_link.parquet",
        crsp_path=lake / "macro" / "sp500_prices.parquet",
    )
    assert "stk_ret" in out.columns
    assert "return_source" in out.columns
    assert (out["return_source"] == "optionmetrics").all()
    assert np.isfinite(out["stk_ret"].to_numpy()).all()


def test_equity_only_name_yields_finite_stk_ret(tmp_path: Path):
    """Critical: no option rows required for finite stk_ret."""
    lake = _write_tiny_lake(tmp_path)
    out_dir = tmp_path / "equity_out"
    result = materialize_equity_panel(
        secids=[999001],
        tickers=["EQONLY"],
        panel_start="2020-01-03",
        panel_end="2020-01-08",
        lake_base_dir=lake,
        out_dir=out_dir,
        universe_mode=UNIVERSE_MODE_EXPLICIT,
        reconcile=True,
        apply_pit_membership=True,
    )
    assert result["n_rows"] >= 1
    wide_path = out_dir / "equity_signals.parquet"
    assert wide_path.is_file()
    wide = pd.read_parquet(wide_path)
    assert "stk_ret_0" in wide.columns
    finite = np.isfinite(wide["stk_ret_0"].to_numpy(dtype=np.float64))
    assert finite.any(), "expected finite stk_ret without any option chain"
    assert finite.sum() == int(np.isfinite(wide["stk_ret_0"]).sum())


def test_explicit_requires_secids_or_selector(tmp_path: Path):
    lake = _write_tiny_lake(tmp_path)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        materialize_equity_panel(
            secids=None,
            panel_start="2020-01-03",
            panel_end="2020-01-08",
            lake_base_dir=lake,
            out_dir=tmp_path / "out",
            universe_mode=UNIVERSE_MODE_EXPLICIT,
        )


def test_selector_result_secids_accepted(tmp_path: Path):
    lake = _write_tiny_lake(tmp_path)
    out_dir = tmp_path / "sel_out"
    result = materialize_equity_panel(
        secids=None,
        tickers=["EQONLY"],
        panel_start="2020-01-03",
        panel_end="2020-01-08",
        lake_base_dir=lake,
        out_dir=out_dir,
        universe_mode=UNIVERSE_MODE_EXPLICIT,
        selector_result={
            "secids": [999001],
            "provenance": {"mode": "explicit", "hash": "abc"},
        },
        reconcile=False,
        apply_pit_membership=False,
        validate_cfadj=False,
    )
    assert result["n_rows"] >= 1
    meta = json.loads((out_dir / "equity_universe.json").read_text())
    assert meta["universe_mode"] == "explicit"
    assert meta["selector_provenance"]["mode"] == "explicit"
