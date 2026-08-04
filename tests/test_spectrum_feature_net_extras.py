"""Spectrum must honor use_feature_net_extras (G0 stays off by default)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.equity_substrate import (
    apply_feature_net_extras_if_enabled,
    stamp_equity_obs_defaults,
)


def test_stamp_defaults_keep_feature_net_off() -> None:
    cfg = stamp_equity_obs_defaults({})
    assert cfg.get("use_feature_net_extras") is False


def test_apply_feature_net_noop_when_flag_false(tmp_path: Path) -> None:
    cfg: dict = {
        "use_feature_net_extras": False,
        "lake_root": str(tmp_path),
        "feature_extras": {"dollar_volume": np.ones((4, 2))},
    }
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    out = apply_feature_net_extras_if_enabled(
        cfg, dates=dates, secids=["a", "b"], panel_source="lake_sp500_sec"
    )
    assert "ohlc" not in (out.get("feature_extras") or {})
    assert out.get("_feature_net_errors") in (None, [])


def test_apply_feature_net_fail_closed_missing_lake() -> None:
    cfg: dict = {
        "use_feature_net_extras": True,
        "feature_extras": {},
    }
    with pytest.raises(RuntimeError, match="lake_root"):
        apply_feature_net_extras_if_enabled(
            cfg,
            dates=["2020-01-01", "2020-01-02"],
            secids=["a"],
            panel_source="lake_sp500_sec",
        )


def test_apply_feature_net_fail_closed_empty_attach(tmp_path: Path) -> None:
    cfg: dict = {
        "use_feature_net_extras": True,
        "lake_root": str(tmp_path),
        "feature_extras": {},
    }
    with pytest.raises(RuntimeError, match="no feature-net panels"):
        apply_feature_net_extras_if_enabled(
            cfg,
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            secids=["100", "200"],
            panel_source="lake_sp500_sec",
        )


def test_apply_feature_net_attaches_ohlc_from_mirror(tmp_path: Path) -> None:
    import pandas as pd

    lake = tmp_path
    panels = lake / "_panels"
    panels.mkdir(parents=True)
    rows = []
    for d in ("2020-01-01", "2020-01-02", "2020-01-03"):
        for sid in ("100", "200"):
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "adj_close": 1.05,
                }
            )
    pd.DataFrame(rows).to_parquet(panels / "feat_ohlc.parquet")
    cfg: dict = {
        "use_feature_net_extras": True,
        "lake_root": str(lake),
        "feature_extras": {},
    }
    out = apply_feature_net_extras_if_enabled(
        cfg,
        dates=["2020-01-01", "2020-01-02", "2020-01-03"],
        secids=["100", "200"],
        panel_source="lake_sp500_sec",
    )
    extras = out["feature_extras"]
    assert "ohlc" in extras
    assert "close" in extras["ohlc"]
    assert extras["ohlc"]["close"].shape == (3, 2)
