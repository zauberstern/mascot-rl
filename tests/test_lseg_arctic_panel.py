"""Long LSEG panels must keep multiple secids on one date."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mascotrl.data.arctic_store import ArcticStateStore


def test_persist_panel_keeps_two_secids_same_date(tmp_path: Path) -> None:
    store = ArcticStateStore(db_path=tmp_path / "arctic", library_name="test_lseg")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "secid": [1, 2],
            "lseg_bid": [1.0, 2.0],
        }
    )
    store.persist_panel("lseg_eq_ohlc_corax", df, metadata={"asof_ts": "2026-08-18"})
    got = store.lib.read("lseg_eq_ohlc_corax").data
    assert len(got) == 2
    with pytest.raises(ValueError, match="P3"):
        store.persist_panel("lseg_p3_worldscope", df)


def test_persist_panel_accepts_pandas_float64_extension(tmp_path: Path) -> None:
    store = ArcticStateStore(db_path=tmp_path / "arctic", library_name="test_lseg")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "secid": [1, 2],
            "lseg_bid": pd.Series([1.0, 2.0], dtype="Float64"),
        }
    )
    store.persist_panel("lseg_eq_ohlc_unadj", df)
    got = store.lib.read("lseg_eq_ohlc_unadj").data
    assert len(got) == 2
