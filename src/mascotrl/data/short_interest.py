"""P3 short-interest disclosure builder (flag-default-off; not feature-admitted)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_SI_PCT_CANDIDATES = (
    "Short Interest Pct",
    "short_interest_pct",
    "short_interest_ratio",
    "si_pct",
)


def build_short_interest_ratio(
    table: pd.DataFrame | str | Path | None,
    *,
    dates: Sequence,
    secids: Sequence,
    enabled: bool = False,
) -> pd.DataFrame:
    """Build ``(secid, date, short_interest_ratio)`` with strict PIT ffill.

    Default ``enabled=False`` returns an empty frame so P3 disclosure never
    leaks into the obs cube unless explicitly opted in. When enabled:

    - values are as-of / forward-filled from the past only (no bfill)
    - missing stays NaN (never zero-filled)
    - ``P3_REFUSED`` in ``lseg_overlay`` is unchanged; this reads the
      disclosure parquet directly behind the flag.
    """
    cols = ["secid", "date", "short_interest_ratio"]
    if not enabled:
        return pd.DataFrame(columns=cols)

    if table is None:
        return pd.DataFrame(columns=cols)
    if isinstance(table, (str, Path)):
        path = Path(table)
        if not path.is_file():
            return pd.DataFrame(columns=cols)
        df = pd.read_parquet(path)
    else:
        df = table.copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    value_col = None
    for c in _SI_PCT_CANDIDATES:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        raise ValueError(
            f"short interest table missing pct column; have {list(df.columns)}"
        )

    sid_col = "secid" if "secid" in df.columns else None
    if sid_col is None:
        raise ValueError("short interest table missing secid")
    df = df[[sid_col, "date", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.rename(columns={sid_col: "secid", value_col: "short_interest_ratio"})
    df["secid"] = df["secid"].map(lambda x: int(x) if pd.notna(x) else x)

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    secid_list = [int(s) for s in secids]
    wide = (
        df.dropna(subset=["short_interest_ratio"])
        .pivot_table(
            index="date",
            columns="secid",
            values="short_interest_ratio",
            aggfunc="last",
        )
        .sort_index()
    )
    # Reindex to calendar then ffill only (PIT); never bfill / fillna(0).
    wide = wide.reindex(date_index)
    wide = wide.reindex(columns=secid_list)
    wide = wide.ffill()

    out = (
        wide.reset_index(names="date")
        .melt(id_vars=["date"], var_name="secid", value_name="short_interest_ratio")
    )
    out["secid"] = out["secid"].astype(int)
    return out[["secid", "date", "short_interest_ratio"]].reset_index(drop=True)
