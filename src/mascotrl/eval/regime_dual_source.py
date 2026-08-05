"""Dual-source H.15 backups for term spread / HY OAS (eval honesty only).

Never mixes fioracle and H.15 into one silent average. Lag-1 so values are
known at t. Does not admit series onto the confirmatory train cube.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_h15_term_oas(
    usb_root: Path | str,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Load USB H.15 candidates ``t10y2y`` and ``bamlh0a0hym2``, lag-1.

    Returns dict with keys ``term_spread``, ``hy_oas`` (Series or None) and
    ``source`` stamps.
    """
    usb = Path(usb_root)
    path = usb / "macro" / "interest_rate.parquet"
    out: dict[str, Any] = {
        "term_spread": None,
        "hy_oas": None,
        "source": {"term": None, "hy_oas": None},
    }
    if not path.is_file():
        return out
    try:
        df = pd.read_parquet(path)
    except Exception:
        return out
    lower = {str(c).lower(): c for c in df.columns}
    date_col = lower.get("date") or ("date" if "date" in df.columns else None)
    if date_col is None:
        return out
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    work = work.set_index(date_col)

    def _series(candidates: tuple[str, ...]) -> pd.Series | None:
        for name in candidates:
            col = lower.get(name.lower())
            if col is None:
                continue
            s = pd.to_numeric(work[col], errors="coerce")
            s = s.reindex(pd.DatetimeIndex(dates))
            # Lag 1 trading day: value at t = yesterday's observation.
            s = s.shift(1)
            if s.notna().sum() < 10:
                return None
            return s.astype(np.float64)

    term = _series(("t10y2y", "T10Y2Y", "term_spread"))
    oas = _series(("bamlh0a0hym2", "BAMLH0A0HYM2", "hy_oas"))
    if term is not None:
        out["term_spread"] = term
        out["source"]["term"] = "h15"
    if oas is not None:
        out["hy_oas"] = oas
        out["source"]["hy_oas"] = "h15"
    return out


def resolve_macro_yt_cols(
    macro: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    *,
    usb_root: Path | str | None = None,
) -> tuple[np.ndarray | None, dict[str, str]]:
    """Build (T, m) macro_cols for turbulence with source stamps.

    Prefer fioracle levels; fall back to H.15 lag-1 for missing term/OAS only.
    Never averages both sources into one column.
    """
    sources: dict[str, str] = {}
    if macro is None or len(macro) == 0:
        return None, sources
    n = len(dates)
    cols: list[np.ndarray] = []

    def _from_macro(level: str, chg: str) -> np.ndarray | None:
        if level in macro.columns:
            a = macro[level].to_numpy(dtype=np.float64)
            if a.shape[0] == n and np.isfinite(a).sum() >= 10:
                return a
        if chg in macro.columns:
            a = macro[chg].to_numpy(dtype=np.float64)
            if a.shape[0] == n and np.isfinite(a).sum() >= 10:
                return a
        return None

    vix = _from_macro("vix_level", "vix_chg_21")
    if vix is not None:
        cols.append(vix)
        sources["vix"] = "fioracle"

    hy = _from_macro("hy_oas_level", "hy_oas_chg_21")
    term = _from_macro("term_spread_level", "term_spread_chg_21")

    h15 = None
    if (hy is None or term is None) and usb_root is not None:
        h15 = load_h15_term_oas(usb_root, dates)

    if hy is not None:
        cols.append(hy)
        sources["hy_oas"] = "fioracle"
    elif h15 is not None and h15.get("hy_oas") is not None:
        cols.append(h15["hy_oas"].to_numpy(dtype=np.float64))
        sources["hy_oas"] = "h15"

    if term is not None:
        cols.append(term)
        sources["term"] = "fioracle"
    elif h15 is not None and h15.get("term_spread") is not None:
        cols.append(h15["term_spread"].to_numpy(dtype=np.float64))
        sources["term"] = "h15"

    if not cols:
        return None, sources
    return np.column_stack(cols), sources
