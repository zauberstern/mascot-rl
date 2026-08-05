"""Equity-factor confrontation for spectrum arms (market / size / value / mom).

Option-factor models (HVX / BCSZ) are the wrong benchmark for an equity book.
Gate 2 for ``eq`` / ``mix`` must confront strategy returns with equity factors.
Minimum viable set is market excess return from the lake CRSP series already
used by :mod:`src.eval.orientation_benchmarks`. SMB / HML / Mom are attached
when ``macro/ff_factors.parquet`` is present; otherwise they are omitted and
recorded on the DataFrame ``attrs['note']``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.paths import LAKE_ROOT
from src.eval.factor_alpha import factor_alpha, hlz_hurdles
from src.eval.orientation_benchmarks import (
    load_cash_daily_returns,
    load_equity_daily_returns,
)
from src.logging_utils import get_logger

log = get_logger("mascotrl.eval.equity_factors")

# Ken French aliases → short names. Magnitude heuristic below handles percent
# vs decimal units depending on how the parquet was written.
_FF_COL_MAP = {
    "mkt": ("Mkt-RF", "Mkt_RF", "mkt", "MKT"),
    "smb": ("SMB", "smb"),
    "hml": ("HML", "hml"),
    "mom": ("Mom", "MOM", "WML", "mom"),
}


def _resolve_lake(lake_or_path: Path | str | None) -> Path:
    if lake_or_path is None:
        return Path(LAKE_ROOT)
    return Path(lake_or_path)


def _try_load_ff_panel(lake: Path) -> pd.DataFrame | None:
    """Load Ken French FF5+Mom panel if present and non-empty."""
    path = lake / "macro" / "ff_factors.parquet"
    if not path.is_file():
        return None
    try:
        ff = pd.read_parquet(path)
    except Exception as exc:
        log.warning("ff_factors unreadable at %s: %s", path, exc)
        return None
    if ff is None or len(ff) == 0:
        return None
    if "date" not in ff.columns:
        return None
    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index("date").sort_index()
    # Detect percent units (Ken French raw) vs already-decimal.
    sample_cols = [c for c in ff.columns if c in ("Mkt-RF", "SMB", "HML", "Mom")]
    if sample_cols:
        med = float(np.nanmedian(np.abs(ff[sample_cols[0]].to_numpy(dtype=np.float64))))
        if med > 0.5:  # typical daily |Mkt-RF| in percent ≈ 0.5–1
            for c in ff.columns:
                ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0
    return ff


def _pick_ff_series(ff: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    for name in aliases:
        if name in ff.columns and ff[name].notna().any():
            s = pd.to_numeric(ff[name], errors="coerce").astype(np.float64)
            s.name = name
            return s
    return None


def build_equity_factors(
    dates: Sequence[Any] | pd.DatetimeIndex,
    *,
    lake_or_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build an equity-factor panel aligned to ``dates``.

    Always attempts ``mkt`` (market excess). Optional ``smb`` / ``hml`` / ``mom``
    come from ``macro/ff_factors.parquet`` when available. Missing optional legs
    are omitted (not fabricated); ``attrs['note']`` documents the gap.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    lake = _resolve_lake(lake_or_path)
    notes: list[str] = []
    cols: dict[str, np.ndarray] = {}

    equity = load_equity_daily_returns(lake)
    equity = equity.copy()
    equity.index = pd.to_datetime(equity.index)
    eq_aligned = equity.reindex(idx).astype(np.float64)

    rf_aligned = None
    try:
        cash = load_cash_daily_returns(lake)
        cash = cash.copy()
        cash.index = pd.to_datetime(cash.index)
        rf_aligned = cash.reindex(idx).astype(np.float64)
    except Exception as exc:
        notes.append(f"RF unavailable ({exc}); mkt uses raw market return not excess")

    if rf_aligned is not None:
        mkt = eq_aligned.to_numpy() - rf_aligned.to_numpy()
        notes.append(
            f"mkt = {equity.name} minus cash RF ({getattr(cash, 'name', 'rf')})"
        )
    else:
        mkt = eq_aligned.to_numpy()
    cols["mkt"] = mkt

    ff = _try_load_ff_panel(lake)
    if ff is None:
        notes.append(
            "smb/hml/mom omitted — macro/ff_factors.parquet missing or empty; "
            "market-only panel is MVP until Ken French factors are downloaded"
        )
    else:
        # Prefer Ken French Mkt-RF when present (cleaner excess definition).
        ff_mkt = _pick_ff_series(ff, _FF_COL_MAP["mkt"])
        if ff_mkt is not None:
            cols["mkt"] = ff_mkt.reindex(idx).to_numpy(dtype=np.float64)
            notes.append("mkt replaced by Ken French Mkt-RF from ff_factors.parquet")
        for short, aliases in (
            ("smb", _FF_COL_MAP["smb"]),
            ("hml", _FF_COL_MAP["hml"]),
            ("mom", _FF_COL_MAP["mom"]),
        ):
            series = _pick_ff_series(ff, aliases)
            if series is None:
                notes.append(f"{short} omitted — column not in ff_factors.parquet")
                continue
            cols[short] = series.reindex(idx).to_numpy(dtype=np.float64)

    out = pd.DataFrame(cols, index=idx)
    out.attrs["note"] = "; ".join(notes) if notes else "equity factors ok"
    out.attrs["source_lake"] = str(lake)
    return out


def attach_equity_factor_suite(
    strategy_returns: pd.Series,
    factor_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Regress strategy returns on equity factors; report HAC alpha + HLZ hurdles.

    Reuses :func:`src.eval.factor_alpha.factor_alpha` and
    :func:`src.eval.factor_alpha.hlz_hurdles`.
    """
    y = pd.Series(strategy_returns).astype(np.float64)
    y.index = pd.to_datetime(y.index)
    fac = factor_df.copy()
    fac.index = pd.to_datetime(fac.index)
    aligned = pd.concat([y.rename("_y"), fac], axis=1, join="inner")
    if aligned.empty:
        return {
            "ok": False,
            "reason": "no overlapping dates between strategy and factors",
            "note": factor_df.attrs.get("note"),
        }

    factors = {
        c: aligned[c].tolist()
        for c in fac.columns
        if c in aligned.columns and aligned[c].notna().any()
    }
    alpha = factor_alpha(aligned["_y"].tolist(), factors)
    t_hac = alpha.get("alpha_t_hac") if alpha.get("ok") else None
    return {
        "ok": bool(alpha.get("ok")),
        "model": "equity_FF_proxy",
        "reference": "Fama–French + Carhart; lake vwretd/sprtrn MVP",
        "alpha": alpha,
        "hlz": hlz_hurdles(t_hac if isinstance(t_hac, (int, float)) else None),
        "factors_used": list(factors.keys()),
        "n": int(len(aligned)),
        "note": factor_df.attrs.get("note"),
        "citation": "Newey and West (1987); Harvey, Liu and Zhu (2016)",
    }
