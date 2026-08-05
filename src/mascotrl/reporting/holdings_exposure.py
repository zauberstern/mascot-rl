"""Holdings-based characteristic exposures (Daniel-Grinblatt-Titman-Wermers).

Interpretation only. Never feeds capital gates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CHAR_IDS: tuple[str, ...] = (
    "log_mktcap",
    "book_to_market",
    "momentum_12_1",
    "roe",
    "realized_vol_21",
)

EXPOSURE_MEASURE_IDS: tuple[str, ...] = (
    "exposure_size",
    "exposure_value",
    "exposure_momentum",
    "exposure_quality",
    "exposure_low_vol",
    "sector_hhi",
)

_CHAR_TO_MEASURE = {
    "log_mktcap": "exposure_size",
    "book_to_market": "exposure_value",
    "momentum_12_1": "exposure_momentum",
    "roe": "exposure_quality",
    "realized_vol_21": "exposure_low_vol",
}


def _winsorize_per_date(x: np.ndarray, *, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if out.ndim != 2:
        return out
    for t in range(out.shape[0]):
        row = out[t]
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            out[t] = row
            continue
        ql, qh = np.quantile(finite, [lo, hi])
        row = np.clip(row, ql, qh)
        out[t] = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _align_long_to_wide(
    df,
    *,
    dates: Sequence,
    secids: Sequence[str],
    value_col: str,
) -> np.ndarray:
    import pandas as pd

    t = len(dates)
    k = len(secids)
    out = np.full((t, k), np.nan, dtype=np.float64)
    if df is None or getattr(df, "empty", True):
        return out
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["secid"] = d["secid"].astype(str)
    date_index = {pd.Timestamp(x).normalize(): i for i, x in enumerate(dates)}
    sec_index = {str(s): j for j, s in enumerate(secids)}
    for _, row in d.iterrows():
        di = date_index.get(pd.Timestamp(row["date"]).normalize())
        sj = sec_index.get(str(row["secid"]))
        if di is None or sj is None:
            continue
        val = row.get(value_col)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        out[di, sj] = float(val)
    return out


def _load_equity_chars(
    lake_root: Path | str,
    dates: Sequence,
    secids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Best-effort wide panels from lake equity tables."""
    import pandas as pd

    from mascotrl.data.paths import LAKE_ROOT

    lake = Path(lake_root or LAKE_ROOT)
    t, k = len(dates), len(secids)
    empty = {
        "mktcap": np.full((t, k), np.nan),
        "ret_12_1": np.full((t, k), np.nan),
        "hv_21": np.full((t, k), np.nan),
    }
    path = lake / "macro" / "sp500_sec.parquet"
    if not path.is_file():
        return empty
    try:
        cols = ["date", "secid", "close", "shrout", "stk_ret"]
        raw = pd.read_parquet(path, columns=[c for c in cols if True])
    except Exception:
        try:
            raw = pd.read_parquet(path)
        except Exception:
            return empty
    if raw.empty:
        return empty
    raw = raw.copy()
    raw["secid"] = raw["secid"].astype(str)
    raw = raw[raw["secid"].isin({str(s) for s in secids})]
    if raw.empty:
        return empty
    raw["date"] = pd.to_datetime(raw["date"])
    # Approximate features when precomputed cols absent.
    if "mktcap" not in raw.columns and {"close", "shrout"}.issubset(raw.columns):
        raw["mktcap"] = raw["close"].astype(float) * raw["shrout"].astype(float)
    if "ret_12_1" not in raw.columns and "stk_ret" in raw.columns:
        raw = raw.sort_values(["secid", "date"])
        g = raw.groupby("secid", group_keys=False)
        r = raw["stk_ret"].astype(float)
        raw["ret_12_1"] = g["stk_ret"].transform(
            lambda s: (1.0 + s.astype(float)).rolling(252, min_periods=60).apply(
                np.prod, raw=True
            )
            / (1.0 + s.astype(float)).rolling(21, min_periods=5).apply(np.prod, raw=True)
            - 1.0
        )
        del r
    if "hv_21" not in raw.columns and "stk_ret" in raw.columns:
        raw = raw.sort_values(["secid", "date"])
        raw["hv_21"] = (
            raw.groupby("secid")["stk_ret"]
            .transform(lambda s: s.astype(float).rolling(21, min_periods=10).std())
        )
    out = {}
    for col in ("mktcap", "ret_12_1", "hv_21"):
        if col in raw.columns:
            out[col] = _align_long_to_wide(
                raw[["date", "secid", col]], dates=dates, secids=secids, value_col=col
            )
        else:
            out[col] = np.full((t, k), np.nan)
    return out


def _load_fundamental_chars(
    lake_root: Path | str,
    dates: Sequence,
    secids: Sequence[str],
) -> dict[str, np.ndarray]:
    t, k = len(dates), len(secids)
    empty = {"bm": np.full((t, k), np.nan), "roe": np.full((t, k), np.nan)}
    try:
        from mascotrl.data.feature_panels import load_ibes_ratios_long

        start = str(dates[0])[:10]
        end = str(dates[-1])[:10]
        df = load_ibes_ratios_long(lake_root, start, end)
    except Exception:
        return empty
    if df is None or getattr(df, "empty", True):
        return empty
    out = {}
    for col in ("bm", "roe"):
        if col in df.columns:
            out[col] = _align_long_to_wide(
                df[["date", "secid", col]], dates=dates, secids=secids, value_col=col
            )
        else:
            out[col] = np.full((t, k), np.nan)
    return out


def _load_gics_onehot(
    lake_root: Path | str,
    secids: Sequence[str],
    *,
    n_sectors: int = 11,
) -> np.ndarray:
    """Return (K, S) one-hot (or soft) GICS sector membership."""
    k = len(secids)
    try:
        from mascotrl.data.feature_panels import load_gics_map

        gics = load_gics_map(lake_root)
    except Exception:
        return np.zeros((k, n_sectors), dtype=np.float64)
    if gics is None or getattr(gics, "empty", True):
        return np.zeros((k, n_sectors), dtype=np.float64)
    g = gics.copy()
    g["secid"] = g["secid"].astype(str)
    industries = (
        g["gics_industry"].astype(str).fillna("unknown").tolist()
        if "gics_industry" in g.columns
        else []
    )
    # Collapse to at most n_sectors buckets by frequency.
    from collections import Counter

    counts = Counter(industries)
    top = [name for name, _ in counts.most_common(n_sectors)]
    if not top:
        return np.zeros((k, n_sectors), dtype=np.float64)
    idx = {name: i for i, name in enumerate(top)}
    s = len(top)
    out = np.zeros((k, s), dtype=np.float64)
    by_sec = {
        str(r["secid"]): str(r.get("gics_industry") or "unknown")
        for _, r in g.iterrows()
    }
    for j, sid in enumerate(secids):
        ind = by_sec.get(str(sid), "unknown")
        if ind in idx:
            out[j, idx[ind]] = 1.0
    return out


def load_characteristic_panel(
    dates: Sequence,
    secids: Sequence[str],
    lake_root: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Return (T,K) characteristic panels + (T,K,S) GICS one-hot broadcast."""
    from mascotrl.data.paths import LAKE_ROOT

    lake = Path(lake_root or LAKE_ROOT)
    dates = list(dates)
    secids = [str(s) for s in secids]
    t, k = len(dates), len(secids)
    if t < 1 or k < 1:
        return {
            "log_mktcap": np.zeros((0, 0)),
            "book_to_market": np.zeros((0, 0)),
            "momentum_12_1": np.zeros((0, 0)),
            "roe": np.zeros((0, 0)),
            "realized_vol_21": np.zeros((0, 0)),
            "gics_onehot": np.zeros((0, 0, 0)),
        }

    eq = _load_equity_chars(lake, dates, secids)
    fund = _load_fundamental_chars(lake, dates, secids)
    gics_k = _load_gics_onehot(lake, secids)

    mktcap = np.asarray(eq["mktcap"], dtype=np.float64)
    log_mktcap = np.log(np.clip(mktcap, 1e-8, None))
    log_mktcap = _winsorize_per_date(log_mktcap)
    bm = _winsorize_per_date(np.asarray(fund["bm"], dtype=np.float64))
    mom = _winsorize_per_date(np.asarray(eq["ret_12_1"], dtype=np.float64))
    roe = _winsorize_per_date(np.asarray(fund["roe"], dtype=np.float64))
    vol = _winsorize_per_date(np.asarray(eq["hv_21"], dtype=np.float64))
    # Broadcast static GICS over T.
    gics = np.broadcast_to(gics_k[None, :, :], (t, k, gics_k.shape[1])).copy()
    return {
        "log_mktcap": log_mktcap,
        "book_to_market": bm,
        "momentum_12_1": mom,
        "roe": roe,
        "realized_vol_21": vol,
        "gics_onehot": gics,
    }


def holdings_exposures(
    weights: np.ndarray,
    char_panels: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Weight-averaged characteristic exposures + sector concentration HHI."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    t, k = w.shape
    out: dict[str, float] = {m: float("nan") for m in EXPOSURE_MEASURE_IDS}
    for cid, mid in _CHAR_TO_MEASURE.items():
        c = np.asarray(char_panels.get(cid), dtype=np.float64) if cid in char_panels else None
        if c is None or c.shape != (t, k):
            continue
        # Low-vol exposure: invert vol so higher = more defensive.
        if cid == "realized_vol_21":
            c = -c
        expo = np.nansum(w * c, axis=1)
        out[mid] = float(np.nanmean(expo)) if expo.size else float("nan")

    gics = char_panels.get("gics_onehot")
    if gics is not None:
        g = np.asarray(gics, dtype=np.float64)
        if g.ndim == 3 and g.shape[0] == t and g.shape[1] == k:
            # sector weights (T, S)
            sw = np.einsum("tk,tks->ts", w, g)
            hhi = np.sum(sw * sw, axis=1)
            out["sector_hhi"] = float(np.nanmean(hhi)) if hhi.size else float("nan")
    return out


def nan_exposures(*, reason: str = "secids_unavailable") -> dict[str, Any]:
    """Stamp NaN exposures with an availability reason (older artifacts)."""
    return {
        **{m: float("nan") for m in EXPOSURE_MEASURE_IDS},
        "data_availability_reason": reason,
    }
