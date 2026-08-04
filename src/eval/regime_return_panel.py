"""Confirmatory return panels for Kritzman / Ch.10 turbulence.

Priority: style-desk panel (campaign y_t) > KPT 10-sector equal-weight from USB
CRSP SICCD. Densest-PERMNO cross-section is legacy diagnostics only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# KPT used 10 S&P 500 sector indices; CRSP SICCD collapsed to 10 buckets.
# Frozen map — not tuned on COVID. Real estate folded into utilities_real_estate.
KPT10_NAMES: tuple[str, ...] = (
    "energy",
    "materials",
    "industrials",
    "consumer_discretionary",
    "consumer_staples",
    "health_care",
    "financials",
    "info_tech",
    "communication",
    "utilities_real_estate",
)

# 2-digit SIC -> KPT10 name (approximate GICS-like collapse; frozen).
SIC_TO_KPT10: dict[int, str] = {
    # Energy / oil & gas / mining fuels
    **{s: "energy" for s in (10, 12, 13, 14, 29)},
    # Materials / chemicals / metals / paper
    **{s: "materials" for s in (8, 9, 24, 26, 32, 33)},
    # Industrials / construction / transport / machinery
    **{s: "industrials" for s in (15, 16, 17, 34, 37, 40, 41, 42, 44, 45, 47)},
    # Consumer discretionary / retail / autos / apparel
    **{s: "consumer_discretionary" for s in (23, 25, 30, 31, 39, 50, 51, 52, 53, 55, 56, 57, 58, 59)},
    # Consumer staples / food / tobacco / grocery
    **{s: "consumer_staples" for s in (1, 2, 7, 20, 21, 54)},
    # Health care / drugs / medical
    **{s: "health_care" for s in (28, 80, 87)},
    # Financials
    **{s: "financials" for s in (60, 61, 62, 63, 64)},
    # Info tech / electronics / software / instruments
    **{s: "info_tech" for s in (35, 36, 38, 73)},
    # Communication / media / broadcasting
    **{s: "communication" for s in (27, 48, 78, 79)},
    # Utilities / REITs / lodging (real estate folded here)
    **{s: "utilities_real_estate" for s in (49, 65, 67, 70)},
}

# Frozen LSEG TR.GICSSector -> KPT10 (not tuned on COVID). Real estate folded.
GICS_SECTOR_TO_KPT10: dict[str, str] = {
    "energy": "energy",
    "materials": "materials",
    "industrials": "industrials",
    "consumer discretionary": "consumer_discretionary",
    "consumer staples": "consumer_staples",
    "health care": "health_care",
    "financials": "financials",
    "information technology": "info_tech",
    "communication services": "communication",
    "utilities": "utilities_real_estate",
    "real estate": "utilities_real_estate",
}

GICS_COVERAGE_GATE = 0.80
GICS_FINITE_GATE = 0.50


def _sic2(raw: Any) -> int | None:
    try:
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            return None
        s = str(raw).strip()
        if not s or s.lower() == "none":
            return None
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) < 2:
            return None
        return int(digits[:2])
    except (TypeError, ValueError):
        return None


def _crsp_ret_to_float(series: pd.Series) -> pd.Series:
    """CRSP RET: leading-minus codes (-66,-77,-99) -> NaN; never invent 0."""
    out = pd.to_numeric(series, errors="coerce")
    bad = out.isin((-66.0, -77.0, -99.0))
    return out.mask(bad, np.nan).astype(np.float64)


def load_kpt_sector_returns(
    usb_root: Path | str,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Equal-weight 10-sector daily returns aligned to ``dates``.

    Source: USB ``macro/sp500_prices.parquet`` with PERMNO / RET / SICCD.
    """
    usb = Path(usb_root)
    path = usb / "macro" / "sp500_prices.parquet"
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    lower = {str(c).lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in df.columns:
                return name
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    date_col = pick("date", "Date", "datadate")
    ret_col = pick("RET", "ret", "RETX")
    id_col = pick("PERMNO", "permno")
    sic_col = pick("SICCD", "siccd", "sic")
    if date_col is None or ret_col is None or id_col is None or sic_col is None:
        return None

    work = df[[date_col, id_col, ret_col, sic_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[ret_col] = _crsp_ret_to_float(work[ret_col])
    work["sic2"] = work[sic_col].map(_sic2)
    work["sector"] = work["sic2"].map(
        lambda x: SIC_TO_KPT10.get(int(x)) if x is not None else None
    )
    work = work.dropna(subset=[date_col, "sector"])
    d0 = pd.Timestamp(dates.min())
    d1 = pd.Timestamp(dates.max())
    work = work[(work[date_col] >= d0) & (work[date_col] <= d1)]
    if work.empty:
        return None

    def _ew_mean(s: pd.Series) -> float:
        a = s.to_numpy(dtype=np.float64)
        n_ok = int(np.isfinite(a).sum())
        if n_ok < 3:
            return float("nan")
        return float(np.nanmean(a))

    means = (
        work.groupby([date_col, "sector"], sort=False)[ret_col]
        .apply(_ew_mean)
        .reset_index()
    )
    means.columns = [date_col, "sector", "ret"]
    wide = means.pivot_table(
        index=date_col, columns="sector", values="ret", aggfunc="last"
    )
    for name in KPT10_NAMES:
        if name not in wide.columns:
            wide[name] = np.nan
    wide = wide[list(KPT10_NAMES)].sort_index()
    wide = wide.reindex(pd.DatetimeIndex(dates))
    arr = wide.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 100 or arr.shape[1] != 10:
        return None
    meta = {
        "source": "usb_sp500_sic10",
        "n_permno": int(work[id_col].nunique()),
        "n_rows": int(arr.shape[0]),
        "n_sectors": 10,
        "finite_frac": float(np.isfinite(arr).mean()),
        "calendar_start": str(pd.Timestamp(dates.min()).date()),
        "calendar_end": str(pd.Timestamp(dates.max()).date()),
    }
    return arr, meta


def _normalize_gics_sector(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return None
    key = str(raw).strip().lower()
    if not key or key in ("none", "nan"):
        return None
    return GICS_SECTOR_TO_KPT10.get(key)


def load_kpt_gics_sector_returns(
    usb_root: Path | str,
    dates: pd.DatetimeIndex,
    *,
    min_mapped_frac: float = GICS_COVERAGE_GATE,
    min_finite_frac: float = GICS_FINITE_GATE,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Equal-weight 10-sector returns via LSEG GICS sector (static as-of map).

    Join: PERMNO -> OptionMetrics link secid -> lseg_ric_map TR.GICSSector.
    Returns None when coverage gates fail (caller falls back to SIC10).
    """
    usb = Path(usb_root)
    prices_path = usb / "macro" / "sp500_prices.parquet"
    link_path = usb / "macro" / "crsp_optionm_link.parquet"
    ric_path = usb / "macro" / "lseg_ric_map.parquet"
    if not (prices_path.is_file() and link_path.is_file() and ric_path.is_file()):
        return None
    try:
        df = pd.read_parquet(prices_path)
        link = pd.read_parquet(link_path)
        ric = pd.read_parquet(ric_path)
    except Exception:
        return None

    lower = {str(c).lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in df.columns:
                return name
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    date_col = pick("date", "Date", "datadate")
    ret_col = pick("RET", "ret", "RETX")
    id_col = pick("PERMNO", "permno")
    if date_col is None or ret_col is None or id_col is None:
        return None
    if "permno" not in {str(c).lower() for c in link.columns}:
        return None
    if "secid" not in {str(c).lower() for c in link.columns}:
        return None
    gics_col = None
    for c in ric.columns:
        if str(c).lower() in ("tr.gicssector", "gicssector", "gics_sector"):
            gics_col = c
            break
    if gics_col is None:
        return None
    sec_col = None
    for c in ric.columns:
        if str(c).lower() == "secid":
            sec_col = c
            break
    if sec_col is None:
        return None

    link_p = link.copy()
    link_p["permno"] = pd.to_numeric(link_p["permno"], errors="coerce")
    link_p["secid"] = link_p["secid"].astype(str)
    if "score" in link_p.columns:
        link_p = link_p.sort_values("score")
    link_p = link_p.dropna(subset=["permno"]).drop_duplicates("permno", keep="first")

    ric_p = ric[[sec_col, gics_col]].copy()
    ric_p["secid"] = ric_p[sec_col].astype(str)
    ric_p["gics_raw"] = ric_p[gics_col]
    asof = None
    if "asof_ts" in ric.columns:
        try:
            asof = str(pd.to_datetime(ric["asof_ts"]).max())
        except Exception:
            asof = str(ric["asof_ts"].iloc[0]) if len(ric) else None

    work = df[[date_col, id_col, ret_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[ret_col] = _crsp_ret_to_float(work[ret_col])
    work["permno"] = pd.to_numeric(work[id_col], errors="coerce")
    n_permno_total = int(work["permno"].nunique())
    work = work.merge(link_p[["permno", "secid"]], on="permno", how="left")
    work = work.merge(ric_p[["secid", "gics_raw"]], on="secid", how="left")
    work["sector"] = work["gics_raw"].map(_normalize_gics_sector)
    mapped = work.dropna(subset=["sector"])
    n_permno_mapped = int(mapped["permno"].nunique())
    mapped_frac = (
        float(n_permno_mapped / n_permno_total) if n_permno_total > 0 else 0.0
    )
    if mapped_frac < float(min_mapped_frac):
        return None

    d0 = pd.Timestamp(dates.min())
    d1 = pd.Timestamp(dates.max())
    mapped = mapped[(mapped[date_col] >= d0) & (mapped[date_col] <= d1)]
    if mapped.empty:
        return None

    def _ew_mean(s: pd.Series) -> float:
        a = s.to_numpy(dtype=np.float64)
        n_ok = int(np.isfinite(a).sum())
        if n_ok < 3:
            return float("nan")
        return float(np.nanmean(a))

    means = (
        mapped.groupby([date_col, "sector"], sort=False)[ret_col]
        .apply(_ew_mean)
        .reset_index()
    )
    means.columns = [date_col, "sector", "ret"]
    wide = means.pivot_table(
        index=date_col, columns="sector", values="ret", aggfunc="last"
    )
    for name in KPT10_NAMES:
        if name not in wide.columns:
            wide[name] = np.nan
    wide = wide[list(KPT10_NAMES)].sort_index()
    wide = wide.reindex(pd.DatetimeIndex(dates))
    arr = wide.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 100 or arr.shape[1] != 10:
        return None
    finite_frac = float(np.isfinite(arr).mean())
    if finite_frac < float(min_finite_frac):
        return None
    meta = {
        "source": "usb_sp500_gics10",
        "n_permno": n_permno_total,
        "n_permno_mapped": n_permno_mapped,
        "permno_mapped_frac": mapped_frac,
        "n_rows": int(arr.shape[0]),
        "n_sectors": 10,
        "finite_frac": finite_frac,
        "gics_asof": asof,
        "calendar_start": str(pd.Timestamp(dates.min()).date()),
        "calendar_end": str(pd.Timestamp(dates.max()).date()),
        "limitation": "GICS sector map is static as-of, not CRSP history",
    }
    return arr, meta


def load_style_desk_returns(
    path: Path | str | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Load (T, K) desk/panel returns from JSON (assemble_regime_desk / campaign)."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    import json

    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict):
        return None
    pr = blob.get("panel_returns")
    if pr is None and isinstance(blob.get("runner_artifact"), dict):
        pr = blob["runner_artifact"].get("panel_returns")
    if pr is None:
        er = blob.get("expert_returns")
        if isinstance(er, dict) and er:
            cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in er.values()]
            if cols and all(c.size == cols[0].size for c in cols):
                arr = np.column_stack(cols)
                if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                    return arr, {
                        "source": "desk_expert_returns",
                        "n_rows": int(arr.shape[0]),
                        "n_assets": int(arr.shape[1]),
                        "path": str(p),
                    }
        return None
    arr = np.asarray(pr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return arr, {
        "source": "desk",
        "n_rows": int(arr.shape[0]),
        "n_assets": int(arr.shape[1]),
        "path": str(p),
    }
