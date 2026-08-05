"""PIT fioracle macro panel: available_date visibility, then causal derived features.

HY OAS note: the source column is named "Total Return Index" but values are OAS
spread levels; ingest stores them under series_id ``hy_oas``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

log = get_logger("volsurf.data.fioracle_macro")

DEFAULT_SERIES: tuple[str, ...] = (
    "vix",
    "gpri",
    "hy_oas",
    "inflation",
    "unemployment",
    "epu",
    "yield_2y",
    "term_spread",
)

FIORACLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "vix_level",
    "vix_z_252",
    "vix_chg_21",
    "hy_oas_level",
    "hy_oas_z_252",
    "hy_oas_chg_21",
    "term_spread_level",
    "epu_z_252",
    "gpri_z_252",
    "unemployment_yoy_chg",
    "inflation_yoy_level",
)


def _resolve_lake_dir(
    lake_root: Path | str,
    lake_subdir: str = "macro/fioracle",
) -> Path:
    root = Path(lake_root)
    # Allow passing the fioracle dir directly
    if (root / "vix.parquet").is_file():
        return root
    candidate = root / lake_subdir
    if candidate.is_dir() and (candidate / "vix.parquet").is_file():
        return candidate
    # Repo-local ingest fallback when the USB lake lacks fioracle parquet.
    try:
        from src.data.paths import MASCOTRL_ROOT

        alt = MASCOTRL_ROOT / "lake" / lake_subdir
        if alt.is_dir() and (alt / "vix.parquet").is_file():
            log.info("fioracle lake fallback → %s", alt)
            return alt
    except Exception:
        pass
    return candidate


def _load_series_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"event_date", "available_date", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("event_date")


def _pit_align_series(
    series_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    use_available_date: bool,
) -> pd.Series:
    """For each calendar day t, last observation with visibility_date <= t."""
    vis_col = "available_date" if use_available_date else "event_date"
    left = pd.DataFrame({"date": calendar})
    right = series_df[["event_date", "available_date", "value"]].copy()
    right = right.sort_values(vis_col)
    right = right.rename(columns={vis_col: "date"})
    merged = pd.merge_asof(
        left,
        right[["date", "value"]],
        on="date",
        direction="backward",
    )
    out = merged.set_index("date")["value"]
    out.index = calendar
    return out


def load_fioracle_macro(
    *,
    lake_root: Path | str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    series: Sequence[str] | None = None,
    as_of: pd.Timestamp | None = None,
    use_available_date: bool = True,
    lake_subdir: str = "macro/fioracle",
) -> pd.DataFrame:
    """Wide daily macro frame indexed by date.

    PIT contract: a row for date t contains only observations whose
    available_date <= t. Set use_available_date=False for the leakage
    ablation only; it must never be the default in a campaign.
    """
    del as_of  # reserved for future restatement shield; lags already in available_date
    lake_dir = _resolve_lake_dir(lake_root, lake_subdir)
    if not lake_dir.is_dir():
        raise FileNotFoundError(f"fioracle lake missing: {lake_dir}")
    ids = list(series) if series is not None else list(DEFAULT_SERIES)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    calendar = pd.date_range(start, end, freq="D")
    cols: dict[str, pd.Series] = {}
    for sid in ids:
        path = lake_dir / f"{sid}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing fioracle series parquet: {path}")
        sdf = _load_series_parquet(path)
        cols[sid] = _pit_align_series(
            sdf, calendar, use_available_date=use_available_date
        )
    wide = pd.DataFrame(cols, index=calendar)
    # Forward-fill on the calendar after PIT visibility (never bfill).
    wide = wide.sort_index().ffill()
    log.info(
        "fioracle macro rows=%d cols=%s use_available_date=%s",
        len(wide),
        list(wide.columns),
        use_available_date,
    )
    return wide


def _rolling_z(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window=window, min_periods=max(20, window // 5)).mean()
    sd = s.rolling(window=window, min_periods=max(20, window // 5)).std(ddof=0)
    z = (s - mu) / sd.where(sd >= 1e-8, np.nan)
    return z


def build_fioracle_feature_frame(levels: pd.DataFrame) -> pd.DataFrame:
    """Causal derived features (rolling / lag only; never full-sample)."""
    out = pd.DataFrame(index=levels.index)

    if "vix" in levels.columns:
        vix = levels["vix"]
        out["vix_level"] = vix
        out["vix_z_252"] = _rolling_z(vix, 252)
        out["vix_chg_21"] = np.log(vix / vix.shift(21))
    if "hy_oas" in levels.columns:
        # Spread levels (see module docstring), not a total-return index.
        oas = levels["hy_oas"]
        out["hy_oas_level"] = oas
        out["hy_oas_z_252"] = _rolling_z(oas, 252)
        out["hy_oas_chg_21"] = np.log(oas.clip(lower=1e-6) / oas.shift(21).clip(lower=1e-6))
    if "term_spread" in levels.columns:
        out["term_spread_level"] = levels["term_spread"]
    if "epu" in levels.columns:
        out["epu_z_252"] = _rolling_z(levels["epu"], 252)
    if "gpri" in levels.columns:
        out["gpri_z_252"] = _rolling_z(levels["gpri"], 252)
    if "unemployment" in levels.columns:
        u = levels["unemployment"]
        out["unemployment_yoy_chg"] = u - u.shift(252)
    if "inflation" in levels.columns:
        out["inflation_yoy_level"] = levels["inflation"]

    # Deterministic column order; only include columns that were produced
    ordered = [c for c in FIORACLE_FEATURE_COLUMNS if c in out.columns]
    return out[ordered]
