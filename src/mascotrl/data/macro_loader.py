"""Load lake / ArcticDB macro features into torch tensors (no synthetic noise)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from mascotrl.data.arctic_store import ArcticStateStore
from mascotrl.data.duckdb_engine import DuckDBFeatureEngine
from mascotrl.data.paths import ARCTIC_ROOT
from mascotrl.logging_utils import get_logger

log = get_logger("volsurf.l5.macro")

MACRO_SYMBOL = "macro_state"


_RATE_COLS = ("sofr", "effr", "dtb3")


def _ffill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strict temporal fill — never Gaussian imputation, never bfill.

    Rate columns (SOFR/EFFR/DTB3) may be null before series inception (SOFR
    ~2018-04). Those nulls must **not** truncate the VIX calendar via
    ``dropna(how='any')``. Leading rate nulls are filled with 0.0 after ffill
    (documented feature: "rate unavailable" → zero before z-score).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"], errors="ignore")
        else:
            raise TypeError("macro frame needs DatetimeIndex or date column")
    df = df.sort_index().replace([np.inf, -np.inf], np.nan).ffill()
    rate_present = [c for c in _RATE_COLS if c in df.columns]
    core = df.drop(columns=rate_present, errors="ignore")
    core = core.dropna(how="any")
    if core.empty:
        raise ValueError("macro frame empty after strict ffill/dropna")
    if rate_present:
        rates = df.loc[core.index, rate_present].ffill()
        # Pre-inception leading NaNs → 0 (not bfill — no lookahead).
        rates = rates.fillna(0.0)
        core = core.join(rates, how="left")
        log.info(
            "macro rates joined cols=%s; leading pre-series nulls→0 (no bfill)",
            rate_present,
        )
    return core


def _frame_to_tensor(
    df: pd.DataFrame,
    macro_dim: int,
    *,
    causal_zscore: bool = True,
    min_periods: int = 60,
) -> torch.Tensor:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        raise ValueError("macro frame has no numeric columns")
    X = num.to_numpy(dtype=np.float64)
    if causal_zscore:
        # Expanding (causal) standardization: row t uses only rows <= t, so a
        # backtest cannot standardize with moments computed from its own future.
        # A full-window mean/std is in-sample normalization and is the classic
        # subtle leak in walk-forward evaluations.
        frame = pd.DataFrame(X)
        mu = frame.expanding(min_periods=1).mean()
        sd = frame.expanding(min_periods=1).std(ddof=0)
        Z = (frame - mu) / sd.where(sd >= 1e-8, 1.0)
        # Before enough history accumulates the estimate is not meaningful;
        # hold those rows at zero rather than emitting noise.
        Z.iloc[: max(0, int(min_periods) - 1)] = 0.0
        X = Z.to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = (X - mu) / sd
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in macro tensor after z-score (ffill incomplete)")

    T, F = X.shape
    out = np.zeros((T, macro_dim), dtype=np.float32)
    out[:, : min(F, macro_dim)] = X[:, : min(F, macro_dim)]
    # Pad unused dims with lagged first factor (deterministic), never randn.
    if F < macro_dim and F > 0:
        for i in range(F, macro_dim):
            lag = min(i - F + 1, T - 1)
            out[lag:, i] = out[: T - lag, 0] if T > lag else out[:, 0]
    return torch.from_numpy(out)


def _duckdb_macro_frame(
    lake_base_dir: str | Path,
    start_date: str,
    end_date: str,
    duckdb_threads: int,
    max_memory: str,
) -> pd.DataFrame:
    eng = DuckDBFeatureEngine(lake_base_dir=lake_base_dir)
    eng.con.execute(f"SET threads = {int(duckdb_threads)};")
    eng.con.execute(f"SET max_memory = '{max_memory}';")
    table = eng.compute_macro_state(start_date, end_date)
    df = table.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.has_duplicates:
        # Belt-and-suspenders: lake may still emit dup event dates after joins.
        n_dup = int(df.index.duplicated().sum())
        log.warning("macro frame had %d duplicate event dates — keeping last", n_dup)
        df = df[~df.index.duplicated(keep="last")].sort_index()
    try:
        disp = eng.compute_iv_dispersion(start_date, end_date)
        if disp.num_rows > 0 and "iv_dispersion" in disp.column_names:
            ddf = disp.to_pandas()
            ddf["date"] = pd.to_datetime(ddf["date"])
            ddf = ddf.set_index("date").sort_index()[["iv_dispersion"]]
            if ddf.index.has_duplicates:
                ddf = ddf[~ddf.index.duplicated(keep="last")].sort_index()
            df = df.join(ddf, how="left")
            log.info("merged iv_dispersion into macro rows=%d", len(df))
    except Exception as exc:
        log.warning("iv_dispersion skip: %s", exc)
    return _ffill_frame(df)


def _maybe_join_fioracle(
    df: pd.DataFrame,
    *,
    lake_base_dir: str | Path,
    start_date: str,
    end_date: str,
    fioracle_enabled: bool,
    fioracle_lake_subdir: str,
    fioracle_series: list[str] | None,
) -> pd.DataFrame:
    """Concatenate PIT fioracle derived features onto the base macro frame.

    Config hook (workflow YAML)::

        feature_extras:
          fioracle_macro:
            enabled: true
            lake_subdir: macro/fioracle
    """
    if not fioracle_enabled:
        return df
    from mascotrl.data.fioracle_macro import (
        DEFAULT_SERIES,
        build_fioracle_feature_frame,
        load_fioracle_macro,
    )

    levels = load_fioracle_macro(
        lake_root=lake_base_dir,
        start_date=start_date,
        end_date=end_date,
        series=fioracle_series if fioracle_series is not None else list(DEFAULT_SERIES),
        use_available_date=True,
        lake_subdir=fioracle_lake_subdir,
    )
    feats = build_fioracle_feature_frame(levels)
    # Align to base index; ffill only (already PIT-gated inside load_fioracle_macro)
    feats = feats.reindex(df.index).ffill()
    overlap = [c for c in feats.columns if c in df.columns]
    if overlap:
        raise ValueError(f"fioracle feature column collision with base macro: {overlap}")
    joined = df.join(feats, how="left")
    # Leading NaNs in fioracle block → 0 after join (same posture as pre-inception rates)
    fio_cols = list(feats.columns)
    joined[fio_cols] = joined[fio_cols].ffill().fillna(0.0)
    log.info(
        "joined fioracle macro features n=%d cols=%s",
        len(fio_cols),
        fio_cols,
    )
    return joined


def load_macro_tensor(
    lake_base_dir: str | Path,
    start_date: str,
    end_date: str,
    macro_dim: int,
    duckdb_threads: int = 2,
    max_memory: str = "4GB",
    arctic_db_path: str | Path | None = None,
    arctic_library: str = "hyper_volanet_features",
    prefer_arctic: bool = True,
    knowledge_time: str | pd.Timestamp | None = ...,  # type: ignore[assignment]
    fioracle_enabled: bool = False,
    fioracle_lake_subdir: str = "macro/fioracle",
    fioracle_series: list[str] | None = None,
    return_meta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Returns float32 tensor [T_days, macro_dim].

    Layer-5 laws
    ------------
    * NEVER impute with ``torch.randn`` / white noise.
    * Prefer ArcticDB bitemporal read: knowledge ``as_of`` + event-time ffill.
    * ``knowledge_time``:
        - default (Ellipsis): try ``as_of=end_date`` then latest (offline lakes)
        - ``None``: force latest library revision
        - Timestamp: strict knowledge PIT (WFO restatement shield)
    * Total failure raises ``RuntimeError`` (no synthetic macro).
    * Optional ``fioracle_enabled`` concatenates PIT fioracle derived features
      before ``_frame_to_tensor`` (see ``feature_extras.fioracle_macro``).
    * When ``return_meta=True``, also returns ``{"macro_column_order": [...], ...}``.
    """
    arctic_path = Path(arctic_db_path) if arctic_db_path else ARCTIC_ROOT
    store: ArcticStateStore | None = None
    try:
        store = ArcticStateStore(db_path=arctic_path, library_name=arctic_library)
    except Exception as exc:
        log.warning("ArcticDB unavailable (%s); will use DuckDB only", exc)
        store = None

    df: pd.DataFrame | None = None
    end_ts = pd.Timestamp(end_date)

    if prefer_arctic and store is not None and MACRO_SYMBOL in store.list_available_features():
        try:
            if knowledge_time is ...:
                # Restatement-safe default with offline-lake fallback inside helper.
                df = store.read_ffill(
                    MACRO_SYMBOL, start=start_date, end=end_date, as_of=end_ts
                )
            elif knowledge_time is None:
                df = store.read_ffill(
                    MACRO_SYMBOL, start=start_date, end=end_date, as_of=None
                )
            else:
                df = store.read_ffill(
                    MACRO_SYMBOL,
                    start=start_date,
                    end=end_date,
                    as_of=pd.Timestamp(knowledge_time),
                )
            log.info(
                "macro from ArcticDB as_of=%s event_end=%s rows=%d cols=%s",
                knowledge_time if knowledge_time is not ... else end_date,
                end_date,
                len(df),
                list(df.columns),
            )
        except Exception as exc:
            if knowledge_time is ...:
                try:
                    df = store.read_ffill(
                        MACRO_SYMBOL, start=start_date, end=end_date, as_of=None
                    )
                    log.info(
                        "macro from ArcticDB latest revision (as_of=end missed: %s) "
                        "event_end=%s rows=%d",
                        exc,
                        end_date,
                        len(df),
                    )
                except Exception as exc2:
                    log.warning("ArcticDB PIT read failed (%s); falling back to DuckDB", exc2)
                    df = None
            else:
                log.warning("ArcticDB PIT read failed (%s); falling back to DuckDB", exc)
                df = None

    if df is None:
        try:
            df = _duckdb_macro_frame(
                lake_base_dir, start_date, end_date, duckdb_threads, max_memory
            )
            log.info("macro from DuckDB rows=%d cols=%s", len(df), list(df.columns))
            if store is not None:
                try:
                    to_store = df.reset_index()
                    date_col = to_store.columns[0]
                    to_store = to_store.rename(columns={date_col: "date"})
                    store.persist_features(
                        MACRO_SYMBOL,
                        pa.Table.from_pandas(to_store),
                        metadata={
                            "event_start": str(start_date),
                            "event_end": str(end_date),
                        },
                    )
                    log.info("persisted %s → ArcticDB for PIT streaming", MACRO_SYMBOL)
                except Exception as exc:
                    log.warning("Arctic persist skipped: %s", exc)
        except Exception as exc:
            raise RuntimeError(
                "Critical Macro Feature Failure: ArcticDB and DuckDB both failed. "
                "Refusing torch.randn synthetic macro (lookahead / causality poison). "
                f"Root error: {exc}"
            ) from exc

    assert df is not None
    df = _maybe_join_fioracle(
        df,
        lake_base_dir=lake_base_dir,
        start_date=start_date,
        end_date=end_date,
        fioracle_enabled=fioracle_enabled,
        fioracle_lake_subdir=fioracle_lake_subdir,
        fioracle_series=fioracle_series,
    )
    num = df.select_dtypes(include=[np.number])
    column_order = list(num.columns)
    tensor = _frame_to_tensor(df, macro_dim)
    if not torch.isfinite(tensor).all():
        raise ValueError("non-finite values in macro tensor after fioracle join")
    log.info("macro tensor shape=%s (strict ffill, no synthetic noise)", tuple(tensor.shape))
    meta = {
        "macro_column_order": column_order,
        "fioracle_enabled": bool(fioracle_enabled),
        "fioracle_lake_subdir": fioracle_lake_subdir,
        "macro_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index],
    }
    if return_meta:
        return tensor, meta
    return tensor


def fioracle_cfg_from_feature_extras(
    cfg: Mapping[str, Any] | None,
) -> tuple[bool, str, list[str] | None]:
    """Read ``feature_extras.fioracle_macro`` from a workflow cfg.

    Returns ``(enabled, lake_subdir, series_or_None)``.
    """
    extras = dict((cfg or {}).get("feature_extras") or {})
    block = extras.get("fioracle_macro")
    if not isinstance(block, dict):
        return False, "macro/fioracle", None
    series = block.get("series")
    series_list = list(series) if isinstance(series, (list, tuple)) else None
    return (
        bool(block.get("enabled", False)),
        str(block.get("lake_subdir") or "macro/fioracle"),
        series_list,
    )


def attach_fioracle_macro_cube(
    cfg: dict[str, Any],
    *,
    lake_base_dir: str | Path,
    start_date: str,
    end_date: str,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
    out_dir: str | Path | None = None,
    prefer_arctic: bool = False,
) -> dict[str, Any]:
    """Load macro under ``feature_extras.fioracle_macro`` and stamp the cube.

    When ``enabled``:
      * calls ``load_macro_tensor(..., fioracle_enabled=True, return_meta=True)``
      * writes ``cfg['feature_extras']['macro']`` as ``(T, F)`` aligned to
        ``dates`` (or the macro calendar when dates is None)
      * records ``macro_column_order`` / ``macro_names``
      * writes ``regime_labels.parquet`` under ``out_dir`` when provided

    When ``enabled`` is false (ablation): returns meta with
    ``fioracle_enabled=False`` and does **not** attach a macro block or
    write regime labels.
    """
    enabled, lake_subdir, series = fioracle_cfg_from_feature_extras(cfg)
    meta_out: dict[str, Any] = {
        "fioracle_enabled": bool(enabled),
        "fioracle_lake_subdir": lake_subdir,
        "macro_column_order": [],
    }
    if not enabled:
        return meta_out

    macro_dim = int(cfg.get("macro_dim") or 16)
    tensor, meta = load_macro_tensor(
        lake_base_dir=lake_base_dir,
        start_date=start_date,
        end_date=end_date,
        macro_dim=macro_dim,
        prefer_arctic=prefer_arctic,
        fioracle_enabled=True,
        fioracle_lake_subdir=lake_subdir,
        fioracle_series=series,
        return_meta=True,
    )
    order = list(meta.get("macro_column_order") or [])
    meta_dates = pd.to_datetime(list(meta.get("macro_dates") or []))
    arr = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    # Keep only columns that exist in the frame (macro_dim may pad).
    n_keep = min(len(order), arr.shape[1])
    order = order[:n_keep]
    arr = arr[:, :n_keep]
    if dates is not None:
        tgt = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        frame = pd.DataFrame(arr, index=meta_dates, columns=order)
        frame = frame.reindex(tgt).ffill().fillna(0.0)
        aligned = frame.to_numpy(dtype=np.float64)
        order = list(frame.columns)
    else:
        aligned = arr

    extras = dict(cfg.get("feature_extras") or {})
    extras["macro"] = aligned
    extras["macro_names"] = list(order)
    # Preserve the fioracle_macro config block beside the arrays.
    extras.setdefault(
        "fioracle_macro",
        {"enabled": True, "lake_subdir": lake_subdir},
    )
    cfg["feature_extras"] = extras
    meta_out["macro_column_order"] = list(order)
    meta_out["macro_shape"] = list(aligned.shape)
    meta_out["fioracle_enabled"] = True

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        from mascotrl.data.fioracle_macro import (
            build_fioracle_feature_frame,
            load_fioracle_macro,
        )
        from mascotrl.data.regime_labels import label_regimes

        levels = load_fioracle_macro(
            lake_root=lake_base_dir,
            start_date=start_date,
            end_date=end_date,
            series=series,
            use_available_date=True,
            lake_subdir=lake_subdir,
        )
        feats = build_fioracle_feature_frame(levels).ffill().fillna(0.0)
        # Label on the full history first so expanding quantiles see pre-panel
        # context; then align the auditable frame to the cube dates.
        _labels, regime_meta = label_regimes(feats)
        if dates is not None:
            tgt = pd.DatetimeIndex(pd.to_datetime(list(dates)))
            regime_meta = regime_meta.reindex(tgt)
            # Warmup / missing calendar rows stay flagged; fill regime id only.
            regime_meta["regime"] = regime_meta["regime"].fillna("calm")
            regime_meta["warmup"] = regime_meta["warmup"].fillna(True).astype(bool)
            regime_meta["date"] = tgt
        regime_path = out_path / "regime_labels.parquet"
        regime_meta = regime_meta.copy()
        regime_meta["date"] = pd.to_datetime(regime_meta["date"]).dt.strftime("%Y-%m-%d")
        regime_meta.to_parquet(regime_path, index=False)
        meta_out["regime_labels_path"] = str(regime_path)
        log.info("wrote regime labels → %s rows=%d", regime_path, len(regime_meta))

    return meta_out


def load_macro_tensor_with_fioracle(
    lake_base_dir: str | Path,
    start_date: str,
    end_date: str,
    macro_dim: int,
    *,
    fioracle_lake_subdir: str = "macro/fioracle",
    fioracle_series: list[str] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Convenience hook: ``load_macro_tensor`` with fioracle enabled + meta.

    Prefer wiring ``feature_extras.fioracle_macro.enabled`` at the campaign
    call site into ``fioracle_enabled=`` on ``load_macro_tensor``.
    """
    return load_macro_tensor(  # type: ignore[return-value]
        lake_base_dir,
        start_date,
        end_date,
        macro_dim,
        fioracle_enabled=True,
        fioracle_lake_subdir=fioracle_lake_subdir,
        fioracle_series=fioracle_series,
        return_meta=True,
        **kwargs,
    )


def make_equity_cholesky(n_assets: int, rho: float = 0.45, seed: int = 42) -> torch.Tensor:
    """Equicorrelation Cholesky so cross-asset vol/spot shocks are linked."""
    rho = float(np.clip(rho, -0.9, 0.9))
    corr = np.full((n_assets, n_assets), rho, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    corr = corr + np.eye(n_assets) * 1e-6
    L = np.linalg.cholesky(corr).astype(np.float32)
    rng = np.random.default_rng(seed)
    scale = 0.85 + 0.3 * rng.random(n_assets, dtype=np.float32)
    L = (L.T * scale).T
    return torch.from_numpy(L.astype(np.float32))
