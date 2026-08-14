"""Lake-backed surface signal and Kelly IV image materialization."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from mascotrl.data.surface_signals_cache import (
    _canonical_secid_key,
    _load_shared_surface_cache,
    _write_shared_surface_cache,
    cache_surface_signals,
    load_surface_signals,
    surface_signals_cache_fingerprint,
)
from mascotrl.data.surface_signals_compute import (
    _finalize_signals_panel,
    build_kelly_iv_images,
    compute_surface_signals_panel,
)
from mascotrl.data.surface_signals_extract import _grouped_signal_rows
from mascotrl.data.surface_signals_grid import (
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
    SURFACE_SIGNAL_NAMES,
)

_LOG = logging.getLogger(__name__)

def _load_vol_surface_raw(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
) -> pd.DataFrame:
    """DuckDB scan of ``vol_surface`` filtered to ``secids`` / ``[start, end]``.

    Shared by :func:`materialize_surface_signals_from_lake` (month-end
    signal computation) and :func:`materialize_kelly_iv_images_from_lake`
    (raw per-date IV grid), so both read the same filtered rows once.
    Raises ``FileNotFoundError`` when the lake root or ``vol_surface`` tree
    is absent. Prefers a filtered parquet scan (secids + date range) over
    loading all rows.
    """
    root = Path(lake_root)
    vs = root / "vol_surface"
    if not root.exists() or not vs.exists():
        raise FileNotFoundError(f"vol_surface lake not found under {root}")

    glob = (vs / "year=*" / "month=*" / "data_0.parquet").as_posix()
    # Also accept any *.parquet under hive partitions.
    alt_glob = (vs / "*" / "*" / "*.parquet").as_posix()
    id_list = ", ".join(str(int(s)) for s in secids)
    if not id_list:
        raise ValueError("secids must be non-empty")

    import duckdb

    # B1 perf: the `date` predicate alone does not let DuckDB skip
    # `year=*/month=*` partition files it still has to open every file in
    # the glob to evaluate it. Filtering directly on the hive partition
    # columns (exposed by `hive_partitioning=1`) lets it prune files
    # outside [start, end] at the year granularity before reading them,
    # which matters at ~17GB / 264 monthly partitions.
    year_lo = int(str(start)[:4])
    year_hi = int(str(end)[:4])

    # An unconfigured connection defaults to DuckDB's own memory/thread
    # auto-detection (a large fraction of total system RAM and all cores),
    # which at a full-universe secid pool (hundreds of names x ~a decade)
    # was observed to balloon host memory and trigger an OOM kill of the
    # whole session. Honor the same env-configured ceiling the lake builder
    # uses so this scan degrades to disk spill instead of OOM.
    mem_limit = os.environ.get("MASCOTRL_DUCKDB_MAX_MEMORY", "4GB")
    n_threads = int(os.environ.get("MASCOTRL_DUCKDB_THREADS", "4"))

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{mem_limit}';")
        con.execute(f"SET threads TO {n_threads};")
        sql = f"""
        SELECT
            TRY_CAST(secid AS BIGINT) AS secid,
            CAST(date AS DATE) AS date,
            TRY_CAST(days AS BIGINT) AS days,
            TRY_CAST(delta AS BIGINT) AS delta,
            CAST(cp_flag AS VARCHAR) AS cp_flag,
            TRY_CAST(impl_volatility AS DOUBLE) AS impl_volatility,
            TRY_CAST(impl_strike AS DOUBLE) AS impl_strike,
            TRY_CAST(impl_premium AS DOUBLE) AS impl_premium,
            TRY_CAST(dispersion AS DOUBLE) AS dispersion
        FROM read_parquet('{alt_glob}', hive_partitioning=1, union_by_name=true)
        WHERE TRY_CAST(secid AS BIGINT) IN ({id_list})
          AND CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
          AND TRY_CAST(year AS BIGINT) BETWEEN {year_lo} AND {year_hi}
        """
        try:
            surface = con.execute(sql).fetch_df()
        except Exception as e:
            # A10: record why the hive glob failed before falling back to
            # explicit data_0 naming; if the fallback also fails the
            # exception propagates (fail-closed).
            _LOG.warning("vol_surface hive glob read failed (%s); retrying with %s", e, glob)
            sql2 = sql.replace(alt_glob, glob)
            surface = con.execute(sql2).fetch_df()
    finally:
        con.close()
    return surface


DEFAULT_SECID_BATCH_SIZE = 60



def _load_surface_aux_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame | None]:
    """Load hv_21 / om_opvold / om_borrate / equity volume for unscored signals.

    ``hv`` prefers annualized trailing stdev (21) from equity returns when the
    security-return panel is available; falls back to lake ``sp500_hv`` tenor 30
    (nearest OM historical-vol tenor to the ATM-30 surface).
    """
    root = Path(lake_root)
    secid_set = set(int(s) for s in secids)
    out: dict[str, pd.DataFrame | None] = {
        "hv": None,
        "option_volume": None,
        "equity_volume": None,
        "borrow": None,
    }

    # --- HV (annualized stdev convention) ---
    hv_df = None
    try:
        from mascotrl.data.equity_panel import load_sp500_security_returns

        raw = load_sp500_security_returns(root, start=start, end=end)
        if raw is not None and len(raw) and "secid" in raw.columns:
            raw = raw.copy()
            raw["secid"] = pd.to_numeric(raw["secid"], errors="coerce")
            raw = raw[raw["secid"].isin(secid_set)]
            raw["date"] = pd.to_datetime(raw["date"])
            ret_col = "stk_ret" if "stk_ret" in raw.columns else "return"
            parts = []
            for sid, g in raw.groupby("secid", sort=False):
                g = g.sort_values("date")
                r = pd.to_numeric(g[ret_col], errors="coerce")
                hv = r.rolling(21, min_periods=21).std(ddof=1) * float(np.sqrt(252.0))
                parts.append(pd.DataFrame({"secid": sid, "date": g["date"].to_numpy(), "hv": hv.to_numpy()}))
            if parts:
                hv_df = pd.concat(parts, ignore_index=True)
                hv_df = hv_df[np.isfinite(hv_df["hv"])]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("surface aux hv_21 from returns failed: %s", exc)

    if hv_df is None or hv_df.empty:
        hv_path = root / "macro" / "sp500_hv.parquet"
        if hv_path.is_file():
            try:
                hv = pd.read_parquet(hv_path, columns=["secid", "date", "days", "volatility"])
                hv["secid"] = pd.to_numeric(hv["secid"], errors="coerce")
                hv["date"] = pd.to_datetime(hv["date"])
                hv["days"] = pd.to_numeric(hv["days"], errors="coerce")
                hv = hv[
                    hv["secid"].isin(secid_set)
                    & (hv["date"] >= pd.Timestamp(start))
                    & (hv["date"] <= pd.Timestamp(end))
                    & (hv["days"] == 30)
                ]
                hv_df = hv.rename(columns={"volatility": "hv"})[["secid", "date", "hv"]]
                hv_df["hv"] = pd.to_numeric(hv_df["hv"], errors="coerce")
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("surface aux sp500_hv load failed: %s", exc)
    out["hv"] = hv_df if hv_df is not None and len(hv_df) else None

    # --- option volume ---
    op_path = root / "macro" / "om_opvold.parquet"
    if op_path.is_file():
        try:
            op = pd.read_parquet(op_path, columns=["secid", "date", "volume"])
            op["secid"] = pd.to_numeric(op["secid"], errors="coerce")
            op["date"] = pd.to_datetime(op["date"])
            op = op[
                op["secid"].isin(secid_set)
                & (op["date"] >= pd.Timestamp(start))
                & (op["date"] <= pd.Timestamp(end))
            ]
            agg = op.groupby(["secid", "date"], as_index=False)["volume"].sum()
            out["option_volume"] = agg.rename(columns={"volume": "option_volume"})
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("surface aux om_opvold load failed: %s", exc)

    # --- borrow ---
    br_path = root / "macro" / "om_borrate.parquet"
    if br_path.is_file():
        try:
            br = pd.read_parquet(br_path, columns=["secid", "date", "days", "borrowrate"])
            br["secid"] = pd.to_numeric(br["secid"], errors="coerce")
            br["date"] = pd.to_datetime(br["date"])
            br["days"] = pd.to_numeric(br["days"], errors="coerce")
            br = br[
                br["secid"].isin(secid_set)
                & (br["date"] >= pd.Timestamp(start))
                & (br["date"] <= pd.Timestamp(end))
                & (br["days"] == 30)
            ]
            # OM uses -99.99 as missing sentinel.
            br["borrowrate"] = pd.to_numeric(br["borrowrate"], errors="coerce")
            br.loc[br["borrowrate"] <= -90.0, "borrowrate"] = np.nan
            out["borrow"] = br.rename(columns={"borrowrate": "borrow_rate"})[
                ["secid", "date", "borrow_rate"]
            ]
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("surface aux om_borrate load failed: %s", exc)

    # --- equity volume (for os_ratio) ---
    try:
        from mascotrl.data.equity_panel import load_sp500_security_returns

        raw = load_sp500_security_returns(root, start=start, end=end)
        if raw is not None and len(raw) and "volume" in raw.columns:
            raw = raw.copy()
            raw["secid"] = pd.to_numeric(raw["secid"], errors="coerce")
            raw = raw[raw["secid"].isin(secid_set)]
            raw["date"] = pd.to_datetime(raw["date"])
            eq = raw[["secid", "date", "volume"]].rename(columns={"volume": "equity_volume"})
            eq["equity_volume"] = pd.to_numeric(eq["equity_volume"], errors="coerce")
            out["equity_volume"] = eq
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("surface aux equity volume load failed: %s", exc)

    return out


def materialize_surface_signals_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
    cache_path: str | Path | None = None,
    hv: pd.DataFrame | None = None,
    option_volume: pd.DataFrame | None = None,
    equity_volume: pd.DataFrame | None = None,
    borrow: pd.DataFrame | None = None,
    month_end_only: bool = True,
    secid_batch_size: int | None = DEFAULT_SECID_BATCH_SIZE,
) -> pd.DataFrame:
    """Load ``vol_surface`` via DuckDB and compute the month-end signal panel.

    ``secid_batch_size`` bounds peak memory: the raw per-quote surface scan
    (``_load_vol_surface_raw``) dominates memory use at a full-universe pool
    (hundreds of names x a decade of daily option chains materialized into
    one pandas DataFrame overflowed host RAM in production). When the pool
    exceeds this size the raw load + per-group signal step run in secid
    batches and are concatenated *before* the single full-pool
    cross-sectional finalize pass, so results are identical to the
    unbatched path (``mw_xs`` still cross-sections over every requested
    secid, never just one batch). Pass ``None`` or a value >= pool size to
    disable batching.

    When ``MASCOTRL_SURFACE_CACHE_DIR`` is set, a fingerprint-keyed parquet
    (sorted secids + date range + signal names) is reused across callers.
    Corrupt shared-cache artifacts raise rather than silently rebuilding.
    """
    secids_list = list(secids)
    if hv is None or option_volume is None or equity_volume is None or borrow is None:
        aux = _load_surface_aux_from_lake(
            lake_root, secids=secids_list, start=start, end=end
        )
        if hv is None:
            hv = aux.get("hv")
        if option_volume is None:
            option_volume = aux.get("option_volume")
        if equity_volume is None:
            equity_volume = aux.get("equity_volume")
        if borrow is None:
            borrow = aux.get("borrow")
    signal_names = list(SURFACE_SIGNAL_NAMES)
    shared_dir = str(os.environ.get("MASCOTRL_SURFACE_CACHE_DIR") or "").strip()
    shared_fp: str | None = None
    if shared_dir:
        shared_fp = surface_signals_cache_fingerprint(
            secids=secids_list,
            start=start,
            end=end,
            signal_names=signal_names,
        )
        hit = _load_shared_surface_cache(shared_dir, shared_fp)
        if hit is not None:
            if cache_path is not None:
                cache_surface_signals(hit, cache_path)
            return hit

    if not secid_batch_size or len(secids_list) <= secid_batch_size:
        from mascotrl.data import surface_signals as _ss

        surface = _ss._load_vol_surface_raw(
            lake_root, secids=secids_list, start=start, end=end
        )
        panel = _ss.compute_surface_signals_panel(
            surface,
            hv=hv,
            option_volume=option_volume,
            equity_volume=equity_volume,
            borrow=borrow,
            month_end_only=month_end_only,
        )
    else:
        from mascotrl.data import surface_signals as _ss

        parts: list[pd.DataFrame] = []
        for i in range(0, len(secids_list), secid_batch_size):
            batch = secids_list[i : i + secid_batch_size]
            surface = _ss._load_vol_surface_raw(
                lake_root, secids=batch, start=start, end=end
            )
            rows = _grouped_signal_rows(
                surface,
                hv=hv,
                option_volume=option_volume,
                equity_volume=equity_volume,
                borrow=borrow,
                month_end_only=month_end_only,
            )
            del surface
            if not rows.empty:
                parts.append(rows)
        rows_panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if rows_panel.empty:
            cols = ["secid", "date", *SURFACE_SIGNAL_NAMES]
            panel = pd.DataFrame(columns=cols)
        else:
            panel = _finalize_signals_panel(rows_panel)
    if shared_dir and shared_fp is not None:
        _write_shared_surface_cache(shared_dir, shared_fp, panel)
    if cache_path is not None:
        cache_surface_signals(panel, cache_path)
    return panel


def _kelly_cache_fingerprint(
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    start: str,
    end: str,
    forward_fill: bool,
) -> str:
    """Stable key so a resumed campaign can reuse a Kelly cube on disk."""
    import hashlib

    payload = {
        "secids": [_canonical_secid_key(s) for s in secids],
        "dates": [str(pd.Timestamp(d).date()) for d in dates],
        "start": str(start),
        "end": str(end),
        "forward_fill": bool(forward_fill),
        "tenors": list(KELLY_TENORS),
        "deltas_put": list(KELLY_DELTAS_PUT),
        "deltas_call": list(KELLY_DELTAS_CALL),
    }
    blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def materialize_kelly_iv_images_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    start: str,
    end: str,
    forward_fill: bool = True,
    secid_batch_size: int | None = DEFAULT_SECID_BATCH_SIZE,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """B3: real per-date Kelly IV-surface image tensor from the lake.

    Returns ``(T, K, 11, 34)`` (``T=len(dates)``, ``K=len(secids)``), the
    layout ``src.eval.ceiling_arms._select_kelly_image_batch`` expects for
    ``kelly_cnn``. ``forward_fill=True`` (default) causally fills a grid
    cell with the most recent *earlier* observed value when the exact
    date has no quote for that (tenor, delta, cp) node -- never from a
    later date, so no look-ahead is introduced.

    ``secid_batch_size`` bounds peak memory the same way
    :func:`materialize_surface_signals_from_lake` does: a single DuckDB
    ``fetch_df`` of every daily option-chain quote for K=100 over a decade
    OOMed the full campaign after the signal gate had already succeeded.
    Batches fill disjoint slices of the output cube and are identical to
    the unbatched path (per-secid grids do not cross-talk).

    ``cache_path`` (optional ``.npz``) stores the cube keyed by a fingerprint
    of secids/dates/window so a killed-and-resumed campaign does not re-pay
    multi-hour lake scans. Cache hit requires an adjacent ``.meta.json`` with
    a matching fingerprint.
    """
    secids_list = list(secids)
    dates_list = list(dates)
    n_t = len(dates_list)
    n_k = len(secids_list)
    n_ten = len(KELLY_TENORS)
    n_del = len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL)
    cube = np.full((n_t, n_k, n_ten, n_del), np.nan, dtype=np.float64)
    if n_t == 0 or n_k == 0:
        return cube

    fp = _kelly_cache_fingerprint(
        secids=secids_list,
        dates=dates_list,
        start=start,
        end=end,
        forward_fill=forward_fill,
    )
    if cache_path is not None:
        cpath = Path(cache_path)
        meta_path = Path(str(cpath) + ".meta.json")
        if cpath.is_file() and meta_path.is_file():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fp:
                    loaded = np.load(cpath)
                    arr = loaded["cube"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
                    if tuple(arr.shape) == (n_t, n_k, n_ten, n_del):
                        _LOG.info(
                            "kelly_images cache hit path=%s shape=%s",
                            cpath,
                            arr.shape,
                        )
                        return np.asarray(arr, dtype=np.float64)
            except Exception as exc:  # pragma: no cover - best-effort cache
                _LOG.warning("kelly_images cache unreadable (%s); rebuilding", exc)

    batch_size = secid_batch_size if secid_batch_size else n_k
    n_batches = (n_k + batch_size - 1) // batch_size
    from mascotrl.data import surface_signals as _ss

    for bi, i in enumerate(range(0, n_k, batch_size)):
        batch = secids_list[i : i + batch_size]
        _LOG.info(
            "kelly_images batch %d/%d secids=%d..%d T=%d",
            bi + 1,
            n_batches,
            i,
            i + len(batch) - 1,
            n_t,
        )
        surface = _ss._load_vol_surface_raw(lake_root, secids=batch, start=start, end=end)
        batch_cube = _ss.build_kelly_iv_images(surface, secids=batch, dates=dates_list)
        del surface
        cube[:, i : i + len(batch)] = batch_cube

    if forward_fill and cube.size:
        # ffill along the date axis (axis=0) per (secid, tenor, delta) cell.
        last = np.full(cube.shape[1:], np.nan, dtype=np.float64)
        for i in range(n_t):
            row = cube[i]
            nan_mask = np.isnan(row)
            row = np.where(nan_mask, last, row)
            cube[i] = row
            last = np.where(np.isnan(row), last, row)
    out = cube
    if cache_path is not None:
        import json

        cpath = Path(cache_path)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, cube=out)
        meta_path = Path(str(cpath) + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "shape": list(out.shape),
                    "start": str(start),
                    "end": str(end),
                    "n_secids": n_k,
                    "n_dates": n_t,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _LOG.info("kelly_images cache wrote path=%s shape=%s", cpath, out.shape)
    return out
