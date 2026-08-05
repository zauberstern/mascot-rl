"""Shared equity observation substrate (H0 / spectrum parity).

Spectrum MLP cells historically fell back to raw returns while H0 ran a
feature cube + geometry_lite surface on lake ``sp500_sec`` + ``dyn_hrp``.
This module is the single attach/load contract both campaigns must use so
those paths cannot silently diverge again.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from src.data.equity_panel import (
    EVAL_END,
    EVAL_START,
    SELECTION_END,
    SELECTION_START,
)
from src.data.paths import CANONICAL_LAKE


GEOMETRY_LITE_CHANNELS: tuple[str, ...] = (
    "mfiv_30",
    "iv_term_slope",
    "iv_skew_30d",
)


def resolve_lake_root(cfg: Mapping[str, Any] | None = None) -> Path:
    """Resolve lake root with Burst alias ``MASCOTRL_LAKE_BASE``.

    Prefer explicit cfg stamps, then ``MASCOTRL_LAKE_DIR``, then the Burst
    extract alias ``MASCOTRL_LAKE_BASE``, then the canonical USB lake.
    """
    cfg = cfg or {}
    for key in ("lake_root", "_lake_root", "lake_base_dir"):
        raw = cfg.get(key)
        if raw:
            return Path(str(raw))
    env = (
        os.environ.get("MASCOTRL_LAKE_DIR")
        or os.environ.get("MASCOTRL_LAKE_BASE")
        or str(CANONICAL_LAKE)
    )
    return Path(env)


def stamp_equity_obs_defaults(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Align spectrum / eq_alloc observation defaults (cube + surface lane)."""
    cfg.setdefault("use_equity_feature_cube", True)
    cfg.setdefault("feature_seq_len", 1)
    # Feature-net extras stay OFF (G0: H0 checkpoints are K*26 without them).
    cfg.setdefault("use_feature_net_extras", False)
    if bool(cfg.get("use_surface_signals", False)):
        cfg.setdefault("surface_obs_lane", "geometry_lite")
        cfg.setdefault(
            "obs_pack_path", "config/obs_packs/surf_geometry_lite.yaml"
        )
    # Stamp lake early so feature-net attach cannot silently no-op.
    lake = resolve_lake_root(cfg)
    cfg.setdefault("_lake_root", str(lake))
    cfg.setdefault("lake_root", str(lake))
    return cfg


_FEATURE_NET_KEYS = (
    "ohlc",
    "microstructure",
    "sentiment",
    "fundamentals_pit",
    "option_flow",
    "jkp",
    "macro",
)


def apply_feature_net_extras_if_enabled(
    cfg: MutableMapping[str, Any],
    *,
    dates: Sequence[Any],
    secids: Sequence[Any],
    panel_source: str = "",
    slots_rows: Sequence[Sequence[Any]] | None = None,
) -> MutableMapping[str, Any]:
    """Opt-in attach of lake feature-net panels into ``feature_extras``.

    Default remains off (G0 K×26 parity). When ``use_feature_net_extras`` is
    true, fail-closed if lake is missing or no panels attach (non-toy).
    """
    cfg.setdefault("use_feature_net_extras", False)
    if not bool(cfg.get("use_feature_net_extras")):
        return cfg

    from src.eval.feature_extras_loader import attach_feature_net_extras

    lake_for_feat = cfg.get("lake_root") or cfg.get("_lake_root")
    if lake_for_feat is None:
        raise RuntimeError(
            "use_feature_net_extras=true but lake_root/_lake_root missing"
        )
    extras = dict(cfg.get("feature_extras") or {})
    extras = attach_feature_net_extras(
        extras,
        lake=lake_for_feat,
        dates=dates,
        secids=secids,
        slots_rows=slots_rows if slots_rows is not None else cfg.get("_slots_rows"),
    )
    if "feature_groups_exclude" in cfg:
        extras["feature_groups_exclude"] = list(cfg.get("feature_groups_exclude") or [])
    if "feature_channels_exclude" in cfg:
        extras["feature_channels_exclude"] = list(
            cfg.get("feature_channels_exclude") or []
        )
    cfg["feature_extras"] = extras
    if not any(k in extras for k in _FEATURE_NET_KEYS):
        if str(panel_source) == "toy":
            cfg.setdefault("_feature_net_errors", []).append(
                "use_feature_net_extras=true but no feature-net panels attached (toy)"
            )
            return cfg
        raise RuntimeError(
            "use_feature_net_extras=true but no feature-net panels attached"
        )
    return cfg


def resolve_substrate_secids(
    cfg: Mapping[str, Any],
    *,
    panel_source: str,
    k: int,
) -> list[Any]:
    """Resolve secids for substrate attach; fail-closed except toy fallback."""
    secids = cfg.get("_universe_secids")
    if panel_source == "toy":
        return list(secids) if secids else list(range(int(k)))
    if not secids:
        raise RuntimeError(
            f"_universe_secids missing for panel_source={panel_source!r}; "
            "refusing range(K) secid fallback"
        )
    return list(secids)


def stamp_lake_universe_secids_for_featnet(
    cfg: MutableMapping[str, Any],
    *,
    k: int,
) -> list[Any]:
    """Stamp lake ``_universe_secids`` for FEATNET when the train panel is non-lake.

    Physics / OM train worlds still need a lake fingerprint so feature-net
    panels can attach by secid. Preserves any pre-existing train-panel slot
    state; when none existed, clears lake slots so OM/physics dates are not
    paired with mismatched lake ``_slots_rows``.
    """
    existing = cfg.get("_universe_secids")
    if existing:
        return list(existing)
    saved_slots = cfg.get("_slots_rows")
    saved_mask = cfg.get("_slot_valid_mask")
    saved_rb = cfg.get("_rebalance_mask")
    had_slots = saved_slots is not None
    load_lake_dyn_hrp_panel(cfg, k=int(k))
    secids = list(cfg.get("_universe_secids") or [])
    if had_slots:
        cfg["_slots_rows"] = saved_slots
        if saved_mask is not None:
            cfg["_slot_valid_mask"] = saved_mask
        if saved_rb is not None:
            cfg["_rebalance_mask"] = saved_rb
    else:
        cfg.pop("_slots_rows", None)
        cfg.pop("_slot_valid_mask", None)
        cfg.pop("_rebalance_mask", None)
    if not secids:
        raise RuntimeError("featnet_lake_universe_empty_after_stamp")
    return secids


def assert_surface_nan_ok(
    iv_surface: Mapping[str, Any],
    *,
    channel_names: Sequence[str] | None = None,
    admitted_channels: Sequence[str] | None = None,
    max_all_nan_frac: float = 0.20,
) -> dict[str, Any]:
    """Fail-closed when admitted surface channels are mostly all-NaN."""
    from src.eval.feature_nan_diagnostics import (
        assert_feature_nan_ok,
        feature_nan_diagnostics,
    )

    chan_names = list(channel_names or iv_surface.keys())
    if not chan_names:
        return {"pass": True, "per_channel": {}, "fail_channels": []}
    cube = np.stack(
        [np.asarray(iv_surface[n], dtype=np.float64) for n in chan_names], axis=-1
    )
    nan_diag = feature_nan_diagnostics(
        cube,
        channel_names=chan_names,
        admitted_channels=admitted_channels or chan_names,
        max_all_nan_frac=float(max_all_nan_frac),
    )
    assert_feature_nan_ok(nan_diag)
    return nan_diag


def _geometry_lite_channel_names(cfg: Mapping[str, Any]) -> list[str]:
    pack_path = cfg.get("obs_pack_path") or "config/obs_packs/surf_geometry_lite.yaml"
    try:
        from src.eval.signal_gate import assert_geometry_pack_valid

        pack = assert_geometry_pack_valid(pack_path)
        names = list(pack.get("channels") or [])
        if names:
            return names
    except Exception:  # noqa: BLE001
        pass
    return list(GEOMETRY_LITE_CHANNELS)


def attach_equity_obs_substrate(
    cfg: MutableMapping[str, Any],
    *,
    dates: Sequence[Any],
    rets: np.ndarray,
    secids: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]] | None,
    dollar_volume: np.ndarray | None = None,
    signals_long: pd.DataFrame | None = None,
    lake_root: Path | str | None = None,
    cache_path: Path | str | None = None,
    fail_closed_surface: bool = True,
) -> dict[str, Any]:
    """Attach dollar_volume + geometry_lite ``iv_surface`` into ``feature_extras``.

    Feature-net extras stay OFF this sprint (H0 parity without re-run).
    Fail-closed when surface is requested but unavailable.
    """
    stamp_equity_obs_defaults(cfg)
    lake = Path(lake_root) if lake_root is not None else resolve_lake_root(cfg)
    cfg["_lake_root"] = str(lake)
    cfg["lake_root"] = str(lake)

    extras: dict[str, Any] = dict(cfg.get("feature_extras") or {})
    meta: dict[str, Any] = {
        "surface_attached": False,
        "n_surface_channels": 0,
        "dollar_volume_attached": False,
        "lake_root": str(lake),
    }

    arr = np.asarray(rets, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"rets must be (T, K), got {arr.shape}")
    t_len, k = int(arr.shape[0]), int(arr.shape[1])
    if dollar_volume is not None:
        dv = np.asarray(dollar_volume, dtype=np.float64)
        if dv.shape != (t_len, k):
            raise ValueError(
                f"dollar_volume shape {dv.shape} != panel {(t_len, k)}"
            )
        extras["dollar_volume"] = dv
        meta["dollar_volume_attached"] = True

    want_surface = bool(cfg.get("use_surface_signals", False))
    if want_surface:
        channel_names = _geometry_lite_channel_names(cfg)
        lane = str(cfg.get("surface_obs_lane") or "geometry_lite")
        if lane not in {"geometry_lite", "geometry"}:
            raise ValueError(
                f"unsupported surface_obs_lane={lane!r} on shared substrate; "
                "expected geometry_lite|geometry"
            )
        cfg["_obs_pack_id"] = "surf_geometry_lite"
        long_df = signals_long
        if long_df is None and cache_path is not None:
            cp = Path(cache_path)
            if cp.is_file():
                long_df = pd.read_parquet(cp)
        if long_df is None:
            for rel in (
                "surface_signals/geometry_lite.parquet",
                "surface_signals/dyn_surface_signals.parquet",
                "_panels/geometry_lite_surface.parquet",
            ):
                cand = lake / rel
                if cand.is_file():
                    long_df = pd.read_parquet(cand)
                    meta["surface_cache"] = str(cand)
                    break
        if long_df is None and (lake / "vol_surface").exists():
            from src.data.surface_signals import materialize_surface_signals_from_lake

            signal_secids = sorted(
                {
                    s
                    for row in (slots_rows or [])
                    for s in row
                    if s is not None
                },
                key=lambda x: str(x),
            ) or list(secids)
            surf_start = str(
                (pd.Timestamp(EVAL_START) - pd.DateOffset(months=2)).date()
            )
            long_df = materialize_surface_signals_from_lake(
                lake,
                secids=signal_secids,
                start=surf_start,
                end=EVAL_END,
                month_end_only=True,
            )
            meta["surface_materialized"] = True
        if long_df is None or getattr(long_df, "empty", True):
            if fail_closed_surface:
                raise ValueError(
                    "use_surface_signals=true but no surface signals available "
                    "(no signals_long, no lake cache, no vol_surface)"
                )
        else:
            if slots_rows is None:
                raise ValueError(
                    "use_surface_signals=true requires slots_rows for slot alignment"
                )
            from src.data.surface_signals import align_signals_to_slots

            iv_surface = align_signals_to_slots(
                long_df,
                list(dates),
                slots_rows,
                lag_days=1,
                signal_names=channel_names,
            )
            nan_diag = assert_surface_nan_ok(
                iv_surface,
                channel_names=channel_names,
                admitted_channels=channel_names,
            )
            extras["iv_surface"] = iv_surface
            meta["surface_attached"] = True
            meta["n_surface_channels"] = int(len(channel_names))
            meta["surface_channels"] = list(channel_names)
            meta["feature_nan_diagnostics"] = nan_diag

    cfg["feature_extras"] = extras
    cfg["_universe_secids"] = list(secids)
    if slots_rows is not None:
        cfg["_slots_rows"] = list(slots_rows)
    return meta


def feature_health_report(
    cube: np.ndarray,
    *,
    channel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per-channel NaN rate, coverage, and variance for feature-health table.

    ``cube`` may be ``(T, K, C)`` or ``(T, C)``. Dead channels (near-zero
    variance or all-NaN) are flagged for the appendix audit.
    """
    x = np.asarray(cube, dtype=np.float64)
    if x.ndim == 2:
        # (T, C) -> treat as K=1
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"feature cube must be (T,K,C) or (T,C); got {x.shape}")
    t, k, c = x.shape
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(c)]
    if len(names) != c:
        raise ValueError(f"channel_names length {len(names)} != C={c}")
    per_channel: dict[str, Any] = {}
    dead: list[str] = []
    for i, name in enumerate(names):
        sl = x[:, :, i]
        finite = np.isfinite(sl)
        n_tot = int(sl.size)
        n_finite = int(np.sum(finite))
        nan_rate = float(1.0 - (n_finite / n_tot)) if n_tot else float("nan")
        # Coverage: non-zero among finite values (zero often = filler).
        if n_finite:
            nz = int(np.sum(np.abs(sl[finite]) > 1e-12))
            coverage = float(nz / n_finite)
            var = float(np.var(sl[finite]))
        else:
            coverage = 0.0
            var = float("nan")
        is_dead = (not np.isfinite(var)) or var < 1e-16 or nan_rate > 0.95
        per_channel[str(name)] = {
            "nan_rate": nan_rate,
            "coverage": coverage,
            "variance": var,
            "n_dates": int(t),
            "n_assets": int(k),
            "dead": bool(is_dead),
        }
        if is_dead:
            dead.append(str(name))
    return {
        "schema_version": 1,
        "n_channels": int(c),
        "per_channel": per_channel,
        "dead_channels": dead,
        "pass": len(dead) == 0,
    }


def _wide_returns_with_availability(
    df: pd.DataFrame,
    *,
    start: str,
    end: str,
    min_cov: float = 0.85,
    ffill_limit: int = 5,
    max_row_nan_frac: float = 0.05,
    keep_partial_rows: bool = True,
) -> tuple[np.ndarray, list, pd.DatetimeIndex, np.ndarray]:
    """Wide returns plus a (T, K) availability mask.

    When ``keep_partial_rows`` is True (availability-mask estimand), every
    business date in the requested window that has at least one finite name
    is kept. Missing names are zero-filled and marked False in ``avail`` so
    peers/policy must renormalize over active slots. When False, rows with
    more than ``max_row_nan_frac`` missing names are dropped, then any
    remaining holes are removed (historical puncture behavior).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]
    d = d.dropna(subset=["return", "secid"])
    wide = d.pivot_table(index="date", columns="secid", values="return", aggfunc="last")
    wide = wide.sort_index()
    cov = wide.notna().mean(axis=0)
    wide = wide.loc[:, cov >= float(min_cov)]
    if int(ffill_limit) > 0:
        wide = wide.ffill(limit=int(ffill_limit))
    if keep_partial_rows:
        wide = wide.loc[~wide.isna().all(axis=1)]
        avail = wide.notna().to_numpy(dtype=bool)
        filled = wide.fillna(0.0)
        return (
            filled.to_numpy(dtype=np.float64),
            list(wide.columns),
            wide.index,
            avail,
        )
    row_ok = wide.isna().mean(axis=1) <= float(max_row_nan_frac)
    wide = wide.loc[row_ok].dropna(how="any")
    avail = np.ones(wide.shape, dtype=bool)
    return wide.to_numpy(dtype=np.float64), list(wide.columns), wide.index, avail


def _wide_returns(
    df: pd.DataFrame,
    *,
    start: str,
    end: str,
    min_cov: float = 0.85,
    ffill_limit: int = 5,
    max_row_nan_frac: float = 0.05,
) -> tuple[np.ndarray, list, pd.DatetimeIndex]:
    rets, secids, idx, _avail = _wide_returns_with_availability(
        df,
        start=start,
        end=end,
        min_cov=min_cov,
        ffill_limit=ffill_limit,
        max_row_nan_frac=max_row_nan_frac,
        keep_partial_rows=False,
    )
    return rets, secids, idx


def _wide_field(
    df: pd.DataFrame,
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    value_col: str,
) -> np.ndarray | None:
    if value_col not in df.columns:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["secid"].isin(list(secids))]
    wide = d.pivot_table(index="date", columns="secid", values=value_col, aggfunc="last")
    wide = wide.reindex(pd.DatetimeIndex(dates))
    cols = []
    for s in secids:
        if s in wide.columns:
            cols.append(wide[s].to_numpy(dtype=np.float64))
        else:
            cols.append(np.full(len(dates), np.nan, dtype=np.float64))
    return np.column_stack(cols) if cols else None


def load_lake_dyn_hrp_panel(
    cfg: MutableMapping[str, Any],
    *,
    k: int,
    lake_root: Path | str | None = None,
    max_pool: int = 400,
) -> tuple[list, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load ``sp500_sec`` eval panel + real ``dyn_hrp`` slotted universe.

    Returns ``(dates, slotted_rets, factors, meta)`` and stamps cfg with
    ``_slots_rows``, ``_slot_valid_mask``, ``_universe_secids``, ``_rebalance_mask``.
    """
    from src.data.dynamic_universe import (
        build_dynamic_universe,
        build_slotted_panel,
        select_universe_corr_cluster,
        selection_turnover,
    )
    from src.data.equity_panel import load_sp500_security_returns
    from src.eval.cadence import (
        assert_universe_subset_of_policy,
        build_rebalance_mask,
        build_universe_cadence_mask,
    )

    stamp_equity_obs_defaults(cfg)
    lake = Path(lake_root) if lake_root is not None else resolve_lake_root(cfg)
    cfg["_lake_root"] = str(lake)
    cfg["lake_root"] = str(lake)

    sel_start = str(cfg.get("selection_start") or SELECTION_START)
    sel_end = str(cfg.get("selection_end") or SELECTION_END)
    eval_start = str(cfg.get("oos_start") or EVAL_START)
    eval_end = str(cfg.get("oos_end") or EVAL_END)

    raw = load_sp500_security_returns(lake, start="2003-01-02", end=eval_end)
    rets_w, secids_w, _idx_w, _ = _wide_returns_with_availability(
        raw, start=sel_start, end=sel_end, min_cov=0.85, keep_partial_rows=False
    )
    if rets_w.shape[1] > int(max_pool):
        activity = np.nanstd(rets_w, axis=0)
        keep = np.argsort(activity)[::-1][: int(max_pool)]
        rets_w = rets_w[:, keep]
        secids_w = [secids_w[i] for i in keep]

    rets_e, secids_e, dates_e, _avail = _wide_returns_with_availability(
        raw,
        start=eval_start,
        end=eval_end,
        min_cov=0.70,
        ffill_limit=5,
        keep_partial_rows=True,
    )
    dates = list(dates_e)
    cadence = str(cfg.get("rebalance_cadence") or "monthly")
    rb_mask = build_rebalance_mask(dates, cadence)
    cfg["_rebalance_mask"] = rb_mask

    # Universe reselect schedule is independent of policy cadence.
    # Daily policy returns rb_mask=None; np.asarray(None) has size 1 and must
    # never be passed to build_dynamic_universe (RC6 canary crash).
    u_mode = str(cfg.get("universe_cadence") or "").strip()
    if rb_mask is None:
        u_mask = build_universe_cadence_mask(dates, u_mode or "quarterly_63d")
    elif u_mode:
        u_mask = build_universe_cadence_mask(dates, u_mode)
        assert_universe_subset_of_policy(u_mask, rb_mask)
    else:
        # Legacy: monthly/weekly policy also drove dyn_hrp reselect days.
        u_mask = rb_mask

    k_i = min(int(k), int(rets_e.shape[1]))
    slots_rows, valid_mask, selection_log = build_dynamic_universe(
        dates=dates,
        rebalance_mask=u_mask,
        wide_returns=rets_e,
        secids=list(secids_e),
        k=k_i,
        select_fn=select_universe_corr_cluster,
        trailing_days=252,
        select_kwargs={},
        eligibility_by_date=None,
    )
    col_map = {s: i for i, s in enumerate(secids_e)}
    slotted = build_slotted_panel(
        dates=dates, slots_rows=slots_rows, wide_returns=rets_e, col_map=col_map
    )
    fingerprint = sorted({s for row in slots_rows for s in row if s is not None})
    cfg["_slots_rows"] = slots_rows
    cfg["_slot_valid_mask"] = valid_mask
    cfg["_universe_secids"] = list(fingerprint)
    cfg.setdefault("universe_arm", "dyn_hrp")
    cfg["_universe_arm_applied"] = str(cfg.get("universe_arm") or "dyn_hrp")

    dv_wide = None
    if "volume" in raw.columns and "close" in raw.columns:
        tmp = raw.copy()
        tmp["dollar_volume"] = pd.to_numeric(tmp["volume"], errors="coerce") * pd.to_numeric(
            tmp["close"], errors="coerce"
        )
        dv_wide = _wide_field(
            tmp, secids=fingerprint, dates=dates, value_col="dollar_volume"
        )
    dollar_volume = None
    if dv_wide is not None:
        from src.features.blocks.liquidity import map_wide_to_slots

        dollar_volume = map_wide_to_slots(
            dv_wide, secids=fingerprint, slots_rows=slots_rows
        )

    # Gate2 / residualization panel: 7-factor when lake has FF5+Mom+PS.
    # Drop RF as a regressor (it is the risk-free rate, not a factor).
    # Prefer Mom as UMD when present; join PS_VWF from pastor_stambaugh.
    factor_names: list[str] = []
    factors = np.zeros((slotted.shape[0], 0), dtype=np.float64)
    try:
        ff_path = lake / "macro" / "ff_factors.parquet"
        date_idx = pd.DatetimeIndex(dates)
        pieces: list[np.ndarray] = []
        names: list[str] = []
        if ff_path.is_file():
            ff = pd.read_parquet(ff_path)
            if "date" in ff.columns:
                ff = ff.copy()
                ff["date"] = pd.to_datetime(ff["date"])
                ff = ff.set_index("date").sort_index()
            # Order: MKT-RF, SMB, HML, RMW, CMA, UMD(Mom), then PS below.
            wanted = [
                ("Mkt-RF", "mkt"),
                ("SMB", "smb"),
                ("HML", "hml"),
                ("RMW", "rmw"),
                ("CMA", "cma"),
            ]
            mom_col = next(
                (c for c in ("Mom", "UMD", "MomRF") if c in ff.columns), None
            )
            for src, alias in wanted:
                if src in ff.columns:
                    col = (
                        ff[src]
                        .reindex(date_idx)
                        .astype(np.float64)
                        .fillna(0.0)
                        .to_numpy()
                    )
                    pieces.append(col)
                    names.append(alias)
            if mom_col is not None:
                col = (
                    ff[mom_col]
                    .reindex(date_idx)
                    .astype(np.float64)
                    .fillna(0.0)
                    .to_numpy()
                )
                pieces.append(col)
                names.append("umd")
        ps_path = lake / "macro" / "pastor_stambaugh.parquet"
        if ps_path.is_file():
            ps = pd.read_parquet(ps_path)
            date_col = next(
                (c for c in ("date", "DATE", "Date") if c in ps.columns), None
            )
            ps_col = next(
                (c for c in ("PS_VWF", "ps_vwf", "PS_INNOV") if c in ps.columns),
                None,
            )
            if date_col is not None and ps_col is not None:
                ps = ps.copy()
                ps[date_col] = pd.to_datetime(ps[date_col])
                ps = ps.set_index(date_col).sort_index()
                col = (
                    ps[ps_col]
                    .reindex(date_idx)
                    .astype(np.float64)
                    .fillna(0.0)
                    .to_numpy()
                )
                pieces.append(col)
                names.append("ps_vwf")
        if pieces:
            arr = np.column_stack(pieces)
            if np.nanmax(np.abs(arr)) > 1.0:
                # FF factors are often stored in percent; PS may already be decimal.
                # Scale only FF-like columns (everything except trailing PS if present).
                scale = np.ones(arr.shape[1], dtype=np.float64)
                for i, n in enumerate(names):
                    if n != "ps_vwf":
                        scale[i] = 0.01
                # Only scale if the FF block looks like percent.
                ff_block = arr[:, [i for i, n in enumerate(names) if n != "ps_vwf"]]
                if ff_block.size and np.nanmax(np.abs(ff_block)) > 1.0:
                    arr = arr * scale
            factors = arr
            factor_names = names
        if factors.size == 0:
            factors = np.zeros((slotted.shape[0], 4), dtype=np.float64)
            factor_names = ["mkt", "smb", "hml", "umd"]
    except Exception:  # noqa: BLE001
        factors = np.zeros((slotted.shape[0], 4), dtype=np.float64)
        factor_names = ["mkt", "smb", "hml", "umd"]
    cfg["_factor_names"] = list(factor_names)

    from src.eval.universe_fingerprint import read_panel_bundle_sha256

    panel_fp = read_panel_bundle_sha256()
    meta = {
        "panel_source": "lake_sp500_sec",
        "universe_arm": "dyn_hrp",
        "k": int(k_i),
        "n_days": int(slotted.shape[0]),
        "fingerprint_size": int(len(fingerprint)),
        "universe_fingerprint": panel_fp or "",
        "universe_fingerprint_kind": "panel_bundle_sha256" if panel_fp else "",
        "n_rebalances": int(len(selection_log)),
        "turnover": selection_turnover(slots_rows),
        "selection_start": sel_start,
        "selection_end": sel_end,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "dollar_volume": dollar_volume,
        "raw": raw,
        "pool_size_selection": int(rets_w.shape[1]),
        "pool_size_eval": int(rets_e.shape[1]),
        "n_factors": int(factors.shape[1]) if factors.ndim == 2 else 0,
        "factor_names": list(factor_names),
    }
    return dates, slotted, factors, meta
