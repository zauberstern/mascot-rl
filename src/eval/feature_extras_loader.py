"""Shared feature-extras assembly for eq_alloc and spectrum campaigns."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.feature_panels import (
    FAMILY_LOADERS,
    load_analyst_long,
    load_compustat_long,
    load_dividend_long,
    load_gics_map,
    load_ibes_ratios_long,
    load_jkp_long,
    load_microstructure_long,
    load_ohlc_long,
    load_option_flow_long,
    load_rates_term_long,
    load_short_interest_long,
    load_worldscope_long,
)
from src.data.paths import LAKE_ROOT
from src.data.surface_signals import _canonical_secid_key
from src.features.blocks.liquidity import map_wide_to_slots
from src.features.groups import resolve_excludes

log = logging.getLogger(__name__)


def _long_to_wide(
    df: pd.DataFrame,
    *,
    value_col: str,
    dates: Sequence[Any],
    secids: Sequence[Any],
) -> np.ndarray:
    if df.empty or value_col not in df.columns:
        t, k = len(dates), len(secids)
        return np.full((t, k), np.nan, dtype=np.float64)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["secid"] = out["secid"].map(_canonical_secid_key)
    dates_idx = pd.to_datetime(list(dates))
    sec_str = [_canonical_secid_key(s) for s in secids]
    wide = out.pivot_table(
        index="date", columns="secid", values=value_col, aggfunc="last"
    )
    wide.columns = [_canonical_secid_key(c) for c in wide.columns]
    wide = wide.reindex(index=dates_idx, columns=sec_str)
    return wide.to_numpy(dtype=np.float64)


def _dict_from_long(
    df: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    dates: Sequence[Any],
    secids: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]] | None = None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for col in value_cols:
        if col not in df.columns:
            continue
        wide = _long_to_wide(df, value_col=col, dates=dates, secids=secids)
        if slots_rows is not None and wide.shape[1] != len(slots_rows[0]):
            # Already (T, N) on fingerprint; map to slots if needed.
            # Here secids are the fingerprint occupants; slots_rows maps day→slots.
            wide = map_wide_to_slots(wide, secids=secids, slots_rows=slots_rows)
        out[col] = wide
    return out


def _load_or_panel(
    lake: Path,
    family: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    mirror = lake / "_panels" / f"feat_{family}.parquet"
    if mirror.is_file():
        try:
            return pd.read_parquet(mirror)
        except Exception as exc:
            log.warning("panel mirror read failed %s: %s", mirror, exc)
    loader = FAMILY_LOADERS.get(family)
    if loader is None:
        return pd.DataFrame()
    return loader(lake, start, end)


def attach_feature_net_extras(
    extras: dict[str, Any],
    *,
    lake: Path | str | None,
    dates: Sequence[Any],
    secids: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Populate ohlc/microstructure/fundamentals_pit/sentiment/option_flow/jkp/macro rates."""
    if lake is None:
        return extras
    lake_p = Path(lake)
    if not lake_p.exists():
        return extras
    dates_list = list(dates)
    if not dates_list:
        return extras
    start_s = start or str(pd.Timestamp(dates_list[0]).date())
    end_s = end or str(pd.Timestamp(dates_list[-1]).date())
    out = dict(extras)

    ohlc_df = _load_or_panel(lake_p, "ohlc", start_s, end_s)
    if not ohlc_df.empty:
        out["ohlc"] = _dict_from_long(
            ohlc_df,
            value_cols=["open", "high", "low", "close", "adj_close"],
            dates=dates_list,
            secids=secids,
            slots_rows=slots_rows,
        )

    micro_df = _load_or_panel(lake_p, "microstructure", start_s, end_s)
    if not micro_df.empty:
        out["microstructure"] = _dict_from_long(
            micro_df,
            value_cols=["eff_spread", "vwap_dev", "block_share", "turnover"],
            dates=dates_list,
            secids=secids,
            slots_rows=slots_rows,
        )

    # Sentiment = short_interest + analyst
    sent: dict[str, np.ndarray] = {}
    si = _load_or_panel(lake_p, "short_interest", start_s, end_s)
    if not si.empty:
        sent.update(
            _dict_from_long(
                si, value_cols=["si_pct"], dates=dates_list, secids=secids, slots_rows=slots_rows
            )
        )
    an = _load_or_panel(lake_p, "analyst", start_s, end_s)
    if not an.empty:
        sent.update(
            _dict_from_long(
                an,
                value_cols=["rec_mean_inv", "pt_gap"],
                dates=dates_list,
                secids=secids,
                slots_rows=slots_rows,
            )
        )
    if sent:
        out["sentiment"] = sent

    # Fundamentals PIT = worldscope + ibes + compustat
    fund: dict[str, np.ndarray] = {}
    for fam, cols in (
        ("worldscope", ["bp", "ep", "ta_growth", "rev_growth"]),
        (
            "ibes_ratios",
            [
                "bm",
                "ep_exi",
                "ps",
                "pcf",
                "dpr",
                "npm",
                "gpm",
                "roa",
                "roe",
                "cfm",
                "evm",
                "capex_inv",
            ],
        ),
        ("compustat", ["at_growth", "sale_growth", "ni_at", "dvc_at"]),
    ):
        df = _load_or_panel(lake_p, fam, start_s, end_s)
        if not df.empty:
            fund.update(
                _dict_from_long(
                    df, value_cols=cols, dates=dates_list, secids=secids, slots_rows=slots_rows
                )
            )
    if fund:
        out["fundamentals_pit"] = fund

    # Option flow + dividend
    oflow: dict[str, np.ndarray] = {}
    op = _load_or_panel(lake_p, "option_flow", start_s, end_s)
    if not op.empty:
        oflow.update(
            _dict_from_long(
                op,
                value_cols=["pc_vol", "pc_oi", "opt_stock_vol", "oi_lvl"],
                dates=dates_list,
                secids=secids,
                slots_rows=slots_rows,
            )
        )
    div = _load_or_panel(lake_p, "dividend", start_s, end_s)
    if not div.empty:
        oflow.update(
            _dict_from_long(
                div,
                value_cols=["div_yield_ttm"],
                dates=dates_list,
                secids=secids,
                slots_rows=slots_rows,
            )
        )
    if oflow:
        out["option_flow"] = oflow

    jkp = _load_or_panel(lake_p, "jkp", start_s, end_s)
    if not jkp.empty:
        out["jkp"] = _dict_from_long(
            jkp,
            value_cols=["log_me", "ivol_capm_21d", "ret_1_0"],
            dates=dates_list,
            secids=secids,
            slots_rows=slots_rows,
        )
        out["include_jkp_lottery"] = True

    # Rates term → merge into macro
    rates = _load_or_panel(lake_p, "rates_term", start_s, end_s)
    if not rates.empty and "date" in rates.columns:
        rates = rates.copy()
        rates["date"] = pd.to_datetime(rates["date"])
        rates = rates.set_index("date").sort_index()
        dates_idx = pd.to_datetime(dates_list)
        value_cols = [c for c in rates.columns if c != "date"]
        aligned = rates.reindex(dates_idx)
        rates_arr = aligned[value_cols].to_numpy(dtype=np.float64)
        macro = out.get("macro")
        macro_names = list(out.get("macro_names") or [])
        if macro is None:
            out["macro"] = rates_arr
            out["macro_names"] = value_cols
        else:
            m = np.asarray(macro, dtype=np.float64)
            if m.ndim == 2 and m.shape[0] == rates_arr.shape[0]:
                out["macro"] = np.concatenate([m, rates_arr], axis=-1)
                if not macro_names:
                    macro_names = [f"macro_{i}" for i in range(m.shape[-1])]
                out["macro_names"] = macro_names + value_cols

    # GICS industry labels aligned to fingerprint secids (static).
    gics = load_gics_map(lake_p)
    if not gics.empty:
        gmap = {
            _canonical_secid_key(r.secid): r.gics_industry
            for r in gics.itertuples(index=False)
        }
        if slots_rows is not None and len(slots_rows) > 0:
            # Per-slot industry of the first day's occupants (approx; dyn rotates).
            row0 = slots_rows[0]
            out["gics_industry"] = [
                gmap.get(_canonical_secid_key(s)) if s is not None else None for s in row0
            ]
        else:
            out["gics_industry"] = [
                gmap.get(_canonical_secid_key(s)) for s in secids
            ]

    return out


def build_feature_extras(
    cfg: Mapping[str, Any],
    *,
    lake: Path | str | None = None,
    dates: Sequence[Any],
    secids: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Merge cfg feature_extras with lake-backed feature-net panels + excludes."""
    extras = dict(cfg.get("feature_extras") or {})
    lake_root = lake or cfg.get("lake_root") or cfg.get("_lake_root") or LAKE_ROOT
    extras = attach_feature_net_extras(
        extras,
        lake=lake_root,
        dates=dates,
        secids=secids,
        slots_rows=slots_rows,
        start=start,
        end=end,
    )
    # Stamp exclude lists into extras so assemble can resolve them.
    if "feature_groups_exclude" in cfg:
        extras["feature_groups_exclude"] = list(cfg.get("feature_groups_exclude") or [])
    if "feature_channels_exclude" in cfg:
        extras["feature_channels_exclude"] = list(cfg.get("feature_channels_exclude") or [])
    # Pre-resolve for callers that want an explicit set (optional).
    try:
        extras["_exclude_channels_resolved"] = resolve_excludes(
            extras.get("feature_groups_exclude"),
            extras.get("feature_channels_exclude"),
            [],  # empty name list: only validates group names / registered channels
        )
    except ValueError:
        # Channel-only excludes need the live name list; assemble resolves again.
        pass
    return extras
