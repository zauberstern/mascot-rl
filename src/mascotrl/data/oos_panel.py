"""Materialize fixed-universe OptionMetrics daily marks into ArcticDB."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from mascotrl.data.arctic_store import ArcticStateStore
from mascotrl.data.duckdb_engine import DuckDBFeatureEngine, OptionFilterConfig
from mascotrl.data.pit_guards import membership_filter
from mascotrl.data.paths import ARCTIC_ROOT, LAKE_ROOT
from mascotrl.logging_utils import get_logger

log = get_logger("volsurf.l5.oos")

SIGNALS_SYMBOL = "constituent_signals"
UNIVERSE_SYMBOL = "oos_universe"
FEATURE_STEMS = (
    "atm_iv",
    "skew_25d",
    "bid_ask_spread",
    "mid",
    "delta",
    "spot",
    "strike",
    "dh_denom",
    "dh_denom_lagdelta",
    "dh_ret",
    "dh_ret_lagdelta",
    "fwd_ret",
    "stk_ret",
    "stk_ret_h_days",
    "volume_imbalance",
    "put_call_oi_ratio",
)

# Headline P&L label: Molnár-lagged delta-hedged call return (IMR-bias corrected).
# Contemporaneous ``dh_ret`` is retained as a robustness stem only.
LABEL_STEM = "dh_ret_lagdelta"
ROBUSTNESS_LABEL_STEM = "dh_ret"
# Legacy naked dollar mid change. Not a return on invested capital; retained
# only for documented robustness rows.
LEGACY_LABEL_STEM = "fwd_ret"
# Equity simple return aligned to the same (date, date_next) pairs as the
# headline option label (spectrum arms eq / mix).
EQUITY_LABEL_STEM = "stk_ret"

# Capital-base column paired with each label stem for hedge-leg cost scaling.
_DENOM_BY_LABEL = {
    "dh_ret_lagdelta": "dh_denom_lagdelta",
    "dh_ret": "dh_denom",
    "stk_ret": "spot",
}

# Realized-return / hold-day stems: never ffill (stale P&L fabrication).
_NO_FFILL_STEMS = frozenset(
    {
        LABEL_STEM,
        ROBUSTNESS_LABEL_STEM,
        LEGACY_LABEL_STEM,
        EQUITY_LABEL_STEM,
        "stk_ret_h_days",
    }
)


def denom_stem_for_label(label_stem: str | None = None) -> str:
    """Return the capital-base feature stem matching ``label_stem``."""
    stem = label_stem or LABEL_STEM
    return _DENOM_BY_LABEL.get(stem, "dh_denom")


def no_ffill_label_columns(wide: pd.DataFrame, n_secids: int) -> list[str]:
    """Wide columns that must stay NaN when missing (labels / hold days)."""
    cols: list[str] = []
    for stem in _NO_FFILL_STEMS:
        for i in range(int(n_secids)):
            c = f"{stem}_{i}"
            if c in wide.columns:
                cols.append(c)
    return cols


def pivot_long_marks_to_wide(
    long_tbl: pa.Table,
    secids: list[int],
) -> pd.DataFrame:
    """
    Long (date, secid, …) → wide DatetimeIndex frame with columns
    ``{feat}_{slot}`` for slot in 0..K-1 following ``secids`` order.

    Feature stems are event-time ffilled within each name. Label stems
    (``dh_ret*`` / ``fwd_ret`` / ``stk_ret`` / ``stk_ret_h_days``) are never
    ffilled: a missing realized return must stay NaN so coverage and OOS masks
    see true absence.
    """
    df = long_tbl.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["secid"] = df["secid"].astype(int)
    slot = {int(s): i for i, s in enumerate(secids)}
    df = df[df["secid"].isin(slot)].copy()
    df["slot"] = df["secid"].map(slot)

    wide_parts: list[pd.DataFrame] = []
    for feat in FEATURE_STEMS:
        if feat not in df.columns:
            continue
        piv = df.pivot_table(index="date", columns="slot", values=feat, aggfunc="last")
        piv = piv.reindex(columns=list(range(len(secids))))
        piv.columns = [f"{feat}_{i}" for i in range(len(secids))]
        if feat not in _NO_FFILL_STEMS:
            # Event-time ffill within each name; leave leading NaNs for strict drop later.
            piv = piv.ffill()
        wide_parts.append(piv)
    if not wide_parts:
        raise ValueError("no feature columns to pivot")
    wide = pd.concat(wide_parts, axis=1).sort_index()
    return wide


# @lat: [[data-rim#Lake to Arctic]]
def materialize_oos_panel(
    *,
    n_assets: int,
    universe_start: str,
    universe_end: str,
    panel_start: str,
    panel_end: str,
    lake_base_dir: str | Path | None = None,
    arctic_db_path: str | Path | None = None,
    arctic_library: str = "hyper_volanet_features",
    duckdb_threads: int = 4,
    max_memory: str = "12GB",
    secids: list[int] | None = None,
    tickers: list[str] | None = None,
    issuers: list[str] | None = None,
    universe_mode: str = "iv_hypergraph",
    liquidity_pool: int = 200,
    iv_corr_threshold: float = 0.35,
    universe_tail_threshold: float = 0.90,
    universe_selection_metric: str = "copula_tail",
    option_filters: OptionFilterConfig | None = None,
    attrition_out_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build IV-hypergraph (or liquid) universe, compute daily ATM marks/signals,
    persist to Arctic as ``constituent_signals`` (+ ``oos_universe`` metadata).

    ``option_filters`` applies the literature-standard chain screens (see
    :class:`OptionFilterConfig`). When ``attrition_out_path`` is set, the
    per-screen attrition table and dividend-bridge coverage are written there
    for the paper's data appendix.
    """
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    eng = DuckDBFeatureEngine(lake_base_dir=lake)
    eng.con.execute(f"SET threads = {int(duckdb_threads)};")
    eng.con.execute(f"SET max_memory = '{max_memory}';")

    store = ArcticStateStore(
        db_path=arctic_db_path or ARCTIC_ROOT,
        library_name=arctic_library,
    )

    if secids is None:
        mode = str(universe_mode or "iv_hypergraph").lower()
        # Alpha v2 primary lock: freeze membership selection end at 2021-12-31.
        if mode in (
            "frozen_at_2021_end",
            "frozen_2021",
            "frozen",
        ) or str(universe_mode).upper() == "FROZEN_AT_2021_END":
            from mascotrl.features.pit_universe import (
                FROZEN_AT_2021_END,
                resolve_universe_end_for_mode,
            )

            universe_end = resolve_universe_end_for_mode(FROZEN_AT_2021_END)
            mode = "iv_hypergraph"  # selection algorithm unchanged; end date frozen
        log.info(
            "Selecting universe mode=%s n=%d window=[%s, %s]",
            mode,
            n_assets,
            universe_start,
            universe_end,
        )
        # Oversample before PIT membership so index drops do not undershoot K.
        n_select = max(int(n_assets) * 2, int(n_assets) + 25)
        if mode in ("liquid", "oi", "open_interest"):
            rows = eng.select_liquid_universe(n_select, universe_start, universe_end)
        else:
            rows = eng.select_iv_hypergraph_universe(
                n_select,
                universe_start,
                universe_end,
                liquidity_pool=max(int(liquidity_pool), n_select),
                corr_threshold=float(iv_corr_threshold),
                tail_threshold=float(universe_tail_threshold),
                selection_metric=str(universe_selection_metric),
            )
        # Index membership as of the selection date. Names that were not in the
        # index at universe_end are dropped so the universe is not built from
        # post-hoc constituents (survivorship).
        members = eng.pit_membership_tickers(universe_end)
        rows, membership_meta = membership_filter(rows, members)
        if membership_meta.get("enforced"):
            log.info(
                "PIT membership as of %s: %d/%d candidates retained",
                universe_end,
                membership_meta["n_out"],
                membership_meta["n_in"],
            )
        else:
            log.warning("PIT membership not enforced: %s", membership_meta.get("reason"))
        rows = rows[: int(n_assets)]
        if len(rows) < n_assets:
            raise RuntimeError(
                f"Universe only returned {len(rows)} names; need {n_assets}"
                + (
                    f" (after PIT membership filter dropped "
                    f"{membership_meta.get('n_dropped_non_member', 0)})"
                    if membership_meta.get("enforced")
                    else ""
                )
            )
        rows = rows[:n_assets]
        secids = [int(r["secid"]) for r in rows]
        tickers = [str(r["ticker"]) for r in rows]
        issuers = [str(r.get("issuer") or "") for r in rows]
        selection_meta = {
            "universe_mode": mode,
            "universe_selection_metric": str(universe_selection_metric),
            "pit_membership": membership_meta,
            "index_iv_corr": [float(r.get("index_iv_corr", float("nan"))) for r in rows],
            "tail_dependence_score": [
                float(r.get("tail_dependence_score", float("nan"))) for r in rows
            ],
        }
    else:
        selection_meta = {"universe_mode": "explicit"}
        secids = [int(s) for s in secids[:n_assets]]
        if tickers is None:
            tickers = [f"SECID_{s}" for s in secids]
        else:
            tickers = [str(t).strip().upper() or f"SECID_{s}" for t, s in zip(tickers, secids)]
        if issuers is None:
            issuers = [""] * len(secids)

    # Disambiguate duplicate tickers with secid suffix.
    seen: dict[str, int] = {}
    display_names: list[str] = []
    for t, s in zip(tickers, secids):
        if t in seen:
            display_names.append(f"{t}_{s}")
        else:
            seen[t] = s
            display_names.append(t)

    log.info(
        "Universe: %s",
        ", ".join(f"{n}({s})" for n, s in zip(display_names, secids)),
    )

    filters = option_filters or OptionFilterConfig()
    attrition: dict[str, Any] | None = None
    if attrition_out_path is not None:
        log.info("Computing filter attrition for data appendix …")
        try:
            att = eng.compute_filter_attrition(
                secids, panel_start, panel_end, filters=filters
            ).to_pandas()
            cov = eng.dividend_bridge_coverage(
                secids, panel_start, panel_end, filters=filters
            ).to_pandas()
            attrition = {
                "screens": {k: int(v) for k, v in att.iloc[0].items()},
                "dividend_bridge": {k: int(v) for k, v in cov.iloc[0].items()},
                "filter_config": asdict(filters),
                "panel_start": panel_start,
                "panel_end": panel_end,
                "note": (
                    "Marginal counts are each screen evaluated independently on "
                    "the base chain window; n_retained is the joint effect. "
                    "Screens follow Cao and Han (2013), Goyal and Saretto (2009)."
                ),
            }
            p = Path(attrition_out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(attrition, indent=2) + "\n")
            log.info("Filter attrition → %s", p)
        except Exception as exc:  # attrition is a disclosure aid, never fatal
            log.warning("filter attrition unavailable: %s", exc)

    log.info("Computing daily marks [%s, %s] …", panel_start, panel_end)
    long_tbl = eng.compute_universe_daily_marks(
        secids, panel_start, panel_end, filters=filters
    )
    log.info("Long marks rows=%d cols=%s", long_tbl.num_rows, long_tbl.column_names)

    from mascotrl.data.duckdb_engine import quote_quality_audit_counts

    quote_audit = quote_quality_audit_counts(long_tbl.to_pandas())
    if int(quote_audit.get("n_crossed_bid_ask") or 0) > 0:
        log.warning(
            "quote quality: n_crossed_bid_ask=%s n_stale_last_date=%s n_zero_volume=%s",
            quote_audit.get("n_crossed_bid_ask"),
            quote_audit.get("n_stale_last_date"),
            quote_audit.get("n_zero_volume"),
        )
    else:
        log.info("quote quality audit: %s", quote_audit)
    if attrition is not None:
        attrition["quote_quality"] = quote_audit
        if attrition_out_path is not None:
            Path(attrition_out_path).write_text(json.dumps(attrition, indent=2) + "\n")

    arb_meta: dict[str, Any] = {
        "calendar_arb_fail": 0,
        "butterfly_arb_fail": 0,
        "n_bad_secid_dates": 0,
        "rows_dropped": 0,
    }
    if filters.no_calendar_arbitrage or filters.no_butterfly_arbitrage:
        from mascotrl.data.arbitrage_screens import filter_long_marks

        log.info("Computing calendar/butterfly arbitrage screens …")
        try:
            arb = eng.compute_surface_arb_violations(
                secids,
                panel_start,
                panel_end,
                calendar=bool(filters.no_calendar_arbitrage),
                butterfly=bool(filters.no_butterfly_arbitrage),
            )
            bad_keys = arb.get("bad_keys") or set()
            arb_meta = {
                "calendar_arb_fail": int(arb.get("n_calendar_fail_days", 0)),
                "butterfly_arb_fail": int(arb.get("n_butterfly_fail_days", 0)),
                "n_bad_secid_dates": int(arb.get("n_bad_secid_dates", 0)),
                "rows_dropped": 0,
                "drop_surface_arb_days": bool(filters.drop_surface_arb_days),
            }
            if bad_keys and filters.drop_surface_arb_days:
                long_df = long_tbl.to_pandas()
                n_before = len(long_df)
                long_df = filter_long_marks(long_df, bad_keys)
                arb_meta["rows_dropped"] = int(n_before - len(long_df))
                long_tbl = pa.Table.from_pandas(long_df, preserve_index=False)
                log.info(
                    "Arb screens dropped %d mark rows (%d bad secid-dates)",
                    arb_meta["rows_dropped"],
                    arb_meta["n_bad_secid_dates"],
                )
            elif bad_keys:
                log.info(
                    "Arb screens attrition-only: %d bad secid-dates "
                    "(drop_surface_arb_days=False)",
                    arb_meta["n_bad_secid_dates"],
                )
        except Exception as exc:
            log.warning("surface arb screens unavailable: %s", exc)

    if attrition is not None:
        attrition["surface_arbitrage"] = arb_meta
        if attrition_out_path is not None:
            Path(attrition_out_path).write_text(json.dumps(attrition, indent=2) + "\n")

    wide = pivot_long_marks_to_wide(long_tbl, secids)
    # Realized-return stems (option + equity) are excluded from ffill:
    # forward-filling a realized return would repeat stale P&L.
    label_cols = [f"{LABEL_STEM}_{i}" for i in range(len(secids))]
    ret_cols = no_ffill_label_columns(wide, len(secids))
    coverage_cols = [c for c in label_cols if c in wide.columns] or ret_cols
    present = wide[coverage_cols].notna().sum(axis=1)
    need = max(1, (len(secids) + 1) // 2)
    n_rows_before = int(len(wide))
    wide = wide.loc[present >= need].copy()
    feat_cols = [c for c in wide.columns if c not in ret_cols]
    wide[feat_cols] = wide[feat_cols].ffill()
    # Missing realized returns are NOT imputed to 0.0. A zero-filled label is a
    # fabricated flat day: it dilutes volatility, inflates Sharpe, and lets the
    # policy be scored on names it had no tradable mark for. Instead the
    # absence is preserved and the evaluator masks those cells per name.
    label_coverage = {
        "rows_before_coverage_filter": n_rows_before,
        "rows_after_coverage_filter": int(len(wide)),
        "label_cell_missing_rate": (
            float(wide[coverage_cols].isna().to_numpy().mean())
            if coverage_cols and len(wide)
            else 0.0
        ),
        "min_names_required": need,
        "zero_imputation": False,
    }
    log.info(
        "Label coverage: rows %d→%d, missing label cells %.4f (no zero-imputation)",
        n_rows_before,
        len(wide),
        label_coverage["label_cell_missing_rate"],
    )
    if wide.empty:
        raise RuntimeError("Wide OOS panel empty after coverage filter")

    meta = {
        "secids": secids,
        "tickers": list(tickers),
        "issuers": list(issuers),
        "display_names": display_names,
        "n_assets": len(secids),
        "universe_start": universe_start,
        "universe_end": universe_end,
        "panel_start": panel_start,
        "panel_end": panel_end,
        "feature_stems": list(FEATURE_STEMS),
        "label_stem": LABEL_STEM,
        "option_filters": asdict(filters),
        "filter_attrition": attrition,
        "quote_quality": quote_audit,
        "label_coverage": label_coverage,
        "knowledge_written_at": pd.Timestamp.utcnow().isoformat(),
        **selection_meta,
    }
    uni_df = pd.DataFrame(
        {"payload": [json.dumps(meta)]},
        index=pd.DatetimeIndex([pd.Timestamp(panel_end)], name="date"),
    )
    if UNIVERSE_SYMBOL in store.lib.list_symbols():
        store.lib.update(UNIVERSE_SYMBOL, uni_df, metadata=meta)
    else:
        store.lib.write(UNIVERSE_SYMBOL, uni_df, metadata=meta)

    out = wide.reset_index()
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    table = pa.Table.from_pandas(out, preserve_index=False)
    store.persist_features(SIGNALS_SYMBOL, table, metadata=meta)
    log.info(
        "Persisted %s rows=%d cols=%d symbols=%s",
        SIGNALS_SYMBOL,
        len(wide),
        wide.shape[1],
        store.list_available_features(),
    )
    return {
        "secids": secids,
        "tickers": list(tickers),
        "display_names": display_names,
        "n_rows": int(len(wide)),
        "n_cols": int(wide.shape[1]),
        "start": str(wide.index.min().date()),
        "end": str(wide.index.max().date()),
        "arctic_symbols": store.list_available_features(),
        "meta": meta,
    }


def load_universe_meta(store: ArcticStateStore) -> dict[str, Any]:
    """Return oos_universe metadata (secids, tickers, display_names)."""
    if UNIVERSE_SYMBOL not in store.list_available_features():
        return {}
    try:
        meta = dict(store.lib.read(UNIVERSE_SYMBOL).metadata or {})
        if meta.get("secids"):
            return meta
    except Exception:
        pass
    try:
        uni = store.read_pit_state(UNIVERSE_SYMBOL, as_of=None)
        return json.loads(uni["payload"].iloc[-1])
    except Exception:
        return {}


def load_oos_panel(
    store: ArcticStateStore,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Load wide constituent_signals + universe secids (knowledge PIT)."""
    if SIGNALS_SYMBOL not in store.list_available_features():
        raise KeyError(
            f"{SIGNALS_SYMBOL} missing in Arctic — run scripts/materialize_oos_arctic.py"
        )
    df = store.read_pit_state(SIGNALS_SYMBOL, as_of=as_of)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("constituent_signals must be DatetimeIndexed")
    df = df.sort_index()
    if start is not None:
        df = df.loc[pd.Timestamp(start) :]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]

    meta = load_universe_meta(store)
    secids = [int(x) for x in (meta.get("secids") or [])]
    if not secids:
        k = sum(1 for c in df.columns if str(c).startswith("atm_iv_"))
        secids = list(range(k))
    df.attrs["pit_as_of"] = as_of.isoformat() if as_of is not None else None
    return df, secids


def wide_feature_matrix(df: pd.DataFrame, stem: str, n_assets: int) -> np.ndarray:
    cols = [f"{stem}_{i}" for i in range(n_assets)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    return df[cols].to_numpy(dtype=np.float64)


def label_matrix(
    df: pd.DataFrame,
    n_assets: int,
    *,
    stem: str | None = None,
) -> np.ndarray:
    """
    Realized P&L label matrix (T, K), defaulting to the delta-hedged return.

    Single source of truth so evaluators cannot silently diverge on which
    column is the paper's return definition. Panels materialized before the
    delta-hedged label existed raise a directed error rather than falling back
    to the legacy dollar mid change, which is not a return on capital.
    """
    stem = stem or LABEL_STEM
    try:
        return wide_feature_matrix(df, stem, n_assets)
    except KeyError as exc:
        raise KeyError(
            f"panel lacks '{stem}' columns — rematerialize the OOS panel "
            "(oos_force_rematerialize: true) so the required label stem is built"
        ) from exc


def extract_om_marks(
    df: pd.DataFrame,
    *,
    n_opt: int,
    n_eq: int = 0,
    label_stem: str | None = None,
) -> dict[str, np.ndarray]:
    """Wide OM mark matrices aligned to ``[opt_0..|eq_0..]`` slot layout.

    Equity-block columns are zero-padded for half_spread/delta/capital_base
    (equity friction uses equity_bps, not OM-touch). Spot for the equity
    block prefers ``spot_*`` when present, else zeros.
    """
    n_opt = int(n_opt)
    n_eq = int(n_eq)
    n_total = n_opt + n_eq
    if n_opt <= 0:
        raise ValueError("extract_om_marks requires n_opt > 0")
    denom = denom_stem_for_label(label_stem or LABEL_STEM)
    half = wide_feature_matrix(df, "bid_ask_spread", n_opt)
    delta = wide_feature_matrix(df, "delta", n_opt)
    spot_opt = wide_feature_matrix(df, "spot", n_opt)
    capital = wide_feature_matrix(df, denom, n_opt)
    T = int(half.shape[0])
    if n_eq <= 0:
        return {
            "half_spread": half,
            "delta": delta,
            "spot": spot_opt,
            "capital_base": capital,
        }
    zeros = np.zeros((T, n_eq), dtype=np.float64)
    degraded = False
    try:
        spot_eq = wide_feature_matrix(df, "spot", n_eq)
    except KeyError:
        spot_eq = zeros.copy()
        degraded = True
    out = {
        "half_spread": np.concatenate([half, zeros], axis=1),
        "delta": np.concatenate([delta, zeros], axis=1),
        "spot": np.concatenate([spot_opt, spot_eq], axis=1),
        "capital_base": np.concatenate([capital, zeros], axis=1),
    }
    if degraded:
        out["om_marks_degraded"] = True
    return out
