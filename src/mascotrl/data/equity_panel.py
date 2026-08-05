"""Standalone equity panel: ``stk_ret`` without option ``label_ok_lag``.

Builds a wide ``{stem}_{slot}`` panel from OptionMetrics security prices
(``macro/sp500_sec.parquet``), optionally reconciled to CRSP RET/DLRET via
``macro/crsp_optionm_link.parquet`` + ``macro/sp500_prices.parquet``.

Calendar (locked for allocation CPCV):
  - Selection W: 2003-01-02 .. 2012-12-31
  - Embargo: 2013 (unused)
  - Eval: 2014-01-01 .. 2024-12-31

``stk_ret`` here is the security simple return (CRSP-compounded when linked,
else OM ``return``). It does **not** require an option chain or ``label_ok_lag``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.paths import ARCTIC_ROOT, LAKE_ROOT
from src.data.slot_mask import build_members_by_date_from_intervals
from src.features.pit_universe import compound_equity_return, validate_delist_handling
from src.logging_utils import get_logger

log = get_logger("volsurf.data.equity_panel")

# Locked allocation calendar (Phase D).
SELECTION_START = "2003-01-02"
SELECTION_END = "2012-12-31"
EMBARGO_YEAR = 2013
EVAL_START = "2014-01-01"
EVAL_END = "2024-12-31"

UNIVERSE_MODE_EQUITY_SP500 = "equity_sp500"
UNIVERSE_MODE_EXPLICIT = "explicit"


def resolve_universe_secids(
    universe_mode: str,
    *,
    secids: list[int] | Sequence[int] | None = None,
    selector_result: dict[str, Any] | None = None,
) -> list[int] | None:
    """Resolve investible secids for materialize (no lake selector fit in unit tests).

    Returns ``None`` for modes that infer membership from the lake. When
    ``secids`` or ``selector_result['secids']`` are provided, those win.
    """
    mode = str(universe_mode or UNIVERSE_MODE_EXPLICIT).lower().strip()
    if secids is not None:
        return [int(s) for s in secids]
    if selector_result is not None:
        if selector_result.get("secids") is not None:
            return [int(s) for s in selector_result["secids"]]
        raise ValueError(
            f"selector_result for universe_mode={mode!r} must include 'secids' "
            "(map column indices to secids before materialize)"
        )
    return None

SIGNALS_SYMBOL = "equity_signals"
UNIVERSE_SYMBOL = "equity_universe"

FEATURE_STEMS = (
    "stk_ret",
    "close",
    "mktcap",
    "dollar_volume",
    "hv_21",
    "hv_63",
    "ret_12_1",
    "ret_1m",
    "amihud_illiq",
)

# Realized-return / liquidity ratios: never ffill.
_NO_FFILL_STEMS = frozenset({"stk_ret", "amihud_illiq", "ret_12_1", "ret_1m", "hv_21", "hv_63"})

_CFADJ_TOL = 1e-6
_CFADJ_SAMPLE = 500
_CFADJ_FAIL_RATE = 0.05


def load_sp500_security_returns(
    lake: str | Path | None,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Read ``macro/sp500_sec.parquet`` for ``[start, end]``.

    Expected columns (when present): secid, date, ticker, close, return,
    shrout, cfadj, volume (plus any extras retained as-is).
    """
    root = Path(lake) if lake is not None else LAKE_ROOT
    path = root / "macro" / "sp500_sec.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing security returns lake file: {path}")
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise KeyError("sp500_sec.parquet missing 'date'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if "secid" in out.columns:
        out["secid"] = pd.to_numeric(out["secid"], errors="coerce").astype("Int64")
    if "return" in out.columns:
        out["return"] = pd.to_numeric(out["return"], errors="coerce")
    for col in ("close", "shrout", "cfadj", "cfret", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    return out.sort_values(["secid", "date"]).reset_index(drop=True)


def _validate_cfadj_return(df: pd.DataFrame) -> dict[str, Any]:
    """Fail closed if OM ``return`` systematically disagrees with cfadj closes."""
    need = {"secid", "date", "close", "return", "cfadj"}
    if not need.issubset(df.columns) or df.empty:
        return {"checked": False, "reason": "missing columns or empty"}
    bad = 0
    checked = 0
    for _, g in df.dropna(subset=["close", "return", "cfadj"]).groupby("secid"):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        prev_c = g["close"].shift(1)
        prev_a = g["cfadj"].shift(1)
        denom = prev_c * prev_a
        implied = (g["close"] * g["cfadj"]) / denom - 1.0
        mask = denom.notna() & (denom != 0) & g["return"].notna() & implied.notna()
        if not mask.any():
            continue
        # Cap per-name contribution so one long series does not dominate.
        take = mask.to_numpy().nonzero()[0][:3]
        for i in take:
            checked += 1
            if abs(float(g["return"].iloc[i]) - float(implied.iloc[i])) >= _CFADJ_TOL:
                bad += 1
            if checked >= _CFADJ_SAMPLE:
                break
        if checked >= _CFADJ_SAMPLE:
            break
    rate = float(bad) / float(checked) if checked else 0.0
    meta = {"checked": True, "n_checked": checked, "n_bad": bad, "bad_rate": rate}
    if checked >= 20 and rate > _CFADJ_FAIL_RATE:
        raise ValueError(
            f"cfadj return validation failed: bad_rate={rate:.3f} "
            f"(n_bad={bad}/{checked}); refuse systematic violation"
        )
    return meta


def reconcile_with_crsp(
    df: pd.DataFrame,
    link_path: str | Path | None,
    *,
    crsp_path: str | Path | None = None,
    lake: str | Path | None = None,
) -> pd.DataFrame:
    """Join OM security rows to CRSP RET/DLRET; fall back to OM ``return``.

    Uses ``compound_equity_return`` and ``validate_delist_handling``. Rows without
    a valid CRSP link keep OM ``return`` and ``return_source='optionmetrics'``.
    """
    out = df.copy()
    if "return" not in out.columns:
        raise KeyError("reconcile_with_crsp requires 'return' column")
    out["date"] = pd.to_datetime(out["date"])
    out["stk_ret"] = pd.to_numeric(out["return"], errors="coerce")
    out["return_source"] = "optionmetrics"

    root = Path(lake) if lake is not None else None
    link_p = Path(link_path) if link_path is not None else None
    if link_p is None and root is not None:
        cand = root / "macro" / "crsp_optionm_link.parquet"
        link_p = cand if cand.is_file() else None
    crsp_p = Path(crsp_path) if crsp_path is not None else None
    if crsp_p is None and root is not None:
        cand = root / "macro" / "sp500_prices.parquet"
        crsp_p = cand if cand.is_file() else None
    if link_p is None or not link_p.is_file() or crsp_p is None or not crsp_p.is_file():
        return out

    link = pd.read_parquet(link_p)
    if link.empty or not {"secid", "permno", "sdate", "edate"}.issubset(link.columns):
        return out
    link = link.copy()
    link["secid"] = pd.to_numeric(link["secid"], errors="coerce").astype("Int64")
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce").astype("Int64")
    link["sdate"] = pd.to_datetime(link["sdate"], errors="coerce")
    link["edate"] = pd.to_datetime(link["edate"], errors="coerce")
    link["edate"] = link["edate"].fillna(pd.Timestamp("2262-04-11"))
    if "score" in link.columns:
        link = link.sort_values(["secid", "score"], ascending=[True, True])
    else:
        link = link.sort_values(["secid", "sdate"])

    crsp = pd.read_parquet(crsp_p)
    # Normalize CRSP column names.
    rename = {}
    for src, dst in (("PERMNO", "permno"), ("date", "date"), ("RET", "RET"), ("DLRET", "DLRET")):
        if src in crsp.columns:
            rename[src] = dst
        elif src.lower() in crsp.columns and src not in crsp.columns:
            rename[src.lower()] = dst
    crsp = crsp.rename(columns=rename)
    need_crsp = {"permno", "date", "RET"}
    if not need_crsp.issubset(crsp.columns) or crsp.empty:
        return out
    crsp = crsp.copy()
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64")
    crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
    crsp["RET"] = pd.to_numeric(crsp["RET"], errors="coerce")
    if "DLRET" in crsp.columns:
        crsp["DLRET"] = pd.to_numeric(crsp["DLRET"], errors="coerce")
    else:
        crsp["DLRET"] = np.nan
    crsp = crsp.dropna(subset=["permno", "date"]).drop_duplicates(
        subset=["permno", "date"], keep="last"
    )

    # Interval join: secid → permno for date in [sdate, edate].
    base = out.reset_index(drop=True)
    base["_row"] = np.arange(len(base))
    merged = base.merge(link[["secid", "permno", "sdate", "edate"]], on="secid", how="left")
    in_win = (
        merged["permno"].notna()
        & merged["sdate"].notna()
        & (merged["date"] >= merged["sdate"])
        & (merged["date"] <= merged["edate"])
    )
    merged = merged.loc[in_win | merged["permno"].isna()].copy()
    # Prefer first link when multiple match (already score-sorted).
    merged = merged.sort_values(["_row", "sdate"]).drop_duplicates("_row", keep="first")

    with_crsp = merged.merge(
        crsp[["permno", "date", "RET", "DLRET"]],
        on=["permno", "date"],
        how="left",
    )
    with_crsp = with_crsp.set_index("_row").reindex(base["_row"]).reset_index(drop=True)

    stk = base["stk_ret"].to_numpy(dtype=np.float64, copy=True)
    src = np.array(["optionmetrics"] * len(base), dtype=object)
    ret = with_crsp["RET"].to_numpy(dtype=np.float64)
    dl = with_crsp["DLRET"].to_numpy(dtype=np.float64)
    has_crsp = np.isfinite(ret)
    if np.any(has_crsp):
        delist_flag = np.isfinite(dl) & has_crsp
        if np.any(delist_flag):
            validate_delist_handling(delist_flag=delist_flag, dlret=dl)
        for i in np.where(has_crsp)[0]:
            d_i = float(dl[i]) if np.isfinite(dl[i]) else None
            stk[i] = compound_equity_return(float(ret[i]), d_i)
            src[i] = "crsp"
    out["stk_ret"] = stk
    out["return_source"] = src
    return out


def _rolling_hv(ret: pd.Series, window: int) -> pd.Series:
    """Annualized trailing sample stdev (ddof=1) x sqrt(252).

    Single convention shared with ``src.features.blocks.volatility_vrp.trailing_hv_panel``.
    """
    import numpy as np

    return ret.rolling(window, min_periods=window).std(ddof=1) * float(np.sqrt(252.0))


def _rolling_prod_return(ret: pd.Series, window: int) -> pd.Series:
    """Trailing compound return over ``window`` days: prod(1+r)-1."""
    return (1.0 + ret).rolling(window, min_periods=window).apply(
        lambda x: float(np.nanprod(x) - 1.0), raw=True
    )


def _add_feature_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-secid features from reconciled long frame (has stk_ret, close, ...)."""
    parts: list[pd.DataFrame] = []
    for _, g in long_df.groupby("secid", sort=False):
        g = g.sort_values("date").copy()
        r = pd.to_numeric(g["stk_ret"], errors="coerce")
        close = pd.to_numeric(g.get("close"), errors="coerce")
        shrout = pd.to_numeric(g.get("shrout"), errors="coerce")
        vol = pd.to_numeric(g.get("volume"), errors="coerce")
        g["mktcap"] = close * shrout
        g["dollar_volume"] = close.abs() * vol
        g["hv_21"] = _rolling_hv(r, 21)
        g["hv_63"] = _rolling_hv(r, 63)
        # 12-1 momentum: compound return over ~252d excluding most recent ~21d.
        ret_12 = _rolling_prod_return(r, 252)
        ret_1 = _rolling_prod_return(r, 21)
        g["ret_1m"] = ret_1
        # (1+r_252)/(1+r_21) - 1  ≈ skip-last-month 12m return
        g["ret_12_1"] = (1.0 + ret_12) / (1.0 + ret_1) - 1.0
        illiq = r.abs() / g["dollar_volume"].replace(0.0, np.nan)
        g["amihud_illiq"] = illiq.rolling(21, min_periods=21).mean()
        parts.append(g)
    if not parts:
        return long_df
    return pd.concat(parts, ignore_index=True)


def _apply_pit_membership_nan(
    long_df: pd.DataFrame,
    members_by_date: dict,
) -> pd.DataFrame:
    """NaN feature/label cells on dates when ticker is not an index member."""
    if not members_by_date or "ticker" not in long_df.columns:
        return long_df
    out = long_df.copy()
    keep_cols = [c for c in FEATURE_STEMS if c in out.columns]
    if not keep_cols:
        return out
    tick = out["ticker"].astype(str).str.strip().str.upper()
    dates = pd.to_datetime(out["date"])
    active = np.ones(len(out), dtype=bool)
    for i, d in enumerate(dates):
        members = None
        if d in members_by_date:
            members = members_by_date[d]
        else:
            ts = pd.Timestamp(d).normalize()
            for key in (ts, str(ts.date()), d):
                if key in members_by_date:
                    members = members_by_date[key]
                    break
        if members is None:
            continue
        member_set = {str(x).strip().upper() for x in members if x is not None}
        if not member_set:
            continue
        active[i] = tick.iloc[i] in member_set
    if np.all(active):
        return out
    out.loc[~active, keep_cols] = np.nan
    return out


def pivot_equity_long_to_wide(
    long_df: pd.DataFrame,
    secids: Sequence[int],
) -> pd.DataFrame:
    """Long (date, secid, stems) → wide ``{stem}_{slot}`` DatetimeIndex frame."""
    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["secid"] = df["secid"].astype(int)
    slot = {int(s): i for i, s in enumerate(secids)}
    df = df[df["secid"].isin(slot)].copy()
    if df.empty:
        raise ValueError("no equity rows for requested secids")
    df["slot"] = df["secid"].map(slot)
    wide_parts: list[pd.DataFrame] = []
    for feat in FEATURE_STEMS:
        if feat not in df.columns:
            continue
        piv = df.pivot_table(index="date", columns="slot", values=feat, aggfunc="last")
        piv = piv.reindex(columns=list(range(len(secids))))
        piv.columns = [f"{feat}_{i}" for i in range(len(secids))]
        if feat not in _NO_FFILL_STEMS:
            piv = piv.ffill()
        wide_parts.append(piv)
    if not wide_parts:
        raise ValueError("no equity feature columns to pivot")
    return pd.concat(wide_parts, axis=1).sort_index()


def materialize_equity_panel(
    *,
    panel_start: str,
    panel_end: str,
    secids: list[int] | None = None,
    tickers: list[str] | None = None,
    lake_base_dir: str | Path | None = None,
    arctic_db_path: str | Path | None = None,
    arctic_library: str = "hyper_volanet_features",
    out_dir: str | Path | None = None,
    universe_mode: str = UNIVERSE_MODE_EXPLICIT,
    reconcile: bool = True,
    apply_pit_membership: bool = True,
    validate_cfadj: bool = True,
    selector_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build wide equity panel; write Arctic symbols or parquet under ``out_dir``.

    ``universe_mode``:
      - ``explicit`` / ``equity_sp500``: require caller ``secids`` (or infer unique
        secids in the window for ``equity_sp500`` when omitted).
    """
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    mode = str(universe_mode or UNIVERSE_MODE_EXPLICIT).lower().strip()

    if mode not in {
        UNIVERSE_MODE_EXPLICIT,
        UNIVERSE_MODE_EQUITY_SP500,
    }:
        raise ValueError(
            f"unsupported universe_mode={universe_mode!r}; "
            f"allowed={[UNIVERSE_MODE_EXPLICIT, UNIVERSE_MODE_EQUITY_SP500]}"
        )

    if secids is None and selector_result is not None:
        secids = resolve_universe_secids(
            mode, secids=secids, selector_result=selector_result
        )

    raw = load_sp500_security_returns(lake, panel_start, panel_end)
    if raw.empty:
        raise RuntimeError(
            f"empty sp500_sec panel for [{panel_start}, {panel_end}] under {lake}"
        )

    cfadj_meta: dict[str, Any] = {"checked": False}
    if validate_cfadj:
        cfadj_meta = _validate_cfadj_return(raw)

    if reconcile:
        long_df = reconcile_with_crsp(
            raw,
            lake / "macro" / "crsp_optionm_link.parquet",
            crsp_path=lake / "macro" / "sp500_prices.parquet",
            lake=lake,
        )
    else:
        long_df = raw.copy()
        long_df["stk_ret"] = pd.to_numeric(long_df["return"], errors="coerce")
        long_df["return_source"] = "optionmetrics"

    if secids is None:
        if mode != UNIVERSE_MODE_EQUITY_SP500:
            raise ValueError("secids required unless universe_mode=equity_sp500")
        secids = sorted({int(s) for s in long_df["secid"].dropna().unique()})
        if not secids:
            raise RuntimeError("equity_sp500 mode found no secids in window")
    else:
        secids = [int(s) for s in secids]

    long_df = long_df[long_df["secid"].isin(secids)].copy()
    if long_df.empty:
        raise RuntimeError("no rows after secid filter")

    if tickers is None:
        tick_map = (
            long_df.dropna(subset=["ticker"])
            .drop_duplicates("secid")
            .set_index("secid")["ticker"]
            .to_dict()
        )
        tickers = [str(tick_map.get(s, f"SECID_{s}")).upper() for s in secids]
    else:
        tickers = [str(t).strip().upper() or f"SECID_{s}" for t, s in zip(tickers, secids)]
        if len(tickers) < len(secids):
            tickers = tickers + [f"SECID_{s}" for s in secids[len(tickers) :]]

    long_df = _add_feature_columns(long_df)

    membership_meta: dict[str, Any] = {"enforced": False}
    if apply_pit_membership:
        mem_path = lake / "macro" / "pit_membership.parquet"
        if mem_path.is_file():
            mem_df = pd.read_parquet(mem_path)
            dates = sorted(pd.to_datetime(long_df["date"]).unique())
            members_by_date = build_members_by_date_from_intervals(dates, mem_df)
            if members_by_date:
                long_df = _apply_pit_membership_nan(long_df, members_by_date)
                membership_meta = {
                    "enforced": True,
                    "path": str(mem_path),
                    "n_dates": len(dates),
                }
            else:
                membership_meta = {"enforced": False, "reason": "empty membership expansion"}
        else:
            membership_meta = {"enforced": False, "reason": "pit_membership.parquet missing"}

    wide = pivot_equity_long_to_wide(long_df, secids)
    if wide.empty:
        raise RuntimeError("wide equity panel empty")

    meta = {
        "secids": secids,
        "tickers": list(tickers),
        "display_names": list(tickers),
        "n_assets": len(secids),
        "panel_start": panel_start,
        "panel_end": panel_end,
        "feature_stems": list(FEATURE_STEMS),
        "label_stem": "stk_ret",
        "universe_mode": mode,
        "selector_provenance": (
            dict(selector_result.get("provenance") or {})
            if isinstance(selector_result, dict)
            else None
        ),
        "pit_membership": membership_meta,
        "cfadj_validation": cfadj_meta,
        "selection_start": SELECTION_START,
        "selection_end": SELECTION_END,
        "embargo_year": EMBARGO_YEAR,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "lake_base_dir": str(lake),
        "knowledge_written_at": pd.Timestamp.utcnow().isoformat(),
    }

    if out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        wide_out = wide.reset_index()
        if "date" not in wide_out.columns:
            wide_out = wide_out.rename(columns={wide_out.columns[0]: "date"})
        wide_out.to_parquet(od / f"{SIGNALS_SYMBOL}.parquet", index=False)
        (od / f"{UNIVERSE_SYMBOL}.json").write_text(json.dumps(meta, indent=2) + "\n")
        arctic_symbols: list[str] = []
    else:
        from src.data.arctic_store import ArcticStateStore

        store = ArcticStateStore(
            db_path=arctic_db_path or ARCTIC_ROOT,
            library_name=arctic_library,
        )
        uni_df = pd.DataFrame(
            {"payload": [json.dumps(meta)]},
            index=pd.DatetimeIndex([pd.Timestamp(panel_end)], name="date"),
        )
        if UNIVERSE_SYMBOL in store.lib.list_symbols():
            store.lib.update(UNIVERSE_SYMBOL, uni_df, metadata=meta)
        else:
            store.lib.write(UNIVERSE_SYMBOL, uni_df, metadata=meta)
        wide_out = wide.reset_index()
        if "date" not in wide_out.columns:
            wide_out = wide_out.rename(columns={wide_out.columns[0]: "date"})
        table = pa.Table.from_pandas(wide_out, preserve_index=False)
        store.persist_features(SIGNALS_SYMBOL, table, metadata=meta)
        arctic_symbols = store.list_available_features()

    log.info(
        "equity panel rows=%d cols=%d secids=%d mode=%s",
        len(wide),
        wide.shape[1],
        len(secids),
        mode,
    )
    return {
        "secids": secids,
        "tickers": list(tickers),
        "n_rows": int(len(wide)),
        "n_cols": int(wide.shape[1]),
        "start": str(wide.index.min().date()),
        "end": str(wide.index.max().date()),
        "arctic_symbols": arctic_symbols,
        "meta": meta,
        "out_dir": str(out_dir) if out_dir is not None else None,
    }


def load_equity_universe_meta(path_or_store: Any) -> dict[str, Any]:
    """Load ``equity_universe`` metadata from a JSON path or Arctic store."""
    if isinstance(path_or_store, (str, Path)):
        p = Path(path_or_store)
        if p.is_dir():
            p = p / f"{UNIVERSE_SYMBOL}.json"
        if p.is_file():
            return json.loads(p.read_text())
        return {}
    store = path_or_store
    try:
        if UNIVERSE_SYMBOL not in store.list_available_features():
            return {}
        meta = dict(store.lib.read(UNIVERSE_SYMBOL).metadata or {})
        if meta.get("secids"):
            return meta
        uni = store.read_pit_state(UNIVERSE_SYMBOL, as_of=None)
        return json.loads(uni["payload"].iloc[-1])
    except Exception:
        return {}
