"""Lake → long (date, secid) feature panels for the equity observation cube.

Each family is PIT-safe at source. Arctic symbols use ``feat_<family>_<stem>``
names so ``ArcticStateStore.persist_panel`` does not refuse P3/Worldscope tokens.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.arctic_store import ArcticStateStore
from mascotrl.data.paths import ARCTIC_ROOT, LAKE_ROOT, assert_lake_mounted
from mascotrl.data.surface_signals import _canonical_secid_key

log = logging.getLogger(__name__)

SHORT_INTEREST_LAG_DAYS = 14
COMPUSTAT_LAG_DAYS = 120
RATES_TENORS: tuple[int, ...] = (30, 91, 182, 365, 730, 1825, 3650)

IBES_CURATED: tuple[str, ...] = (
    "bm",
    "pe_exi",
    "ps",
    "pcf",
    "dpr",
    "npm",
    "gpm",
    "roa",
    "roe",
    "cfm",
    "evm",
    "CAPEI",
)


def _canon_secid_series(s: pd.Series) -> pd.Series:
    return s.map(_canonical_secid_key)


def _read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def _sec_price_frame(lake: Path, start: str, end: str) -> pd.DataFrame:
    path = lake / "macro" / "sp500_sec.parquet"
    cols = ["secid", "date", "close", "volume", "cfadj", "open", "high", "low"]
    df = _read_parquet(path, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["secid"] = _canon_secid_series(df["secid"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    for c in ("close", "volume", "cfadj", "open", "high", "low"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["adj_close"] = df["close"] * df["cfadj"].fillna(1.0)
    return df


def load_ohlc_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    """OHLC panels: prefer corax for adj prices, unadj for microstructure range."""
    lake = Path(lake)
    unadj = _read_parquet(
        lake / "macro" / "lseg_eq_ohlc_unadj.parquet",
        columns=[
            "date",
            "secid",
            "OPEN_PRC",
            "HIGH_1",
            "LOW_1",
            "TRDPRC_1",
        ],
    )
    corax = _read_parquet(
        lake / "macro" / "lseg_eq_ohlc_corax.parquet",
        columns=["date", "secid", "OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
    )
    sec = _sec_price_frame(lake, start, end)
    if unadj.empty and corax.empty and sec.empty:
        return pd.DataFrame(
            columns=["date", "secid", "open", "high", "low", "close", "adj_close"]
        )
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    def _prep(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["secid"] = _canon_secid_series(out["secid"])
        out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
        rename = {
            "OPEN_PRC": f"{prefix}_open",
            "HIGH_1": f"{prefix}_high",
            "LOW_1": f"{prefix}_low",
            "TRDPRC_1": f"{prefix}_close",
        }
        return out.rename(columns=rename)

    u = _prep(unadj, "u")
    c = _prep(corax, "c")
    base = sec[["date", "secid", "open", "high", "low", "close", "adj_close"]].copy()
    if not u.empty:
        base = base.merge(u, on=["date", "secid"], how="outer")
    else:
        for col in ("u_open", "u_high", "u_low", "u_close"):
            base[col] = np.nan
    if not c.empty:
        base = base.merge(c, on=["date", "secid"], how="outer")
    else:
        for col in ("c_open", "c_high", "c_low", "c_close"):
            base[col] = np.nan
    out = pd.DataFrame(
        {
            "date": base["date"],
            "secid": base["secid"],
            "open": base["u_open"].fillna(base["c_open"]).fillna(base["open"]),
            "high": base["u_high"].fillna(base["c_high"]).fillna(base["high"]),
            "low": base["u_low"].fillna(base["c_low"]).fillna(base["low"]),
            "close": base["u_close"].fillna(base["c_close"]).fillna(base["close"]),
            "adj_close": base["c_close"].fillna(base["adj_close"]).fillna(base["close"]),
        }
    )
    return out.dropna(subset=["date", "secid"]).sort_values(["secid", "date"])


def load_microstructure_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    ohlc = _read_parquet(
        lake / "macro" / "lseg_eq_ohlc_unadj.parquet",
        columns=[
            "date",
            "secid",
            "BID",
            "ASK",
            "TRDPRC_1",
            "VWAP",
            "ACVOL_UNS",
            "BLKVOLUM",
        ],
    )
    size = _read_parquet(
        lake / "macro" / "lseg_eq_size.parquet",
        columns=["date", "secid", "Outstanding Shares"],
    )
    if ohlc.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "secid",
                "eff_spread",
                "vwap_dev",
                "block_share",
                "turnover",
            ]
        )
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ohlc = ohlc.copy()
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    ohlc["secid"] = _canon_secid_series(ohlc["secid"])
    ohlc = ohlc[(ohlc["date"] >= start_ts) & (ohlc["date"] <= end_ts)]
    for col in ("BID", "ASK", "TRDPRC_1", "VWAP", "ACVOL_UNS", "BLKVOLUM"):
        ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")
    mid = (ohlc["ASK"] + ohlc["BID"]) / 2.0
    ohlc["eff_spread"] = (ohlc["ASK"] - ohlc["BID"]) / mid.replace(0.0, np.nan)
    ohlc["vwap_dev"] = ohlc["TRDPRC_1"] / ohlc["VWAP"].replace(0.0, np.nan) - 1.0
    ohlc["block_share"] = ohlc["BLKVOLUM"] / ohlc["ACVOL_UNS"].replace(0.0, np.nan)
    if not size.empty:
        size = size.copy()
        size["date"] = pd.to_datetime(size["date"])
        size["secid"] = _canon_secid_series(size["secid"])
        size["Outstanding Shares"] = pd.to_numeric(
            size["Outstanding Shares"], errors="coerce"
        )
        size = size.sort_values(["secid", "date"]).drop_duplicates(
            ["secid", "date"], keep="last"
        )
        ohlc = ohlc.merge(
            size[["date", "secid", "Outstanding Shares"]],
            on=["date", "secid"],
            how="left",
        )
        ohlc = ohlc.sort_values(["secid", "date"])
        ohlc["Outstanding Shares"] = ohlc.groupby("secid")[
            "Outstanding Shares"
        ].ffill()
        ohlc["turnover"] = ohlc["ACVOL_UNS"] / ohlc["Outstanding Shares"].replace(
            0.0, np.nan
        )
    else:
        ohlc["turnover"] = np.nan
    return ohlc[
        ["date", "secid", "eff_spread", "vwap_dev", "block_share", "turnover"]
    ].sort_values(["secid", "date"])


def load_short_interest_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    path = lake / "macro" / "p3" / "lseg_short_interest.parquet"
    df = _read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "secid", "si_pct"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]) + pd.Timedelta(days=SHORT_INTEREST_LAG_DAYS)
    out["secid"] = _canon_secid_series(out["secid"])
    out["si_pct"] = pd.to_numeric(out.get("Short Interest Pct"), errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out[["date", "secid", "si_pct"]].sort_values(["secid", "date"])


def load_analyst_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    path = lake / "macro" / "p3" / "lseg_ibes.parquet"
    df = _read_parquet(path)
    sec = _sec_price_frame(lake, start, end)
    if df.empty:
        return pd.DataFrame(columns=["date", "secid", "rec_mean_inv", "pt_gap"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["secid"] = _canon_secid_series(out["secid"])
    rec = pd.to_numeric(out.get("Recommendation - Mean (1-5)"), errors="coerce")
    out["rec_mean_inv"] = 6.0 - rec
    pt = pd.to_numeric(out.get("Price Target - Median"), errors="coerce")
    if not sec.empty:
        prices = sec[["date", "secid", "close"]].rename(columns={"close": "_px"})
        out = out.merge(prices, on=["date", "secid"], how="left")
        out["pt_gap"] = pt / out["_px"].replace(0.0, np.nan) - 1.0
    else:
        out["pt_gap"] = np.nan
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out[["date", "secid", "rec_mean_inv", "pt_gap"]].sort_values(["secid", "date"])


def load_worldscope_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    path = lake / "macro" / "p3" / "lseg_worldscope.parquet"
    df = _read_parquet(path)
    sec = _sec_price_frame(lake, start, end)
    if df.empty:
        return pd.DataFrame(
            columns=["date", "secid", "bp", "ep", "ta_growth", "rev_growth"]
        )
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["secid"] = _canon_secid_series(out["secid"])
    bv = pd.to_numeric(out.get("Book Value Per Share"), errors="coerce")
    pe = pd.to_numeric(out.get("P/E (Daily Time Series Ratio)"), errors="coerce")
    ta = pd.to_numeric(out.get("Total Assets"), errors="coerce")
    rev = pd.to_numeric(out.get("Revenue"), errors="coerce")
    if not sec.empty:
        prices = sec[["date", "secid", "close"]].rename(columns={"close": "_px"})
        out = out.merge(prices, on=["date", "secid"], how="left")
        out["bp"] = bv / out["_px"].replace(0.0, np.nan)
    else:
        out["bp"] = np.nan
    out["ep"] = 1.0 / pe.replace(0.0, np.nan)
    out = out.sort_values(["secid", "date"])
    out["_ta"] = ta
    out["_rev"] = rev

    def _yoy(series: pd.Series, dates: pd.Series) -> pd.Series:
        """YoY pct change on sparse daily stamps (300–450d lookback)."""
        vals = series.to_numpy(dtype=float)
        dts = pd.to_datetime(dates).to_numpy()
        out_v = np.full(len(vals), np.nan, dtype=float)
        for i in range(len(vals)):
            if not np.isfinite(vals[i]):
                continue
            target = dts[i] - np.timedelta64(365, "D")
            # search backward for nearest obs in [target-90d, target+90d]
            lo = target - np.timedelta64(90, "D")
            hi = target + np.timedelta64(90, "D")
            best = None
            best_dist = None
            for j in range(i - 1, -1, -1):
                if dts[j] < lo:
                    break
                if lo <= dts[j] <= hi and np.isfinite(vals[j]) and vals[j] != 0:
                    dist = abs(dts[j] - target)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = vals[j]
            if best is not None:
                out_v[i] = vals[i] / best - 1.0
        return pd.Series(out_v, index=series.index)

    parts = []
    for _, g in out.groupby("secid", sort=False):
        g = g.copy()
        g["ta_growth"] = _yoy(g["_ta"], g["date"])
        g["rev_growth"] = _yoy(g["_rev"], g["date"])
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else out
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    return out[["date", "secid", "bp", "ep", "ta_growth", "rev_growth"]].sort_values(
        ["secid", "date"]
    )


def load_ibes_ratios_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    """PIT as-of join: public_date <= t via expanding merge onto daily sec calendar."""
    lake = Path(lake)
    path = lake / "macro" / "ibes_financial_ratios.parquet"
    link = _read_parquet(
        lake / "macro" / "crsp_optionm_link.parquet",
        columns=["secid", "permno", "sdate", "edate"],
    )
    ratios = _read_parquet(path)
    sec = _sec_price_frame(lake, start, end)
    if ratios.empty or sec.empty:
        cols = ["date", "secid", "bm", "ep_exi", "ps", "pcf", "dpr", "npm", "gpm", "roa", "roe", "cfm", "evm", "capex_inv"]
        return pd.DataFrame(columns=cols)
    ratios = ratios.copy()
    ratios["public_date"] = pd.to_datetime(ratios["public_date"])
    ratios["permno"] = pd.to_numeric(ratios["permno"], errors="coerce").astype("int64")
    for c in IBES_CURATED:
        if c in ratios.columns:
            ratios[c] = pd.to_numeric(ratios[c], errors="coerce")
    if link.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "secid",
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
            ]
        )
    link = link.copy()
    link["secid"] = _canon_secid_series(link["secid"])
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce").astype("int64")
    link["sdate"] = pd.to_datetime(link["sdate"])
    link["edate"] = pd.to_datetime(link["edate"])
    cal = sec[["date", "secid"]].drop_duplicates()
    cal = cal.merge(link, on="secid", how="left")
    cal = cal[
        (cal["date"] >= cal["sdate"].fillna(pd.Timestamp.min))
        & (cal["date"] <= cal["edate"].fillna(pd.Timestamp.max))
    ]
    ratios_s = ratios[
        ["permno", "public_date"] + [c for c in IBES_CURATED if c in ratios.columns]
    ].sort_values(["permno", "public_date"])
    cal = cal.sort_values(["permno", "date"])
    left = cal[["date", "secid", "permno"]].dropna(subset=["permno"]).copy()
    left["permno"] = left["permno"].astype("int64")
    left = left.drop_duplicates(["permno", "date", "secid"], keep="last")
    right = ratios_s.rename(columns={"public_date": "date"}).copy()
    right["permno"] = right["permno"].astype("int64")
    right = right.drop_duplicates(["permno", "date"], keep="last")
    pieces: list[pd.DataFrame] = []
    right_by = {p: g.sort_values("date") for p, g in right.groupby("permno", sort=False)}
    for permno, gleft in left.groupby("permno", sort=False):
        gright = right_by.get(int(permno))
        gleft = gleft.sort_values("date")
        if gright is None or gright.empty:
            tmp = gleft.copy()
            for c in IBES_CURATED:
                if c not in tmp.columns:
                    tmp[c] = np.nan
            pieces.append(tmp)
            continue
        pieces.append(
            pd.merge_asof(
                gleft,
                gright.drop(columns=["permno"], errors="ignore"),
                on="date",
                direction="backward",
            )
        )
    merged = pd.concat(pieces, ignore_index=True) if pieces else left

    merged["ep_exi"] = 1.0 / merged["pe_exi"].replace(0.0, np.nan) if "pe_exi" in merged.columns else np.nan
    merged["capex_inv"] = 1.0 / merged["CAPEI"].replace(0.0, np.nan) if "CAPEI" in merged.columns else np.nan
    cols = ["date", "secid", "bm", "ep_exi", "ps", "pcf", "dpr", "npm", "gpm", "roa", "roe", "cfm", "evm", "capex_inv"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan
    return merged[cols].sort_values(["secid", "date"])


def load_compustat_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    path = lake / "macro" / "compustat_funda_enrich.parquet"
    df = _read_parquet(path)
    sec = _sec_price_frame(lake, start, end)
    if df.empty or sec.empty:
        return pd.DataFrame(
            columns=["date", "secid", "at_growth", "sale_growth", "ni_at", "dvc_at"]
        )
    fund = df.copy()
    fund["avail"] = pd.to_datetime(fund["datadate"]) + pd.Timedelta(days=COMPUSTAT_LAG_DAYS)
    fund["secid"] = _canon_secid_series(fund["secid"])
    for c in ("at", "sale", "ni", "dvc"):
        fund[c] = pd.to_numeric(fund[c], errors="coerce")
    fund = fund.sort_values(["secid", "avail"])
    fund["at_growth"] = fund.groupby("secid")["at"].pct_change(1)
    fund["sale_growth"] = fund.groupby("secid")["sale"].pct_change(1)
    fund["ni_at"] = fund["ni"] / fund["at"].replace(0.0, np.nan)
    fund["dvc_at"] = fund["dvc"] / fund["at"].replace(0.0, np.nan)
    cal = sec[["date", "secid"]].drop_duplicates()
    fund_s = fund[
        ["avail", "secid", "at_growth", "sale_growth", "ni_at", "dvc_at"]
    ].rename(columns={"avail": "date"})
    pieces: list[pd.DataFrame] = []
    fund_by = {s: g.sort_values("date") for s, g in fund_s.groupby("secid", sort=False)}
    for sid, gleft in cal.groupby("secid", sort=False):
        gright = fund_by.get(sid)
        gleft = gleft.sort_values("date")
        if gright is None or gright.empty:
            tmp = gleft.copy()
            for c in ("at_growth", "sale_growth", "ni_at", "dvc_at"):
                tmp[c] = np.nan
            pieces.append(tmp)
            continue
        pieces.append(
            pd.merge_asof(
                gleft,
                gright.drop(columns=["secid"], errors="ignore"),
                on="date",
                direction="backward",
            )
        )
    merged = pd.concat(pieces, ignore_index=True) if pieces else cal
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)]
    return merged[
        ["date", "secid", "at_growth", "sale_growth", "ni_at", "dvc_at"]
    ].sort_values(["secid", "date"])


def load_option_flow_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    opv = _read_parquet(lake / "macro" / "om_opvold.parquet")
    sec = _sec_price_frame(lake, start, end)
    if opv.empty:
        return pd.DataFrame(
            columns=["date", "secid", "pc_vol", "pc_oi", "opt_stock_vol", "oi_lvl"]
        )
    opv = opv.copy()
    opv["date"] = pd.to_datetime(opv["date"])
    opv["secid"] = _canon_secid_series(opv["secid"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    opv = opv[(opv["date"] >= start_ts) & (opv["date"] <= end_ts)]
    opv["volume"] = pd.to_numeric(opv["volume"], errors="coerce")
    opv["open_interest"] = pd.to_numeric(opv["open_interest"], errors="coerce")
    cp = opv["cp_flag"].astype(str).str.upper().str[:1]
    puts = opv.loc[cp == "P"].groupby(["date", "secid"], as_index=False).agg(
        put_vol=("volume", "sum"), put_oi=("open_interest", "sum")
    )
    calls = opv.loc[cp == "C"].groupby(["date", "secid"], as_index=False).agg(
        call_vol=("volume", "sum"), call_oi=("open_interest", "sum")
    )
    wide = puts.merge(calls, on=["date", "secid"], how="outer")
    wide["pc_vol"] = wide["put_vol"] / wide["call_vol"].replace(0.0, np.nan)
    wide["pc_oi"] = wide["put_oi"] / wide["call_oi"].replace(0.0, np.nan)
    wide["oi_lvl"] = wide["put_oi"].fillna(0.0) + wide["call_oi"].fillna(0.0)
    if not sec.empty:
        vol = sec[["date", "secid", "volume"]].rename(columns={"volume": "stock_vol"})
        wide = wide.merge(vol, on=["date", "secid"], how="left")
        wide["opt_stock_vol"] = (
            wide["put_vol"].fillna(0.0) + wide["call_vol"].fillna(0.0)
        ) / wide["stock_vol"].replace(0.0, np.nan)
    else:
        wide["opt_stock_vol"] = np.nan
    return wide[["date", "secid", "pc_vol", "pc_oi", "opt_stock_vol", "oi_lvl"]].sort_values(
        ["secid", "date"]
    )


def load_dividend_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    dist = _read_parquet(
        lake / "macro" / "om_distrd.parquet",
        columns=["secid", "ex_date", "amount"],
    )
    sec = _sec_price_frame(lake, start, end)
    if dist.empty or sec.empty:
        return pd.DataFrame(columns=["date", "secid", "div_yield_ttm"])
    dist = dist.copy()
    dist["ex_date"] = pd.to_datetime(dist["ex_date"])
    dist["secid"] = _canon_secid_series(dist["secid"])
    dist["amount"] = pd.to_numeric(dist["amount"], errors="coerce")
    dist = dist.dropna(subset=["ex_date", "secid", "amount"])
    # Expand each distribution onto calendar dates in (ex_date, ex_date+252d]
    # via a daily cumulative sum keyed by (secid, date), then take 252d diffs.
    cal = sec[["date", "secid", "close"]].sort_values(["secid", "date"]).copy()
    daily = (
        dist.groupby(["secid", "ex_date"], as_index=False)["amount"]
        .sum()
        .rename(columns={"ex_date": "date"})
    )
    cal = cal.merge(daily, on=["date", "secid"], how="left")
    cal["amount"] = cal["amount"].fillna(0.0)
    cal["cum"] = cal.groupby("secid")["amount"].cumsum()
    # TTM = cum[t] - cum[t-252] (approx trading-day lag via shift 252 rows).
    cal["cum_lag"] = cal.groupby("secid")["cum"].shift(252)
    ttm = cal["cum"] - cal["cum_lag"].fillna(0.0)
    # Before 252 history, require at least some dividends observed.
    hist = cal.groupby("secid").cumcount()
    ttm = ttm.where(hist >= 21, np.nan)
    cal["div_yield_ttm"] = ttm / cal["close"].replace(0.0, np.nan)
    return cal[["date", "secid", "div_yield_ttm"]].sort_values(["secid", "date"])


def load_rates_term_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    """Macro-level zero-curve features (date only; no secid)."""
    lake = Path(lake)
    path = lake / "macro" / "om_zerocd.parquet"
    if not path.is_file():
        path = lake / "macro" / "spx_zerocd.parquet"
    df = _read_parquet(path)
    if df.empty:
        cols = [f"zrate_{t}" for t in RATES_TENORS] + [
            "term_slope",
            "term_curv",
            "d_term_slope_21",
        ]
        return pd.DataFrame(columns=["date"] + cols)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    rows = []
    for dt, g in df.groupby("date", sort=True):
        days = g["days"].to_numpy(dtype=float)
        rates = g["rate"].to_numpy(dtype=float)
        order = np.argsort(days)
        days, rates = days[order], rates[order]
        rec: dict[str, Any] = {"date": dt}
        for t in RATES_TENORS:
            if len(days) == 0:
                rec[f"zrate_{t}"] = np.nan
            else:
                rec[f"zrate_{t}"] = float(np.interp(t, days, rates))
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("date")
    out["term_slope"] = out["zrate_3650"] - out["zrate_91"]
    out["term_curv"] = out["zrate_730"] - 0.5 * (out["zrate_91"] + out["zrate_3650"])
    out["d_term_slope_21"] = out["term_slope"] - out["term_slope"].shift(21)
    return out


def load_jkp_long(lake: Path | str, start: str, end: str) -> pd.DataFrame:
    lake = Path(lake)
    jkp = _read_parquet(lake / "factors" / "jkp_chars.parquet")
    link = _read_parquet(
        lake / "macro" / "crsp_optionm_link.parquet",
        columns=["secid", "permno", "sdate", "edate"],
    )
    if jkp.empty:
        return pd.DataFrame(columns=["date", "secid", "log_me", "ivol_capm_21d", "ret_1_0"])
    jkp = jkp.copy()
    jkp["date"] = pd.to_datetime(jkp["date"])
    jkp["permno"] = pd.to_numeric(jkp["permno"], errors="coerce")
    jkp["me"] = pd.to_numeric(jkp["me"], errors="coerce")
    jkp["ivol_capm_21d"] = pd.to_numeric(jkp["ivol_capm_21d"], errors="coerce")
    jkp["ret_1_0"] = pd.to_numeric(jkp["ret_1_0"], errors="coerce")
    jkp["log_me"] = np.log(jkp["me"].replace(0.0, np.nan))
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    jkp = jkp[(jkp["date"] >= start_ts) & (jkp["date"] <= end_ts)]
    if link.empty:
        return pd.DataFrame(columns=["date", "secid", "log_me", "ivol_capm_21d", "ret_1_0"])
    link = link.copy()
    link["secid"] = _canon_secid_series(link["secid"])
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    link["sdate"] = pd.to_datetime(link["sdate"])
    link["edate"] = pd.to_datetime(link["edate"])
    merged = jkp.merge(link, on="permno", how="left")
    merged = merged[
        (merged["date"] >= merged["sdate"].fillna(pd.Timestamp.min))
        & (merged["date"] <= merged["edate"].fillna(pd.Timestamp.max))
    ]
    return (
        merged[["date", "secid", "log_me", "ivol_capm_21d", "ret_1_0"]]
        .dropna(subset=["secid"])
        .drop_duplicates(["date", "secid"])
        .sort_values(["secid", "date"])
    )


def load_gics_map(lake: Path | str) -> pd.DataFrame:
    """Static secid → GICS industry map (via RIC map when needed)."""
    lake = Path(lake)
    gics = _read_parquet(lake / "macro" / "lseg_gics.parquet")
    ric_map = _read_parquet(
        lake / "macro" / "lseg_ric_map.parquet",
        columns=["secid", "ric", "TR.GICSIndustry"],
    )
    if not ric_map.empty and "TR.GICSIndustry" in ric_map.columns:
        out = ric_map.copy()
        out["secid"] = _canon_secid_series(out["secid"])
        out = out.rename(columns={"TR.GICSIndustry": "gics_industry"})
        return out[["secid", "gics_industry"]].drop_duplicates("secid")
    if gics.empty:
        return pd.DataFrame(columns=["secid", "gics_industry"])
    # Instrument is RIC; join via ric_map if present.
    if not ric_map.empty:
        m = ric_map[["secid", "ric"]].copy()
        m["secid"] = _canon_secid_series(m["secid"])
        g = gics.rename(
            columns={"Instrument": "ric", "TR.GICSIndustry": "gics_industry"}
        )
        out = m.merge(g[["ric", "gics_industry"]], on="ric", how="left")
        return out[["secid", "gics_industry"]].drop_duplicates("secid")
    return pd.DataFrame(columns=["secid", "gics_industry"])


FAMILY_LOADERS: dict[str, Callable[..., pd.DataFrame]] = {
    "ohlc": load_ohlc_long,
    "microstructure": load_microstructure_long,
    "short_interest": load_short_interest_long,
    "analyst": load_analyst_long,
    "worldscope": load_worldscope_long,
    "ibes_ratios": load_ibes_ratios_long,
    "compustat": load_compustat_long,
    "option_flow": load_option_flow_long,
    "dividend": load_dividend_long,
    "rates_term": load_rates_term_long,
    "jkp": load_jkp_long,
}

# Raw lake paths used when a ``_panels/feat_{family}.parquet`` mirror is absent.
# Shared files may appear in multiple families; collectors de-dupe by arcname.
FAMILY_RAW_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ohlc": (
        "macro/lseg_eq_ohlc_unadj.parquet",
        "macro/lseg_eq_ohlc_corax.parquet",
        "macro/sp500_sec.parquet",
    ),
    "microstructure": (
        "macro/lseg_eq_ohlc_unadj.parquet",
        "macro/lseg_eq_size.parquet",
    ),
    "short_interest": ("macro/p3/lseg_short_interest.parquet",),
    "analyst": ("macro/p3/lseg_ibes.parquet",),
    "worldscope": ("macro/p3/lseg_worldscope.parquet",),
    "ibes_ratios": (
        "macro/ibes_financial_ratios.parquet",
        "macro/crsp_optionm_link.parquet",
    ),
    "compustat": ("macro/compustat_funda_enrich.parquet",),
    "option_flow": ("macro/om_opvold.parquet", "macro/sp500_sec.parquet"),
    "dividend": ("macro/om_distrd.parquet", "macro/sp500_sec.parquet"),
    "rates_term": ("macro/om_zerocd.parquet", "macro/spx_zerocd.parquet"),
    "jkp": ("factors/jkp_chars.parquet", "macro/crsp_optionm_link.parquet"),
}

# GICS is never consumed via feat_* mirror at train time; always raw.
GICS_RAW_PATHS: tuple[str, ...] = (
    "macro/lseg_gics.parquet",
    "macro/lseg_ric_map.parquet",
)

OPTIONAL_BUNDLE_PATHS: tuple[str, ...] = ("macro/ff_factors.parquet",)

# Equity substrate parity (spectrum = H0): returns panel + geometry_lite cache.
# vol_surface (17GB) is NOT shipped; precomputed surface signals (~11MB) are.
SUBSTRATE_BUNDLE_PATHS: tuple[tuple[str, str], ...] = (
    ("macro/sp500_sec.parquet", "sp500_sec"),
    ("surface_signals/geometry_lite.parquet", "geometry_lite"),
    ("_panels/geometry_lite_surface.parquet", "geometry_lite"),
)


def required_panel_families() -> list[str]:
    """Eleven feature-cube families that AWS must ship (mirrors or raw)."""
    return list(FAMILY_LOADERS)


def required_panel_mirror_names() -> list[str]:
    return [f"feat_{f}.parquet" for f in required_panel_families()]


def collect_panel_bundle_paths(
    lake: Path | str,
    *,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate lake files to ship for the equity feature cube.

    Preference order per family: ``_panels/feat_{family}.parquet`` mirror, else
    that family's ``FAMILY_RAW_FALLBACKS``. GICS raw paths are always required.
    ``ff_factors`` is included when present (optional for OM CPCV).

    Each entry: ``{arcname, abs_path, source, family}``.
    """
    lake_root = Path(lake)
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    missing_families: list[str] = []

    def _add(rel: str, *, source: str, family: str | None) -> bool:
        if rel in seen:
            return True
        abs_path = lake_root / rel
        if not abs_path.is_file():
            return False
        seen.add(rel)
        entries.append(
            {
                "arcname": rel,
                "abs_path": abs_path,
                "source": source,
                "family": family,
            }
        )
        return True

    for family in required_panel_families():
        mirror_rel = f"_panels/feat_{family}.parquet"
        if _add(mirror_rel, source="mirror", family=family):
            continue
        got_any = False
        for raw in FAMILY_RAW_FALLBACKS.get(family, ()):
            if _add(raw, source="raw", family=family):
                got_any = True
        if not got_any:
            missing_families.append(family)

    for rel in GICS_RAW_PATHS:
        if not _add(rel, source="gics_raw", family="gics"):
            if "gics" not in missing_families:
                missing_families.append("gics")

    for rel in OPTIONAL_BUNDLE_PATHS:
        _add(rel, source="optional", family=None)

    # Substrate parity: sp500_sec + geometry_lite surface cache (required when
    # require_complete so Burst cells cannot silently fall back to raw returns).
    substrate_missing: list[str] = []
    got_surface = False
    for rel, family in SUBSTRATE_BUNDLE_PATHS:
        if family == "sp500_sec":
            if not _add(rel, source="substrate", family=family):
                substrate_missing.append(family)
        elif family == "geometry_lite":
            if _add(rel, source="substrate", family=family):
                got_surface = True
    if not got_surface:
        substrate_missing.append("geometry_lite")

    if require_complete and missing_families:
        raise ValueError(
            "panel_bundle_missing_families: " + ",".join(sorted(missing_families))
        )
    if require_complete and substrate_missing:
        raise ValueError(
            "panel_bundle_missing_substrate: " + ",".join(sorted(set(substrate_missing)))
        )
    return entries


def _arctic_symbol(family: str, stem: str) -> str:
    # persist_panel refuses symbols containing "worldscope" / "p3".
    fam = {"worldscope": "ws", "ibes_ratios": "ibesrat"}.get(family, family)
    return f"feat_{fam}_{stem}"


def materialize_feature_panels(
    *,
    start: str = "2003-01-01",
    end: str = "2024-12-31",
    lake: Path | str | None = None,
    arctic_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    families: Sequence[str] | None = None,
    persist_arctic: bool = True,
) -> dict[str, Any]:
    """Materialize all families to parquet mirrors (+ optional Arctic)."""
    lake_root = assert_lake_mounted(Path(lake) if lake else LAKE_ROOT)
    panel_dir = Path(out_dir) if out_dir else lake_root / "_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    store = None
    if persist_arctic:
        store = ArcticStateStore(db_path=str(arctic_path or ARCTIC_ROOT))
    selected = list(families) if families is not None else list(FAMILY_LOADERS) + ["gics_map"]
    provenance: dict[str, Any] = {
        "start": start,
        "end": end,
        "lake": str(lake_root),
        "panel_dir": str(panel_dir),
        "families": {},
    }
    for family in selected:
        if family == "gics_map":
            df = load_gics_map(lake_root)
            path = panel_dir / "feat_gics_map.parquet"
            df.to_parquet(path, index=False)
            if store is not None and not df.empty and "secid" in df.columns:
                # Synthetic date for static map so persist_panel accepts it.
                tmp = df.copy()
                tmp["date"] = pd.Timestamp("2003-01-01")
                try:
                    store.persist_panel(_arctic_symbol("gics", "map"), tmp)
                except Exception as exc:
                    log.warning("arctic gics_map skip: %s", exc)
            provenance["families"]["gics_map"] = {
                "rows": int(len(df)),
                "path": str(path),
                "nan_rates": {},
            }
            continue
        loader = FAMILY_LOADERS[family]
        df = loader(lake_root, start, end)
        path = panel_dir / f"feat_{family}.parquet"
        df.to_parquet(path, index=False)
        value_cols = [c for c in df.columns if c not in ("date", "secid")]
        nan_rates = {
            c: float(df[c].isna().mean()) if len(df) else 1.0 for c in value_cols
        }
        if store is not None and not df.empty:
            if family == "rates_term":
                # Macro panel: assign sentinel secid=0 for Arctic long format.
                tmp = df.copy()
                tmp["secid"] = 0
                try:
                    store.persist_panel(_arctic_symbol(family, "panel"), tmp)
                except Exception as exc:
                    log.warning("arctic %s skip: %s", family, exc)
            else:
                try:
                    store.persist_panel(_arctic_symbol(family, "panel"), df)
                except Exception as exc:
                    log.warning("arctic %s skip: %s", family, exc)
        provenance["families"][family] = {
            "rows": int(len(df)),
            "path": str(path),
            "nan_rates": nan_rates,
            "columns": list(df.columns),
        }
        log.info("materialized %s rows=%d", family, len(df))
    prov_path = panel_dir / "feature_panels_provenance.json"
    provenance["knowledge_written_at"] = pd.Timestamp.utcnow().isoformat()
    prov_path.write_text(json.dumps(provenance, indent=2, default=str))
    provenance["provenance_path"] = str(prov_path)
    return provenance
