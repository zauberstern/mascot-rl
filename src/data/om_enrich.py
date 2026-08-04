"""OptionMetrics WRDS enrichment beyond the existing options_panel / vol_surface lake.

Pulls distributions, equity option volume, borrow rates, security master,
standardized options, index dividends, and zero curve for lake secids.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.paths import LAKE_ROOT
from src.data.wrds_enrich import ADV_NAME, connect_wrds
from src.logging_utils import get_logger

log = get_logger("volsurf.om_enrich")

OM_DISTRD = "om_distrd.parquet"
OM_OPVOLD = "om_opvold.parquet"
OM_BORRATE = "om_borrate.parquet"
OM_STDBRTE = "om_stdbrte.parquet"
OM_SECURD = "om_securd.parquet"
OM_SECNMD = "om_secnmd.parquet"
OM_STDOPD = "om_stdopd.parquet"
OM_IDXDVD = "om_idxdvd.parquet"
OM_ZEROCD = "om_zerocd.parquet"
OM_PROV = "om_enrichment_provenance.json"

YEAR_START = 2003
YEAR_END = 2024


def _chunks(xs: Sequence[int], n: int = 200) -> Iterable[list[int]]:
    buf = list(xs)
    for i in range(0, len(buf), n):
        yield buf[i : i + n]


def lake_secids(lake_base_dir: Path) -> list[int]:
    adv_path = lake_base_dir / "macro" / ADV_NAME
    if adv_path.is_file():
        s = pd.read_parquet(adv_path, columns=["secid"])["secid"]
        return sorted({int(x) for x in s.dropna().astype(int).tolist()})
    link = lake_base_dir / "macro" / "crsp_optionm_link.parquet"
    if link.is_file():
        s = pd.read_parquet(link, columns=["secid"])["secid"]
        return sorted({int(x) for x in s.dropna().astype(int).tolist()})
    raise FileNotFoundError("need crsp_om_adv or crsp_optionm_link to scope OM pulls")


def _sql_in(ids: Sequence[int]) -> str:
    return ",".join(str(int(i)) for i in ids)


def fetch_distrd(conn, secids: Sequence[int]) -> pd.DataFrame:
    parts = []
    for chunk in _chunks(secids, 250):
        sql = f"""
            select secid, record_date, seq_num, ex_date, amount, adj_factor,
                   declare_date, payment_date, link_secid, distr_type, frequency,
                   currency, approx_flag, cancel_flag, liquid_flag
            from optionm.distrd
            where secid in ({_sql_in(chunk)})
        """
        parts.append(conn.raw_sql(sql))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
        for c in ("record_date", "ex_date", "declare_date", "payment_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    return df


def fetch_opvold(conn, secids: Sequence[int]) -> pd.DataFrame:
    parts = []
    for chunk in _chunks(secids, 200):
        sql = f"""
            select secid, date, cp_flag, volume, open_interest
            from optionm.opvold
            where secid in ({_sql_in(chunk)})
              and date between '{YEAR_START}-01-01' and '{YEAR_END}-12-31'
        """
        parts.append(conn.raw_sql(sql))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    return df


def fetch_year_partitioned(
    conn,
    *,
    table_prefix: str,
    secids: Sequence[int],
    id_col: str = "securityid",
    years: range | None = None,
    columns: str = "*",
    extra_where: str = "",
) -> pd.DataFrame:
    years = years or range(YEAR_START, YEAR_END + 1)
    parts: list[pd.DataFrame] = []
    for y in years:
        table = f"optionm.{table_prefix}{y}"
        n_before = sum(len(p) for p in parts)
        for chunk in _chunks(secids, 200):
            sql = f"""
                select {columns} from {table}
                where {id_col} in ({_sql_in(chunk)})
                {extra_where}
            """
            try:
                parts.append(conn.raw_sql(sql))
            except Exception as exc:  # noqa: BLE001 — year may be empty/missing
                log.warning("%s chunk failed: %s", table, str(exc).split("\n")[0][:120])
        n_after = sum(len(p) for p in parts)
        log.info("pulled %s (+%d rows)", table, n_after - n_before)
        print(f"pulled {table} (+{n_after - n_before} rows)", flush=True)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # Normalize securityid -> secid
    if "securityid" in df.columns and "secid" not in df.columns:
        df = df.rename(columns={"securityid": "secid"})
    if "secid" in df.columns:
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_securd(conn, secids: Sequence[int]) -> pd.DataFrame:
    parts = []
    for chunk in _chunks(secids, 300):
        parts.append(
            conn.raw_sql(
                f"""
                select secid, cusip, ticker, sic, index_flag, exchange_d, class,
                       issue_type, industry_group
                from optionm.securd
                where secid in ({_sql_in(chunk)})
                """
            )
        )
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
    return df.drop_duplicates(subset=["secid"])


def fetch_secnmd(conn, secids: Sequence[int]) -> pd.DataFrame:
    parts = []
    for chunk in _chunks(secids, 300):
        parts.append(
            conn.raw_sql(
                f"""
                select secid, effect_date, cusip, ticker, class, issuer, issue, sic
                from optionm.secnmd
                where secid in ({_sql_in(chunk)})
                """
            )
        )
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
        df["effect_date"] = pd.to_datetime(df["effect_date"])
    return df


def fetch_stdopd_near_atm(conn, secids: Sequence[int]) -> pd.DataFrame:
    """Standardized options near 50-delta calls/puts (hedge / skew friendly)."""
    parts: list[pd.DataFrame] = []
    for y in range(YEAR_START, YEAR_END + 1):
        table = f"optionm.stdopd{y}"
        for chunk in _chunks(secids, 80):
            sql = f"""
                select secid, date, days, forward_price, strike_price, premium,
                       impl_volatility, delta, gamma, theta, vega, cp_flag
                from {table}
                where secid in ({_sql_in(chunk)})
                  and abs(delta - 0.5) <= 0.15
                  and days between 10 and 60
            """
            try:
                parts.append(conn.raw_sql(sql))
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", table, str(exc).split("\n")[0][:120])
        log.info("stdopd %s rows so far %d", y, sum(len(p) for p in parts))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_idxdvd(conn) -> pd.DataFrame:
    df = conn.raw_sql(
        f"""
        select secid, date, rate
        from optionm.idxdvd
        where date between '{YEAR_START}-01-01' and '{YEAR_END}-12-31'
        """
    )
    df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_zerocd(conn) -> pd.DataFrame:
    df = conn.raw_sql(
        f"""
        select date, days, rate
        from optionm.zerocd
        where date between '{YEAR_START}-01-01' and '{YEAR_END}-12-31'
        """
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_option_adv_from_opvold(opvold: pd.DataFrame) -> pd.DataFrame:
    """Daily total option contracts volume + OI per secid (cp_flag null = aggregate)."""
    if opvold is None or len(opvold) == 0:
        return pd.DataFrame(columns=["date", "secid", "opt_volume", "opt_open_interest"])
    df = opvold.copy()
    # Prefer aggregate rows (cp_flag null); else sum C+P
    agg = df[df["cp_flag"].isna()] if "cp_flag" in df.columns else df.iloc[0:0]
    if len(agg) == 0:
        g = (
            df.groupby(["secid", "date"], as_index=False)[["volume", "open_interest"]]
            .sum(min_count=1)
        )
    else:
        g = agg.rename(columns={"volume": "volume", "open_interest": "open_interest"})[
            ["secid", "date", "volume", "open_interest"]
        ]
    g = g.rename(columns={"volume": "opt_volume", "open_interest": "opt_open_interest"})
    return g.sort_values(["secid", "date"]).reset_index(drop=True)


def materialize_optionmetrics_enrichment(
    lake_base_dir: str | Path | None = None,
    *,
    years: range | None = None,
    skip_stdopd: bool = False,
    skip_borrow: bool = False,
) -> dict[str, Any]:
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    macro = lake / "macro"
    macro.mkdir(parents=True, exist_ok=True)
    secids = lake_secids(lake)
    log.info("OM enrich for %d secids → %s", len(secids), macro)

    conn = connect_wrds()
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    try:
        distrd_path = macro / OM_DISTRD
        opvold_path = macro / OM_OPVOLD
        if distrd_path.is_file() and distrd_path.stat().st_size > 1000:
            log.info("reusing existing %s", distrd_path)
            print(f"reusing {distrd_path.name}", flush=True)
            paths["distrd"] = distrd_path
            counts["distrd"] = int(len(pd.read_parquet(distrd_path)))
        else:
            log.info("distrd …")
            print("distrd …", flush=True)
            distrd = fetch_distrd(conn, secids)
            distrd.to_parquet(distrd_path, index=False)
            paths["distrd"] = distrd_path
            counts["distrd"] = int(len(distrd))

        if opvold_path.is_file() and opvold_path.stat().st_size > 1000:
            log.info("reusing existing %s", opvold_path)
            print(f"reusing {opvold_path.name}", flush=True)
            opvold = pd.read_parquet(opvold_path)
            paths["opvold"] = opvold_path
            counts["opvold"] = int(len(opvold))
        else:
            log.info("opvold …")
            print("opvold …", flush=True)
            opvold = fetch_opvold(conn, secids)
            opvold.to_parquet(opvold_path, index=False)
            paths["opvold"] = opvold_path
            counts["opvold"] = int(len(opvold))

        opt_adv = build_option_adv_from_opvold(opvold)
        p = macro / "om_option_adv.parquet"
        opt_adv.to_parquet(p, index=False)
        paths["option_adv"] = p
        counts["option_adv"] = int(len(opt_adv))

        if not skip_borrow:
            log.info("borrate …")
            print("borrate …", flush=True)
            borr = fetch_year_partitioned(
                conn,
                table_prefix="borrate",
                secids=secids,
                years=years,
                columns="securityid, date, days, borrowrate",
                extra_where="and days between 10 and 60",
            )
            p = macro / OM_BORRATE
            borr.to_parquet(p, index=False)
            paths["borrate"] = p
            counts["borrate"] = int(len(borr))

            log.info("stdbrte …")
            print("stdbrte …", flush=True)
            stdb = fetch_year_partitioned(
                conn,
                table_prefix="stdbrte",
                secids=secids,
                years=years,
                columns="securityid, date, days, borrowrate",
                extra_where="and days between 10 and 60",
            )
            p = macro / OM_STDBRTE
            stdb.to_parquet(p, index=False)
            paths["stdbrte"] = p
            counts["stdbrte"] = int(len(stdb))

        log.info("securd / secnmd …")
        securd = fetch_securd(conn, secids)
        p = macro / OM_SECURD
        securd.to_parquet(p, index=False)
        paths["securd"] = p
        counts["securd"] = int(len(securd))

        secnmd = fetch_secnmd(conn, secids)
        p = macro / OM_SECNMD
        secnmd.to_parquet(p, index=False)
        paths["secnmd"] = p
        counts["secnmd"] = int(len(secnmd))

        if not skip_stdopd:
            log.info("stdopd near-ATM …")
            print("stdopd near-ATM …", flush=True)
            stdopd = fetch_stdopd_near_atm(conn, secids)
            p = macro / OM_STDOPD
            stdopd.to_parquet(p, index=False)
            paths["stdopd"] = p
            counts["stdopd"] = int(len(stdopd))

        # Skip re-pull of distrd/opvold if already present and non-empty
        log.info("idxdvd / zerocd …")
        print("idxdvd / zerocd …", flush=True)
        idxdvd = fetch_idxdvd(conn)
        p = macro / OM_IDXDVD
        idxdvd.to_parquet(p, index=False)
        paths["idxdvd"] = p
        counts["idxdvd"] = int(len(idxdvd))

        zerocd = fetch_zerocd(conn)
        p = macro / OM_ZEROCD
        zerocd.to_parquet(p, index=False)
        paths["zerocd"] = p
        counts["zerocd"] = int(len(zerocd))
    finally:
        conn.close()

    prov = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "lake": str(lake),
        "n_secids": len(secids),
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "redistributable": False,
        "sources": [
            "optionm.distrd",
            "optionm.opvold",
            "optionm.borrateYYYY",
            "optionm.stdbrteYYYY",
            "optionm.securd",
            "optionm.secnmd",
            "optionm.stdopdYYYY",
            "optionm.idxdvd",
            "optionm.zerocd",
        ],
    }
    prov_path = macro / OM_PROV
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    paths["provenance"] = prov_path
    return {"paths": {k: str(v) for k, v in paths.items()}, "provenance": prov}


def median_borrow_bps(
    stdbrte_path: str | Path,
    *,
    secid: int,
    start: str | None = None,
    end: str | None = None,
) -> float:
    """Median standardized borrow rate (percent) → bps; 0 if missing."""
    path = Path(stdbrte_path)
    if not path.is_file():
        return 0.0
    df = pd.read_parquet(path)
    if "secid" not in df.columns or "borrowrate" not in df.columns:
        return 0.0
    sub = df[df["secid"].astype(int) == int(secid)]
    if start:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    if sub.empty or not np.isfinite(sub["borrowrate"]).any():
        return 0.0
    rates = sub["borrowrate"].to_numpy(dtype=np.float64)
    # OM uses large-magnitude sentinels for missing / unavailable borrow.
    rates = rates[np.isfinite(rates) & (np.abs(rates) < 50.0) & (rates >= 0.0)]
    if rates.size == 0:
        return 0.0
    # OM borrowrate is typically in percent (e.g. 0.5 = 0.5%)
    return float(np.nanmedian(rates) * 100.0)


def attach_episode_om_fields(
    episodes: list,
    lake_base_dir: str | Path | None = None,
) -> dict[str, float]:
    """
    Attach OM borrow (stdbrte), option ADV, dividend flags, and stdopd ATM IV.

    Updates ``estimand_residuals['borrow_state']`` / American residual tags.
    """
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    std_path = lake / "macro" / OM_STDBRTE
    opt_path = lake / "macro" / "om_option_adv.parquet"
    distrd_path = lake / "macro" / OM_DISTRD
    stdopd_path = lake / "macro" / OM_STDOPD
    borrow_vals: list[float] = []
    opt_vals: list[float] = []
    stdopd_vals: list[float] = []

    opt_df = None
    if opt_path.is_file():
        opt_df = pd.read_parquet(opt_path)
    distrd = None
    if distrd_path.is_file():
        distrd = pd.read_parquet(distrd_path)
    stdopd = None
    if stdopd_path.is_file():
        # Keep memory bounded: load only columns we need
        stdopd = pd.read_parquet(
            stdopd_path, columns=["secid", "date", "impl_volatility", "cp_flag", "days"]
        )

    for ep in episodes:
        try:
            sid = int(float(str(ep.secid)))
        except (TypeError, ValueError):
            continue
        dates = list(getattr(ep, "dates", []) or [])
        start = str(dates[0])[:10] if dates else None
        end = str(dates[-1])[:10] if dates else None
        bps = median_borrow_bps(std_path, secid=sid, start=start, end=end)
        ep.borrow_bps_annual = float(bps)
        res = dict(getattr(ep, "estimand_residuals", {}) or {})
        if bps > 0:
            borrow_vals.append(bps)
            res["borrow_state"] = "optionmetrics_stdbrte"
        if distrd is not None and start and end:
            dsub = distrd[distrd["secid"].astype(int) == sid]
            if "ex_date" in dsub.columns:
                dsub = dsub[
                    (dsub["ex_date"] >= pd.Timestamp(start))
                    & (dsub["ex_date"] <= pd.Timestamp(end))
                ]
            if len(dsub) > 0:
                res["american_residual"] = "om_distrd_exdiv_in_window"
                res["n_exdiv_in_window"] = int(len(dsub))
            else:
                res.setdefault("american_residual", "disclosed")
        if opt_df is not None:
            sub = opt_df[opt_df["secid"].astype(int) == sid]
            if start:
                sub = sub[sub["date"] >= pd.Timestamp(start)]
            if end:
                sub = sub[sub["date"] <= pd.Timestamp(end)]
            if not sub.empty and np.isfinite(sub["opt_volume"]).any():
                v = float(np.nanmedian(sub["opt_volume"].to_numpy(dtype=np.float64)))
                ep.option_adv_volume = v
                opt_vals.append(v)
                res["option_adv_source"] = "om_option_adv"
        if stdopd is not None and start and end:
            ssub = stdopd[stdopd["secid"].astype(int) == sid]
            ssub = ssub[
                (ssub["date"] >= pd.Timestamp(start))
                & (ssub["date"] <= pd.Timestamp(end))
                & (ssub["cp_flag"].astype(str).str.upper() == "C")
            ]
            if not ssub.empty and np.isfinite(ssub["impl_volatility"]).any():
                iv = float(np.nanmedian(ssub["impl_volatility"].to_numpy(dtype=np.float64)))
                res["stdopd_atm_iv"] = iv
                res["stdopd_state"] = "om_stdopd_near_atm"
                stdopd_vals.append(iv)
        ep.estimand_residuals = res
    return {
        "median_borrow_bps_annual": float(np.median(borrow_vals)) if borrow_vals else 0.0,
        "median_option_adv_volume": float(np.median(opt_vals)) if opt_vals else 0.0,
        "median_stdopd_atm_iv": float(np.median(stdopd_vals)) if stdopd_vals else 0.0,
        "n_with_borrow": float(len(borrow_vals)),
        "n_with_option_adv": float(len(opt_vals)),
        "n_with_stdopd": float(len(stdopd_vals)),
    }
