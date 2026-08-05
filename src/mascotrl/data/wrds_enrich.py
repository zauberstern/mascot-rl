"""WRDS / local-lake enrichment: CRSP–OM link, dollar ADV, Compustat funda.

Vendor extracts are written under ``{lake}/macro/`` and are not redistributable.
Credentials: ``WRDS_USERNAME`` / ``WRDS_PW`` in env or gitignored ``.env``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.paths import LAKE_ROOT
from src.logging_utils import get_logger

log = get_logger("volsurf.wrds_enrich")

LINK_NAME = "crsp_optionm_link.parquet"
ADV_NAME = "crsp_om_adv.parquet"
COMP_NAME = "compustat_funda_enrich.parquet"
PROVENANCE_NAME = "wrds_enrichment_provenance.json"
WRDS_SUBDIR = "wrds"

# Canonical annual STD/INDL panel: identifiers + BS + IS + CF line items.
FUNDA_FULL_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "datadate",
    "fyear",
    "tic",
    "cusip",
    "conm",
    "exchg",
    "fic",
    # income statement
    "sale",
    "cogs",
    "xopr",
    "xsga",
    "oiadp",
    "oibdp",
    "ebit",
    "ebitda",
    "xint",
    "txt",
    "ib",
    "ni",
    "epsfx",
    "epspx",
    "dvp",
    "dvc",
    "dv",
    # balance sheet
    "at",
    "act",
    "che",
    "rect",
    "invt",
    "ppent",
    "intan",
    "ao",
    "lt",
    "lct",
    "dlc",
    "dltt",
    "ap",
    "txditc",
    "ceq",
    "seq",
    "pstk",
    "csho",
    "prcc_f",
    "mkvalt",
    # cash flow
    "oancf",
    "ivncf",
    "fincf",
    "capx",
    "sppe",
    "aqc",
    "fiao",
    "sstk",
    "prstkc",
    "dltis",
    "dltr",
)

FUNDQ_FULL_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "datadate",
    "fyearq",
    "fqtr",
    "tic",
    "cusip",
    "conm",
    "saleq",
    "cogsq",
    "xoprq",
    "oiadpq",
    "oibdpq",
    "niq",
    "ibq",
    "xintq",
    "txtq",
    "atq",
    "actq",
    "cheq",
    "rectq",
    "invtq",
    "ppentq",
    "ltq",
    "lctq",
    "dlcq",
    "dlttq",
    "ceqq",
    "seqq",
    "cshoq",
    "prccq",
    "oancfy",
    "capxy",
    "dvpsxq",
)


def _load_dotenv_files() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / ".env",
        root.parent / "VolSurf_PY_prototype" / ".env",
        root.parents[1] / "VolSurf_PY_prototype" / ".env",
        Path.home() / "Desktop" / "volsurf" / "VolSurf_PY_prototype" / ".env",
        Path.home() / "Desktop" / "VolSurf_PY_prototype" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def connect_wrds():
    """Authenticated wrds.Connection; raises if credentials missing.

    Never prompts on stdin (fail-closed for automation / seals).
    """
    _load_dotenv_files()
    user = os.environ.get("WRDS_USERNAME")
    pwd = os.environ.get("WRDS_PW")
    if not user:
        raise EnvironmentError(
            "WRDS_USERNAME not set (put credentials in volsurf/.env — gitignored)"
        )
    if not pwd:
        raise EnvironmentError(
            "WRDS_PW not set (required for non-interactive WRDS connect)"
        )
    import wrds

    # Refuse interactive username/password / .pgpass prompts.
    def _refuse_input(*_a, **_k):  # type: ignore[no-untyped-def]
        raise EnvironmentError(
            "WRDS interactive prompt refused; check credentials, network, or .pgpass"
        )

    import builtins

    connect_args = {
        # Duo Push needs wall-clock; default 30s is too short for phone approve.
        "connect_timeout": int(os.environ.get("WRDS_CONNECT_TIMEOUT", "120")),
        "sslmode": os.environ.get("WRDS_SSLMODE", "require"),
    }
    real_input = builtins.input
    builtins.input = _refuse_input  # type: ignore[assignment]
    try:
        conn = wrds.Connection(
            autoconnect=False,
            wrds_username=user,
            wrds_password=pwd,
            wrds_connect_args=connect_args,
        )
        # Prefer explicit raise on first engine attempt over credential re-prompt.
        try:
            conn._Connection__make_sa_engine_conn(raise_err=True)
        except Exception as exc:
            raise EnvironmentError(f"WRDS connection failed: {exc}") from exc
        if conn.engine is None:
            raise EnvironmentError("WRDS connection failed: engine is None")
        conn.load_library_list()
        return conn
    finally:
        builtins.input = real_input


def fetch_crsp_optionm_link(conn=None) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect_wrds()
    try:
        df = conn.raw_sql(
            """
            select secid, permno, sdate, edate, score
            from wrdsapps_link_crsp_optionm.opcrsphist
            """
        )
    finally:
        if own:
            conn.close()
    df = df.copy()
    df["secid"] = pd.to_numeric(df["secid"], errors="coerce").astype("Int64")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["sdate"] = pd.to_datetime(df["sdate"])
    df["edate"] = pd.to_datetime(df["edate"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["secid", "permno"])


def fetch_compustat_funda(
    conn=None,
    *,
    gvkeys: list[str] | None = None,
    start: str = "2003-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Annual Compustat fundamentals useful for dividend / American residuals."""
    own = conn is None
    conn = conn or connect_wrds()
    try:
        sql = f"""
            select gvkey, datadate, fyear, tic, cusip, conm,
                   dvc, dv, prcc_f, csho, at, sale, ni
            from comp.funda
            where indfmt = 'INDL'
              and datafmt = 'STD'
              and consol = 'C'
              and popsrc in ('D', 'A')
              and datadate between '{start}' and '{end}'
        """
        if gvkeys:
            keys = ",".join(f"'{g}'" for g in gvkeys)
            sql += f" and gvkey in ({keys})"
        df = conn.raw_sql(sql)
    finally:
        if own:
            conn.close()
    df = df.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def fetch_compustat_funda_full(
    conn=None,
    *,
    columns: tuple[str, ...] = FUNDA_FULL_COLUMNS,
    start: str = "1950-01-01",
    end: str = "2099-12-31",
    year: int | None = None,
) -> pd.DataFrame:
    """Full-entitlement annual Compustat funda (BS/IS/CF line items)."""
    own = conn is None
    conn = conn or connect_wrds()
    cols = ", ".join(columns)
    try:
        sql = f"""
            select {cols}
            from comp.funda
            where indfmt = 'INDL'
              and datafmt = 'STD'
              and consol = 'C'
              and popsrc in ('D', 'A')
              and datadate between '{start}' and '{end}'
        """
        if year is not None:
            sql += f" and fyear = {int(year)}"
        df = conn.raw_sql(sql)
    finally:
        if own:
            conn.close()
    df = df.copy()
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def fetch_compustat_fundq_full(
    conn=None,
    *,
    columns: tuple[str, ...] = FUNDQ_FULL_COLUMNS,
    start: str = "1950-01-01",
    end: str = "2099-12-31",
    year: int | None = None,
) -> pd.DataFrame:
    """Full-entitlement quarterly Compustat fundq."""
    own = conn is None
    conn = conn or connect_wrds()
    cols = ", ".join(columns)
    try:
        sql = f"""
            select {cols}
            from comp.fundq
            where indfmt = 'INDL'
              and datafmt = 'STD'
              and consol = 'C'
              and popsrc in ('D', 'A')
              and datadate between '{start}' and '{end}'
        """
        if year is not None:
            sql += f" and fyearq = {int(year)}"
        df = conn.raw_sql(sql)
    finally:
        if own:
            conn.close()
    df = df.copy()
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def fetch_compustat_company(conn=None) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect_wrds()
    try:
        df = conn.raw_sql(
            """
            select gvkey, conm, tic, cusip, cik, sic, naics, state, fyrc, ipodate
            from comp.company
            """
        )
    finally:
        if own:
            conn.close()
    return df.copy()


def fetch_ccm_link_full(conn=None) -> pd.DataFrame:
    """Full CCM link history (not restricted to lake permnos)."""
    own = conn is None
    conn = conn or connect_wrds()
    try:
        df = conn.raw_sql(
            """
            select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
            from crsp.ccmxpf_lnkhist
            where linktype in ('LU', 'LC')
              and linkprim in ('P', 'C')
            """
        )
    finally:
        if own:
            conn.close()
    df = df.copy()
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["linkdt"] = pd.to_datetime(df["linkdt"], errors="coerce")
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce")
    return df


def project_spine_compustat_funda(
    funda_full: pd.DataFrame,
    ccm: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """Thin spine enrich projection for ``attach_episode_compustat`` compatibility."""
    need = ["gvkey", "datadate", "fyear", "tic", "cusip", "conm", "dvc", "dv", "prcc_f", "csho", "at", "sale", "ni"]
    missing = [c for c in need if c not in funda_full.columns]
    if missing:
        raise ValueError(f"funda_full missing columns for spine projection: {missing}")
    comp = funda_full[need].copy()
    comp["gvkey"] = comp["gvkey"].astype(str).str.zfill(6)
    ccm2 = ccm.copy()
    ccm2["gvkey"] = ccm2["gvkey"].astype(str).str.zfill(6)
    ccm2["permno"] = pd.to_numeric(ccm2["permno"], errors="coerce")
    comp = comp.merge(ccm2[["gvkey", "permno"]].drop_duplicates(), on="gvkey", how="left")
    lk = link[["permno", "secid"]].drop_duplicates().copy()
    lk["permno"] = pd.to_numeric(lk["permno"], errors="coerce")
    comp = comp.merge(lk, on="permno", how="inner")
    return comp


def materialize_wrds_fundamentals_full(
    lake_base_dir: str | Path | None = None,
    *,
    years: range | None = None,
    include_fundq: bool = True,
    regenerate_spine_enrich: bool = True,
) -> dict[str, Any]:
    """Pull full Compustat entitlement into ``macro/wrds/``; optionally refresh thin spine enrich."""
    import json
    from datetime import datetime, timezone

    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    wrds_dir = lake / "macro" / WRDS_SUBDIR
    wrds_dir.mkdir(parents=True, exist_ok=True)
    year_iter = list(years) if years is not None else list(range(1950, datetime.now(timezone.utc).year + 1))

    conn = connect_wrds()
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    try:
        log.info("comp.company …")
        company = fetch_compustat_company(conn)
        p = wrds_dir / "comp_company.parquet"
        company.to_parquet(p, index=False)
        paths["company"] = p
        counts["company"] = int(len(company))

        log.info("crsp.ccmxpf_lnkhist full …")
        ccm = fetch_ccm_link_full(conn)
        p = wrds_dir / "comp_ccm_link.parquet"
        ccm.to_parquet(p, index=False)
        paths["ccm"] = p
        counts["ccm"] = int(len(ccm))

        funda_chunks: list[pd.DataFrame] = []
        for y in year_iter:
            log.info("comp.funda year=%d …", y)
            chunk = fetch_compustat_funda_full(conn, year=y)
            if len(chunk):
                funda_chunks.append(chunk)
        funda = pd.concat(funda_chunks, ignore_index=True) if funda_chunks else pd.DataFrame()
        p = wrds_dir / "comp_funda_full.parquet"
        funda.to_parquet(p, index=False)
        paths["funda_full"] = p
        counts["funda_full"] = int(len(funda))

        fundq = pd.DataFrame()
        if include_fundq:
            fundq_chunks: list[pd.DataFrame] = []
            for y in year_iter:
                log.info("comp.fundq year=%d …", y)
                try:
                    chunk = fetch_compustat_fundq_full(conn, year=y)
                except Exception as exc:
                    log.warning("fundq year=%d skipped: %s", y, exc)
                    continue
                if len(chunk):
                    fundq_chunks.append(chunk)
            fundq = pd.concat(fundq_chunks, ignore_index=True) if fundq_chunks else pd.DataFrame()
            p = wrds_dir / "comp_fundq_full.parquet"
            fundq.to_parquet(p, index=False)
            paths["fundq_full"] = p
            counts["fundq_full"] = int(len(fundq))

        if regenerate_spine_enrich and len(funda):
            link_path = lake / "macro" / LINK_NAME
            if link_path.is_file():
                link = pd.read_parquet(link_path)
            else:
                link = fetch_crsp_optionm_link(conn)
            spine = project_spine_compustat_funda(funda, ccm, link)
            p = lake / "macro" / COMP_NAME
            spine.to_parquet(p, index=False)
            paths["spine_enrich"] = p
            counts["spine_enrich"] = int(len(spine))
            p2 = wrds_dir / "comp_funda_spine.parquet"
            spine.to_parquet(p2, index=False)
            paths["funda_spine"] = p2
    finally:
        conn.close()

    prov = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "lake": str(lake),
        "years": [int(y) for y in year_iter],
        "funda_columns": list(FUNDA_FULL_COLUMNS),
        "fundq_columns": list(FUNDQ_FULL_COLUMNS) if include_fundq else [],
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "redistributable": False,
    }
    prov_path = wrds_dir / "wrds_fundamentals_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    paths["provenance"] = prov_path
    return {"paths": paths, "counts": counts, "provenance": prov}


def fetch_ccm_links_for_permnos(conn, permnos: list[int]) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame()
    ids = ",".join(str(int(p)) for p in sorted(set(int(p) for p in permnos)))
    return conn.raw_sql(
        f"""
        select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
        from crsp.ccmxpf_lnkhist
        where lpermno in ({ids})
          and linktype in ('LU', 'LC')
          and linkprim in ('P', 'C')
        """
    )


def build_adv_panel_from_crsp_and_link(
    crsp: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """Dollar ADV = |PRC| * VOL joined to OM secid via link windows."""
    c = crsp.copy()
    # Normalize column case
    colmap = {x: x.upper() if isinstance(x, str) else x for x in c.columns}
    c = c.rename(columns=colmap)
    need = {"PERMNO", "date", "PRC", "VOL"}
    missing = need - set(c.columns)
    if missing:
        # try lowercase
        c = crsp.copy()
        c.columns = [str(x).lower() for x in c.columns]
        rename = {
            "permno": "PERMNO",
            "date": "date",
            "prc": "PRC",
            "vol": "VOL",
        }
        c = c.rename(columns={k: v for k, v in rename.items() if k in c.columns})
        missing = need - set(c.columns)
        if missing:
            raise ValueError(f"CRSP frame missing columns {missing}")
    c["date"] = pd.to_datetime(c["date"])
    c["PERMNO"] = pd.to_numeric(c["PERMNO"], errors="coerce").astype("Int64")
    c["PRC"] = pd.to_numeric(c["PRC"], errors="coerce").abs()
    c["VOL"] = pd.to_numeric(c["VOL"], errors="coerce")
    c["adv"] = c["PRC"] * c["VOL"]
    c = c.dropna(subset=["PERMNO", "date", "adv"])

    lk = link.copy()
    lk["secid"] = pd.to_numeric(lk["secid"], errors="coerce").astype("Int64")
    lk["permno"] = pd.to_numeric(lk["permno"], errors="coerce").astype("Int64")
    lk["sdate"] = pd.to_datetime(lk["sdate"])
    lk["edate"] = pd.to_datetime(lk["edate"]).fillna(pd.Timestamp("2099-12-31"))
    lk = lk.dropna(subset=["secid", "permno"])

    merged = c.merge(lk, left_on="PERMNO", right_on="permno", how="inner")
    ok = (merged["date"] >= merged["sdate"]) & (merged["date"] <= merged["edate"])
    merged = merged.loc[ok]
    # Prefer best score when overlapping links
    if "score" in merged.columns:
        merged = merged.sort_values(["secid", "date", "score"])
        merged = merged.drop_duplicates(subset=["secid", "date"], keep="first")
    else:
        merged = merged.drop_duplicates(subset=["secid", "date"], keep="first")

    out = pd.DataFrame(
        {
            "date": merged["date"].values,
            "secid": merged["secid"].astype(int).values,
            "permno": merged["PERMNO"].astype(int).values,
            "adv": merged["adv"].astype(float).values,
            "prc": merged["PRC"].astype(float).values,
            "vol": merged["VOL"].astype(float).values,
        }
    )
    return out.sort_values(["secid", "date"]).reset_index(drop=True)


def load_adv_for_secid(
    adv_path: str | Path,
    *,
    secid: int | str,
    start: str | None = None,
    end: str | None = None,
) -> float:
    """Median dollar ADV over [start, end] for one secid; 0 if missing."""
    path = Path(adv_path)
    if not path.is_file():
        return 0.0
    df = pd.read_parquet(path)
    sid = int(float(secid))
    sub = df[df["secid"].astype(int) == sid]
    if start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    if sub.empty or not np.isfinite(sub["adv"]).any():
        return 0.0
    return float(np.nanmedian(sub["adv"].to_numpy(dtype=np.float64)))


def write_enrichment_parquets(
    lake_base_dir: str | Path,
    *,
    link: pd.DataFrame | None = None,
    adv: pd.DataFrame | None = None,
    compustat: pd.DataFrame | None = None,
) -> dict[str, Path]:
    macro = Path(lake_base_dir) / "macro"
    macro.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if link is not None:
        p = macro / LINK_NAME
        link.to_parquet(p, index=False)
        paths["link"] = p
    if adv is not None:
        p = macro / ADV_NAME
        adv.to_parquet(p, index=False)
        paths["adv"] = p
    if compustat is not None:
        p = macro / COMP_NAME
        compustat.to_parquet(p, index=False)
        paths["compustat"] = p
    return paths


def materialize_wrds_enrichment(
    lake_base_dir: str | Path | None = None,
    *,
    use_local_crsp: bool = True,
    fetch_compustat: bool = True,
) -> dict[str, Any]:
    """
    Pull CRSP–OM link (+ optional Compustat); build ADV from local CRSP parquet.

    Writes under ``{lake}/macro/`` and a small provenance JSON.
    """
    import json
    from datetime import datetime, timezone

    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    macro = lake / "macro"
    crsp_path = macro / "sp500_prices.parquet"
    if use_local_crsp and not crsp_path.is_file():
        raise FileNotFoundError(f"local CRSP parquet missing: {crsp_path}")

    conn = connect_wrds()
    try:
        log.info("Fetching CRSP–OptionMetrics link …")
        link = fetch_crsp_optionm_link(conn)
        log.info("link rows=%d secids=%d", len(link), link["secid"].nunique())

        crsp = pd.read_parquet(crsp_path)
        log.info("Building ADV from local CRSP %s …", crsp_path)
        adv = build_adv_panel_from_crsp_and_link(crsp, link)
        log.info(
            "adv rows=%d secids=%d median_adv=%.3g",
            len(adv),
            adv["secid"].nunique(),
            float(np.nanmedian(adv["adv"])) if len(adv) else float("nan"),
        )

        comp = None
        if fetch_compustat:
            permnos = adv["permno"].dropna().astype(int).unique().tolist()
            log.info("Fetching CCM + Compustat funda for %d permnos …", len(permnos))
            ccm = fetch_ccm_links_for_permnos(conn, permnos)
            if len(ccm):
                gvkeys = sorted({str(g).zfill(6) for g in ccm["gvkey"].astype(str)})
                # Chunk to keep SQL small
                chunks = []
                for i in range(0, len(gvkeys), 400):
                    chunks.append(
                        fetch_compustat_funda(conn, gvkeys=gvkeys[i : i + 400])
                    )
                comp = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                # attach permno via CCM (latest link)
                ccm2 = ccm.copy()
                ccm2["permno"] = pd.to_numeric(ccm2["permno"], errors="coerce")
                ccm2["gvkey"] = ccm2["gvkey"].astype(str).str.zfill(6)
                comp["gvkey"] = comp["gvkey"].astype(str).str.zfill(6)
                comp = comp.merge(
                    ccm2[["gvkey", "permno"]].drop_duplicates(),
                    on="gvkey",
                    how="left",
                )
                # map permno -> secid via link
                lk = link[["permno", "secid"]].drop_duplicates()
                comp = comp.merge(lk, on="permno", how="left")
                log.info("compustat rows=%d", len(comp))
            else:
                comp = pd.DataFrame()
                log.warning("CCM returned no rows; skipping Compustat write body")
    finally:
        conn.close()

    paths = write_enrichment_parquets(lake, link=link, adv=adv, compustat=comp)
    prov = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "lake": str(lake),
        "sources": {
            "crsp_local": str(crsp_path),
            "wrds_link": "wrdsapps_link_crsp_optionm.opcrsphist",
            "wrds_compustat": "comp.funda" if fetch_compustat else None,
            "wrds_ccm": "crsp.ccmxpf_lnkhist" if fetch_compustat else None,
        },
        "n_link": int(len(link)),
        "n_adv": int(len(adv)),
        "n_adv_secid": int(adv["secid"].nunique()) if len(adv) else 0,
        "n_compustat": int(len(comp)) if comp is not None else 0,
        "paths": {k: str(v) for k, v in paths.items()},
        "redistributable": False,
    }
    prov_path = macro / PROVENANCE_NAME
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    paths["provenance"] = prov_path
    return {"paths": {k: str(v) for k, v in paths.items()}, "provenance": prov}


def attach_episode_adv(
    episodes: list,
    adv_path: str | Path | None = None,
    lake_base_dir: str | Path | None = None,
) -> float:
    """
    Set ``episode.hedge_adv`` from the ADV parquet (median over episode dates).

    Returns the median ADV across episodes (for OmHedgeMDPConfig.hedge_adv).
    """
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    path = Path(adv_path) if adv_path else lake / "macro" / ADV_NAME
    if not path.is_file():
        log.warning("ADV parquet missing at %s — leaving hedge_adv=0", path)
        return 0.0
    adv_df = pd.read_parquet(path)
    vals: list[float] = []
    for ep in episodes:
        try:
            sid = int(float(str(ep.secid)))
        except (TypeError, ValueError):
            continue
        dates = list(getattr(ep, "dates", []) or [])
        start = dates[0] if dates else None
        end = dates[-1] if dates else None
        sub = adv_df[adv_df["secid"].astype(int) == sid]
        if start is not None:
            sub = sub[sub["date"] >= pd.Timestamp(str(start)[:10])]
        if end is not None:
            sub = sub[sub["date"] <= pd.Timestamp(str(end)[:10])]
        if sub.empty:
            # fallback: all-history median for secid
            sub = adv_df[adv_df["secid"].astype(int) == sid]
        if sub.empty or not np.isfinite(sub["adv"]).any():
            ep.hedge_adv = 0.0
            continue
        v = float(np.nanmedian(sub["adv"].to_numpy(dtype=np.float64)))
        ep.hedge_adv = v
        vals.append(v)
    return float(np.median(vals)) if vals else 0.0


def attach_episode_compustat(
    episodes: list,
    lake_base_dir: str | Path | None = None,
) -> dict[str, float]:
    """
    Attach Compustat annual funda onto episodes (dividend / size residuals).

    Sets ``estimand_residuals['compustat_state']`` and optional ``dvc_annual``.
    Requires ``compustat_funda_enrich.parquet`` with a ``secid`` column.
    """
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    path = lake / "macro" / COMP_NAME
    if not path.is_file():
        log.warning("Compustat enrich missing at %s", path)
        return {"n_with_compustat": 0.0}
    df = pd.read_parquet(path)
    if "secid" not in df.columns:
        return {"n_with_compustat": 0.0}
    n_hit = 0
    for ep in episodes:
        try:
            sid = int(float(str(ep.secid)))
        except (TypeError, ValueError):
            continue
        dates = list(getattr(ep, "dates", []) or [])
        end = str(dates[-1])[:10] if dates else None
        sub = df[df["secid"].astype(int) == sid]
        if end and "datadate" in sub.columns:
            sub = sub[sub["datadate"] <= pd.Timestamp(end)]
        if sub.empty:
            continue
        row = sub.sort_values("datadate").iloc[-1]
        res = dict(getattr(ep, "estimand_residuals", {}) or {})
        res["compustat_state"] = "compustat_funda_enrich"
        if "dvc" in row.index and pd.notna(row["dvc"]):
            res["dvc_annual"] = float(row["dvc"])
            if float(row["dvc"]) > 0:
                res.setdefault("american_residual", "disclosed")
                # Strengthen American tag when Compustat shows cash dividends
                if res.get("american_residual") == "disclosed":
                    res["american_residual"] = "compustat_dvc_positive"
        if "prcc_f" in row.index and "csho" in row.index:
            try:
                mcap = float(row["prcc_f"]) * float(row["csho"]) * 1e6
                if np.isfinite(mcap) and mcap > 0:
                    res["mktcap_approx"] = mcap
            except (TypeError, ValueError):
                pass
        ep.estimand_residuals = res
        n_hit += 1
    return {"n_with_compustat": float(n_hit)}
