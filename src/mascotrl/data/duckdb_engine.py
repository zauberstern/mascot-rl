"""Phase B: DuckDB feature engine over Parquet lakes → Arrow tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from mascotrl.data.paths import LAKE_ROOT
from mascotrl.logging_utils import get_logger

log = get_logger("volsurf.data.duckdb")


@dataclass(frozen=True)
class OptionFilterConfig:
    """
    Literature-standard OptionMetrics chain screens.

    Defaults follow Cao and Han (2013, JFE 108(1)) Section 2, with the
    contract-standardization screens of Goyal and Saretto (2009) and the
    Dallas Fed IPCA study (WP 2214): drop non-standard settlement,
    non-common-stock underliers, index options, quotes violating basic
    no-arbitrage bounds, zero-volume and sub-tick quotes, and options whose
    life spans a cash dividend on the underlying.

    ``calls_only`` restricts the selected contract to calls, which is the
    convention for delta-hedged *call* returns and also guarantees a positive
    capital base (Δ·S − C) for the scaling denominator.
    """

    calls_only: bool = True
    moneyness_lo: float = 0.8
    moneyness_hi: float = 1.2
    require_iv: bool = True
    min_volume: float = 0.0
    min_mid: float = 0.125
    require_no_arb: bool = True
    no_calendar_arbitrage: bool = True
    no_butterfly_arbitrage: bool = True
    # Surface calendar/butterfly checks are reported for the data appendix.
    # Dropping whole (secid, date) rows on *any* surface violation is too
    # aggressive for EOD mid noise (empirically wiped ~99% of marks). Default
    # is attrition-only; set True only for a documented hard-screen robustness.
    drop_surface_arb_days: bool = False
    standard_settlement_only: bool = True
    common_stock_only: bool = True
    exclude_index_options: bool = True
    exclude_dividend_in_life: bool = True
    dte_lo: int = 14
    dte_hi: int = 45
    # Opt-in stale-quote screen (OM ``last_date``). Default False preserves
    # literature screen set; set True to fail-closed when last_date is present.
    require_fresh_quotes: bool = False

    @classmethod
    def disabled(cls) -> "OptionFilterConfig":
        """Pre-remediation behaviour, for attrition baselines and A/B rows."""
        return cls(
            calls_only=False,
            moneyness_lo=0.0,
            moneyness_hi=1e9,
            require_iv=False,
            min_volume=-1.0,
            min_mid=0.0,
            require_no_arb=False,
            no_calendar_arbitrage=False,
            no_butterfly_arbitrage=False,
            drop_surface_arb_days=False,
            standard_settlement_only=False,
            common_stock_only=False,
            exclude_index_options=False,
            exclude_dividend_in_life=False,
            require_fresh_quotes=False,
        )

    def screens(self) -> list[tuple[str, str]]:
        """
        Ordered ``(name, sql_predicate)`` pairs.

        Single source of truth: the marks query filters on these and the
        attrition report counts failures against the same expressions, so the
        published table cannot drift from what was actually applied.
        """
        out: list[tuple[str, str]] = []
        if self.require_iv:
            out.append(("iv_present", "impl_volatility IS NOT NULL"))
        if self.min_volume >= 0.0:
            out.append(("volume_positive", f"volume > {float(self.min_volume)}"))
        if self.min_mid > 0.0:
            out.append(("mid_above_tick", f"mid >= {float(self.min_mid)}"))
        if self.moneyness_lo > 0.0 or self.moneyness_hi < 1e8:
            out.append(
                (
                    "moneyness_band",
                    f"spot IS NOT NULL AND strike > 0 AND "
                    f"(spot / strike) BETWEEN {float(self.moneyness_lo)} "
                    f"AND {float(self.moneyness_hi)}",
                )
            )
        if self.require_no_arb:
            # Calls: max(0, S-K) <= C <= S. Puts: max(0, K-S) <= P <= K.
            # Undiscounted bounds are deliberately loose so that only genuine
            # quote errors are removed, not carry effects.
            out.append(
                (
                    "no_arbitrage_bounds",
                    "spot IS NOT NULL AND ("
                    "(cp_flag = 'C' AND mid <= spot * 1.0001 "
                    "  AND mid >= GREATEST(0.0, spot - strike) - 0.0001) OR "
                    "(cp_flag = 'P' AND mid <= strike * 1.0001 "
                    "  AND mid >= GREATEST(0.0, strike - spot) - 0.0001))",
                )
            )
        if self.standard_settlement_only:
            out.append(("standard_settlement", "COALESCE(ss_flag, 0) = 0"))
        if self.common_stock_only:
            out.append(("common_stock", "COALESCE(issue_type, '0') = '0'"))
        if self.exclude_index_options:
            out.append(("not_index_option", "COALESCE(index_flag, 0) = 0"))
        if self.exclude_dividend_in_life:
            out.append(("no_dividend_in_life", "no_dividend_in_life"))
        if self.require_fresh_quotes:
            # Fail closed when last_date is present and older than the quote date.
            # Rows with NULL last_date pass (column may be absent / unmapped).
            out.append(
                (
                    "fresh_quotes",
                    "last_date IS NULL OR CAST(last_date AS DATE) >= CAST(date AS DATE)",
                )
            )
        return out

    def selection_screens(self) -> list[tuple[str, str]]:
        """Screens applied only when picking the traded contract."""
        out: list[tuple[str, str]] = []
        if self.calls_only:
            out.append(("calls_only", "cp_flag = 'C'"))
        return out


def quote_quality_audit_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count crossed quotes, zero-volume, and stale last_date rows (audit only).

    Does not mutate ``df``. Missing columns contribute 0 for that counter.
    """
    n = int(len(df)) if df is not None else 0
    out: dict[str, int] = {
        "n_rows": n,
        "n_crossed_bid_ask": 0,
        "n_zero_volume": 0,
        "n_stale_last_date": 0,
        "n_missing_iv": 0,
    }
    if df is None or n == 0:
        return out
    if "best_bid" in df.columns and "best_offer" in df.columns:
        bid = pd.to_numeric(df["best_bid"], errors="coerce")
        offer = pd.to_numeric(df["best_offer"], errors="coerce")
        out["n_crossed_bid_ask"] = int(((offer < bid) | (bid <= 0)).fillna(False).sum())
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        out["n_zero_volume"] = int((vol.fillna(0) <= 0).sum())
    if "impl_volatility" in df.columns:
        iv = pd.to_numeric(df["impl_volatility"], errors="coerce")
        out["n_missing_iv"] = int(iv.isna().sum())
    if "last_date" in df.columns and "date" in df.columns:
        last = pd.to_datetime(df["last_date"], errors="coerce")
        dt = pd.to_datetime(df["date"], errors="coerce")
        stale = last.notna() & dt.notna() & (last < dt)
        out["n_stale_last_date"] = int(stale.sum())
    return out


class DuckDBFeatureEngine:
    def __init__(self, lake_base_dir: str | Path | None = None):
        self.lake_base_dir = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("SET threads = 8;")
        self.con.execute("SET max_memory = '24GB';")

    def query_arrow(self, sql_query: str) -> pa.Table:
        result = self.con.execute(sql_query)
        if hasattr(result, "to_arrow_table"):
            return result.to_arrow_table()
        if hasattr(result, "fetch_arrow_table"):
            return result.fetch_arrow_table()
        out = result.arrow()
        if isinstance(out, pa.Table):
            return out
        return out.read_all()

    def compute_iv_dispersion(self, start_date: str, end_date: str) -> pa.Table:
        glob = (self.lake_base_dir / "vol_surface" / "*" / "*" / "*.parquet").as_posix()
        # Also allow flat parquet
        flat = (self.lake_base_dir / "vol_surface" / "data.parquet").as_posix()
        sql = f"""
        WITH src AS (
            SELECT * FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
            WHERE CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{flat}', union_by_name=true)
            WHERE CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        )
        SELECT
            CAST(date AS DATE) AS date,
            AVG(impl_volatility) AS mean_iv,
            (STDDEV_SAMP(impl_volatility) / NULLIF(AVG(impl_volatility), 0)) AS iv_dispersion,
            COUNT(secid) AS n_stocks
        FROM src
        WHERE days = 30 AND delta = 50 AND cp_flag = 'C'
        GROUP BY 1
        ORDER BY 1
        """
        try:
            return self.query_arrow(sql)
        except Exception:
            # Schema-flexible fallback for smoke / alternate column names
            sql_fb = f"""
            SELECT CAST(date AS DATE) AS date,
                   AVG(TRY_CAST(impl_volatility AS DOUBLE)) AS mean_iv,
                   COUNT(*) AS n_stocks
            FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
            WHERE CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            GROUP BY 1 ORDER BY 1
            """
            return self.query_arrow(sql_fb)

    def _options_glob(self, year: int | None = None) -> str:
        if year is None:
            return (self.lake_base_dir / "options_panel" / "*" / "*" / "*.parquet").as_posix()
        return (
            self.lake_base_dir / "options_panel" / f"year={int(year)}" / "*" / "*.parquet"
        ).as_posix()

    def select_liquid_universe(
        self,
        n_assets: int,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Liquidity shortlist: top-n_assets by mean open interest (Jan/Jul samples).

        Prefer :meth:`select_iv_hypergraph_universe` for training universes —
        this remains the liquidity prefilter / fallback.
        """
        y0 = int(str(start_date)[:4])
        y1 = int(str(end_date)[:4])
        parts = []
        for y in range(y0, y1 + 1):
            for month in (1, 7):
                g = (
                    self.lake_base_dir
                    / "options_panel"
                    / f"year={y}"
                    / f"month={month}"
                    / "*.parquet"
                ).as_posix()
                parts.append(
                    f"""
                    SELECT
                        secid,
                        ticker,
                        issuer,
                        TRY_CAST(open_interest AS DOUBLE) AS open_interest
                    FROM read_parquet('{g}', union_by_name=true)
                    WHERE CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                      AND open_interest IS NOT NULL AND open_interest > 0
                      AND ticker IS NOT NULL AND length(trim(ticker)) > 0
                    """
                )
        union = " UNION ALL BY NAME ".join(parts)
        sql = f"""
        SELECT
            secid,
            upper(trim(mode(ticker))) AS ticker,
            mode(issuer) AS issuer,
            AVG(open_interest) AS mean_oi
        FROM ({union})
        GROUP BY secid
        ORDER BY mean_oi DESC
        LIMIT {int(n_assets)}
        """
        tbl = self.query_arrow(sql)
        out: list[dict] = []
        for i in range(tbl.num_rows):
            secid = int(tbl.column("secid")[i].as_py())
            ticker = str(tbl.column("ticker")[i].as_py() or "").strip().upper()
            issuer = str(tbl.column("issuer")[i].as_py() or "").strip()
            if not ticker:
                ticker = f"SECID_{secid}"
            out.append(
                {
                    "secid": secid,
                    "ticker": ticker,
                    "issuer": issuer,
                    "mean_oi": float(tbl.column("mean_oi")[i].as_py() or 0.0),
                }
            )
        return out

    def _membership_parquet(self) -> str:
        return (self.lake_base_dir / "macro" / "pit_membership.parquet").as_posix()

    def pit_membership_tickers(self, as_of: str) -> set[str]:
        """
        Index members as of ``as_of`` from the PIT membership snapshot.

        Returns an empty set when the snapshot is unavailable, so callers can
        degrade to "membership not enforced" and disclose it rather than
        silently applying a survivorship-biased universe.
        """
        path = Path(self._membership_parquet())
        if not path.is_file():
            return set()
        sql = f"""
        SELECT DISTINCT upper(trim(ticker)) AS ticker
        FROM read_parquet('{path.as_posix()}', union_by_name=true)
        WHERE TRY_CAST(start_date AS DATE) <= DATE '{as_of}'
          AND (
              end_date IS NULL
              OR trim(CAST(end_date AS VARCHAR)) = ''
              OR TRY_CAST(end_date AS DATE) >= DATE '{as_of}'
          )
        """
        try:
            tbl = self.query_arrow(sql)
        except Exception:
            return set()
        return {
            str(tbl.column("ticker")[i].as_py() or "").strip().upper()
            for i in range(tbl.num_rows)
        }

    def select_iv_hypergraph_universe(
        self,
        n_assets: int,
        start_date: str,
        end_date: str,
        *,
        liquidity_pool: int = 200,
        corr_threshold: float = 0.35,
        tail_threshold: float = 0.90,
        min_obs: int = 40,
        selection_metric: str = "copula_tail",
    ) -> list[dict]:
        """
        Select equities that fuel a dense DHGNN-friendly IV hypergraph.

        Default metric is empirical-copula **upper-tail dependence** (λ_U),
        matching Layer-3 SpatialDHGNN incidence (Pearson is banned online and
        must not pre-filter the offline universe either).

        Protocol:
          1) Liquidity prefilter: top ``liquidity_pool`` by mean OI.
          2) Build 30d ATM call IV panels from ``vol_surface``.
          3) Rank transform → empirical copula U(0,1); λ_U at ``tail_threshold``.
          4) Score vs equal-weight index IV tail co-exceedances.
          5) Greedy densest subgraph on pairwise λ_U.

        ``corr_threshold`` / ``selection_metric='pearson'`` remain available as
        an explicit legacy override only.
        """
        import numpy as np

        pool_n = max(int(liquidity_pool), int(n_assets))
        liquid = self.select_liquid_universe(pool_n, start_date, end_date)
        if len(liquid) < n_assets:
            return liquid

        secids = [int(r["secid"]) for r in liquid]
        id_list = ", ".join(str(s) for s in secids)
        y0 = int(str(start_date)[:4])
        y1 = int(str(end_date)[:4])
        parts = []
        for y in range(y0, y1 + 1):
            for month in (1, 6, 7, 12):
                g = (
                    self.lake_base_dir
                    / "vol_surface"
                    / f"year={y}"
                    / f"month={month}"
                    / "*.parquet"
                ).as_posix()
                parts.append(
                    f"""
                    SELECT
                        secid,
                        CAST(date AS DATE) AS date,
                        TRY_CAST(impl_volatility AS DOUBLE) AS atm_iv
                    FROM read_parquet('{g}', union_by_name=true)
                    WHERE secid IN ({id_list})
                      AND CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                      AND TRY_CAST(days AS INTEGER) = 30
                      AND TRY_CAST(delta AS INTEGER) = 50
                      AND cp_flag = 'C'
                      AND impl_volatility IS NOT NULL
                    """
                )
        union = " UNION ALL BY NAME ".join(parts)
        sql = f"""
        SELECT secid, date, AVG(atm_iv) AS atm_iv
        FROM ({union})
        GROUP BY secid, date
        ORDER BY date, secid
        """
        try:
            long_tbl = self.query_arrow(sql)
        except Exception:
            return liquid[:n_assets]

        df = long_tbl.to_pandas()
        if df.empty:
            return liquid[:n_assets]
        wide = (
            df.pivot_table(index="date", columns="secid", values="atm_iv", aggfunc="last")
            .sort_index()
            .astype(float)
        )
        wide = wide.loc[:, wide.notna().sum(axis=0) >= int(min_obs)]
        meta = {int(r["secid"]): r for r in liquid}
        if wide.shape[1] < n_assets:
            have = set(int(c) for c in wide.columns)
            pad = [r for r in liquid if int(r["secid"]) not in have]
            out = [meta[int(c)] for c in wide.columns if int(c) in meta]
            for r in pad:
                if len(out) >= n_assets:
                    break
                out.append(r)
            return out[:n_assets]

        # Forward-fill only. A bfill here would score universe membership using
        # IV observed *after* the gap, importing future information into a
        # selection that is applied from the start of the panel.
        wide = wide.ffill()
        # Names still missing history at the front are dropped rather than
        # back-filled; the padding path below tops the universe back up.
        wide = wide.loc[:, wide.notna().sum(axis=0) >= int(min_obs)]
        wide = wide.dropna(how="any")
        if wide.empty or wide.shape[1] < n_assets:
            have = set(int(c) for c in wide.columns)
            out = [meta[int(c)] for c in wide.columns if int(c) in meta]
            for r in liquid:
                if len(out) >= n_assets:
                    break
                if int(r["secid"]) not in have:
                    out.append(r)
            return out[:n_assets]
        cols = [int(c) for c in wide.columns]
        arr = wide.to_numpy(dtype=np.float64)
        T, Kp = arr.shape
        metric = str(selection_metric or "copula_tail").lower()

        if metric in ("pearson", "corr", "linear"):
            # Legacy override — not used by default overnight configs.
            index_iv = arr.mean(axis=1)
            idx_score = np.zeros(Kp, dtype=np.float64)
            for i in range(Kp):
                s = arr[:, i]
                if s.std() < 1e-12 or index_iv.std() < 1e-12:
                    idx_score[i] = 0.0
                else:
                    c = float(np.corrcoef(s, index_iv)[0, 1])
                    idx_score[i] = c if np.isfinite(c) else 0.0
            mu = arr.mean(axis=0, keepdims=True)
            sd = np.where(arr.std(axis=0, keepdims=True) < 1e-12, 1.0, arr.std(axis=0, keepdims=True))
            z = (arr - mu) / sd
            edge = (z.T @ z) / max(T - 1, 1)
            np.fill_diagonal(edge, 0.0)
            edge = (edge >= float(corr_threshold)).astype(np.float64)
            selection_tag = "iv_hypergraph_pearson"
            score_key = "index_iv_corr"
        else:
            # Empirical copula ranks → upper-tail dependence λ_U (DHGNN-aligned).
            # Average ranks for ties via argsort-of-argsort (stable enough for IV).
            ranks = np.argsort(np.argsort(arr, axis=0), axis=0).astype(np.float64) + 1.0
            U = ranks / (T + 1.0)
            thr = float(tail_threshold)
            exceed = U > thr  # (T, K)
            # Pairwise P(U_i>θ, U_j>θ) / P(U_j>θ) then symmetrize.
            joint = np.logical_and(exceed[:, :, None], exceed[:, None, :]).sum(
                axis=0, dtype=np.float64
            )
            marg = np.maximum(exceed.sum(axis=0, dtype=np.float64), 1.0)
            lam = joint / marg[:, None]
            edge = 0.5 * (lam + lam.T)
            np.fill_diagonal(edge, 0.0)

            index_iv = arr.mean(axis=1)
            idx_ranks = np.argsort(np.argsort(index_iv)).astype(np.float64) + 1.0
            U_idx = idx_ranks / (T + 1.0)
            idx_exceed = U_idx > thr
            idx_joint = np.logical_and(exceed, idx_exceed[:, None])
            idx_score = idx_joint.sum(axis=0, dtype=np.float64) / max(
                float(idx_exceed.sum()), 1.0
            )
            selection_tag = "copula_tail_hypergraph"
            score_key = "tail_dependence_score"

        # Soft survivor pool: top by index co-movement score.
        order = np.argsort(-idx_score)
        pool_idx = order[: max(n_assets * 2, n_assets)].tolist()
        if len(pool_idx) < n_assets:
            pool_idx = list(range(Kp))

        chosen: list[int] = []
        remaining = list(pool_idx)
        seed = max(remaining, key=lambda i: float(idx_score[i]))
        chosen.append(seed)
        remaining.remove(seed)
        while len(chosen) < n_assets and remaining:
            def score(i: int) -> tuple[float, float]:
                e = float(edge[i, chosen].sum()) if chosen else 0.0
                return (e, float(idx_score[i]))

            nxt = max(remaining, key=score)
            chosen.append(nxt)
            remaining.remove(nxt)

        out: list[dict] = []
        for i in chosen:
            sid = cols[i]
            row = dict(meta.get(sid, {"secid": sid, "ticker": f"SECID_{sid}", "issuer": ""}))
            row[score_key] = float(idx_score[i])
            row["selection"] = selection_tag
            out.append(row)
        return out[:n_assets]

    def _sec_prices_parquet(self) -> str:
        return (self.lake_base_dir / "macro" / "sp500_sec.parquet").as_posix()

    def _rates_parquet(self) -> str:
        return (self.lake_base_dir / "macro" / "interest_rate.parquet").as_posix()

    def _crsp_prices_parquet(self) -> str:
        return (self.lake_base_dir / "macro" / "sp500_prices.parquet").as_posix()

    def _marks_base_sql(
        self,
        id_list: str,
        start_date: str,
        end_date: str,
        cfg: "OptionFilterConfig",
    ) -> str:
        """
        CTE prefix shared by the marks query and the filter-attrition report.

        Ends with a ``screened`` CTE carrying every column the screens need,
        including the ``no_dividend_in_life`` flag, but applying none of them.
        """
        y0 = int(str(start_date)[:4])
        y1 = int(str(end_date)[:4])
        # Pull one extra year of lead so the last OOS day can still form a label.
        y_lead = min(y1 + 1, 2024)
        parts = []
        for y in range(y0, y_lead + 1):
            g = self._options_glob(y)
            # DTE window is widened vs the selection band so a contract selected
            # at DTE=dte_lo still has a next-session quote one day later.
            parts.append(
                f"""
                SELECT
                    secid,
                    CAST(date AS DATE) AS date,
                    CAST(exdate AS DATE) AS exdate,
                    optionid,
                    cp_flag,
                    cusip,
                    TRY_CAST(strike_price AS DOUBLE) / 1000.0 AS strike,
                    TRY_CAST(delta AS DOUBLE) AS delta,
                    TRY_CAST(impl_volatility AS DOUBLE) AS impl_volatility,
                    TRY_CAST(best_bid AS DOUBLE) AS best_bid,
                    TRY_CAST(best_offer AS DOUBLE) AS best_offer,
                    TRY_CAST(volume AS DOUBLE) AS volume,
                    TRY_CAST(open_interest AS DOUBLE) AS open_interest,
                    TRY_CAST(cfadj AS DOUBLE) AS cfadj,
                    TRY_CAST(ss_flag AS DOUBLE) AS ss_flag,
                    TRY_CAST(issue_type AS VARCHAR) AS issue_type,
                    TRY_CAST(index_flag AS DOUBLE) AS index_flag,
                    TRY_CAST(last_date AS DATE) AS last_date,
                    date_diff('day', CAST(date AS DATE), CAST(exdate AS DATE)) AS dte
                FROM read_parquet('{g}', hive_partitioning=1, union_by_name=true)
                WHERE secid IN ({id_list})
                  AND CAST(date AS DATE) BETWEEN DATE '{start_date}'
                      AND DATE '{end_date}' + INTERVAL 10 DAY
                  AND date_diff('day', CAST(date AS DATE), CAST(exdate AS DATE))
                      BETWEEN {max(1, int(cfg.dte_lo) - 7)} AND {int(cfg.dte_hi) + 5}
                  AND best_bid IS NOT NULL AND best_offer IS NOT NULL
                  AND best_offer >= best_bid
                  AND best_bid > 0
                """
            )
        union = " UNION ALL BY NAME ".join(parts)
        sec = self._sec_prices_parquet()
        rates = self._rates_parquet()
        crsp = self._crsp_prices_parquet()
        return f"""
        WITH raw AS (
            {union}
        ),
        spot AS (
            SELECT
                TRY_CAST(secid AS BIGINT) AS secid,
                CAST(date AS DATE) AS date,
                TRY_CAST(close AS DOUBLE) AS spot,
                TRY_CAST(cfadj AS DOUBLE) AS sec_cfadj
            FROM read_parquet('{sec}', union_by_name=true)
            WHERE TRY_CAST(secid AS BIGINT) IN ({id_list})
              AND CAST(date AS DATE) BETWEEN DATE '{start_date}'
                  AND DATE '{end_date}' + INTERVAL 10 DAY
        ),
        rates AS (
            SELECT
                CAST(date AS DATE) AS date,
                TRY_CAST(dtb3 AS DOUBLE) AS rf_pct
            FROM read_parquet('{rates}', union_by_name=true)
            WHERE TRY_CAST(dtb3 AS DOUBLE) IS NOT NULL
        ),
        om_cusip AS (
            SELECT DISTINCT secid, SUBSTRING(cusip, 1, 8) AS cusip8 FROM raw
        ),
        -- CRSP cash-dividend ex-dates (DISTCD family 1) bridged to OM secid on
        -- historical CUSIP. Coverage is reported, not assumed: names absent
        -- from the bridge keep no_dividend_in_life = TRUE rather than being
        -- dropped on a linkage artifact.
        divs AS (
            SELECT DISTINCT o.secid, CAST(c.date AS DATE) AS ex_date
            FROM read_parquet('{crsp}', union_by_name=true) c
            JOIN om_cusip o
              ON SUBSTRING(c.NCUSIP, 1, 8) = o.cusip8
            WHERE TRY_CAST(c.DIVAMT AS DOUBLE) > 0
              AND SUBSTRING(CAST(c.DISTCD AS VARCHAR), 1, 1) = '1'
        ),
        quotes AS (
            SELECT
                r.*,
                (r.best_bid + r.best_offer) * 0.5 AS mid,
                (r.best_offer - r.best_bid) * 0.5 AS half_spread,
                s.spot,
                s.sec_cfadj
            FROM raw r
            LEFT JOIN spot s USING (secid, date)
        ),
        screened AS (
            SELECT
                q.*,
                NOT EXISTS (
                    SELECT 1 FROM divs d
                    WHERE d.secid = q.secid
                      AND d.ex_date > q.date
                      AND d.ex_date <= q.exdate
                ) AS no_dividend_in_life
            FROM quotes q
        )"""

    def compute_filter_attrition(
        self,
        secids: list[int],
        start_date: str,
        end_date: str,
        filters: OptionFilterConfig | None = None,
    ) -> pa.Table:
        """
        Per-screen attrition for the paper's data appendix.

        For each screen reports the marginal number of chain observations it
        rejects (evaluated independently on the base window) alongside the
        cumulative retained count, so reviewers can see both the standalone
        bite of each filter and the joint effect.
        """
        if not secids:
            raise ValueError("secids must be non-empty")
        cfg = filters or OptionFilterConfig()
        id_list = ", ".join(str(int(s)) for s in secids)
        base = self._marks_base_sql(id_list, start_date, end_date, cfg)
        screens = cfg.screens() + cfg.selection_screens()
        if not screens:
            raise ValueError("no screens configured")
        marginal = ",\n            ".join(
            f"COUNT(*) FILTER (WHERE NOT COALESCE(({p}), FALSE)) AS fail_{n}"
            for n, p in screens
        )
        cumulative = " AND ".join(f"COALESCE(({p}), FALSE)" for _, p in screens)
        sql = f"""
        {base},
        counts AS (
            SELECT
                COUNT(*) AS n_base,
                {marginal},
                COUNT(*) FILTER (WHERE {cumulative}) AS n_retained,
                COUNT(DISTINCT secid) AS n_secids,
                COUNT(*) FILTER (WHERE spot IS NULL) AS fail_spot_missing
            FROM screened
            WHERE date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        )
        SELECT * FROM counts
        """
        return self.query_arrow(sql)

    def dividend_bridge_coverage(
        self,
        secids: list[int],
        start_date: str,
        end_date: str,
        filters: OptionFilterConfig | None = None,
    ) -> pa.Table:
        """CUSIP-bridge coverage for the dividend screen (disclosure honesty)."""
        if not secids:
            raise ValueError("secids must be non-empty")
        cfg = filters or OptionFilterConfig()
        id_list = ", ".join(str(int(s)) for s in secids)
        base = self._marks_base_sql(id_list, start_date, end_date, cfg)
        sql = f"""
        {base},
        bridged AS (
            SELECT DISTINCT secid FROM divs
        )
        SELECT
            (SELECT COUNT(DISTINCT secid) FROM screened) AS n_universe_secids,
            (SELECT COUNT(*) FROM bridged) AS n_secids_with_dividend_records
        """
        return self.query_arrow(sql)

    def compute_universe_daily_marks(
        self,
        secids: list[int],
        start_date: str,
        end_date: str,
        filters: OptionFilterConfig | None = None,
    ) -> pa.Table:
        """
        Daily ATM option marks + delta-hedged returns for a fixed secid universe.

        Per (secid, date): nearest |Δ|≈0.5 call with DTE∈[14,45], after the
        literature-standard chain screens in :class:`OptionFilterConfig`.

        Headline label ``dh_ret_lagdelta`` is the Bakshi–Kapadia / Cao–Han
        delta-hedged call return with the Molnár et al. (FMA Derivatives 2025)
        one-step lagged hedge ratio, which breaks the IMR bias between noisy
        contemporaneous delta and the subsequent underlying return::

            denom_lag = Δ_{t-1}·S_t − C_t
            gain_lag  = (C' − C) − Δ_{t-1}·(S' − S) − r·(days/365)·denom_lag
            dh_ret_lagdelta = gain_lag / denom_lag,   requires denom_lag > 0

        ``dh_ret`` retains the contemporaneous-Δ formula as a robustness column
        so the paper can report the estimated IMR bias magnitude. Lag and lead
        are taken on the **same optionid** window.

        ``fwd_ret`` (naked dollar mid change) is retained as a legacy column for
        robustness rows only; it is not a return on invested capital.
        """
        if not secids:
            raise ValueError("secids must be non-empty")
        cfg = filters or OptionFilterConfig()
        id_list = ", ".join(str(int(s)) for s in secids)
        base = self._marks_base_sql(id_list, start_date, end_date, cfg)
        screen_sql = " AND ".join(f"({p})" for _, p in cfg.screens()) or "TRUE"
        sel_sql = " AND ".join(f"({p})" for _, p in cfg.selection_screens()) or "TRUE"
        sql = f"""
        {base},
        quotes_ok AS (
            SELECT * FROM screened WHERE {screen_sql}
        ),
        cont AS (
            SELECT
                secid,
                optionid,
                date,
                LAG(delta) OVER w AS delta_lag,
                LEAD(date) OVER w AS date_next,
                LEAD(mid) OVER w AS mid_next,
                LEAD(spot) OVER w AS spot_next,
                LEAD(cfadj) OVER w AS cfadj_next,
                LEAD(sec_cfadj) OVER w AS sec_cfadj_next
            FROM quotes_ok
            WINDOW w AS (PARTITION BY secid, optionid ORDER BY date)
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY secid, date
                    ORDER BY
                        CASE WHEN cp_flag = 'C' AND delta BETWEEN 0.35 AND 0.65 THEN 0 ELSE 1 END,
                        ABS(ABS(COALESCE(delta, 0.0)) - 0.5),
                        open_interest DESC NULLS LAST
                ) AS rn
            FROM quotes_ok
            WHERE delta IS NOT NULL
              AND dte BETWEEN {int(cfg.dte_lo)} AND {int(cfg.dte_hi)}
              AND {sel_sql}
        ),
        daily AS (
            SELECT
                secid,
                date,
                optionid,
                mid,
                delta,
                strike,
                spot,
                sec_cfadj,
                cfadj,
                impl_volatility AS atm_iv,
                half_spread AS bid_ask_spread,
                volume,
                open_interest
            FROM ranked
            WHERE rn = 1
        ),
        skew AS (
            SELECT
                secid,
                date,
                AVG(impl_volatility) FILTER (
                    WHERE cp_flag = 'P' AND delta BETWEEN -0.30 AND -0.20
                )
                - AVG(impl_volatility) FILTER (
                    WHERE cp_flag = 'C' AND delta BETWEEN 0.20 AND 0.30
                ) AS skew_25d
            -- quotes_ok carries the quality screens but not calls_only, so the
            -- 25-delta skew still has both wings available.
            FROM quotes_ok
            WHERE dte BETWEEN {int(cfg.dte_lo)} AND {int(cfg.dte_hi)}
            GROUP BY secid, date
        ),
        flow AS (
            SELECT
                secid,
                date,
                (SUM(volume) FILTER (WHERE cp_flag = 'C')
                    - SUM(volume) FILTER (WHERE cp_flag = 'P'))
                  / NULLIF(SUM(volume), 0) AS volume_imbalance,
                SUM(open_interest) FILTER (WHERE cp_flag = 'P')
                  / NULLIF(SUM(open_interest) FILTER (WHERE cp_flag = 'C'), 0)
                    AS put_call_oi_ratio
            FROM quotes_ok
            WHERE dte BETWEEN {int(cfg.dte_lo)} AND {int(cfg.dte_hi)}
            GROUP BY secid, date
        ),
        joined AS (
            SELECT
                d.*,
                c.delta_lag,
                c.date_next,
                c.mid_next,
                c.spot_next,
                c.cfadj_next,
                c.sec_cfadj_next,
                s.skew_25d,
                f.volume_imbalance,
                f.put_call_oi_ratio
            FROM daily d
            LEFT JOIN cont c USING (secid, optionid, date)
            LEFT JOIN skew s USING (secid, date)
            LEFT JOIN flow f USING (secid, date)
        ),
        withrf AS (
            SELECT j.*, r.rf_pct
            FROM joined j
            ASOF LEFT JOIN rates r ON j.date >= r.date
        ),
        labelled AS (
            SELECT
                *,
                (delta * spot - mid) AS dh_denom,
                (delta_lag * spot - mid) AS dh_denom_lagdelta,
                date_diff('day', date, date_next) AS hold_days,
                (
                    date_next IS NOT NULL
                    AND date_diff('day', date, date_next) BETWEEN 1 AND 5
                    AND spot IS NOT NULL AND spot_next IS NOT NULL
                    AND mid_next IS NOT NULL
                    AND cfadj IS NOT DISTINCT FROM cfadj_next
                    AND sec_cfadj IS NOT DISTINCT FROM sec_cfadj_next
                    AND (delta * spot - mid) > 0
                ) AS label_ok,
                (
                    date_next IS NOT NULL
                    AND date_diff('day', date, date_next) BETWEEN 1 AND 5
                    AND spot IS NOT NULL AND spot_next IS NOT NULL
                    AND mid_next IS NOT NULL
                    AND cfadj IS NOT DISTINCT FROM cfadj_next
                    AND sec_cfadj IS NOT DISTINCT FROM sec_cfadj_next
                    AND delta_lag IS NOT NULL
                    AND (delta_lag * spot - mid) > 0
                ) AS label_ok_lag
            FROM withrf
        )
        SELECT
            secid,
            date,
            mid,
            delta,
            atm_iv,
            bid_ask_spread,
            skew_25d,
            spot,
            strike,
            volume_imbalance,
            put_call_oi_ratio,
            dh_denom,
            dh_denom_lagdelta,
            CASE
                WHEN label_ok THEN (
                    (mid_next - mid)
                    - delta * (spot_next - spot)
                    - (COALESCE(rf_pct, 0.0) / 100.0) * (hold_days / 365.0)
                      * (delta * spot - mid)
                ) / (delta * spot - mid)
                ELSE NULL
            END AS dh_ret,
            CASE
                WHEN label_ok_lag THEN (
                    (mid_next - mid)
                    - delta_lag * (spot_next - spot)
                    - (COALESCE(rf_pct, 0.0) / 100.0) * (hold_days / 365.0)
                      * (delta_lag * spot - mid)
                ) / (delta_lag * spot - mid)
                ELSE NULL
            END AS dh_ret_lagdelta,
            CASE WHEN label_ok THEN mid_next - mid ELSE NULL END AS fwd_ret,
            -- Equity simple return on the SAME (date, date_next) pairs and
            -- label_ok_lag gate as dh_ret_lagdelta (spectrum arm alignment).
            CASE
                WHEN label_ok_lag AND spot > 0 AND spot_next IS NOT NULL THEN
                    (spot_next - spot) / spot
                ELSE NULL
            END AS stk_ret,
            CASE
                WHEN label_ok_lag THEN date_diff('day', date, date_next)
                ELSE NULL
            END AS stk_ret_h_days
        FROM labelled
        WHERE date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY date, secid
        """
        return self.query_arrow(sql)

    def compute_surface_arb_violations(
        self,
        secids: list[int],
        start_date: str,
        end_date: str,
        *,
        calendar: bool = True,
        butterfly: bool = True,
    ) -> dict[str, Any]:
        """
        Calendar / butterfly violation keys for a universe × window.

        Scoped to ``secids`` only (not the full SPX lake). Returns counts and a
        set of ``(secid, date_str)`` keys to anti-join from daily marks.
        """
        from mascotrl.data.arbitrage_screens import (
            butterfly_violations,
            calendar_violations,
            merge_violation_keys,
        )

        if not secids:
            raise ValueError("secids must be non-empty")
        id_list = ", ".join(str(int(s)) for s in secids)
        cal_df = pd.DataFrame()
        bf_df = pd.DataFrame()
        if calendar:
            glob = (self.lake_base_dir / "vol_surface" / "*" / "*" / "*.parquet").as_posix()
            sql = f"""
            SELECT
                TRY_CAST(secid AS BIGINT) AS secid,
                CAST(date AS DATE) AS date,
                TRY_CAST(days AS BIGINT) AS days,
                TRY_CAST(delta AS BIGINT) AS delta,
                TRY_CAST(impl_volatility AS DOUBLE) AS impl_volatility,
                CAST(cp_flag AS VARCHAR) AS cp_flag
            FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
            WHERE TRY_CAST(secid AS BIGINT) IN ({id_list})
              AND CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
              AND impl_volatility IS NOT NULL
              AND days IS NOT NULL
            """
            try:
                cal_df = self.query_arrow(sql).to_pandas()
            except Exception as exc:
                log.warning("calendar arb surface read failed: %s", exc)
                cal_df = pd.DataFrame()
            if not cal_df.empty:
                cal_df = calendar_violations(cal_df)
        if butterfly:
            glob = self._options_glob()
            sql = f"""
            SELECT
                TRY_CAST(secid AS BIGINT) AS secid,
                CAST(date AS DATE) AS date,
                CAST(exdate AS DATE) AS exdate,
                CAST(cp_flag AS VARCHAR) AS cp_flag,
                TRY_CAST(strike_price AS DOUBLE) / 1000.0 AS strike,
                (TRY_CAST(best_bid AS DOUBLE) + TRY_CAST(best_offer AS DOUBLE)) / 2.0 AS mid
            FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
            WHERE TRY_CAST(secid AS BIGINT) IN ({id_list})
              AND CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
              AND best_bid IS NOT NULL AND best_offer IS NOT NULL
              AND best_offer >= best_bid
              AND strike_price IS NOT NULL
            """
            try:
                bf_raw = self.query_arrow(sql).to_pandas()
            except Exception as exc:
                log.warning("butterfly arb chain read failed: %s", exc)
                bf_raw = pd.DataFrame()
            if not bf_raw.empty:
                bf_df = butterfly_violations(bf_raw)
        keys = merge_violation_keys(cal_df, bf_df)
        return {
            "calendar_violations": cal_df,
            "butterfly_violations": bf_df,
            "bad_keys": keys,
            "n_calendar_fail_days": int(len(cal_df)) if cal_df is not None else 0,
            "n_butterfly_fail_days": int(len(bf_df)) if bf_df is not None else 0,
            "n_bad_secid_dates": int(len(keys)),
        }

    def compute_macro_state(self, start_date: str, end_date: str) -> pa.Table:
        macro = self.lake_base_dir / "macro"
        vix = macro / "cboe_vix.parquet"
        rates = macro / "interest_rate.parquet"
        # Prefer lake parquet; fall back to FRB CSV so SOFR join works offline.
        rates_src = rates.as_posix()
        rates_reader = f"read_parquet('{rates_src}', union_by_name=true)"
        if not rates.exists():
            try:
                from mascotrl.data.paths import TIER_B

                csv_path = TIER_B.get("interest_rate")
                if csv_path is not None and Path(csv_path).exists():
                    rates_src = Path(csv_path).as_posix()
                    rates_reader = (
                        f"read_csv_auto('{rates_src}', header=true, sample_size=-1)"
                    )
            except Exception:
                pass
        # QUALIFY/ROW_NUMBER: lake VIX has known duplicate calendar rows
        # (2003-09-22, 2003-10-30). Propagating them poisons Arctic reindex/ffill.
        # Rates join emits SOFR / EFFR / DTB3 for funding plugins + macro features.
        sql = f"""
        WITH vix_raw AS (
            SELECT CAST("Date" AS DATE) AS date,
                   TRY_CAST(vix AS DOUBLE) AS vix,
                   TRY_CAST(vxn AS DOUBLE) AS vxn,
                   TRY_CAST(vxd AS DOUBLE) AS vxd
            FROM read_parquet('{vix.as_posix()}', union_by_name=true)
        ),
        vix AS (
            SELECT date, vix, vxn, vxd
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY date
                           ORDER BY vix NULLS LAST, vxn NULLS LAST, vxd NULLS LAST
                       ) AS _rn
                FROM vix_raw
            )
            WHERE _rn = 1
        ),
        rates_raw AS (
            SELECT CAST(COALESCE(TRY_CAST(date AS DATE), TRY_CAST("Date" AS DATE)) AS DATE) AS date,
                   TRY_CAST(sofr AS DOUBLE) AS sofr,
                   TRY_CAST(effr AS DOUBLE) AS effr,
                   TRY_CAST(dtb3 AS DOUBLE) AS dtb3
            FROM {rates_reader}
        ),
        rates AS (
            SELECT date, sofr, effr, dtb3
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY date
                           ORDER BY sofr NULLS LAST, effr NULLS LAST
                       ) AS _rn
                FROM rates_raw
            )
            WHERE _rn = 1
        )
        SELECT v.date, v.vix, v.vxn, v.vxd, r.sofr, r.effr, r.dtb3
        FROM vix v
        LEFT JOIN rates r ON v.date = r.date
        WHERE v.date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY v.date
        """
        try:
            return self.query_arrow(sql)
        except Exception:
            # Last-resort: raw VIX window (still de-duplicated by date)
            return self.query_arrow(
                f"""
                SELECT date, vix FROM (
                    SELECT CAST("Date" AS DATE) AS date,
                           TRY_CAST(vix AS DOUBLE) AS vix,
                           ROW_NUMBER() OVER (
                               PARTITION BY CAST("Date" AS DATE)
                               ORDER BY TRY_CAST(vix AS DOUBLE) NULLS LAST
                           ) AS _rn
                    FROM read_parquet('{vix.as_posix()}', union_by_name=true)
                    WHERE CAST("Date" AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                )
                WHERE _rn = 1
                ORDER BY 1
                """
            )
