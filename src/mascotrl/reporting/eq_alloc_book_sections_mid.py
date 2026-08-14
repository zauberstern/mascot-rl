"""Equity allocation book sections 4-5 (cost/capacity and attribution)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.reporting.book_style import (
    C_ACCENT,
    C_GRAY,
    C_NAVY,
    C_NEG,
    C_POS,
    C_STEEL,
    C_ZERO,
    CMAP_DIVERGING,
    HEIGHT_DEFAULT,
    HEIGHT_SINGLE,
    WIDTH_DOUBLE_IN,
    WIDTH_SINGLE_IN,
    PdfBook,
    caption,
    family_color,
    finalize_figure,
    place_legend,
    stamp_footer,
    stamp_n,
    table_figure,
)
from mascotrl.reporting.book_style import save_pdf_png, use_agg
from mascotrl.reporting.eq_alloc_book_primitives import (
    _annualized_sharpe,
    _entry,
    _finish,
    _finite,
    _focus_frame,
    _new_fig,
    _weight_cols,
    _write_table,
)
from mascotrl.reporting.eq_alloc_book_stats import _hac_ols_full

def _section4_cost_capacity(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
    strategy_frames: Mapping[str, pd.DataFrame],
    focus: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    conf = results.get("confirmatory") or {}
    fill_ladder = dict(conf.get("fill_ladder") or {})
    if fill_ladder:
        fig, ax = _new_fig(width="single", height_in=3.8)
        labs = list(fill_ladder.keys())
        vals = [_finite(fill_ladder[k]) for k in labs]
        ax.bar(labs, vals, color=C_ACCENT)
        ax.set_ylabel("Path Sharpe at fill rung (dimensionless)")
        ax.set_title("Cost / fill ladder (policy)")
        paths = _finish(fig, stem=out_dir / "S4_cost_ladder", manifest=manifest, book=book)
        entries.append(_entry("S4.1", "Cost ladder", status="written", paths=paths))
    else:
        entries.append(_entry("S4.1", "Cost ladder", status="skipped", note="no fill_ladder in results"))

    name, df = _focus_frame(strategy_frames, focus)
    have_cols = df is not None and {"gross", "total_net", "residual"}.issubset(df.columns)
    if have_cols and df["total_net"].notna().any():
        dates = pd.to_datetime(df["date"])
        fig, ax = _new_fig(width="double", height_in=4.0, legend_space=True)
        ax.plot(dates, np.cumsum(df["gross"].fillna(0.0)), label="gross", color=C_NAVY)
        ax.plot(dates, np.cumsum(df["total_net"].fillna(0.0)), label="net (total_net)", color=C_STEEL)
        if df["residual"].notna().any():
            ax.plot(dates, np.cumsum(df["residual"].fillna(0.0)), label="residual", color=C_POS, ls="--")
        ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (fraction of NAV)")
        ax.set_title(f"Cumulative gross vs net vs residual ({name})")
        place_legend(ax, loc="outside right")
        paths = _finish(fig, stem=out_dir / "S4_cum_gross_net_residual", manifest=manifest, book=book, legend_space=True)
        entries.append(_entry("S4.2", "Cumulative gross/net/residual", status="written", paths=paths))

        cost = df.get("cost")
        if cost is not None and cost.notna().any():
            gross_abs = df["gross"].abs().replace(0.0, np.nan)
            share = (cost / gross_abs).rolling(21, min_periods=5).mean()
            fig, ax = _new_fig(width="double", height_in=3.6)
            ax.plot(dates, share, color=C_NEG, lw=1.0)
            ax.set_xlabel("Date")
            ax.set_ylabel("21d rolling mean cost / |gross| (fraction)")
            ax.set_title(f"Cost as a share of gross ({name})")
            paths = _finish(fig, stem=out_dir / "S4_cost_share", manifest=manifest, book=book)
            entries.append(_entry("S4.3", "Cost as share of gross", status="written", paths=paths))
        else:
            entries.append(_entry("S4.3", "Cost as share of gross", status="skipped", note="no cost column"))
    else:
        entries.append(_entry("S4.2", "Cumulative gross/net/residual", status="skipped", note="no cost columns"))
        entries.append(_entry("S4.3", "Cost as share of gross", status="skipped", note="no cost columns"))

    gate1 = (conf.get("gates") or {}).get("gate1") or {}
    be = gate1.get("break_even_spread_multiplier")
    if be is not None and np.isfinite(_finite(be)):
        fig, ax = _new_fig(width="single", height_in=3.4)
        val = _finite(be)
        color = C_POS if val >= 0.25 else C_NEG
        ax.bar(["eq"], [val], color=color)
        ax.axhline(0.25, color=C_ZERO, ls="--", lw=0.9, label="min 0.25")
        ax.set_ylabel("Break-even spread multiplier (dimensionless)")
        ax.set_title("Gate1: break-even spread multiplier")
        place_legend(ax, loc="outside top")
        paths = _finish(fig, stem=out_dir / "S4_gate1_break_even", manifest=manifest, book=book)
        entries.append(_entry("S4.4", "Gate1 break-even multiplier", status="written", paths=paths))
    else:
        entries.append(_entry("S4.4", "Gate1 break-even multiplier", status="skipped", note="gate1 undefined"))

    if df is not None and {"total_net", "turnover"}.issubset(df.columns) and df["total_net"].notna().any():
        from mascotrl.reporting.capital_gates import capacity_curve_from_daily

        cap = capacity_curve_from_daily(
            df["total_net"].fillna(0.0).to_numpy(), df["turnover"].fillna(0.0).to_numpy()
        )
        rows = cap.get("rows") or []
        if rows:
            fig, ax = _new_fig(width="single", height_in=3.8)
            mult = [r["aum_multiplier"] for r in rows]
            sh = [r["sharpe"] for r in rows]
            ax.plot(mult, sh, marker="o", color=C_NAVY)
            ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("AUM multiplier (dimensionless, log scale)")
            ax.set_ylabel("Annualized Sharpe at scaled AUM (dimensionless)")
            ax.set_title(
                f"Capacity curve ({name}); ceiling ~{cap.get('capacity_ceiling_multiplier'):.2f}x"
            )
            paths = _finish(
                fig, stem=out_dir / "S4_capacity_curve", manifest=manifest, book=book,
                caption_text="Post-hoc sqrt-impact + linear-spread capacity model; not an ADV participation curve (macro/crsp_om_adv.parquet not wired here).",
            )
            entries.append(_entry("S4.5", "Capacity curve", status="written", paths=paths, note="sqrt-impact proxy, not ADV-participation"))
        else:
            entries.append(_entry("S4.5", "Capacity curve", status="skipped", note="capacity_curve_from_daily produced no rows"))
    else:
        entries.append(_entry("S4.5", "Capacity curve", status="skipped", note="no total_net/turnover columns"))

    return entries


# --------------------------------------------------------------------------
# Section 5: attribution
# --------------------------------------------------------------------------


def _hac_ols_full(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Newey-West HAC OLS for every coefficient (intercept + factors).

    Same Bartlett-kernel sandwich methodology as
    ``src.eval.signal_gate.ff_alpha``, generalized to report every
    coefficient's t-stat (not only the intercept).
    """
    from mascotrl.eval.stats_inference import newey_west_lag

    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    yy, xx = yy[mask], xx[mask]
    n = int(yy.size)
    p = int(xx.shape[1]) + 1
    if n < p + 2:
        return {"n": n, "coef": None, "t_stat": None}
    design = np.column_stack([np.ones(n), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    l_bw = int(newey_west_lag(n))
    scores = design * resid[:, None]
    s_mat = scores.T @ scores
    for j in range(1, l_bw + 1):
        if j >= n:
            break
        w = 1.0 - j / (l_bw + 1.0)
        gamma_j = scores[j:].T @ scores[:-j]
        s_mat = s_mat + w * (gamma_j + gamma_j.T)
    cov = xtx_inv @ s_mat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 1e-12)
    return {"n": n, "coef": beta, "t_stat": t_stat, "lags": l_bw}


def _section5_attribution(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    strategy_frames: Mapping[str, pd.DataFrame],
    focus: str,
    ff4_factors: np.ndarray | None,
    industry_group: Mapping[str, str] | None,
    returns_panel: np.ndarray | None,
    secids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    name, df = _focus_frame(strategy_frames, focus)
    have_series = df is not None and "total_net" in df.columns and df["total_net"].notna().any()

    if have_series and ff4_factors is not None and len(ff4_factors) == len(df):
        r = df["total_net"].fillna(0.0).to_numpy()
        fac = np.asarray(ff4_factors, dtype=np.float64)
        fit = _hac_ols_full(r, fac)
        fac_names = ["alpha", "Mkt-RF", "SMB", "HML", "Mom"]
        if fit["coef"] is not None:
            fig, ax = _new_fig(width="single", height_in=4.0)
            coef, t = fit["coef"], fit["t_stat"]
            colors = [C_POS if abs(tt) >= 2.0 else C_GRAY for tt in t]
            ax.bar(fac_names, coef, color=colors)
            for i, tt in enumerate(t):
                ax.text(i, coef[i], f"t={tt:.2f}", ha="center", va="bottom" if coef[i] >= 0 else "top", fontsize=7)
            ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
            ax.set_ylabel("FF4 loading (alpha in return units; betas dimensionless)")
            ax.set_title(f"FF4 loadings with Newey-West t ({name}, n={fit['n']})")
            paths = _finish(fig, stem=out_dir / "S5_ff4_loadings", manifest=manifest, book=book)
            entries.append(_entry("S5.1", "FF4 loadings", status="written", paths=paths))

            win = 126
            if r.size > win + 5:
                betas = np.full((r.size, 4), np.nan)
                for i in range(win, r.size):
                    f = _hac_ols_full(r[i - win : i], fac[i - win : i])
                    if f["coef"] is not None:
                        betas[i] = f["coef"][1:5]
                fig, ax = _new_fig(width="double", height_in=4.0, legend_space=True)
                dates = pd.to_datetime(df["date"])
                for j, fn in enumerate(("Mkt-RF", "SMB", "HML", "Mom")):
                    ax.plot(dates, betas[:, j], label=fn, lw=1.0)
                ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
                ax.set_xlabel("Date")
                ax.set_ylabel(f"Rolling {win}d FF4 beta (dimensionless)")
                ax.set_title(f"Rolling betas ({name})")
                place_legend(ax, loc="outside right")
                paths = _finish(fig, stem=out_dir / "S5_rolling_betas", manifest=manifest, book=book, legend_space=True)
                entries.append(_entry("S5.2", "Rolling betas", status="written", paths=paths))
            else:
                entries.append(_entry("S5.2", "Rolling betas", status="skipped", note="insufficient history for rolling window"))

            mean_gross = float(np.mean(r))
            factor_contrib = coef[1:5] * np.mean(fac, axis=0)
            fig, ax = _new_fig(width="single", height_in=3.8)
            labels = ["alpha"] + list(fac_names[1:]) + ["mean total_net"]
            vals = [coef[0]] + list(factor_contrib) + [mean_gross]
            bottoms = np.concatenate([[0.0], np.cumsum(vals[:-2])])
            colors_w = [C_POS] + [C_STEEL] * 4 + [C_NAVY]
            ax.bar(labels[:-1], vals[:-1], bottom=list(bottoms), color=colors_w[:-1])
            ax.bar(labels[-1], mean_gross, color=colors_w[-1])
            ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
            ax.set_ylabel("Mean daily return contribution (fraction of NAV)")
            ax.set_title(f"Alpha decomposition waterfall ({name})")
            paths = _finish(fig, stem=out_dir / "S5_alpha_waterfall", manifest=manifest, book=book)
            entries.append(_entry("S5.3", "Alpha decomposition waterfall", status="written", paths=paths))
        else:
            for i in (1, 2, 3):
                entries.append(_entry(f"S5.{i}", ["FF4 loadings", "Rolling betas", "Alpha waterfall"][i - 1], status="skipped", note="insufficient observations for HAC OLS"))
    else:
        for i, t in enumerate(["FF4 loadings", "Rolling betas", "Alpha decomposition waterfall"], start=1):
            entries.append(_entry(f"S5.{i}", t, status="skipped", note="no policy return series and/or FF4 factors supplied"))

    if industry_group and _weight_cols(df or pd.DataFrame()):
        w_cols = _weight_cols(df)
        mean_w = df[w_cols].mean(axis=0)
        sec_w: dict[str, float] = {}
        for c in w_cols:
            sid = c[2:]
            sec = industry_group.get(sid, "unknown")
            sec_w[sec] = sec_w.get(sec, 0.0) + float(mean_w[c])
        fig, ax = _new_fig(width="double", height_in=4.0)
        labs = sorted(sec_w, key=lambda k: -abs(sec_w[k]))
        ax.bar(labs, [sec_w[k] for k in labs], color=C_STEEL)
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, rotation=75, fontsize=7)
        ax.set_ylabel("Mean weight by OptionMetrics industry_group (fraction of NAV)")
        ax.set_title(f"Sector exposure ({name})")
        paths = _finish(
            fig, stem=out_dir / "S5_sector_exposure", manifest=manifest, book=book,
            caption_text="GICS sector codes are absent from the lake; grouped by OptionMetrics industry_group (32 coarse codes) instead.",
        )
        entries.append(_entry("S5.4", "Sector exposure", status="written", paths=paths, note="uses OptionMetrics industry_group, not GICS"))
    else:
        entries.append(_entry("S5.4", "Sector exposure", status="skipped", note="GICS not in lake; no industry_group mapping supplied"))

    if returns_panel is not None and secids and df is not None and _weight_cols(df):
        w_cols = _weight_cols(df)
        W = df[w_cols].to_numpy(dtype=np.float64)
        rp = np.asarray(returns_panel, dtype=np.float64)
        n = min(W.shape[0], rp.shape[0])
        contrib = np.nanmean(W[:n] * rp[:n], axis=0)
        order = np.argsort(-contrib)
        top = order[:10]
        bot = order[-10:]
        idx = list(top) + list(bot)
        fig, ax = _new_fig(width="double", height_in=4.2)
        colors_c = [C_POS if contrib[j] >= 0 else C_NEG for j in idx]
        idx_labels = [secids[j] if j < len(secids) else str(j) for j in idx]
        ax.bar(idx_labels, contrib[idx], color=colors_c)
        ax.set_xticks(np.arange(len(idx_labels)))
        ax.set_xticklabels(idx_labels, rotation=75, fontsize=7)
        ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
        ax.set_ylabel("Mean daily contribution to return (fraction of NAV)")
        ax.set_title(f"Contribution to return: top/bottom 10 names ({name})")
        paths = _finish(fig, stem=out_dir / "S5_contribution", manifest=manifest, book=book)
        entries.append(_entry("S5.5", "Contribution to return", status="written", paths=paths))
    else:
        entries.append(_entry("S5.5", "Contribution to return", status="skipped", note="no raw per-asset returns panel supplied"))

    return entries


# --------------------------------------------------------------------------
# Section 6: signals
# --------------------------------------------------------------------------


