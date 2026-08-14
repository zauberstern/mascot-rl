"""Equity allocation book sections 0-3 (provenance through holdings)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    section_divider,
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

def _section0_provenance(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    run_meta: Mapping[str, Any],
    results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    fig = section_divider(
        "Equity Allocation Reporting Book",
        subtitle=f"{manifest.get('date_start') or '?'} .. {manifest.get('date_end') or '?'}",
    )
    paths = _finish(fig, stem=out_dir / "S0_title", manifest=manifest, book=book)
    entries.append(_entry("S0.1", "Title page", status="written", paths=paths))

    run_rows = [
        {"field": k, "value": str(v)} for k, v in run_meta.items() if v is not None
    ]
    df = pd.DataFrame(run_rows) if run_rows else pd.DataFrame(columns=["field", "value"])
    fig = table_figure(df, "Run manifest", width="single")
    paths = _finish(fig, stem=out_dir / "S0_run_manifest", manifest=manifest, book=book)
    _write_table(df, out_dir, "S0_run_manifest")
    entries.append(
        _entry(
            "S0.2",
            "Run manifest",
            status="written" if run_rows else "skipped",
            paths=paths,
            note=None if run_rows else "no run metadata supplied",
        )
    )

    conf = results.get("confirmatory") or {}
    all_hashes = set()
    all_hashes.update((conf.get("benchmark_estimand_hashes") or {}).values())
    all_hashes.update((conf.get("olps_estimand_hashes") or {}).values())
    all_hashes.update((conf.get("ceiling_estimand_hashes") or {}).values())
    stats_hash = (conf.get("stats_table") or {}).get("estimand_hash")
    if stats_hash:
        all_hashes.add(stats_hash)
    uniform = len(all_hashes) <= 1 if all_hashes else None
    neg = conf.get("negative_controls_prelim") or {}
    integrity_rows = [
        {"check": "estimand_hash_uniform", "value": str(uniform)},
        {"check": "n_distinct_estimand_hashes", "value": str(len(all_hashes))},
        {"check": "yaml_honesty_ok", "value": str("yaml_honesty_error" not in results)},
        {
            "check": "shuffled_panel_ew_sharpe",
            "value": str(neg.get("shuffled_panel_ew_sharpe")),
        },
    ]
    df = pd.DataFrame(integrity_rows)
    fig = table_figure(df, "Integrity checks", width="single")
    paths = _finish(fig, stem=out_dir / "S0_integrity", manifest=manifest, book=book)
    _write_table(df, out_dir, "S0_integrity")
    entries.append(_entry("S0.3", "Integrity checks", status="written", paths=paths))

    return entries


# --------------------------------------------------------------------------
# Section 1: headline
# --------------------------------------------------------------------------


def _section1_headline(
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
    bench_tn = dict(conf.get("benchmark_sharpes") or {})
    bench_res = dict(conf.get("benchmark_sharpes_residual") or {})
    pol_sharpe = _finite((conf.get("path_summary") or {}).get("sharpe_mean"))

    names = list(bench_tn.keys())
    if names or np.isfinite(pol_sharpe):
        fig, ax = _new_fig(width="double", height_in=4.5, legend_space=True)
        labels = (["policy"] if np.isfinite(pol_sharpe) else []) + names
        tn_vals = ([pol_sharpe] if np.isfinite(pol_sharpe) else []) + [
            bench_tn[n] for n in names
        ]
        res_vals = ([float("nan")] if np.isfinite(pol_sharpe) else []) + [
            bench_res.get(n, float("nan")) for n in names
        ]
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, tn_vals, width=w, label="total_net", color=C_NAVY)
        ax.bar(x + w / 2, res_vals, width=w, label="residual", color=C_STEEL)
        ax.axhline(0.0, color=C_ZERO, lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Annualized Sharpe (dimensionless)")
        ax.set_title("Dual scorecard: total_net vs residual Sharpe")
        stamp_n(ax, len(labels))
        place_legend(ax, loc="outside right")
        paths = _finish(
            fig,
            stem=out_dir / "S1_dual_scorecard",
            manifest=manifest,
            book=book,
            legend_space=True,
            caption_text="Policy residual Sharpe is not tracked per-date (see D2 scope note).",
        )
        entries.append(_entry("S1.1", "Dual scorecard bars", status="written", paths=paths))
    else:
        entries.append(
            _entry("S1.1", "Dual scorecard bars", status="skipped", note="no benchmark/policy sharpes")
        )

    pol_name, pol_frame = _focus_frame(strategy_frames, focus)
    ew_frame = strategy_frames.get("equal_weight")
    if pol_frame is not None and "total_net" in pol_frame.columns and pol_frame["total_net"].notna().any():
        fig, ax = _new_fig(width="double", height_in=4.2, legend_space=True)
        pol_r = pol_frame["total_net"].fillna(0.0).to_numpy()
        ax.plot(pol_frame["date"], np.cumsum(pol_r), label=pol_name, color=family_color(pol_name), lw=1.5)
        if ew_frame is not None and not ew_frame.empty and "total_net" in ew_frame.columns:
            ew_r = ew_frame["total_net"].fillna(0.0).to_numpy()
            ax.plot(
                ew_frame["date"], np.cumsum(ew_r), label="equal_weight",
                color=family_color("equal_weight"), lw=1.2, ls="--",
            )
        if bench_tn:
            best_bench = max(bench_tn, key=lambda n: bench_tn[n] if np.isfinite(bench_tn[n]) else -np.inf)
            bf = strategy_frames.get(best_bench)
            if bf is not None and not bf.empty and "total_net" in bf.columns:
                bf_r = bf["total_net"].fillna(0.0).to_numpy()
                ax.plot(
                    bf["date"], np.cumsum(bf_r), label=f"best bench: {best_bench}",
                    color=family_color(best_bench), lw=1.2, ls=":",
                )
        ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative net return (sum of daily total_net)")
        ax.set_title("Cumulative NAV: policy vs equal-weight vs best benchmark")
        stamp_n(ax, len(pol_frame))
        place_legend(ax, loc="outside right")
        paths = _finish(
            fig, stem=out_dir / "S1_cumulative_nav", manifest=manifest, book=book, legend_space=True,
        )
        entries.append(_entry("S1.2", "Cumulative NAV", status="written", paths=paths))
    else:
        entries.append(
            _entry("S1.2", "Cumulative NAV", status="skipped", note="no policy total_net series")
        )

    path_pnls = dict(conf.get("path_pnls") or {})
    if path_pnls:
        fig, ax = _new_fig(width="double", height_in=4.2, legend_space=True)
        for i, (pid, series) in enumerate(sorted(path_pnls.items())):
            arr = np.nan_to_num(np.asarray(series, dtype=float))
            ax.plot(np.cumsum(arr), alpha=0.55, lw=1.0, label=f"path {pid}" if i < 8 else None)
        ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
        ax.set_xlabel("OOS step index")
        ax.set_ylabel("Cumulative path PnL (sum of total_net)")
        ax.set_title("CPCV path fan chart (policy)")
        place_legend(ax, loc="outside right")
        paths = _finish(
            fig, stem=out_dir / "S1_path_fan", manifest=manifest, book=book, legend_space=True,
        )
        entries.append(_entry("S1.3", "CPCV path fan chart", status="written", paths=paths))
    else:
        entries.append(_entry("S1.3", "CPCV path fan chart", status="skipped", note="no path_pnls"))

    path_sharpes = [
        _finite(s) for s in ((conf.get("path_summary") or {}).get("path_sharpes") or [])
    ]
    path_sharpes = [s for s in path_sharpes if np.isfinite(s)]
    if len(path_sharpes) >= 2:
        fig, ax = _new_fig(width="single", height_in=4.0)
        arr = np.asarray(path_sharpes)
        try:
            import seaborn as sns

            sns.violinplot(y=arr, ax=ax, color=family_color("policy"), inner="box", cut=0)
            ax.set_xticks([])
        except Exception:
            parts = ax.violinplot([arr], showmeans=False, showmedians=True, showextrema=False)
            for body in parts.get("bodies", []):
                body.set_facecolor(family_color("policy"))
                body.set_alpha(0.35)
            jitter = np.linspace(-0.06, 0.06, arr.size)
            ax.scatter(np.full(arr.size, 1) + jitter, arr, s=18, color=family_color("policy"), zorder=3)
            ax.set_xticks([1])
            ax.set_xticklabels(["policy paths"])
        ax.axhline(0.0, color=C_ZERO, lw=0.8, ls="--")
        ax.set_ylabel("Path Sharpe (dimensionless)")
        ax.set_title("CPCV path Sharpe distribution")
        stamp_n(ax, arr.size)
        paths = _finish(fig, stem=out_dir / "S1_path_sharpe_violin", manifest=manifest, book=book)
        entries.append(_entry("S1.4", "Path Sharpe violin", status="written", paths=paths))
    else:
        entries.append(
            _entry("S1.4", "Path Sharpe violin", status="skipped", note="fewer than 2 finite path sharpes")
        )

    return entries


# --------------------------------------------------------------------------
# Section 2: risk
# --------------------------------------------------------------------------


def _section2_risk(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    strategy_frames: Mapping[str, pd.DataFrame],
    focus: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    name, df = _focus_frame(strategy_frames, focus)
    if df is None or "total_net" not in df.columns or not df["total_net"].notna().any():
        note = "no focus strategy total_net series"
        return [
            _entry(f"S2.{i}", t, status="skipped", note=note)
            for i, t in enumerate(
                ["Underwater drawdown", "Rolling Sharpe/vol", "Return distribution",
                 "Monthly heatmap", "Regime-conditional Sharpe"], start=1,
            )
        ]

    dates = pd.to_datetime(df["date"])
    r = df["total_net"].fillna(0.0).to_numpy()
    cum = np.cumsum(r)

    fig, ax = _new_fig(width="double", height_in=3.8)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ax.fill_between(dates, dd, 0.0, color=C_NEG, alpha=0.75)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from running peak (cumulative return units)")
    ax.set_title(f"Underwater drawdown ({name})")
    stamp_n(ax, r.size)
    paths = _finish(fig, stem=out_dir / "S2_drawdown", manifest=manifest, book=book)
    entries.append(_entry("S2.1", "Underwater drawdown", status="written", paths=paths))

    fig, axes = _new_fig(width="double", height_in=5.6, nrows=2, ncols=1, sharex=True)
    for win, ax, lab in ((63, axes[0], "63d"), (252, axes[1], "252d")):
        if r.size > win + 5:
            roll_sh, roll_vol = [], []
            for i in range(win, r.size):
                w = r[i - win : i]
                mu, sd = float(np.mean(w)), float(np.std(w))
                roll_sh.append((mu / sd) * np.sqrt(252.0) if sd > 0 else np.nan)
                roll_vol.append(sd * np.sqrt(252.0))
            ax2 = ax.twinx()
            ax.plot(dates.iloc[win:], roll_sh, color=C_NAVY, lw=1.1, label=f"{lab} Sharpe")
            ax2.plot(dates.iloc[win:], roll_vol, color=C_STEEL, lw=1.0, ls="--", label=f"{lab} vol")
            ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
            ax.set_ylabel(f"{lab} Sharpe (dimensionless)")
            ax2.set_ylabel(f"{lab} ann. vol (fraction)")
        else:
            ax.text(0.5, 0.5, f"insufficient history for {lab} window", transform=ax.transAxes, ha="center")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Rolling Sharpe and volatility ({name})", fontsize=11)
    paths = _finish(fig, stem=out_dir / "S2_rolling_sharpe_vol", manifest=manifest, book=book)
    entries.append(_entry("S2.2", "Rolling Sharpe and volatility", status="written", paths=paths))

    fig, ax = _new_fig(width="single", height_in=4.0)
    ax.hist(r, bins=40, color=C_STEEL, alpha=0.85)
    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean()) if np.any(r <= var95) else float("nan")
    ax.axvline(var95, color=C_NEG, ls="--", lw=1.2, label=f"VaR95={var95:.4f}")
    ax.axvline(cvar95, color=C_ZERO, ls=":", lw=1.2, label=f"CVaR95={cvar95:.4f}")
    ax.set_xlabel("Daily total_net return (fraction of NAV)")
    ax.set_ylabel("Count (days)")
    ax.set_title(f"Return distribution ({name})")
    stamp_n(ax, r.size)
    place_legend(ax, loc="outside top")
    paths = _finish(fig, stem=out_dir / "S2_return_distribution", manifest=manifest, book=book)
    entries.append(_entry("S2.3", "Return distribution", status="written", paths=paths))

    s = pd.Series(r, index=dates)
    piv = s.groupby([s.index.year, s.index.month]).sum().unstack(fill_value=np.nan)
    if piv.shape[0] >= 1:
        fig, ax = _new_fig(width="double", height_in=float(min(max(2.5 + 0.3 * piv.shape[0], 3.0), 8.0)))
        vmax = float(np.nanmax(np.abs(piv.to_numpy()))) if np.isfinite(piv.to_numpy()).any() else 1e-6
        vmax = max(vmax, 1e-8)
        im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels([str(y) for y in piv.index])
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels([str(m) for m in piv.columns])
        ax.set_xlabel("Calendar month")
        ax.set_ylabel("Calendar year")
        ax.set_title(f"Year-by-month return heatmap ({name})")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Monthly return sum (fraction)")
        paths = _finish(fig, stem=out_dir / "S2_month_heatmap", manifest=manifest, book=book)
        entries.append(_entry("S2.4", "Year-by-month return heatmap", status="written", paths=paths))
    else:
        entries.append(_entry("S2.4", "Year-by-month return heatmap", status="skipped", note="no monthly data"))

    win = 21
    if r.size > 2 * win:
        trail_vol = pd.Series(r).rolling(win).std().to_numpy()
        med = np.nanmedian(trail_vol)
        hi_mask = trail_vol >= med
        lo_mask = ~hi_mask & np.isfinite(trail_vol)
        hi_mask = hi_mask & np.isfinite(trail_vol)
        sh_hi = _annualized_sharpe(r[hi_mask]) if hi_mask.sum() > 5 else float("nan")
        sh_lo = _annualized_sharpe(r[lo_mask]) if lo_mask.sum() > 5 else float("nan")
        fig, ax = _new_fig(width="single", height_in=3.6)
        ax.bar(["high trailing vol", "low trailing vol"], [sh_hi, sh_lo], color=[C_NEG, C_POS])
        ax.axhline(0.0, color=C_ZERO, lw=0.8, ls="--")
        ax.set_ylabel("Annualized Sharpe (dimensionless)")
        ax.set_title(f"Regime-conditional Sharpe, split by trailing {win}d vol median ({name})")
        paths = _finish(fig, stem=out_dir / "S2_regime_sharpe", manifest=manifest, book=book)
        entries.append(_entry("S2.5", "Regime-conditional Sharpe", status="written", paths=paths))
    else:
        entries.append(
            _entry("S2.5", "Regime-conditional Sharpe", status="skipped", note="insufficient history")
        )

    return entries


# --------------------------------------------------------------------------
# Section 3: the book itself (holdings)
# --------------------------------------------------------------------------


def _section3_holdings(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    strategy_frames: Mapping[str, pd.DataFrame],
    focus: str,
) -> list[dict[str, Any]]:
    name, df = _focus_frame(strategy_frames, focus)
    w_cols = _weight_cols(df) if df is not None else []
    if df is None or not w_cols:
        note = "no focus strategy weight columns"
        titles = [
            "Weight time series (top-N + other)", "Weight heatmap", "Exposure timelines",
            "HHI and effective N", "Position count / largest share", "Mean abs weight by name",
            "Turnover time series", "Turnover decomposition", "Trade blotter",
        ]
        return [_entry(f"S3.{i}", t, status="skipped", note=note) for i, t in enumerate(titles, start=1)]

    entries: list[dict[str, Any]] = []
    dates = pd.to_datetime(df["date"])
    W = df[w_cols].to_numpy(dtype=np.float64)
    secids = [c[2:] for c in w_cols]
    n = W.shape[0]

    var_rank = np.argsort(-np.nanvar(W, axis=0))
    top_n = min(12, len(secids))
    top_idx = var_rank[:top_n]
    other_idx = var_rank[top_n:]

    fig, ax = _new_fig(width="double", height_in=4.6, legend_space=True)
    ax.stackplot(
        dates,
        *[W[:, j] for j in top_idx],
        labels=[secids[j] for j in top_idx],
    )
    if other_idx.size:
        ax.plot(dates, W[:, other_idx].sum(axis=1), color=C_GRAY, lw=1.0, ls="--", label="other (remainder)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (fraction of NAV)")
    ax.set_title(f"Weight time series: top {top_n} by variance + remainder ({name})")
    stamp_n(ax, n)
    place_legend(ax, loc="outside right")
    paths = _finish(fig, stem=out_dir / "S3_weight_ts", manifest=manifest, book=book, legend_space=True)
    entries.append(_entry("S3.1", "Weight time series", status="written", paths=paths))

    h_names = min(40, len(secids))
    h_idx = var_rank[:h_names]
    fig, ax = _new_fig(width="double", height_in=float(min(max(2.5 + 0.18 * h_names, 3.0), 8.5)))
    im = ax.imshow(W[:, h_idx].T, aspect="auto", cmap=CMAP_DIVERGING, vmin=-np.nanmax(np.abs(W)), vmax=np.nanmax(np.abs(W)))
    ax.set_yticks(np.arange(h_names))
    ax.set_yticklabels([secids[j] for j in h_idx], fontsize=6)
    step = max(1, n // 10)
    ax.set_xticks(np.arange(0, n, step))
    ax.set_xticklabels([str(pd.Timestamp(d).date()) for d in dates.iloc[::step]], rotation=60, fontsize=6)
    ax.set_title(f"Weight heatmap: date x name (top {h_names} by variance, {name})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Weight (fraction of NAV)")
    paths = _finish(fig, stem=out_dir / "S3_weight_heatmap", manifest=manifest, book=book)
    entries.append(_entry("S3.2", "Weight heatmap", status="written", paths=paths))

    gross = np.nansum(np.abs(W), axis=1)
    net = np.nansum(W, axis=1)
    long_exp = np.nansum(np.clip(W, 0, None), axis=1)
    short_exp = np.nansum(np.clip(W, None, 0), axis=1)
    fig, ax = _new_fig(width="double", height_in=4.0, legend_space=True)
    ax.plot(dates, gross, label="gross", color=C_NAVY)
    ax.plot(dates, net, label="net", color=C_STEEL)
    ax.plot(dates, long_exp, label="long", color=C_POS)
    ax.plot(dates, short_exp, label="short", color=C_NEG)
    ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure (fraction of NAV)")
    ax.set_title(f"Gross/net/long/short exposure ({name})")
    stamp_n(ax, n)
    place_legend(ax, loc="outside right")
    paths = _finish(fig, stem=out_dir / "S3_exposure", manifest=manifest, book=book, legend_space=True)
    entries.append(_entry("S3.3", "Exposure timelines", status="written", paths=paths))

    hhi = np.nansum(W * W, axis=1)
    neff = np.where(hhi > 0, 1.0 / hhi, np.nan)
    fig, axes = _new_fig(width="double", height_in=4.0, nrows=2, ncols=1, sharex=True)
    axes[0].plot(dates, hhi, color=C_NAVY)
    axes[0].set_ylabel("HHI (sum w^2, dimensionless)")
    axes[1].plot(dates, neff, color=C_STEEL)
    axes[1].set_ylabel("Effective N positions (1/HHI, count)")
    axes[1].set_xlabel("Date")
    fig.suptitle(f"Concentration: HHI and effective N ({name})", fontsize=11)
    paths = _finish(fig, stem=out_dir / "S3_hhi_neff", manifest=manifest, book=book)
    entries.append(_entry("S3.4", "HHI and effective N", status="written", paths=paths))

    pos_count = np.sum(np.abs(W) > 1e-6, axis=1)
    largest_share = np.nanmax(np.abs(W), axis=1)
    fig, axes = _new_fig(width="double", height_in=4.0, nrows=2, ncols=1, sharex=True)
    axes[0].plot(dates, pos_count, color=C_NAVY)
    axes[0].set_ylabel("Position count (names)")
    axes[1].plot(dates, largest_share, color=C_NEG)
    axes[1].set_ylabel("Largest position share (fraction of NAV)")
    axes[1].set_xlabel("Date")
    fig.suptitle(f"Position count and largest-position share ({name})", fontsize=11)
    paths = _finish(fig, stem=out_dir / "S3_position_count", manifest=manifest, book=book)
    entries.append(_entry("S3.5", "Position count / largest share", status="written", paths=paths))

    mean_abs = np.nanmean(np.abs(W), axis=0)
    top_mabs = np.argsort(-mean_abs)[: min(20, len(secids))]
    fig, ax = _new_fig(width="double", height_in=4.2)
    ax.bar([secids[j] for j in top_mabs], mean_abs[top_mabs], color=C_STEEL)
    ax.set_xticks(np.arange(len(top_mabs)))
    ax.set_xticklabels([secids[j] for j in top_mabs], rotation=75, fontsize=7)
    ax.set_ylabel("Mean |weight| over sample (fraction of NAV)")
    ax.set_title(f"Mean absolute weight by name, top {len(top_mabs)} ({name})")
    paths = _finish(fig, stem=out_dir / "S3_mean_abs_weight", manifest=manifest, book=book)
    entries.append(_entry("S3.6", "Mean absolute weight by name", status="written", paths=paths))

    if "turnover" in df.columns:
        turn = df["turnover"].fillna(0.0).to_numpy()
        fig, ax = _new_fig(width="double", height_in=3.6)
        ax.plot(dates, turn, color=C_NAVY, lw=1.0)
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover (fraction of NAV traded)")
        ax.set_title(f"Turnover time series ({name})")
        stamp_n(ax, n)
        paths = _finish(fig, stem=out_dir / "S3_turnover_ts", manifest=manifest, book=book)
        entries.append(_entry("S3.7", "Turnover time series", status="written", paths=paths))

        # Rebalance turnover is the recorded harness turnover; drift turnover is
        # the day-to-day weight change on days the harness reports zero
        # turnover (i.e. organic price drift between rebalances, not a trade).
        rebal_mask = turn > 1e-9
        dW = np.zeros(n)
        if n > 1:
            dW[1:] = 0.5 * np.nansum(np.abs(np.diff(W, axis=0)), axis=1)
        drift = np.where(rebal_mask, 0.0, dW)
        fig, ax = _new_fig(width="double", height_in=3.8, legend_space=True)
        ax.plot(dates, turn, label="rebalance turnover", color=C_NAVY)
        ax.plot(dates, drift, label="drift-implied turnover", color=C_ACCENT, ls="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover (fraction of NAV)")
        ax.set_title(f"Turnover decomposition: rebalance vs drift ({name})")
        place_legend(ax, loc="outside right")
        paths = _finish(
            fig, stem=out_dir / "S3_turnover_decomp", manifest=manifest, book=book, legend_space=True,
            caption_text="Drift-implied turnover approximates organic weight change on non-rebalance days from the weight series itself.",
        )
        entries.append(_entry("S3.8", "Turnover decomposition", status="written", paths=paths))
    else:
        entries.append(_entry("S3.7", "Turnover time series", status="skipped", note="no turnover column"))
        entries.append(_entry("S3.8", "Turnover decomposition", status="skipped", note="no turnover column"))

    if n > 1:
        dW_full = np.diff(W, axis=0)
        flat = np.abs(dW_full).ravel()
        order = np.argsort(-flat)[:20]
        rows = []
        for k in order:
            ti, j = divmod(int(k), W.shape[1])
            rows.append(
                {
                    "date": str(pd.Timestamp(dates.iloc[ti + 1]).date()),
                    "secid": secids[j],
                    "delta_weight": float(dW_full[ti, j]),
                }
            )
        blotter = pd.DataFrame(rows)
        fig = table_figure(blotter, f"Trade blotter: top 20 |delta weight| ({name})", width="single", max_rows=20)
        paths = _finish(fig, stem=out_dir / "S3_trade_blotter", manifest=manifest, book=book)
        _write_table(blotter, out_dir, "S3_trade_blotter")
        entries.append(_entry("S3.9", "Trade blotter", status="written", paths=paths))
    else:
        entries.append(_entry("S3.9", "Trade blotter", status="skipped", note="insufficient history"))

    return entries


# --------------------------------------------------------------------------
# Section 4: cost and capacity
# --------------------------------------------------------------------------


