"""D3/D4: the equity allocation reporting book.

Renders the ten sections specified in Workstream D (provenance, headline,
risk, holdings, cost and capacity, attribution, signals, learning,
statistical rigor, spectrum, limitations) from the artifacts the eq
allocation campaign (``scripts/run_eq_alloc_campaign.py``) already produces:
the ``results`` dict it writes to ``cpcv_path_summary.json`` plus the
per-strategy parquet frames from ``src/reporting/strategy_persistence.py``.

Every figure degrades to a ``"skipped"`` manifest entry with a ``note``
rather than fabricating data when its inputs are absent (the same contract
``src/reporting/figures/core_suite.py`` already uses); the one hard failure
is missing weights entirely, since without at least one strategy's holdings
there is no book to render.
"""
from __future__ import annotations

import hashlib
import json
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
    build_manifest,
    caption,
    family_color,
    finalize_figure,
    place_legend,
    section_divider,
    stamp_footer,
    stamp_n,
    table_figure,
)
from mascotrl.reporting.book_style import save_pdf_png, use_agg

SECTION_TITLES: tuple[str, ...] = (
    "Provenance",
    "Headline",
    "Risk",
    "Holdings",
    "Cost and Capacity",
    "Attribution",
    "Signals",
    "Learning",
    "Statistical Rigor",
    "Spectrum",
    "Limitations",
)


# --------------------------------------------------------------------------
# Generic figure/table primitives
# --------------------------------------------------------------------------


def _new_fig(
    *,
    width: str = "double",
    height_in: float | None = None,
    legend_space: bool = False,
    nrows: int = 1,
    ncols: int = 1,
    **kw: Any,
):
    import matplotlib.pyplot as plt

    use_agg()
    w = WIDTH_DOUBLE_IN if width == "double" else WIDTH_SINGLE_IN
    if legend_space:
        w += 1.6
    h = height_in if height_in is not None else (
        HEIGHT_SINGLE if width == "single" else HEIGHT_DEFAULT
    )
    h = float(min(max(h, 2.5), 9.5))
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **kw)


def _finish(
    fig,
    *,
    stem: Path,
    manifest: Mapping[str, Any],
    book: Any | None,
    legend_space: bool = False,
    caption_text: str | None = None,
) -> dict[str, str]:
    """Stamp footer + optional caption, save PNG (book.pdf via PdfBook), close."""
    import matplotlib.pyplot as plt

    finalize_figure(fig, legend_space=legend_space)
    if caption_text:
        caption(fig, caption_text)
    stamp_footer(fig, manifest)
    paths = save_pdf_png(fig, stem, pdf=False)
    if book is not None:
        book.add(fig)
    plt.close(fig)
    return paths


def _entry(
    fig_id: str,
    title: str,
    *,
    status: str,
    paths: dict[str, str] | None = None,
    sources: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": fig_id,
        "title": title,
        "status": status,
        "sources": sources or [],
    }
    if paths:
        out.update(paths)
    if note:
        out["note"] = note
    return out


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _annualized_sharpe(r: np.ndarray, periods: int = 252) -> float:
    from mascotrl.eval.stats_rigor import annualized_sharpe

    return float(annualized_sharpe(np.asarray(r, dtype=np.float64)))


def _focus_frame(
    strategy_frames: Mapping[str, pd.DataFrame], focus: str
) -> tuple[str, pd.DataFrame] | tuple[None, None]:
    if focus in strategy_frames and not strategy_frames[focus].empty:
        return focus, strategy_frames[focus]
    for name, df in strategy_frames.items():
        if df is not None and not df.empty:
            return name, df
    return None, None


def _weight_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("w_")]


# --------------------------------------------------------------------------
# Section 0: provenance
# --------------------------------------------------------------------------


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


def _section6_signals(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
    signal_gate_result: Mapping[str, Any] | None,
    signal_panels: Mapping[str, np.ndarray] | None,
    iv_surface_grid: np.ndarray | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    if signal_gate_result and signal_gate_result.get("stats"):
        rows = [
            {"signal": k, **v} for k, v in signal_gate_result["stats"].items()
        ]
        df = pd.DataFrame(rows)
        fig = table_figure(df, "Signal gate scorecard (Fama-MacBeth t, IC)", width="double")
        paths = _finish(fig, stem=out_dir / "S6_gate_scorecard", manifest=manifest, book=book)
        _write_table(df, out_dir, "S6_gate_scorecard")
        entries.append(_entry("S6.1", "Signal gate scorecard", status="written", paths=paths))
    else:
        entries.append(_entry("S6.1", "Signal gate scorecard", status="skipped", note="no signal_gate_result supplied"))

    allowlist = list((results.get("surface_signals") or {}).get("allowlist") or [])
    df = pd.DataFrame({"signal": allowlist}) if allowlist else pd.DataFrame(columns=["signal"])
    fig = table_figure(df, "Signal allowlist" + ("" if allowlist else " (empty)"), width="single")
    paths = _finish(fig, stem=out_dir / "S6_allowlist", manifest=manifest, book=book)
    _write_table(df, out_dir, "S6_allowlist")
    entries.append(
        _entry(
            "S6.2", "Signal allowlist", status="written", paths=paths,
            note=None if allowlist else "allowlist is genuinely empty at this pool size (B2 fail-closed gate)",
        )
    )

    breadth = dict(results.get("breadth") or {})
    if breadth:
        labs = list(breadth.keys())
        vals = [_finite((breadth[k] or {}).get("n_eff_enb")) for k in labs]
        fig, ax = _new_fig(width="single", height_in=3.8)
        ax.bar(labs, vals, color=C_STEEL)
        ax.set_ylabel("Effective number of bets (N_eff_ENB, count)")
        ax.set_title("Effective breadth by universe")
        paths = _finish(fig, stem=out_dir / "S6_breadth", manifest=manifest, book=book)
        entries.append(_entry("S6.3", "Effective breadth", status="written", paths=paths))
    else:
        entries.append(_entry("S6.3", "Effective breadth", status="skipped", note="no breadth data in results"))

    if signal_panels:
        from mascotrl.eval.signal_gate import effective_breadth, _signal_corr_matrix  # type: ignore

        names = list(signal_panels.keys())
        corr = _signal_corr_matrix(signal_panels)
        fig, ax = _new_fig(width="single", height_in=float(min(max(2.5 + 0.3 * len(names), 3.0), 8.0)))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=CMAP_DIVERGING)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=75, fontsize=7)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Signal correlation heatmap")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Pearson correlation (dimensionless)")
        paths = _finish(fig, stem=out_dir / "S6_signal_corr", manifest=manifest, book=book)
        entries.append(_entry("S6.4", "Signal correlation heatmap", status="written", paths=paths))
    else:
        entries.append(_entry("S6.4", "Signal correlation heatmap", status="skipped", note="no raw signal panels supplied"))

    if iv_surface_grid is not None:
        grid = np.asarray(iv_surface_grid, dtype=np.float64)
        fig, ax = _new_fig(width="single", height_in=4.0)
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xlabel("Tenor bucket (index)")
        ax.set_ylabel("Delta bucket (index)")
        ax.set_title("Sampled IV surface (delta x tenor grid)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Implied volatility (fraction)")
        paths = _finish(fig, stem=out_dir / "S6_iv_surface", manifest=manifest, book=book)
        entries.append(_entry("S6.5", "IV surface heatmap", status="written", paths=paths))
    else:
        entries.append(_entry("S6.5", "IV surface heatmap", status="skipped", note="no iv_surface_grid supplied"))

    return entries


# --------------------------------------------------------------------------
# Section 7: learning
# --------------------------------------------------------------------------


def _section7_learning(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
    learning_curves: Mapping[str, Sequence[float]] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    curve_keys = ["reward", "entropy", "explained_variance", "kl", "clip_fraction", "grad_norm"]
    if learning_curves and any(k in learning_curves for k in curve_keys):
        present = [k for k in curve_keys if k in learning_curves]
        fig, axes = _new_fig(width="double", height_in=6.5, nrows=len(present), ncols=1, sharex=True)
        axes_list = np.atleast_1d(axes)
        for ax, k in zip(axes_list, present):
            ax.plot(learning_curves[k], color=C_NAVY, lw=1.0)
            ax.set_ylabel(k)
        axes_list[-1].set_xlabel("Optimizer update index")
        fig.suptitle("Per-fold learning curves", fontsize=11)
        paths = _finish(fig, stem=out_dir / "S7_learning_curves", manifest=manifest, book=book)
        entries.append(_entry("S7.1", "Learning curves", status="written", paths=paths))
    else:
        entries.append(_entry("S7.1", "Learning curves", status="skipped", note="no learning_curves supplied"))

    entries.append(
        _entry("S7.2", "Optimizer step ledger", status="skipped", note="optimizer step floor not persisted by the current campaign artifact")
    )

    per_seed = [_finite(s) for s in ((results.get("confirmatory") or {}).get("path_summary") or {}).get("per_seed") or []]
    per_seed = [s for s in per_seed if np.isfinite(s)]
    if per_seed:
        fig, ax = _new_fig(width="single", height_in=3.6)
        ax.bar([f"seed {i}" for i in range(len(per_seed))], per_seed, color=C_STEEL)
        ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
        ax.set_ylabel("Mean path Sharpe (dimensionless)")
        ax.set_title("Seed dispersion")
        paths = _finish(fig, stem=out_dir / "S7_seed_dispersion", manifest=manifest, book=book)
        entries.append(_entry("S7.3", "Seed dispersion", status="written", paths=paths))
    else:
        entries.append(_entry("S7.3", "Seed dispersion", status="skipped", note="no per-seed sharpes in results"))

    return entries


# --------------------------------------------------------------------------
# Section 8: statistical rigor
# --------------------------------------------------------------------------


def _section8_stat_rigor(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    stats_tbl = dict((results.get("confirmatory") or {}).get("stats_table") or {})

    dsr = dict(stats_tbl.get("deflated_sharpe") or {})
    if dsr:
        df = pd.DataFrame(
            [
                {"metric": "sharpe_ann", "value": dsr.get("sharpe_ann")},
                {"metric": "psr", "value": dsr.get("psr")},
                {"metric": "dsr", "value": dsr.get("dsr")},
                {"metric": "n_trials", "value": dsr.get("n_trials")},
                {"metric": "n_obs", "value": dsr.get("n_obs")},
                {"metric": "significant_05", "value": dsr.get("significant_05")},
            ]
        )
        fig = table_figure(df, "Deflated / probabilistic Sharpe ratio", width="single")
        paths = _finish(fig, stem=out_dir / "S8_dsr_psr", manifest=manifest, book=book)
        _write_table(df, out_dir, "S8_dsr_psr")
        entries.append(_entry("S8.1", "DSR and PSR", status="written", paths=paths))
    else:
        entries.append(_entry("S8.1", "DSR and PSR", status="skipped", note="no deflated_sharpe in stats_table"))

    spa = dict(stats_tbl.get("hansen_spa_vs_ew") or {})
    if spa.get("ok"):
        df = pd.DataFrame(
            [{"quantity": k, "value": v} for k, v in spa.items() if k not in ("mean_pnl_diff_vs_bench", "rivals")]
        )
        fig = table_figure(df, "Hansen SPA test vs equal weight", width="single")
        paths = _finish(fig, stem=out_dir / "S8_hansen_spa", manifest=manifest, book=book)
        _write_table(df, out_dir, "S8_hansen_spa")
        entries.append(_entry("S8.2", "Hansen SPA", status="written", paths=paths))
    else:
        entries.append(_entry("S8.2", "Hansen SPA", status="skipped", note=spa.get("reason", "no hansen_spa_vs_ew in stats_table")))

    rw = dict(stats_tbl.get("romano_wolf_vs_ew") or {})
    if rw.get("results"):
        df = pd.DataFrame(rw["results"]) if isinstance(rw["results"], list) else pd.DataFrame([rw["results"]])
        fig = table_figure(df, "Romano-Wolf stepdown vs equal weight", width="single")
        paths = _finish(fig, stem=out_dir / "S8_romano_wolf", manifest=manifest, book=book)
        _write_table(df, out_dir, "S8_romano_wolf")
        entries.append(_entry("S8.3", "Romano-Wolf", status="written", paths=paths))
    else:
        entries.append(_entry("S8.3", "Romano-Wolf", status="skipped", note=rw.get("reason", "no romano_wolf_vs_ew in stats_table")))

    pbo = dict(stats_tbl.get("cscv_pbo") or {})
    if pbo and np.isfinite(_finite(pbo.get("pbo"))):
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in pbo.items() if k not in ("citation", "interpretation")])
        fig = table_figure(df, "CSCV probability of backtest overfitting", width="single")
        paths = _finish(fig, stem=out_dir / "S8_cscv_pbo", manifest=manifest, book=book)
        _write_table(df, out_dir, "S8_cscv_pbo")
        entries.append(
            _entry(
                "S8.4", "CSCV PBO", status="written", paths=paths,
                note="raw per-combination logits are not persisted; table reports pbo/median_logit only, not a full histogram",
            )
        )
    else:
        entries.append(_entry("S8.4", "CSCV PBO", status="skipped", note="no cscv_pbo in stats_table"))

    neg = dict((results.get("confirmatory") or {}).get("negative_controls_prelim") or {})
    neg_full = dict((results.get("confirmatory") or {}).get("negative_controls") or {})
    if neg_full.get("checks"):
        rows = [{"control": k, **v} for k, v in neg_full["checks"].items()]
        df = pd.DataFrame(rows)
        note = None
    elif neg:
        df = pd.DataFrame([{"control": "shuffled_labels_ew_only", "sharpe": neg.get("shuffled_panel_ew_sharpe")}])
        note = "only the preliminary EW-on-shuffled-panel check ran; the full policy-level three-control battery (shuffled/permuted/date-shifted) is not wired into the campaign"
    else:
        df = pd.DataFrame(columns=["control"])
        note = "no negative control data in results"
    fig = table_figure(df, "Negative controls", width="single")
    paths = _finish(fig, stem=out_dir / "S8_negative_controls", manifest=manifest, book=book)
    _write_table(df, out_dir, "S8_negative_controls")
    entries.append(_entry("S8.5", "Negative controls", status="written" if not df.empty else "skipped", paths=paths, note=note))

    return entries


# --------------------------------------------------------------------------
# Section 9: spectrum
# --------------------------------------------------------------------------


def _section9_spectrum(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
    arms_root: Path | None,
    artifacts_flat: Path | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    if arms_root is not None:
        from mascotrl.reporting.figures.core_suite import plot_f01_sharpe_ladder, plot_f26_attrition_funnel

        f01 = plot_f01_sharpe_ladder(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat)
        entries.append({**f01, "id": "S9.1"})
        f26 = plot_f26_attrition_funnel(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat)
        entries.append({**f26, "id": "S9.3"})
    else:
        entries.append(_entry("S9.1", "Sharpe ladder per axis", status="skipped", note="no arms_root supplied"))
        entries.append(_entry("S9.3", "Attrition funnel", status="skipped", note="no arms_root supplied"))

    gates = dict((results.get("confirmatory") or {}).get("gates") or {})
    if gates:
        rows = []
        for gname in ("gate1", "gate2", "gate3"):
            g = gates.get(gname) or {}
            rows.append({"gate": gname, "pass": g.get("pass"), "decision": g.get("decision")})
        df = pd.DataFrame(rows)
        fig = table_figure(df, "Decision matrix: gate1/gate2/gate3", width="single")
        paths = _finish(fig, stem=out_dir / "S9_decision_matrix", manifest=manifest, book=book)
        _write_table(df, out_dir, "S9_decision_matrix")
        entries.append(_entry("S9.2", "Decision matrix", status="written", paths=paths))
    else:
        entries.append(_entry("S9.2", "Decision matrix", status="skipped", note="no gates in results"))

    return entries


# --------------------------------------------------------------------------
# Section 10: limitations
# --------------------------------------------------------------------------


def _section10_limitations(
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    book: Any,
    results: Mapping[str, Any],
    known_limitations: Sequence[str] | None,
) -> list[dict[str, Any]]:
    from mascotrl.reporting.capital_gates import KNOWN_UNMODELED_RISKS

    entries: list[dict[str, Any]] = []

    df = pd.DataFrame(KNOWN_UNMODELED_RISKS)
    fig = table_figure(df, "Known unmodeled risks", width="double", max_rows=len(KNOWN_UNMODELED_RISKS))
    paths = _finish(fig, stem=out_dir / "S10_unmodeled_risks", manifest=manifest, book=book)
    _write_table(df, out_dir, "S10_unmodeled_risks")
    entries.append(_entry("S10.1", "Known unmodeled risks", status="written", paths=paths))

    if known_limitations:
        df = pd.DataFrame({"limitation": list(known_limitations)})
        note = None
    else:
        df = pd.DataFrame(columns=["limitation"])
        note = "no known_limitations list supplied by caller"
    fig = table_figure(df, "Known limitations", width="single")
    paths = _finish(fig, stem=out_dir / "S10_known_limitations", manifest=manifest, book=book)
    _write_table(df, out_dir, "S10_known_limitations")
    entries.append(_entry("S10.2", "Known limitations", status="written" if known_limitations else "skipped", paths=paths, note=note))

    gates = dict((results.get("confirmatory") or {}).get("gates") or {})
    df = pd.DataFrame(
        [
            {"field": "gate1_pass", "value": (gates.get("gate1") or {}).get("pass")},
            {"field": "gate2_pass", "value": (gates.get("gate2") or {}).get("pass")},
            {"field": "gate3_pass", "value": (gates.get("gate3") or {}).get("pass")},
        ]
    )
    fig = table_figure(df, "Capital gate verdict", width="single")
    paths = _finish(fig, stem=out_dir / "S10_capital_verdict", manifest=manifest, book=book)
    _write_table(df, out_dir, "S10_capital_verdict")
    entries.append(_entry("S10.3", "Capital gate verdict", status="written", paths=paths))

    return entries


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def render_eq_alloc_book(
    *,
    strategy_frames: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    results: Mapping[str, Any] | None = None,
    focus_strategy: str = "policy",
    cfg: Mapping[str, Any] | None = None,
    scorecard: str = "total_net",
    signal_gate_result: Mapping[str, Any] | None = None,
    signal_panels: Mapping[str, np.ndarray] | None = None,
    iv_surface_grid: np.ndarray | None = None,
    ff4_factors: np.ndarray | None = None,
    industry_group: Mapping[str, str] | None = None,
    returns_panel: np.ndarray | None = None,
    returns_panel_secids: Sequence[str] | None = None,
    learning_curves: Mapping[str, Sequence[float]] | None = None,
    known_limitations: Sequence[str] | None = None,
    arms_root: str | Path | None = None,
    artifacts_flat: str | Path | None = None,
) -> dict[str, Any]:
    """Render the full ten-section book; fail closed if no weights exist."""
    non_empty = {k: v for k, v in strategy_frames.items() if v is not None and not v.empty}
    has_weights = any(_weight_cols(v) for v in non_empty.values())
    if not non_empty or not has_weights:
        raise ValueError(
            "render_eq_alloc_book: no strategy carries any weight columns "
            "(w_<secid>); refusing to render a book with no holdings evidence"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = dict(results or {})
    conf = results.get("confirmatory") or {}

    focus_name, focus_df = _focus_frame(non_empty, focus_strategy)
    date_start = date_end = None
    if focus_df is not None and "date" in focus_df.columns and len(focus_df):
        dts = pd.to_datetime(focus_df["date"])
        date_start, date_end = str(dts.min().date()), str(dts.max().date())

    manifest = build_manifest(
        cfg=cfg,
        estimand_hash=conf.get("estimand_hash"),
        scorecard=scorecard,
        date_start=date_start,
        date_end=date_end,
    )

    run_meta = {
        "focus_strategy": focus_name,
        "n_strategies": len(non_empty),
        "k": results.get("k"),
        "pool": results.get("pool"),
        "date_start": date_start,
        "date_end": date_end,
        "wall_total_s": results.get("wall_total_s"),
    }

    arms_root_p = Path(arms_root) if arms_root is not None else None
    artifacts_flat_p = Path(artifacts_flat) if artifacts_flat is not None else None

    entries: list[dict[str, Any]] = []
    with PdfBook(out_dir / "book.pdf") as book:
        for idx, title in enumerate(SECTION_TITLES):
            fig = section_divider(f"Section {idx}: {title}")
            _finish(fig, stem=out_dir / f"divider_S{idx}", manifest=manifest, book=book)

            if idx == 0:
                entries += _section0_provenance(out_dir=out_dir, manifest=manifest, book=book, run_meta=run_meta, results=results)
            elif idx == 1:
                entries += _section1_headline(out_dir=out_dir, manifest=manifest, book=book, results=results, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 2:
                entries += _section2_risk(out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 3:
                entries += _section3_holdings(out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 4:
                entries += _section4_cost_capacity(out_dir=out_dir, manifest=manifest, book=book, results=results, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 5:
                entries += _section5_attribution(
                    out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy,
                    ff4_factors=ff4_factors, industry_group=industry_group, returns_panel=returns_panel,
                    secids=returns_panel_secids,
                )
            elif idx == 6:
                entries += _section6_signals(
                    out_dir=out_dir, manifest=manifest, book=book, results=results,
                    signal_gate_result=signal_gate_result, signal_panels=signal_panels,
                    iv_surface_grid=iv_surface_grid,
                )
            elif idx == 7:
                entries += _section7_learning(out_dir=out_dir, manifest=manifest, book=book, results=results, learning_curves=learning_curves)
            elif idx == 8:
                entries += _section8_stat_rigor(out_dir=out_dir, manifest=manifest, book=book, results=results)
            elif idx == 9:
                entries += _section9_spectrum(out_dir=out_dir, manifest=manifest, book=book, results=results, arms_root=arms_root_p, artifacts_flat=artifacts_flat_p)
            elif idx == 10:
                entries += _section10_limitations(out_dir=out_dir, manifest=manifest, book=book, results=results, known_limitations=known_limitations)

        n_pages = book.n_pages

    n_written = sum(1 for e in entries if e.get("status") == "written")
    n_skipped = sum(1 for e in entries if e.get("status") == "skipped")
    # Write BOOK.md before hashing so the sha256 map covers every on-disk
    # artifact (index.json itself is excluded as the digest container).
    md_lines = [
        "# Equity Allocation Reporting Book",
        "",
        f"Date range: {date_start} .. {date_end}",
        f"Focus strategy: {focus_name}",
        f"Figures written: {n_written} / {len(entries)} (skipped: {n_skipped})",
        "",
    ]
    for idx, title in enumerate(SECTION_TITLES):
        md_lines.append(f"## Section {idx}: {title}")
        md_lines.append("")
        for e in entries:
            if str(e.get("id", "")).startswith(f"S{idx}."):
                status = e.get("status")
                note = f" -- {e['note']}" if e.get("note") else ""
                md_lines.append(f"- `{e['id']}` {e['title']} [{status}]{note}")
        md_lines.append("")
    (out_dir / "BOOK.md").write_text("\n".join(md_lines))

    # W5: loud list of skipped figures so a silent empty section is obvious.
    skipped_lines = [
        "# Skipped figures",
        "",
        "Each entry names the figure id and the reason (missing input key).",
        "",
    ]
    for e in entries:
        if e.get("status") == "skipped":
            skipped_lines.append(
                f"- `{e.get('id')}` {e.get('title')}: {e.get('note') or 'missing input'}"
            )
    if n_skipped == 0:
        skipped_lines.append("_None skipped._")
    (out_dir / "SKIPPED.md").write_text("\n".join(skipped_lines) + "\n")

    index: dict[str, Any] = {
        "manifest": manifest,
        "n_pages": n_pages,
        "n_figures": len(entries),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "figures": entries,
        "sha256": {},
    }
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "index.json":
            index["sha256"][str(p.relative_to(out_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, default=str, sort_keys=True))

    return index
