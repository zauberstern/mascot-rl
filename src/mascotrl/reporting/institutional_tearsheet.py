"""Institutional tearsheet plots for MascotRL HAPPO runs.

Maps the idea brief onto metrics this stack emits: synthetic-path episodes,
CMDP delta/turnover, train vs held-out eval. Comparative alpha vs non-trivial
vol baselines (carry / GARCH / Heston) is publication plot 30_*, not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mascotrl.reporting.book_style import (
    C_ACCENT,
    C_BLUE,
    C_GRAY,
    C_LIGHT,
    C_NAVY,
    C_NEG,
    C_POS,
    C_STEEL,
    C_ZERO,
    CMAP_DIVERGING,
    academic_figure,
    finalize_figure,
    mark_oos_boundary,
    place_legend,
    save_figure,
    shade_insample,
)
from mascotrl.reporting.viz_ingest import (
    DEFAULT_FRICTION_PER_TURNOVER,
    REGIME_IS,
    REGIME_OOS,
    build_nav_series,
    enrich_episodes,
    herfindahl,
    is_oos_split_index,
    load_episode_frames,
    load_report_arrays,
    rolling_sharpe,
    rolling_sortino,
    write_metrics_parquet,
)


def _finish(
    fig,
    ax,
    out_stem: Path,
    *,
    legend: bool = True,
    legend_loc: str = "outside right",
    ncol: int = 1,
) -> list[str]:
    """Legend outside data area + safe layout + PNG-only save."""
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            place_legend(ax, loc=legend_loc, ncol=ncol)
            finalize_figure(fig, legend_space=legend_loc.startswith("outside"))
        else:
            finalize_figure(fig, legend_space=False)
    else:
        finalize_figure(fig, legend_space=False)
    return save_figure(fig, out_stem, pdf=False)


def _apply_regime_band(ax, split: int | None) -> None:
    """Shade IS / mark OOS *after* data are plotted (avoids bbox explosion)."""
    if split is not None and split > 0:
        shade_insample(ax, -0.5, split - 0.5)
        mark_oos_boundary(ax, split - 0.5)


def _safe_kde(ax, values: np.ndarray, color: str, label: str, alpha: float = 0.35) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return
    try:
        from scipy import stats

        xs = np.linspace(np.percentile(values, 1), np.percentile(values, 99), 200)
        kde = stats.gaussian_kde(values)
        ax.fill_between(xs, kde(xs), color=color, alpha=alpha, label=label, lw=0)
        ax.plot(xs, kde(xs), color=color, lw=1.0)
    except Exception:
        ax.hist(
            values,
            bins=min(30, max(8, values.size // 5)),
            density=True,
            color=color,
            alpha=alpha,
            label=label,
        )


def plot_equity_curve(nav: pd.DataFrame, out: Path) -> list[str]:
    if nav.empty:
        return []
    with academic_figure("double", height_in=4.2, legend_space=True) as (fig, ax):
        split = is_oos_split_index(nav)
        ax.plot(nav["idx"], nav["nav"], color=C_BLUE, lw=1.6, label="HAPPO NAV (base 100)")
        ax.axhline(100.0, color=C_GRAY, lw=0.7, ls=":")
        _apply_regime_band(ax, split)
        ax.set_xlabel("Episode index (train → held-out eval)")
        ax.set_ylabel("NAV")
        ax.set_title("Continuous equity curve — in-sample shaded")
        return _finish(fig, ax, out / "01_equity_curve")


def plot_rolling_risk(nav: pd.DataFrame, out: Path, window: int | None = None) -> list[str]:
    if nav.empty or len(nav) < 10:
        return []
    rets = nav["ret_proxy"].to_numpy()
    win = window or min(50, max(10, len(rets) // 8))
    sh = rolling_sharpe(rets, win)
    so = rolling_sortino(rets, win)
    if sh.size == 0:
        return []
    xs = np.arange(win - 1, win - 1 + sh.size)
    split = is_oos_split_index(nav)
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        finite_so = np.where(np.isfinite(so), so, np.nan)
        ax.plot(xs, sh, color=C_NAVY, lw=1.4, label="Rolling Sharpe")
        ax.plot(xs, finite_so, color=C_ACCENT, lw=1.2, label="Rolling Sortino")
        ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
        _apply_regime_band(ax, split)
        ax.set_title(f"Rolling risk-adjusted returns (window={win})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Annualized ratio (proxy)")
        return _finish(fig, ax, out / "02_rolling_sharpe_sortino")


def plot_return_distributions(episodes: pd.DataFrame, out: Path) -> list[str]:
    if episodes.empty:
        return []
    is_pnl = episodes[(episodes["regime"] == REGIME_IS) & (episodes["mode"] == "happo")]["pnl"].to_numpy()
    oos_pnl = episodes[(episodes["regime"] == REGIME_OOS) & (episodes["mode"] == "happo")]["pnl"].to_numpy()
    if is_pnl.size < 3 and oos_pnl.size < 3:
        return []
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        if is_pnl.size >= 3:
            _safe_kde(ax, is_pnl, C_GRAY, "In-sample (train)", alpha=0.4)
        if oos_pnl.size >= 3:
            _safe_kde(ax, oos_pnl, C_NEG, "Out-of-sample (eval)", alpha=0.35)
        ax.axvline(0.0, color=C_ZERO, lw=0.7, ls="--")
        ax.set_title("Episodic PnL distributions")
        ax.set_xlabel("Episode PnL")
        ax.set_ylabel("Density")
        return _finish(fig, ax, out / "03_return_distributions")


def plot_underwater(nav: pd.DataFrame, out: Path) -> list[str]:
    if nav.empty:
        return []
    dd = nav["drawdown"].to_numpy() * 100.0
    split = is_oos_split_index(nav)
    with academic_figure("double", height_in=3.8) as (fig, ax):
        ax.fill_between(nav["idx"], dd, 0.0, color=C_NEG, alpha=0.75, lw=0)
        ax.set_ylim(min(float(dd.min()) * 1.15, -0.1), 0.5)
        _apply_regime_band(ax, split)
        ax.set_title("Underwater drawdown curve")
        ax.set_xlabel("Episode index")
        ax.set_ylabel("Drawdown (%)")
        if split is not None and split < len(dd):
            oos_dd = dd[split:]
            if oos_dd.size:
                i = int(np.argmin(oos_dd)) + split
                ax.annotate(
                    f"OOS max DD {dd[i]:.1f}%",
                    xy=(float(nav["idx"].iloc[i]), float(dd[i])),
                    xytext=(8, -14),
                    textcoords="offset points",
                    fontsize=8,
                    color=C_NEG,
                )
        return _finish(fig, ax, out / "04_underwater_drawdown", legend=False)


def plot_episode_block_heatmap(episodes: pd.DataFrame, out: Path) -> list[str]:
    """Episode-block grid (not calendar months)."""
    sub = episodes[(episodes["regime"] == REGIME_IS) & (episodes["mode"] == "happo")]
    if sub.empty or len(sub) < 12:
        return []
    pnl = sub.sort_values("ep")["pnl"].to_numpy(dtype=float)
    cols = 12
    # Cap rows so the figure stays screen-sized (downsample if needed).
    max_rows = 20
    rows_full = int(np.ceil(pnl.size / cols))
    if rows_full > max_rows:
        # Average into max_rows blocks.
        block = int(np.ceil(pnl.size / (max_rows * cols)))
        n = (pnl.size // block) * block
        pnl = pnl[:n].reshape(-1, block).mean(axis=1)
        rows_full = int(np.ceil(pnl.size / cols))
    rows = min(rows_full, max_rows)
    grid = np.full((rows, cols), np.nan)
    grid.flat[: min(pnl.size, rows * cols)] = pnl[: rows * cols]
    with academic_figure("double", height_in=min(6.0, 2.5 + 0.18 * rows)) as (fig, ax):
        vmax = np.nanpercentile(np.abs(pnl), 95) if np.isfinite(pnl).any() else 1.0
        vmax = max(float(vmax), 1e-6)
        im = ax.imshow(grid, aspect="auto", cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_title("Training PnL by episode block (12-wide; not calendar months)")
        ax.set_xlabel("Block column")
        ax.set_ylabel("Block row")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        finalize_figure(fig, legend_space=False)
        return save_figure(fig, out / "05_episode_block_heatmap", pdf=False)


def plot_delta_tracking(episodes: pd.DataFrame, out: Path) -> list[str]:
    hap = episodes[episodes["mode"] == "happo"].copy()
    if hap.empty or "mean_abs_delta" not in hap.columns:
        return []
    hap = hap.reset_index(drop=True)
    split = int((hap["regime"] == REGIME_IS).sum())
    xs = np.arange(len(hap))
    mean_d = hap["mean_abs_delta"].astype(float).to_numpy()
    max_d = hap["max_abs_delta"].astype(float).to_numpy() if "max_abs_delta" in hap.columns else mean_d
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.fill_between(xs, 0.0, max_d, color=C_LIGHT, alpha=0.9, label="Max |δ|", lw=0)
        ax.plot(xs, mean_d, color=C_ZERO, lw=1.2, label="Mean |δ|")
        ax.axhline(0.0, color=C_GRAY, lw=0.5)
        ax.set_yscale("symlog", linthresh=1e-6)
        _apply_regime_band(ax, split if split > 0 else None)
        ax.set_title("Delta exposure tracking (portfolio w·Δ)")
        ax.set_xlabel("Episode (train → eval)")
        ax.set_ylabel("|δ|")
        return _finish(fig, ax, out / "06_delta_exposure")


def plot_var_exceedance(episodes: pd.DataFrame, out: Path) -> list[str]:
    hap = episodes[episodes["mode"] == "happo"]
    if hap.empty:
        return []
    pnl = hap["pnl"].astype(float).to_numpy()
    if pnl.size < 20:
        return []
    win = min(40, max(10, pnl.size // 5))
    var = np.full(pnl.size, np.nan)
    for i in range(win, pnl.size):
        var[i] = np.percentile(pnl[i - win : i], 5)
    xs = np.arange(pnl.size)
    exceed = pnl < var
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.scatter(xs[~exceed], pnl[~exceed], s=8, c=C_GRAY, alpha=0.4, rasterized=True, label="PnL")
        ax.scatter(
            xs[exceed],
            pnl[exceed],
            s=22,
            c=C_NEG,
            alpha=0.9,
            rasterized=True,
            label="VaR exceedance",
            zorder=3,
        )
        ax.plot(xs, var, color=C_NAVY, lw=1.3, label=f"Rolling 5% VaR (w={win})")
        ax.axhline(0.0, color=C_ZERO, lw=0.6, ls=":")
        ax.set_title("VaR exceedance profile (episodic)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("PnL")
        return _finish(fig, ax, out / "07_var_exceedance")


def plot_vol_vs_macro(episodes: pd.DataFrame, report: dict[str, Any], out: Path) -> list[str]:
    hap = episodes[(episodes["regime"] == REGIME_IS) & (episodes["mode"] == "happo")]
    if hap.empty:
        return []
    pnl = hap.sort_values("ep")["pnl"].astype(float).to_numpy()
    win = min(30, max(8, len(pnl) // 10))
    if len(pnl) < win:
        return []
    strat_vol = np.array([pnl[i : i + win].std() for i in range(len(pnl) - win + 1)])
    macro = report.get("macro_sample")
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.plot(
            np.arange(strat_vol.size),
            strat_vol,
            color=C_BLUE,
            lw=1.4,
            label="Strategy PnL vol (rolling)",
        )
        if macro is not None:
            arr = np.asarray(macro, dtype=float)
            if arr.ndim == 2 and arr.shape[1] > 0 and arr.shape[0] > 5:
                m = np.abs(arr[: strat_vol.size, 0])
                if m.size:
                    m_s = m * (strat_vol.std() / (m.std() + 1e-8))
                    ax.plot(
                        np.arange(min(m_s.size, strat_vol.size)),
                        m_s[: strat_vol.size],
                        color=C_GRAY,
                        lw=1.0,
                        ls="--",
                        label="Macro f0 |z| (rescaled)",
                    )
        ax.set_title("Volatility dynamics vs macro stress proxy")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Volatility")
        return _finish(fig, ax, out / "08_vol_dynamics")


def plot_constraint_slack(episodes: pd.DataFrame, out: Path, turnover_limit: float = 0.15) -> list[str]:
    hap = episodes[episodes["mode"] == "happo"]
    if hap.empty:
        return []
    hap = hap.reset_index(drop=True)
    turn_slack = hap.get("turnover_breach", pd.Series(np.zeros(len(hap)))).astype(float).to_numpy()
    delta_slack = hap.get("delta_slack_proxy", pd.Series(np.zeros(len(hap)))).astype(float).to_numpy()
    xs = np.arange(len(hap))
    split = int((hap["regime"] == REGIME_IS).sum())
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        # Downsample stems for readability on long runs.
        step = max(1, len(xs) // 800)
        ax.vlines(
            xs[::step],
            0,
            turn_slack[::step],
            color=C_ACCENT,
            lw=0.8,
            label="Turnover breach (max−τ)",
        )
        ax.plot(xs, delta_slack, color=C_NAVY, lw=1.1, alpha=0.9, label="Delta slack proxy max|δ|")
        _apply_regime_band(ax, split if split > 0 else None)
        ax.set_title(f"CMDP constraint stress (τ={turnover_limit})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Slack / breach magnitude")
        return _finish(fig, ax, out / "09_constraint_slack")


def plot_baseline_alpha_attribution(episodes: pd.DataFrame, out: Path) -> list[str]:
    """Cumulative HAPPO PnL (academic baseline peers live in publication suite)."""
    oos = episodes[episodes["regime"] == REGIME_OOS]
    if oos.empty:
        return []
    modes = {m: g.sort_values("ep")["pnl"].astype(float).to_numpy() for m, g in oos.groupby("mode")}
    if "happo" not in modes:
        return []
    h = modes["happo"]
    if len(h) < 2:
        return []
    cum = np.cumsum(h)
    xs = np.arange(len(h))
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.fill_between(xs, 0, cum, color=C_STEEL, alpha=0.55, label="Σ HAPPO", lw=0)
        ax.plot(xs, cum, color=C_NAVY, lw=1.4, label="Σ HAPPO")
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_title("HAPPO cumulative PnL (held-out; peers = BASELINE_NAMES)")
        ax.set_xlabel("Eval episode")
        ax.set_ylabel("Cumulative PnL")
        return _finish(fig, ax, out / "10_baseline_alpha")


def plot_turnover_adherence(episodes: pd.DataFrame, out: Path, turnover_limit: float = 0.15) -> list[str]:
    hap = episodes[episodes["mode"] == "happo"].reset_index(drop=True)
    if hap.empty or "mean_turnover" not in hap.columns:
        return []
    xs = np.arange(len(hap))
    mean_t = hap["mean_turnover"].astype(float).to_numpy()
    max_t = hap["max_turnover"].astype(float).to_numpy() if "max_turnover" in hap.columns else mean_t
    split = int((hap["regime"] == REGIME_IS).sum())
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        # Plot data first, lock a sensible ylim, THEN regime markers.
        ax.step(xs, mean_t, where="mid", color=C_NAVY, lw=1.2, label="Mean turnover")
        ax.plot(xs, max_t, color=C_STEEL, lw=0.9, alpha=0.85, label="Max turnover")
        ax.axhline(turnover_limit, color=C_NEG, lw=1.4, label=f"Limit τ={turnover_limit}")
        y_hi = max(float(np.nanmax(max_t)) * 1.15, turnover_limit * 1.25, 0.05)
        ax.set_ylim(0.0, y_hi)
        _apply_regime_band(ax, split if split > 0 else None)
        ax.set_title("L1 turnover adherence")
        ax.set_xlabel("Episode")
        ax.set_ylabel("‖Δw‖₁")
        return _finish(fig, ax, out / "11_turnover_adherence")


def plot_friction_drag(episodes: pd.DataFrame, out: Path) -> list[str]:
    hap = episodes[episodes["mode"] == "happo"].reset_index(drop=True)
    if hap.empty or "gross_pnl" not in hap.columns:
        return []
    split = int((hap["regime"] == REGIME_IS).sum())
    g = build_nav_series(hap, mode="happo", pnl_col="gross_pnl")
    n = build_nav_series(hap, mode="happo", pnl_col="net_pnl_stylized")
    if g.empty or n.empty:
        return []
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.plot(g["idx"], g["nav"], color=C_BLUE, lw=1.4, label="Gross MTM NAV")
        ax.plot(n["idx"], n["nav"], color=C_ACCENT, lw=1.3, label="Stylized net (linear turnover drag)")
        ax.fill_between(g["idx"], n["nav"], g["nav"], color=C_LIGHT, alpha=0.7, lw=0)
        _apply_regime_band(ax, split if split > 0 else None)
        ax.set_title(
            f"Execution friction drag (illustrative; cost={DEFAULT_FRICTION_PER_TURNOVER}×Σturnover)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("NAV")
        return _finish(fig, ax, out / "12_friction_drag")


def plot_allocation_and_hhi(steps: pd.DataFrame, out: Path) -> list[str]:
    written: list[str] = []
    if steps.empty:
        return written
    wcols = [c for c in steps.columns if c.startswith("w_")]
    if not wcols:
        return written
    steps = steps.copy()
    steps["ep"] = steps["ep"].astype(int)
    k = len(wcols)
    n_grp = min(5, k)
    edges = np.linspace(0, k, n_grp + 1).astype(int)
    grp_mat = []
    labels = []
    for g in range(n_grp):
        cols = wcols[edges[g] : edges[g + 1]]
        grp_mat.append(steps[cols].abs().sum(axis=1))
        labels.append(f"Cluster {g}")
    G = pd.DataFrame({lab: s for lab, s in zip(labels, grp_mat)})
    G["ep"] = steps["ep"]
    by_ep = G.groupby("ep", as_index=True)[labels].mean()
    row_sum = by_ep.sum(axis=1).replace(0.0, np.nan)
    comp = by_ep.div(row_sum, axis=0).fillna(0.0)
    xs = np.arange(len(comp))
    with academic_figure("double", height_in=4.2, legend_space=True) as (fig, ax):
        colors = [C_NAVY, C_BLUE, C_STEEL, C_ACCENT, C_GRAY][:n_grp]
        ax.stackplot(xs, *[comp[c].to_numpy() for c in labels], labels=labels, colors=colors, lw=0)
        ax.set_ylim(0, 1)
        ax.set_title("Dynamic capital allocation (asset clusters; |w| share)")
        ax.set_xlabel("Sampled episode order")
        ax.set_ylabel("Share")
        written += _finish(fig, ax, out / "13_capital_allocation", legend_loc="outside right")

    hhi = [herfindahl(row.to_numpy()) for _, row in by_ep.iterrows()]
    eps = list(by_ep.index)
    with academic_figure("double", height_in=3.6) as (fig, ax):
        ax.axhspan(0.15, 0.35, color=C_LIGHT, alpha=0.55, zorder=0, label="Diversification band")
        ax.plot(eps, hhi, color=C_NAVY, lw=1.3, label="HHI")
        ax.set_title("Asset concentration (HHI of |w|)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("HHI")
        ax.set_ylim(0, 1)
        written += _finish(fig, ax, out / "14_concentration_hhi", legend_loc="outside top")
    return written


def plot_regime_pca(episodes: pd.DataFrame, out: Path) -> list[str]:
    """PCA of episode health features — substitute for embedding t-SNE."""
    cols = [
        c
        for c in ("pnl", "mean_abs_delta", "mean_turnover", "weight_l1_mean", "max_abs_delta")
        if c in episodes.columns
    ]
    if len(cols) < 2 or len(episodes) < 30:
        return []
    df = episodes[episodes["mode"] == "happo"].copy()
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 30:
        return []
    regimes = df.loc[X.index, "regime"]
    Z = (X - X.mean()) / X.std().replace(0, 1.0)
    U, S, _Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        is_mask = regimes.to_numpy() == REGIME_IS
        ax.scatter(
            pcs[is_mask, 0],
            pcs[is_mask, 1],
            s=10,
            c=C_GRAY,
            alpha=0.35,
            rasterized=True,
            label="In-sample",
        )
        ax.scatter(
            pcs[~is_mask, 0],
            pcs[~is_mask, 1],
            s=18,
            c=C_NAVY,
            alpha=0.85,
            rasterized=True,
            label="OOS eval",
        )
        ax.set_title("State-space regime map (PCA of episode metrics)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        return _finish(fig, ax, out / "15_regime_pca")


def plot_alpha_spread(episodes: pd.DataFrame, out: Path) -> list[str]:
    oos = episodes[episodes["regime"] == REGIME_OOS]
    modes = {m: g.sort_values("ep")["pnl"].astype(float).to_numpy() for m, g in oos.groupby("mode")}
    if "happo" not in modes:
        return []
    h = modes["happo"]
    if len(h) < 2:
        return []
    spread = np.cumsum(h)
    with academic_figure("double", height_in=3.8) as (fig, ax):
        ax.plot(np.arange(len(h)), spread, color=C_NAVY, lw=1.6, label="Σ HAPPO")
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_title("Cumulative HAPPO PnL (held-out; academic peers elsewhere)")
        ax.set_xlabel("Eval episode")
        ax.set_ylabel("Σ HAPPO")
        return _finish(fig, ax, out / "16_alpha_spread", legend=False)


def plot_ir_migration(episodes: pd.DataFrame, out: Path) -> list[str]:
    def _ir_stats(active: np.ndarray) -> tuple[float, float]:
        active = np.asarray(active, dtype=float)
        if active.size < 2:
            return 0.0, 0.0
        te = float(active.std() + 1e-12)
        ar = float(active.mean())
        return ar * np.sqrt(252.0 / max(active.size, 1)), te * np.sqrt(252.0 / max(active.size, 1))

    hap_is = episodes[(episodes["regime"] == REGIME_IS) & (episodes["mode"] == "happo")]["pnl"].to_numpy()
    oos = episodes[episodes["regime"] == REGIME_OOS]
    if oos.empty or hap_is.size < 5:
        return []
    by = {m: g.sort_values("ep")["pnl"].astype(float).to_numpy() for m, g in oos.groupby("mode")}
    if "happo" not in by:
        return []
    ar_is, te_is = _ir_stats(hap_is)
    ar_oos, te_oos = _ir_stats(by["happo"])
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.scatter([te_is], [ar_is], marker="s", s=70, c=C_GRAY, label="In-sample", zorder=3)
        ax.scatter([te_oos], [ar_oos], marker="*", s=140, c=C_NAVY, label="Out-of-sample", zorder=3)
        ax.annotate(
            "",
            xy=(te_oos, ar_oos),
            xytext=(te_is, ar_is),
            arrowprops=dict(arrowstyle="->", color=C_ZERO, lw=1.1),
        )
        te_max = max(te_is, te_oos, 1e-6) * 1.3
        ax.plot([0, te_max], [0, 0.5 * te_max], color=C_LIGHT, ls="--", lw=1.1, label="IR = 0.5")
        ax.axhline(0.0, color=C_GRAY, lw=0.5)
        ax.axvline(0.0, color=C_GRAY, lw=0.5)
        ax.set_title("Information ratio migration")
        ax.set_xlabel("Tracking error (ann. proxy)")
        ax.set_ylabel("Active return (ann. proxy)")
        return _finish(fig, ax, out / "17_ir_migration")


def plot_capture_ratios(episodes: pd.DataFrame, out: Path) -> list[str]:
    """Up/down capture self-benchmark (random peer removed; rich baselines elsewhere)."""
    oos = episodes[episodes["regime"] == REGIME_OOS]
    by = {m: g.sort_values("ep")["pnl"].astype(float).to_numpy() for m, g in oos.groupby("mode")}
    if "happo" not in by:
        return []
    h = by["happo"]

    def self_cap(x, positive: bool):
        mask = x > 0 if positive else x < 0
        if mask.sum() == 0:
            return 0.0
        return float(x[mask].mean() / (np.mean(np.abs(x[mask])) + 1e-12))

    train = episodes[(episodes["regime"] == REGIME_IS) & (episodes["mode"] == "happo")]["pnl"].to_numpy()
    vals = {
        "IS up": self_cap(train, True) if train.size else 0.0,
        "IS down": abs(self_cap(train, False)) if train.size else 0.0,
        "OOS up": self_cap(h, True),
        "OOS down": abs(self_cap(h, False)),
    }
    with academic_figure("double", height_in=3.8) as (fig, ax):
        labels = list(vals.keys())
        heights = [vals[k] for k in labels]
        colors = [C_POS, C_NEG, C_POS, C_NEG]
        ax.barh(labels, heights, color=colors, height=0.55)
        ax.axvline(1.0, color=C_GRAY, ls="--", lw=0.8)
        ax.set_title("Up / down capture (self-benchmark; no random peer)")
        ax.set_xlabel("Capture ratio")
        return _finish(fig, ax, out / "18_capture_ratios", legend=False)


def plot_signal_case_study(steps: pd.DataFrame, out: Path) -> list[str]:
    if steps.empty or "spot_mean" not in steps.columns:
        return []
    sub = steps.copy()
    if "mode" in sub.columns:
        sub = sub[sub["mode"] == "happo"]
    if "regime" in sub.columns and (sub["regime"] == REGIME_OOS).any():
        sub = sub[sub["regime"] == REGIME_OOS]
    if sub.empty:
        return []
    counts = sub.groupby("ep").size()
    if counts.empty:
        return []
    ep = int(counts.idxmax())
    path = sub[sub["ep"] == ep].sort_values("step")
    if len(path) < 5:
        return []
    spot = path["spot_mean"].astype(float).to_numpy()
    spot_n = spot / (spot[0] + 1e-12)
    w_l1 = path["weight_l1"].astype(float).to_numpy() if "weight_l1" in path.columns else None
    with academic_figure("double", height_in=4.0, legend_space=True) as (fig, ax):
        ax.plot(path["step"], spot_n, color=C_GRAY, lw=1.3, label="Spot mean (norm)")
        ax.set_ylabel("Normalized spot")
        ax.set_xlabel("Step")
        ax.set_title(f"Signal activation case study — eval ep={ep}")
        if w_l1 is not None:
            ax2 = ax.twinx()
            ax2.fill_between(path["step"], 0, w_l1, color=C_BLUE, alpha=0.30, lw=0)
            ax2.plot([], [], color=C_BLUE, alpha=0.6, lw=6, label="‖w‖₁")
            ax2.set_ylabel("‖w‖₁")
            ax2.spines["top"].set_visible(False)
            # Merge legends from both axes onto the primary, outside.
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                bbox_to_anchor=(1.12, 1.0),
                frameon=False,
            )
            finalize_figure(fig, legend_space=True)
        else:
            return _finish(fig, ax, out / "19_signal_case_study")
        return save_figure(fig, out / "19_signal_case_study", pdf=False)


def plot_incubation(episodes: pd.DataFrame, report: dict[str, Any], out: Path) -> list[str]:
    pol = np.asarray(report.get("policy_losses") or [], dtype=float)
    val = np.asarray(report.get("value_losses") or [], dtype=float)
    if pol.size == 0 and not episodes.empty and "train_policy_loss" in episodes.columns:
        train = episodes[episodes["regime"] == REGIME_IS].sort_values("ep")
        pol = train["train_policy_loss"].astype(float).to_numpy()
        val = (
            train["train_value_loss"].astype(float).to_numpy()
            if "train_value_loss" in train.columns
            else pol
        )
    if pol.size < 5:
        return []

    def ma(x, w=21):
        if x.size < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    sp = ma(pol)
    sv = ma(val)
    with academic_figure("double", height_in=5.2, nrows=2, sharex=True) as (fig, axes):
        axes[0].plot(np.arange(pol.size), pol, color=C_LIGHT, lw=0.5)
        axes[0].plot(np.arange(len(sp)) + (pol.size - len(sp)), sp, color=C_NAVY, lw=1.3)
        axes[0].set_ylabel("Signal variance\n(policy loss)")
        axes[0].set_title("Strategy incubation / convergence")
        axes[1].plot(np.arange(val.size), val, color=C_LIGHT, lw=0.5)
        axes[1].plot(np.arange(len(sv)) + (val.size - len(sv)), sv, color=C_POS, lw=1.3)
        axes[1].set_ylabel("Strategy calibration\n(value loss)")
        axes[1].set_xlabel("Incubation epoch")
        if len(sp) > 50:
            d = np.abs(np.diff(sp))
            thresh = np.percentile(d, 20)
            ready_rel = int(np.argmax(d < thresh))
            ready = ready_rel + (pol.size - len(sp))
            for ax in axes:
                ax.axvline(ready, color=C_ZERO, ls="--", lw=0.9)
            axes[0].text(
                0.01,
                0.92,
                "ready≈",
                transform=axes[0].transAxes,
                fontsize=8,
                ha="left",
                va="top",
            )
        finalize_figure(fig, legend_space=False)
        return save_figure(fig, out / "20_incubation_convergence", pdf=False)


def plot_eval_baselines_box(episodes: pd.DataFrame, out: Path) -> list[str]:
    oos = episodes[episodes["regime"] == REGIME_OOS]
    if oos.empty:
        return []
    with academic_figure("double", height_in=4.0) as (fig, ax):
        data, labels = [], []
        for m in ("happo",):
            vals = oos[oos["mode"] == m]["pnl"].astype(float).to_numpy()
            if vals.size:
                data.append(vals)
                labels.append(m)
        if not data:
            return []
        try:
            bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, m in zip(bp["boxes"], labels):
            patch.set_facecolor(C_LIGHT)
            patch.set_edgecolor(C_NAVY if m == "happo" else C_GRAY)
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_title("Held-out PnL by policy")
        ax.set_ylabel("Episode PnL")
        return _finish(fig, ax, out / "21_eval_boxplot", legend=False)


def render_tearsheet(
    run_dir: Path | str,
    *,
    turnover_limit: float = 0.15,
    cost_per_turnover: float = DEFAULT_FRICTION_PER_TURNOVER,
    write_parquet: bool = True,
) -> dict[str, Any]:
    """
    Build the institutional tearsheet under ``report/tearsheet/``.

    Returns metadata including written paths and skipped plot reasons.
    """
    run_dir = Path(run_dir)
    out = run_dir / "report" / "tearsheet"
    out.mkdir(parents=True, exist_ok=True)

    frames = load_episode_frames(run_dir)
    if write_parquet:
        write_metrics_parquet(run_dir, frames)

    episodes = enrich_episodes(
        frames["train"],
        frames["eval"],
        cost_per_turnover=cost_per_turnover,
        turnover_limit=turnover_limit,
    )
    report = load_report_arrays(run_dir)
    nav = build_nav_series(episodes, mode="happo", pnl_col="pnl")
    steps = frames["steps"]

    written: list[str] = []
    skipped: list[str] = []

    def _run(name: str, fn, *args, **kwargs):
        nonlocal written
        try:
            paths = fn(*args, **kwargs)
            if paths:
                written.extend(paths)
            else:
                skipped.append(name)
        except Exception as exc:
            skipped.append(f"{name}: {exc}")

    _run("equity", plot_equity_curve, nav, out)
    _run("rolling", plot_rolling_risk, nav, out)
    _run("distributions", plot_return_distributions, episodes, out)
    _run("underwater", plot_underwater, nav, out)
    _run("heatmap", plot_episode_block_heatmap, episodes, out)
    _run("delta", plot_delta_tracking, episodes, out)
    _run("var", plot_var_exceedance, episodes, out)
    _run("vol", plot_vol_vs_macro, episodes, report, out)
    _run("slack", plot_constraint_slack, episodes, out, turnover_limit)
    # Obsolete vs zero/random "baseline" suite removed — publication plots 30+
    # (short_vol_carry / GARCH / Heston) are the comparative surface.
    _run("turnover", plot_turnover_adherence, episodes, out, turnover_limit)
    _run("friction", plot_friction_drag, episodes, out)
    _run("alloc", plot_allocation_and_hhi, steps, out)
    _run("pca", plot_regime_pca, episodes, out)
    _run("case", plot_signal_case_study, steps, out)
    _run("incubate", plot_incubation, episodes, report, out)

    # Index + methodology note
    note = {
        "run_dir": str(run_dir),
        "n_plots_files": len(written),
        "written": written,
        "skipped": skipped,
        "regime_definition": {
            "in_sample": "training episodes on synthetic rBergomi paths",
            "out_of_sample": "held-out eval with frozen policy",
            "not_used": "calendar CRSP IS/OOS (paths are not a single market timeline)",
        },
        "friction_note": (
            f"Gross vs net uses stylized linear drag cost_per_turnover={cost_per_turnover}; "
            "training reward remains clean MTM (CMDP enforces turnover, does not subtract fees)."
        ),
        "omitted_from_brief": [
            "Fama–French factor stack (no equity factor panel in this vol-arb stack)",
            "t-SNE of neural hidden states (embeddings not persisted; PCA of episode metrics used)",
            "Calendar monthly heatmap (episode-block heatmap used instead)",
            "Almgren–Chriss impact model (stylized linear turnover drag only)",
            "Tearsheet plots vs synthetic zero/random (10/16/17/18/21) — obsolete; "
            "use publication 30_baseline_equity (short_vol_carry / GARCH / Heston)",
        ],
        "obsolete_tearsheet_removed": [
            "10_baseline_alpha",
            "16_alpha_spread",
            "17_ir_migration",
            "18_capture_ratios",
            "21_eval_boxplot",
        ],
        "nav_n": int(len(nav)),
        "episodes_n": int(len(episodes)),
        "steps_n": int(len(steps)),
    }
    (out / "index.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Institutional tearsheet",
                "",
                "Screen-oriented PNG figures for this run (no PDF).",
                "Gray bands mark **in-sample (train)**; the dashed line marks **held-out eval**.",
                "Legends sit outside the data area.",
                "",
                "Comparative alpha vs **short_vol_carry / GARCH / Heston** lives in "
                "`report/plots/30_baseline_equity.png` (publication suite), not in this "
                "tearsheet. Synthetic zero/random legs are eval diagnostics only.",
                "",
                "See `index.json` for methodology notes and intentional omissions vs the idea brief.",
                "",
                f"- Plot files: {len(written)}",
                f"- Skipped: {len(skipped)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note
