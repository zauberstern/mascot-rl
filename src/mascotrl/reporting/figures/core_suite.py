"""Core spectrum figures (F01/F02/F03/F05/F09/F16/F26 MVP)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mascotrl.reporting.figures import loaders as L
from mascotrl.reporting.book_style import (
    ARM_COLORS,
    ARM_ORDER,
    C_BLUE,
    C_GRAY,
    C_NAVY,
    C_NEG,
    C_POS,
    C_STEEL,
    C_ZERO,
    CMAP_DIVERGING,
    academic_figure,
    finalize_figure,
    place_legend,
    save_pdf_png,
    use_agg,
    mirror_files,
)


def _entry(
    fig_id: str,
    title: str,
    *,
    status: str,
    paths: dict[str, str] | None = None,
    sources: list[str] | None = None,
    note: str | None = None,
    caption: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": fig_id,
        "title": title,
        "status": status,
        "sources": sources or [],
        "caption": caption or title,
    }
    if paths:
        out.update(paths)
    if note:
        out["note"] = note
    return out


def _stem(out_dir: Path, fig_id: str, slug: str) -> Path:
    return Path(out_dir) / f"{fig_id}_{slug}"


def plot_f01_sharpe_ladder(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F01: grouped bars arm x (RL, best industry, best ML, equal weight)."""
    use_agg()
    title = "Spectrum Sharpe ladder"
    spectrum = L.load_spectrum_summary(arms_root=arms_root, artifacts_flat=artifacts_flat)
    sources: list[str] = []
    rows: dict[str, dict[str, float]] = {}

    def _f(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    if spectrum and isinstance(spectrum.get("arms"), dict):
        sources.append("spectrum_summary.json")
        for arm, payload in spectrum["arms"].items():
            if not isinstance(payload, dict):
                continue
            rl = payload.get("sharpe_mean", payload.get("rl"))
            rows[str(arm)] = {
                "rl": _f(rl),
                "best_industry": _f(payload.get("best_industry")),
                "best_ml": _f(payload.get("best_ml")),
                "equal_weight": _f(payload.get("equal_weight")),
            }

    if not rows:
        for arm in ARM_ORDER:
            cpcv = L.load_cpcv(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
            gate3 = L.load_gate3(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
            ps = L.path_summary(cpcv)
            bas = L.gate3_baselines(gate3)
            if not ps and not bas:
                continue
            industry_keys = [
                k
                for k in bas
                if k
                not in (
                    "xgb",
                    "mlp",
                    "happo_cpcv_mean_path_sharpe",
                    "equal_weight",
                )
                and not str(k).startswith("happo")
            ]
            ml_keys = [k for k in ("xgb", "mlp") if k in bas]
            best_ind = max((bas[k] for k in industry_keys), default=float("nan"))
            best_ml = max((bas[k] for k in ml_keys), default=float("nan"))
            rl = float(ps.get("sharpe_mean") if ps else bas.get("happo_cpcv_mean_path_sharpe", np.nan))
            rows[arm] = {
                "rl": rl,
                "best_industry": float(best_ind),
                "best_ml": float(best_ml),
                "equal_weight": float(bas.get("equal_weight", np.nan)),
            }
            if cpcv:
                sources.append(f"arms/{arm}/cpcv_path_summary.json")
            if gate3:
                sources.append(f"arms/{arm}/gate3_same_fold.json")

    if not rows:
        return _entry("F01", title, status="skipped", note="no spectrum or arm CPCV/gate3 data")

    series_keys = ("rl", "best_industry", "best_ml", "equal_weight")
    series_labels = ("RL (HAPPO)", "Best industry", "Best ML", "Equal weight")
    colors = (C_NAVY, C_STEEL, C_BLUE, C_GRAY)
    arms = [a for a in ARM_ORDER if a in rows] or sorted(rows)
    x = np.arange(len(arms), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(series_keys)) * width

    with academic_figure(width="double", height_in=4.2) as (fig, ax):
        for off, key, lab, col in zip(offsets, series_keys, series_labels, colors):
            vals = [rows[a].get(key, float("nan")) for a in arms]
            ax.bar(x + off, vals, width=width, label=lab, color=col, edgecolor="white", linewidth=0.4)
            # Whiskers from path Sharpe dispersion when available
            if key == "rl":
                for i, arm in enumerate(arms):
                    cpcv = L.load_cpcv(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
                    sh = L.path_summary(cpcv).get("path_sharpes") or []
                    arr = np.asarray([float(s) for s in sh if np.isfinite(float(s))], dtype=float)
                    if arr.size >= 2:
                        ax.vlines(x[i] + off, arr.min(), arr.max(), colors=C_ZERO, lw=0.9, zorder=3)
                        ax.plot([x[i] + off], [float(np.median(arr))], "o", color=C_ZERO, ms=3, zorder=4)
        ax.axhline(0.0, color=C_ZERO, lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_ylabel("Sharpe")
        ax.set_title(title)
        place_legend(ax, loc="outside right")
        finalize_figure(fig, legend_space=True)
        paths = save_pdf_png(fig, _stem(out_dir, "F01", "spectrum_sharpe_ladder"))

    return _entry("F01", title, status="written", paths=paths, sources=sources)


def plot_f02_path_sharpe_violin(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F02: violin/strip of CPCV path Sharpes per arm."""
    use_agg()
    title = "CPCV path Sharpe distribution"
    data: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    for arm in ARM_ORDER:
        cpcv = L.load_cpcv(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
        sh = L.path_summary(cpcv).get("path_sharpes") or []
        arr = np.asarray([float(s) for s in sh if np.isfinite(float(s))], dtype=float)
        if arr.size == 0:
            continue
        data.append(arr)
        labels.append(arm)
        sources.append(f"arms/{arm}/cpcv_path_summary.json")

    if not data:
        return _entry("F02", title, status="skipped", note="no path_sharpes found")

    with academic_figure(width="double", height_in=4.0) as (fig, ax):
        parts = ax.violinplot(data, positions=np.arange(1, len(data) + 1), showmeans=False, showmedians=True, showextrema=False)
        for i, body in enumerate(parts.get("bodies", [])):
            arm = labels[i]
            body.set_facecolor(ARM_COLORS.get(arm, C_STEEL))
            body.set_alpha(0.35)
            body.set_edgecolor(ARM_COLORS.get(arm, C_STEEL))
        if "cmedians" in parts:
            parts["cmedians"].set_color(C_ZERO)
        for i, arr in enumerate(data):
            jitter = (np.linspace(-0.08, 0.08, arr.size) if arr.size > 1 else np.zeros(1))
            ax.scatter(
                np.full(arr.size, i + 1) + jitter,
                arr,
                s=22,
                color=ARM_COLORS.get(labels[i], C_NAVY),
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.axhline(0.0, color=C_ZERO, lw=0.8, ls="--")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Path Sharpe")
        ax.set_title(title)
        finalize_figure(fig)
        paths = save_pdf_png(fig, _stem(out_dir, "F02", "cpcv_path_sharpe_violin"))

    return _entry("F02", title, status="written", paths=paths, sources=sources)


def plot_f03_cumulative_pnl(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F03: cumulative net path PnL (mean across paths) per arm."""
    use_agg()
    title = "Cumulative CPCV path PnL"
    sources: list[str] = []
    series: dict[str, np.ndarray] = {}

    for arm in ARM_ORDER:
        cpcv = L.load_cpcv(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
        paths = L.cpcv_paths(cpcv)
        if not paths:
            continue
        # Align by index (paths can differ in length); truncate to min length.
        pnls = []
        for p in paths:
            arr = np.asarray(p.get("pnl") or p.get("returns") or [], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                pnls.append(arr)
        if not pnls:
            continue
        n = int(min(a.size for a in pnls))
        stacked = np.vstack([a[:n] for a in pnls])
        series[arm] = np.cumsum(stacked.mean(axis=0))
        sources.append(f"arms/{arm}/cpcv_path_summary.json")

    if not series:
        return _entry("F03", title, status="skipped", note="no CPCV path pnl series")

    with academic_figure(width="double", height_in=4.0, legend_space=True) as (fig, ax):
        for arm, cum in series.items():
            ax.plot(cum, label=arm, color=ARM_COLORS.get(arm, C_NAVY), lw=1.4)
        ax.axhline(0.0, color=C_ZERO, lw=0.7, ls="--")
        ax.set_xlabel("Path day index")
        ax.set_ylabel("Cumulative mean path PnL")
        ax.set_title(title)
        place_legend(ax, loc="outside right")
        finalize_figure(fig, legend_space=True)
        paths = save_pdf_png(fig, _stem(out_dir, "F03", "cumulative_path_pnl"))

    return _entry("F03", title, status="written", paths=paths, sources=sources)


def plot_f05_baseline_heatmap(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F05: baseline-by-arm Sharpe heatmap."""
    use_agg()
    title = "Baseline-by-arm Sharpe heatmap"
    sources: list[str] = []
    arm_bas: dict[str, dict[str, float]] = {}
    for arm in ARM_ORDER:
        gate3 = L.load_gate3(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
        bas = L.gate3_baselines(gate3)
        if not bas:
            continue
        arm_bas[arm] = bas
        sources.append(f"arms/{arm}/gate3")

    if not arm_bas:
        return _entry("F05", title, status="skipped", note="no gate3 baselines")

    # Stable baseline order: non-happo first, happo last
    names: list[str] = []
    for bas in arm_bas.values():
        for k in bas:
            if k not in names:
                names.append(k)
    names = sorted(names, key=lambda n: (str(n).startswith("happo"), n))
    arms = [a for a in ARM_ORDER if a in arm_bas]
    mat = np.full((len(names), len(arms)), np.nan, dtype=float)
    for j, arm in enumerate(arms):
        for i, name in enumerate(names):
            if name in arm_bas[arm]:
                mat[i, j] = arm_bas[arm][name]

    h = float(min(max(2.8 + 0.28 * len(names), 3.5), 8.0))
    with academic_figure(width="double", height_in=h) as (fig, ax):
        finite = mat[np.isfinite(mat)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(mat, aspect="auto", cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(arms)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sharpe")
        finalize_figure(fig)
        paths = save_pdf_png(fig, _stem(out_dir, "F05", "baseline_by_arm_heatmap"))

    return _entry("F05", title, status="written", paths=paths, sources=sources)


def plot_f09_break_even(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F09: break-even spread multiplier bars; annotate undefined."""
    use_agg()
    title = "Break-even spread multiplier"
    sources: list[str] = []
    arms: list[str] = []
    values: list[float] = []
    undefined: list[bool] = []

    for arm in ARM_ORDER:
        cpcv = L.load_cpcv(arm, arms_root=arms_root, artifacts_flat=artifacts_flat)
        if cpcv is None:
            continue
        be = L.break_even(cpcv)
        arms.append(arm)
        sources.append(f"arms/{arm}/cpcv_path_summary.json")
        if be is None:
            values.append(0.0)
            undefined.append(True)
        else:
            values.append(float(be))
            undefined.append(False)

    if not arms:
        return _entry("F09", title, status="skipped", note="no CPCV gate1/cost_ladder")

    with academic_figure(width="single", height_in=3.8) as (fig, ax):
        colors = [C_GRAY if u else (C_POS if v >= 0.25 else C_NEG) for v, u in zip(values, undefined)]
        bars = ax.bar(np.arange(len(arms)), values, color=colors, edgecolor="white")
        ax.axhline(0.25, color=C_ZERO, ls="--", lw=0.9, label="min 0.25")
        for i, (bar, undef) in enumerate(zip(bars, undefined)):
            if undef:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "undefined",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=C_NEG,
                    rotation=90,
                )
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(arms)
        ax.set_ylabel("Break-even spread multiplier")
        ax.set_title(title)
        place_legend(ax, loc="outside top")
        finalize_figure(fig)
        paths = save_pdf_png(fig, _stem(out_dir, "F09", "break_even_bar"))

    return _entry(
        "F09",
        title,
        status="written",
        paths=paths,
        sources=sources,
        note="undefined = gross mean path PnL <= 0 (option-hedging OM backtest)",
    )


def plot_f16_dh_surface(
    *,
    out_dir: Path,
    lake_panel: Path | None = None,
) -> dict[str, Any]:
    """F16: 3D moneyness x DTE x mean DH return when panel has required columns."""
    use_agg()
    title = "Moneyness x DTE x mean DH return"
    panel = Path(lake_panel) if lake_panel is not None else Path("lake/panels/dh_cross_section.parquet")
    if not panel.is_file():
        return _entry(
            "F16",
            title,
            status="skipped",
            note=f"missing panel: {panel}",
            sources=[],
        )

    try:
        import pandas as pd
    except ImportError:
        return _entry("F16", title, status="skipped", note="pandas unavailable")

    cols = None
    try:
        import pyarrow.parquet as pq

        cols = list(pq.read_schema(panel).names)
    except Exception:
        cols = None

    need_ret = "dh_ret_lagdelta" if (cols is None or "dh_ret_lagdelta" in cols) else "dh_ret"
    use_cols = [c for c in ("spot", "strike", need_ret, "dte", "ttm", "tau", "days_to_expiry") if cols is None or c in (cols or [])]
    # Always try spot/strike/ret
    for c in ("spot", "strike", need_ret):
        if c not in use_cols:
            use_cols.append(c)

    try:
        df = pd.read_parquet(panel, columns=[c for c in use_cols if cols is None or c in cols])
    except Exception as exc:
        return _entry("F16", title, status="skipped", note=f"failed to read panel: {exc}", sources=[str(panel)])

    if "spot" not in df.columns or "strike" not in df.columns:
        return _entry("F16", title, status="skipped", note="panel lacks spot/strike for moneyness", sources=[str(panel)])

    dte_col = next((c for c in ("dte", "ttm", "tau", "days_to_expiry") if c in df.columns), None)
    if dte_col is None:
        return _entry(
            "F16",
            title,
            status="skipped",
            note="panel lacks DTE/ttm column; rematerialize with tenor to enable F16",
            sources=[str(panel)],
        )

    ret_col = need_ret if need_ret in df.columns else ("dh_ret" if "dh_ret" in df.columns else None)
    if ret_col is None:
        return _entry("F16", title, status="skipped", note="panel lacks DH return column", sources=[str(panel)])

    m = (pd.to_numeric(df["strike"], errors="coerce") / pd.to_numeric(df["spot"], errors="coerce")).to_numpy()
    dte = pd.to_numeric(df[dte_col], errors="coerce").to_numpy()
    ret = pd.to_numeric(df[ret_col], errors="coerce").to_numpy()
    mask = np.isfinite(m) & np.isfinite(dte) & np.isfinite(ret) & (m > 0) & (dte > 0)
    m, dte, ret = m[mask], dte[mask], ret[mask]
    if m.size < 50:
        return _entry("F16", title, status="skipped", note="insufficient finite rows for surface", sources=[str(panel)])

    # Coarse grid mean
    m_edges = np.quantile(m, np.linspace(0.05, 0.95, 9))
    d_edges = np.quantile(dte, np.linspace(0.05, 0.95, 9))
    m_edges = np.unique(m_edges)
    d_edges = np.unique(d_edges)
    if m_edges.size < 3 or d_edges.size < 3:
        return _entry("F16", title, status="skipped", note="could not form moneyness/DTE grid", sources=[str(panel)])

    Z = np.full((d_edges.size - 1, m_edges.size - 1), np.nan)
    for i in range(len(d_edges) - 1):
        for j in range(len(m_edges) - 1):
            sel = (
                (dte >= d_edges[i])
                & (dte < d_edges[i + 1])
                & (m >= m_edges[j])
                & (m < m_edges[j + 1])
            )
            if np.any(sel):
                Z[i, j] = float(np.mean(ret[sel]))

    m_c = 0.5 * (m_edges[:-1] + m_edges[1:])
    d_c = 0.5 * (d_edges[:-1] + d_edges[1:])
    MM, DD = np.meshgrid(m_c, d_c)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    finite = Z[np.isfinite(Z)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1e-4
    vmax = max(vmax, 1e-8)
    norm = plt.Normalize(-vmax, vmax)
    surf = ax.plot_surface(MM, DD, np.nan_to_num(Z, nan=0.0), cmap=CMAP_DIVERGING, linewidth=0, antialiased=True, alpha=0.9)
    surf.set_array(Z.ravel())
    surf.set_norm(norm)
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel(f"DTE ({dte_col})")
    ax.set_zlabel(f"Mean {ret_col}")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label=ret_col)
    paths = save_pdf_png(fig, _stem(out_dir, "F16", "moneyness_dte_dh_surface"))
    plt.close(fig)

    return _entry("F16", title, status="written", paths=paths, sources=[str(panel)])


def plot_f26_attrition_funnel(
    *,
    arms_root: Path | None,
    out_dir: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any]:
    """F26: attrition funnel from filter_attrition.json."""
    use_agg()
    title = "Filter attrition funnel"
    attr = L.load_attrition(arms_root=arms_root, artifacts_flat=artifacts_flat)
    if not attr or not isinstance(attr.get("screens"), dict):
        return _entry("F26", title, status="skipped", note="missing filter_attrition.json")

    s = attr["screens"]
    n_base = int(s.get("n_base") or 0)
    if n_base <= 0:
        return _entry("F26", title, status="skipped", note="n_base missing/zero")

    # Sequential remaining estimate: start at base, subtract each fail_* in order.
    fail_order = [
        ("fail_iv_present", "IV present"),
        ("fail_volume_positive", "Volume > 0"),
        ("fail_mid_above_tick", "Mid >= tick"),
        ("fail_moneyness_band", "Moneyness band"),
        ("fail_no_arbitrage_bounds", "No-arb bounds"),
        ("fail_standard_settlement", "Std settlement"),
        ("fail_common_stock", "Common stock"),
        ("fail_not_index_option", "Not index"),
        ("fail_no_dividend_in_life", "No dividend"),
        ("fail_calls_only", "Calls only"),
        ("fail_spot_missing", "Spot present"),
    ]
    labels = ["Base"]
    counts = [n_base]
    remaining = n_base
    for key, lab in fail_order:
        if key not in s:
            continue
        remaining = max(0, remaining - int(s.get(key) or 0))
        labels.append(lab)
        counts.append(remaining)
    retained = int(s.get("n_retained") or remaining)
    labels.append("Retained")
    counts.append(retained)

    y = np.arange(len(labels))[::-1]
    with academic_figure(width="double", height_in=float(min(max(3.5, 0.35 * len(labels) + 1.5), 8.0))) as (fig, ax):
        ax.barh(y, counts, color=C_NAVY, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Rows remaining (sequential approx.)")
        ax.set_title(title)
        for yi, c in zip(y, counts):
            ax.text(c, yi, f" {c:,}", va="center", ha="left", fontsize=7, color=C_GRAY)
        finalize_figure(fig)
        paths = save_pdf_png(fig, _stem(out_dir, "F26", "attrition_funnel"))

    return _entry(
        "F26",
        title,
        status="written",
        paths=paths,
        sources=["filter_attrition.json"],
        note="Sequential remaining is illustrative; screens are not strictly nested.",
    )


def render_core_suite(
    *,
    arms_root: Path,
    out_dir: Path,
    artifacts_flat: Path | None = None,
    lake_panel: Path | None = None,
) -> list[dict[str, Any]]:
    """Render MVP figures; return list of manifest entries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_f01_sharpe_ladder(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
        plot_f02_path_sharpe_violin(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
        plot_f03_cumulative_pnl(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
        plot_f05_baseline_heatmap(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
        plot_f09_break_even(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
        plot_f16_dh_surface(out_dir=out_dir, lake_panel=lake_panel),
        plot_f26_attrition_funnel(arms_root=arms_root, out_dir=out_dir, artifacts_flat=artifacts_flat),
    ]


def render_spectrum_figures(
    arms_root: str | Path,
    out_dir: str | Path,
    *,
    artifacts_flat: str | Path | None = None,
    artifacts_fig_dir: str | Path | None = None,
    lake_panel: str | Path | None = None,
) -> dict[str, Any]:
    """Render MVP spectrum figures and write ``figure_manifest.json``."""
    use_agg()
    arms_root = Path(arms_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if artifacts_flat is None:
        artifacts_flat = arms_root.parent if arms_root.name == "arms" else arms_root
    artifacts_flat = Path(artifacts_flat)

    if artifacts_fig_dir is None:
        artifacts_fig_dir = artifacts_flat / "figures"
    artifacts_fig_dir = Path(artifacts_fig_dir)
    artifacts_fig_dir.mkdir(parents=True, exist_ok=True)

    if lake_panel is None:
        lake_panel = Path("lake/panels/dh_cross_section.parquet")
    lake_panel = Path(lake_panel)

    entries = render_core_suite(
        arms_root=arms_root,
        out_dir=out_dir,
        artifacts_flat=artifacts_flat,
        lake_panel=lake_panel,
    )

    for entry in entries:
        if entry.get("status") != "written":
            continue
        src_paths = {k: entry[k] for k in ("png", "pdf") if k in entry}
        mirrored = mirror_files(src_paths, artifacts_fig_dir)
        entry["latex"] = dict(src_paths)
        entry["artifacts"] = mirrored
        for k, v in src_paths.items():
            entry[k] = v

    manifest: dict[str, Any] = {
        "schema": "mascotrl.figure_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms_root": str(arms_root),
        "out_dir": str(out_dir),
        "artifacts_fig_dir": str(artifacts_fig_dir),
        "figures": entries,
    }

    for dest in (out_dir / "figure_manifest.json", artifacts_fig_dir / "figure_manifest.json"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    return manifest
