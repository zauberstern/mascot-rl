"""Equity allocation book sections 6-10 (signals through limitations)."""
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
    _new_fig,
    _write_table,
)

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

