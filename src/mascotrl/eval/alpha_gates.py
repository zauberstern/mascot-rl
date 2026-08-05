"""Alpha claim gates vs rich BASELINE_NAMES peers only.

Zero and random are not peers. delta-hedged option allocator next: friction XOR (critique 03) + pack
hygiene (07).
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Nonsense comparators — never gate or SPA against these.
NONSENSE_PEERS = frozenset({"zero", "random", "happo", "happo_gross"})


def pick_best_baseline(
    baselines: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return (name, summary) for the richest mean_pnl peer in suite summary."""
    if not isinstance(baselines, dict):
        return None, None
    summary = baselines.get("summary") or {}
    if not isinstance(summary, dict) or not summary:
        return None, None
    best_name: str | None = None
    best_mu = float("-inf")
    best_sm: dict[str, Any] | None = None
    for name, sm in summary.items():
        if not isinstance(sm, dict):
            continue
        if str(name) in NONSENSE_PEERS:
            continue
        mu = float(sm.get("mean_pnl", float("nan")))
        if np.isfinite(mu) and mu > best_mu:
            best_mu = mu
            best_name = str(name)
            best_sm = sm
    return best_name, best_sm


def rich_baseline_alpha_ok(
    *,
    happo_mean: float,
    happo_sharpe: float,
    baselines: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Fail-closed: HAPPO mean>0 and beats best academic baseline on mean+Sharpe."""
    meta: dict[str, Any] = {
        "baselines_available": False,
        "alpha_pending_baselines": True,
        "best_baseline": None,
        "edge_vs_best_baseline": float("nan"),
        "sharpe_beats_best_baseline": False,
        "positive_mean": False,
    }
    positive = np.isfinite(happo_mean) and happo_mean > 0.0
    meta["positive_mean"] = bool(positive)
    best_name, best_sm = pick_best_baseline(baselines)
    if best_name is None or best_sm is None:
        return False, meta
    meta["baselines_available"] = True
    meta["alpha_pending_baselines"] = False
    meta["best_baseline"] = best_name
    best_mu = float(best_sm.get("mean_pnl", float("nan")))
    best_sh = float(best_sm.get("sharpe", float("nan")))
    edge = (
        float(happo_mean - best_mu)
        if np.isfinite(happo_mean) and np.isfinite(best_mu)
        else float("nan")
    )
    meta["edge_vs_best_baseline"] = edge
    sharpe_ok = (
        np.isfinite(happo_sharpe)
        and np.isfinite(best_sh)
        and float(happo_sharpe) > float(best_sh)
    )
    meta["sharpe_beats_best_baseline"] = bool(sharpe_ok)
    mean_ok = np.isfinite(edge) and edge > 0.0
    ok = bool(positive and mean_ok and sharpe_ok)
    return ok, meta


def apply_rich_baseline_alpha_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Stamp report + historical_oos alpha from rich baselines only."""
    ho = report.get("historical_oos")
    if not isinstance(ho, dict):
        ho = {}
        report["historical_oos"] = ho
    summary = ho.get("summary") or {}
    happo = summary.get("happo") or {}
    ok, meta = rich_baseline_alpha_ok(
        happo_mean=float(happo.get("mean_pnl", float("nan"))),
        happo_sharpe=float(happo.get("sharpe", float("nan"))),
        baselines=report.get("baselines"),
    )
    report["best_baseline"] = meta["best_baseline"]
    report["edge_vs_best_baseline"] = meta["edge_vs_best_baseline"]
    for dead in ("edge_vs_random", "edge_vs_zero"):
        report.pop(dead, None)
    ho["alpha_found_historical"] = bool(ok)
    ho["alpha_pending_baselines"] = bool(meta["alpha_pending_baselines"])
    ho["baselines_available"] = bool(meta["baselines_available"])
    ho["best_baseline"] = meta["best_baseline"]
    ho["edge_vs_best_baseline"] = meta["edge_vs_best_baseline"]
    ho["sharpe_beats_best_baseline"] = bool(meta["sharpe_beats_best_baseline"])
    ho["positive_mean"] = bool(meta["positive_mean"])
    for dead in (
        "sharpe_beats_random",
        "edge_vs_random",
        "sharpe_edge_vs_random",
        "edge_vs_zero",
        "beats_zero_and_positive",
        "beats_sanity_legs",
    ):
        ho.pop(dead, None)
    # Drop nonsense peers from attached hist summary if a legacy run left them.
    if isinstance(summary, dict):
        for dead in ("zero", "random"):
            summary.pop(dead, None)
    ho["alpha_claim_status"] = (
        "claim requires beating the best rich BASELINE_NAMES peer on mean AND "
        "Sharpe (fail-closed until baselines stamped), plus factor-adjusted "
        "after-cost alpha and DSR when stamped. Zero/random are not peers."
    )
    report["alpha_found"] = bool(ok)
    return report
