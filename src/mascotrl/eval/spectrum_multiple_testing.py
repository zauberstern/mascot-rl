"""Spectrum multiple-testing helpers: Romano-Wolf stepdown, MDE, trial counts."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def paired_mde(sigma_d: float, n: int) -> float:
    """Two-sided 5% test at 80% power: Delta_min = 2.802 * sigma_d / sqrt(n)."""
    n_i = int(n)
    if n_i <= 0:
        return float("nan")
    s = float(sigma_d)
    if not (s == s) or s < 0:
        return float("nan")
    return 2.802 * s / float(np.sqrt(n_i))


def n_trials_breakdown(
    n_cells: int,
    n_seeds: int,
    n_cost_rungs: int,
) -> dict[str, Any]:
    """DSR trial-count ledger for a spectrum campaign."""
    c = max(0, int(n_cells))
    s = max(0, int(n_seeds))
    r = max(0, int(n_cost_rungs))
    n_trials = c * s * r
    return {
        "n_cells": c,
        "cells": c,  # B-DSR: capital gate also looks for ``cells``
        "n_seeds": s,
        "n_cost_rungs": r,
        "n_trials": int(n_trials),
        "formula": "n_cells * n_seeds * n_cost_rungs",
        "source": "spectrum_campaign_ledger",
    }


def _stationary_bootstrap_indices(
    n: int,
    *,
    rng: np.random.Generator,
    block_mean: int = 10,
) -> np.ndarray:
    """Politis–Romano stationary bootstrap index path of length n."""
    p_block = 1.0 / max(1, int(block_mean))
    idx = np.empty(n, dtype=int)
    i = int(rng.integers(0, n))
    for j in range(n):
        idx[j] = i
        if rng.random() < p_block:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
    return idx


def romano_wolf_stepdown(
    diffs_by_cell: Mapping[str, Sequence[float]],
    block_indices: Sequence[int] | None = None,
    *,
    n_boot: int = 200,
    block_mean: int = 10,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Simplified Romano–Wolf stepdown on cell-minus-reference differences.

    ``diffs_by_cell`` maps cell_id -> paired per-episode (or per-path) differences
    vs the reference cell. Uses a stationary bootstrap of the aligned series.
    Returns adjusted p-values per cell (family-wise error control).
    """
    names = [k for k, v in diffs_by_cell.items() if len(list(v)) > 0]
    if not names:
        return {
            "protocol": "romano_wolf_stepdown_spectrum",
            "citation": "Romano and Wolf (2005, Econometrica 73(4))",
            "reason": "no non-empty cell difference series",
            "adjusted_pvalues": {},
            "rejected": [],
            "results": [],
            "n_boot": int(n_boot),
        }

    series = {k: np.asarray(list(diffs_by_cell[k]), dtype=np.float64).reshape(-1) for k in names}
    # Align to common length (truncate to min) for joint bootstrap.
    n = int(min(v.size for v in series.values()))
    if block_indices is not None:
        bi = np.asarray(list(block_indices), dtype=int).reshape(-1)
        if bi.size > 0:
            n = min(n, int(bi.max()) + 1) if bi.size else n
    if n < 2:
        return {
            "protocol": "romano_wolf_stepdown_spectrum",
            "citation": "Romano and Wolf (2005, Econometrica 73(4))",
            "reason": "need >=2 aligned observations",
            "adjusted_pvalues": {k: 1.0 for k in names},
            "rejected": [],
            "results": [],
            "n_boot": int(n_boot),
        }

    for k in names:
        series[k] = series[k][:n]

    obs_mean = {k: float(series[k].mean()) for k in names}
    # Studentize by SE of mean.
    obs_t: dict[str, float] = {}
    for k in names:
        se = float(series[k].std(ddof=1) / np.sqrt(n))
        obs_t[k] = obs_mean[k] / se if se > 0 else (np.inf if obs_mean[k] > 0 else -np.inf)

    rng = np.random.default_rng(int(seed))
    boot: dict[str, np.ndarray] = {k: np.empty(int(n_boot)) for k in names}
    for b in range(int(n_boot)):
        idx = _stationary_bootstrap_indices(n, rng=rng, block_mean=block_mean)
        for k in names:
            d = series[k][idx]
            mu = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(n))
            centre = obs_mean[k]
            boot[k][b] = (mu - centre) / se if se > 0 else 0.0

    remaining = sorted(names, key=lambda k: obs_t[k], reverse=True)
    rejected: list[str] = []
    results: list[dict[str, Any]] = []
    adjusted: dict[str, float] = {}
    step = 0
    while remaining:
        step += 1
        mat = np.vstack([boot[k] for k in remaining])
        max_stat = mat.max(axis=0)
        crit = float(np.quantile(max_stat, 1.0 - float(alpha)))
        top = remaining[0]
        p_adj = float(np.mean(max_stat >= obs_t[top])) if np.isfinite(obs_t[top]) else 1.0
        # Monotone stepdown: later p cannot fall below earlier.
        if adjusted:
            prev = max(adjusted.values())
            p_adj = max(p_adj, prev)
        adjusted[top] = p_adj
        row = {
            "cell": top,
            "step": step,
            "t_stat": obs_t[top],
            "mean_diff": obs_mean[top],
            "critical_value": crit,
            "p_adjusted": p_adj,
            "rejected": bool(np.isfinite(obs_t[top]) and obs_t[top] > crit),
        }
        results.append(row)
        if row["rejected"]:
            rejected.append(top)
            remaining = remaining[1:]
            continue
        for k in remaining[1:]:
            p_k = float(np.mean(max_stat >= obs_t[k])) if np.isfinite(obs_t[k]) else 1.0
            p_k = max(p_k, p_adj)
            adjusted[k] = p_k
            results.append(
                {
                    "cell": k,
                    "step": step,
                    "t_stat": obs_t[k],
                    "mean_diff": obs_mean[k],
                    "critical_value": crit,
                    "p_adjusted": p_k,
                    "rejected": False,
                }
            )
        break

    return {
        "protocol": "romano_wolf_stepdown_spectrum",
        "citation": "Romano and Wolf (2005, Econometrica 73(4))",
        "h0_statement": "H_k: cell_k does not beat reference (mean diff <= 0)",
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "bootstrap": "stationary (Politis and Romano 1994)",
        "n": n,
        "n_cells": len(names),
        "adjusted_pvalues": adjusted,
        "rejected": rejected,
        "n_rejected": len(rejected),
        "results": results,
    }
