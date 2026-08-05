"""Policy diagnostics: HHI/L1-vs-EW/max-weight/turnover/entropy in one report.

Rolls up per-rebalance weight geometry with the collapse guards so a
campaign artifact can carry a single ``policy_diagnostics`` block instead of
scattering ad hoc concentration stats across the report.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mascotrl.eval.collapse_guard import collapse_guard, equal_weight_collapse_guard


def summarize_policy_diagnostics(
    *,
    weights: np.ndarray,
    turnovers: Sequence[float] | None = None,
    entropies: Sequence[float] | None = None,
    log_std_mean: float | None = None,
    logit_std: float | None = None,
) -> dict[str, Any]:
    """Summarize a policy's realized weight rows plus optional training stats.

    ``weights`` is ``(T, K)`` (or a list of equal-length weight vectors).
    ``turnovers``/``entropies`` are optional per-step series; when omitted
    the corresponding mean is NaN rather than raising, so this can be
    called with only the weights available (e.g. eval-only rollouts).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        raise ValueError(f"weights must be a non-empty (T, K) array; got shape {w.shape}")

    hhi = np.sum(w * w, axis=1)
    k = int(w.shape[1])
    ew = np.full(k, 1.0 / k, dtype=np.float64)
    l1_vs_ew = np.sum(np.abs(w - ew[None, :]), axis=1)
    max_weight = np.max(np.abs(w), axis=1)

    turnover_mean = float("nan")
    if turnovers is not None:
        t = np.asarray(turnovers, dtype=np.float64).reshape(-1)
        turnover_mean = float(np.nanmean(t)) if t.size else float("nan")

    entropy_mean = float("nan")
    if entropies is not None:
        e = np.asarray(entropies, dtype=np.float64).reshape(-1)
        entropy_mean = float(np.nanmean(e)) if e.size else float("nan")

    cg = collapse_guard(t if turnovers is not None else np.asarray([]))
    ewcg = equal_weight_collapse_guard(w)

    return {
        "hhi_mean": float(np.nanmean(hhi)),
        "hhi_max": float(np.nanmax(hhi)),
        "l1_vs_ew_mean": float(np.nanmean(l1_vs_ew)),
        "l1_vs_ew_max": float(np.nanmax(l1_vs_ew)),
        "max_weight_mean": float(np.nanmean(max_weight)),
        "max_weight_max": float(np.nanmax(max_weight)),
        "turnover_mean": turnover_mean,
        "entropy_mean": entropy_mean,
        "entropy_series": e.tolist() if entropies is not None else [],
        "log_std_mean": None if log_std_mean is None else float(log_std_mean),
        "logit_std": None if logit_std is None else float(logit_std),
        "collapse_guard": cg,
        "equal_weight_collapse_guard": ewcg,
        "equal_weight_collapse_detected": bool(ewcg["collapse_detected"]),
    }


def _ann_sharpe(x: np.ndarray) -> float:
    r = np.asarray(x, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=0))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(252.0))


def selection_vs_sizing_attribution(
    policy_returns: np.ndarray | Sequence[float],
    ew_crucible_returns: np.ndarray | Sequence[float],
    ew_parent_returns: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Decompose policy - parent EW into name-set + sizing + interaction.

    Name-set effect: EW(crucible) - EW(parent).
    Sizing effect: policy - EW(crucible).
    Total: policy - EW(parent).
    Interaction is the residual so the three legs sum to total within 1e-9
    for both mean return and Sharpe-of-level differences.
    """
    pol = np.asarray(policy_returns, dtype=np.float64).reshape(-1)
    ew_c = np.asarray(ew_crucible_returns, dtype=np.float64).reshape(-1)
    ew_p = np.asarray(ew_parent_returns, dtype=np.float64).reshape(-1)
    n = min(pol.size, ew_c.size, ew_p.size)
    pol, ew_c, ew_p = pol[:n], ew_c[:n], ew_p[:n]

    name_set_mean = float(np.mean(ew_c - ew_p))
    sizing_mean = float(np.mean(pol - ew_c))
    total_mean = float(np.mean(pol - ew_p))
    interaction_mean = float(total_mean - name_set_mean - sizing_mean)

    sharpe_pol = _ann_sharpe(pol)
    sharpe_c = _ann_sharpe(ew_c)
    sharpe_p = _ann_sharpe(ew_p)
    name_set_sharpe = float(sharpe_c - sharpe_p)
    sizing_sharpe = float(sharpe_pol - sharpe_c)
    total_sharpe = float(sharpe_pol - sharpe_p)
    interaction_sharpe = float(total_sharpe - name_set_sharpe - sizing_sharpe)

    return {
        "name_set_mean": name_set_mean,
        "sizing_mean": sizing_mean,
        "interaction_mean": interaction_mean,
        "total_mean": total_mean,
        "name_set_sharpe": name_set_sharpe,
        "sizing_sharpe": sizing_sharpe,
        "interaction_sharpe": interaction_sharpe,
        "total_sharpe": total_sharpe,
    }
