"""Zero-action collapse detector (measurable Buehler soft-fee failure mode)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def collapse_guard(
    turnovers: Sequence[float] | np.ndarray,
    *,
    action_l1: Sequence[float] | np.ndarray | None = None,
    residual_vs_bs: Sequence[float] | np.ndarray | None = None,
    turnover_floor: float = 1e-4,
    action_dispersion_floor: float = 1e-6,
    residual_vs_bs_floor: float = 1e-3,
) -> dict[str, Any]:
    """Fail closed if mean turnover, action dispersion, or residual-vs-BS collapses."""
    t = np.asarray(turnovers, dtype=np.float64).reshape(-1)
    mean_to = float(np.nanmean(t)) if t.size else 0.0
    failures: list[str] = []
    if not (mean_to == mean_to) or mean_to < float(turnover_floor):
        failures.append(f"turnover_below_floor (mean={mean_to}, floor={turnover_floor})")

    disp = float("nan")
    if action_l1 is not None:
        a = np.asarray(action_l1, dtype=np.float64).reshape(-1)
        disp = float(np.nanstd(a)) if a.size else 0.0
        if not (disp == disp) or disp < float(action_dispersion_floor):
            failures.append(
                f"action_dispersion_below_floor (std={disp}, floor={action_dispersion_floor})"
            )

    resid_disp = float("nan")
    if residual_vs_bs is not None:
        r = np.asarray(residual_vs_bs, dtype=np.float64).reshape(-1)
        resid_disp = float(np.nanstd(r)) if r.size else 0.0
        if not (resid_disp == resid_disp) or resid_disp < float(residual_vs_bs_floor):
            failures.append("residual_collapsed_to_bs")

    return {
        "mean_turnover": mean_to,
        "action_dispersion": disp,
        "residual_vs_bs_dispersion": resid_disp,
        "turnover_floor": float(turnover_floor),
        "action_dispersion_floor": float(action_dispersion_floor),
        "residual_vs_bs_floor": float(residual_vs_bs_floor),
        "collapse_detected": bool(failures),
        "collapse_failures": failures,
        "ok": not failures,
    }


def assert_collapse_guard_ok(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise if collapse_detected is true."""
    if bool(report.get("collapse_detected")):
        fails = report.get("collapse_failures") or ["collapse"]
        raise ValueError(f"zero-action collapse detected: {fails}")
    return report


def equal_weight_collapse_guard(
    weights: Sequence[Sequence[float]] | np.ndarray,
    *,
    l1_vs_ew_floor: float = 0.05,
) -> dict[str, Any]:
    """Fail closed if a policy's holdings never leave the equal-weight point.

    A policy that always emits ``w = 1/K`` produces credit-assignment-free
    "alpha" that is really just the equal-weight benchmark relabeled. For
    each rebalance row this computes the Herfindahl-Hirschman Index (HHI),
    the L1 distance to equal-weight (``1/K``), and the max weight; the mean
    L1-vs-EW distance across rows must clear ``l1_vs_ew_floor`` or the run
    is flagged as collapsed.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        raise ValueError(f"weights must be a non-empty (T, K) array; got shape {w.shape}")
    k = int(w.shape[1])
    ew = np.full(k, 1.0 / k, dtype=np.float64)

    hhi = np.sum(w * w, axis=1)
    l1_vs_ew = np.sum(np.abs(w - ew[None, :]), axis=1)
    max_weight = np.max(np.abs(w), axis=1)

    mean_hhi = float(np.nanmean(hhi))
    mean_l1_vs_ew = float(np.nanmean(l1_vs_ew))
    mean_max_weight = float(np.nanmean(max_weight))

    failures: list[str] = []
    if not (mean_l1_vs_ew == mean_l1_vs_ew) or mean_l1_vs_ew < float(l1_vs_ew_floor):
        failures.append(
            f"l1_vs_ew_below_floor (mean={mean_l1_vs_ew}, floor={l1_vs_ew_floor})"
        )

    return {
        "mean_hhi": mean_hhi,
        "mean_l1_vs_ew": mean_l1_vs_ew,
        "mean_max_weight": mean_max_weight,
        "l1_vs_ew_floor": float(l1_vs_ew_floor),
        "collapse_detected": bool(failures),
        "failures": failures,
        "ok": not failures,
    }
