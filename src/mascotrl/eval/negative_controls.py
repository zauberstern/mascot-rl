"""Negative controls for surface-alpha campaign (fail-closed).

Controls score a *signal pathway* (not the RL policy by default). The
headline estimand for the verdict is beta-free: long-short demeaned weights
scored on residual PnL, compared to an identically built uncorrupted arm via
a degradation ratio. Absolute Sharpe floors on long-only ``total_net`` are
kept only as continuity fields — market beta alone can breach them.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def shuffled_return_rows(
    returns: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Permute return rows across time (destroys temporal alignment)."""
    r = np.asarray(returns, dtype=np.float64).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(r, axis=0)
    return r


# Backward-compatible alias (name historically implied "labels").
shuffled_label_control = shuffled_return_rows


def permute_signals_across_names(
    signals: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Permute each signal's columns (names) independently."""
    rng = np.random.default_rng(int(seed))
    out: dict[str, np.ndarray] = {}
    for name, arr in signals.items():
        a = np.asarray(arr, dtype=np.float64).copy()
        if a.ndim != 2:
            out[str(name)] = a
            continue
        perm = rng.permutation(a.shape[1])
        out[str(name)] = a[:, perm]
    return out


def date_shift_signals(
    signals: Mapping[str, np.ndarray],
    *,
    shift: int = 21,
) -> dict[str, np.ndarray]:
    """Shift signals forward by ``shift`` days (look-ahead contamination control).

    A pipeline that still shows edge after a large positive shift is leaking.
    """
    out: dict[str, np.ndarray] = {}
    s = int(shift)
    for name, arr in signals.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2:
            out[str(name)] = a
            continue
        z = np.full_like(a, np.nan)
        if s >= 0:
            if s < a.shape[0]:
                z[s:] = a[: a.shape[0] - s]
        else:
            ss = -s
            if ss < a.shape[0]:
                z[: a.shape[0] - ss] = a[ss:]
        out[str(name)] = z
    return out


def negative_control_should_fail(
    control_sharpe: float,
    *,
    max_abs_sharpe: float = 0.5,
) -> bool:
    """Return True if an absolute |Sharpe| floor is breached (legacy path)."""
    try:
        s = float(control_sharpe)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(s):
        return False
    return abs(s) > float(max_abs_sharpe)


def degradation_ratio(
    control_sharpe: float,
    *,
    clean_sharpe: float,
) -> float:
    """``|control| / max(|clean|, eps)`` — near 1 means corruption did nothing."""
    try:
        c = abs(float(control_sharpe))
        u = abs(float(clean_sharpe))
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(c) and np.isfinite(u)):
        return float("nan")
    return float(c / max(u, 1e-8))


def degradation_should_fail(
    control_sharpe: float,
    *,
    clean_sharpe: float,
    max_degradation_ratio: float = 0.5,
    min_clean_abs_sharpe: float = 0.2,
) -> bool:
    """Fail closed when corruption fails to destroy a clean residual edge.

    If the clean arm itself has negligible |Sharpe|, the ratio is undefined
    for a fail-closed verdict — treat as non-failure (nothing to destroy).
    """
    try:
        clean = abs(float(clean_sharpe))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(clean) or clean < float(min_clean_abs_sharpe):
        return False
    ratio = degradation_ratio(control_sharpe, clean_sharpe=clean_sharpe)
    if not np.isfinite(ratio):
        return False
    return ratio > float(max_degradation_ratio)


def policy_level_negative_control_stamp(
    *,
    control_sharpe: float,
    clean_sharpe: float,
    seed: int,
    fold_id: int,
    max_degradation_ratio: float = 0.5,
    min_clean_abs_sharpe: float = 0.2,
) -> dict[str, Any]:
    """Stamp one policy rerun using the shared degradation-ratio verdict."""
    ratio = degradation_ratio(control_sharpe, clean_sharpe=clean_sharpe)
    return {
        "sharpe": float(control_sharpe),
        "seed": int(seed),
        "fold_id": int(fold_id),
        "degradation_ratio": float(ratio) if np.isfinite(ratio) else None,
        "failed": degradation_should_fail(
            control_sharpe,
            clean_sharpe=clean_sharpe,
            max_degradation_ratio=max_degradation_ratio,
            min_clean_abs_sharpe=min_clean_abs_sharpe,
        ),
    }


def run_negative_controls(
    *,
    control_sharpe_on_shuffled: float | None = None,
    control_sharpe_on_permuted_signals: float | None = None,
    control_sharpe_on_date_shifted: float | None = None,
    clean_sharpe: float | None = None,
    max_abs_sharpe: float = 0.5,
    max_degradation_ratio: float = 0.5,
    min_clean_abs_sharpe: float = 0.2,
    # Backward-compatible aliases (misleading: no policy is re-run).
    policy_sharpe_on_shuffled: float | None = None,
    policy_sharpe_on_permuted_signals: float | None = None,
    policy_sharpe_on_date_shifted: float | None = None,
) -> dict[str, Any]:
    """Aggregate fail-closed verdict for the three mandatory controls.

    When ``clean_sharpe`` is provided, the primary verdict uses degradation
    ratios on the beta-free estimand. Absolute floors remain as continuity
    fields under each check's ``abs_floor_failed`` key.
    """
    sh = (
        control_sharpe_on_shuffled
        if control_sharpe_on_shuffled is not None
        else policy_sharpe_on_shuffled
    )
    perm = (
        control_sharpe_on_permuted_signals
        if control_sharpe_on_permuted_signals is not None
        else policy_sharpe_on_permuted_signals
    )
    shift = (
        control_sharpe_on_date_shifted
        if control_sharpe_on_date_shifted is not None
        else policy_sharpe_on_date_shifted
    )
    if sh is None or perm is None or shift is None:
        raise TypeError(
            "run_negative_controls requires shuffled, permuted, and date-shifted "
            "control Sharpes (control_sharpe_* or legacy policy_sharpe_*)"
        )

    use_degradation = clean_sharpe is not None and np.isfinite(float(clean_sharpe))

    def _check(name: str, value: float) -> dict[str, Any]:
        abs_failed = negative_control_should_fail(value, max_abs_sharpe=max_abs_sharpe)
        entry: dict[str, Any] = {
            "sharpe": float(value),
            "abs_floor_failed": bool(abs_failed),
        }
        if use_degradation:
            ratio = degradation_ratio(value, clean_sharpe=float(clean_sharpe))
            deg_failed = degradation_should_fail(
                value,
                clean_sharpe=float(clean_sharpe),
                max_degradation_ratio=max_degradation_ratio,
                min_clean_abs_sharpe=min_clean_abs_sharpe,
            )
            entry["degradation_ratio"] = float(ratio) if np.isfinite(ratio) else None
            entry["failed"] = bool(deg_failed)
        else:
            entry["failed"] = bool(abs_failed)
        return entry

    checks = {
        "shuffled_labels": _check("shuffled_labels", float(sh)),
        "permuted_signals": _check("permuted_signals", float(perm)),
        "date_shifted_signals": _check("date_shifted_signals", float(shift)),
    }
    any_fail = any(v["failed"] for v in checks.values())
    out: dict[str, Any] = {
        "checks": checks,
        "pipeline_broken": bool(any_fail),
        "max_abs_sharpe": float(max_abs_sharpe),
        "verdict_mode": "degradation_ratio" if use_degradation else "abs_floor",
    }
    if use_degradation:
        out["clean_sharpe"] = float(clean_sharpe)
        out["max_degradation_ratio"] = float(max_degradation_ratio)
        out["min_clean_abs_sharpe"] = float(min_clean_abs_sharpe)
    return out
