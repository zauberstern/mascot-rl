"""Reward-to-noise diagnostic: max feasible EW tilt vs reward std (RC6)."""
from __future__ import annotations

import numpy as np


def concentrated_vs_ew_gap(
    returns: np.ndarray,
    *,
    turnover_limit: float,
) -> float:
    """Mean |r_t · (w_concentrated - w_ew)| under a one-step L1 budget.

    From equal weight, the largest long-only L1 move of size ``turnover_limit``
    puts ``tau/2`` onto the day's best name and takes ``tau/2`` from the worst.
    That is the max feasible deviation from EW in one rebalance.
    """
    rets = np.asarray(returns, dtype=np.float64)
    if rets.ndim != 2 or rets.shape[0] == 0 or rets.shape[1] == 0:
        return float("nan")
    t_n, k = rets.shape
    if k < 2:
        return 0.0
    tau = max(float(turnover_limit), 0.0)
    delta = min(tau / 2.0, 1.0 - 1.0 / float(k))
    if delta <= 0.0:
        return 0.0
    # Per day: |delta * (r_best - r_worst)|
    r_best = np.nanmax(rets, axis=1)
    r_worst = np.nanmin(rets, axis=1)
    gaps = np.abs(delta * (r_best - r_worst))
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return float("nan")
    return float(np.mean(gaps))


def reward_to_noise_diagnostic(
    returns: np.ndarray,
    daily_rewards: np.ndarray,
    *,
    turnover_limit: float,
) -> dict[str, float | str | bool]:
    """Compare concentrated-vs-EW edge to reward noise.

    If ``signal_to_noise < 1``, the max feasible tilt is smaller than one
    standard deviation of the observed reward series: likely unlearnable.
    """
    rew = np.asarray(daily_rewards, dtype=np.float64).reshape(-1)
    rew = rew[np.isfinite(rew)]
    reward_std = float(np.std(rew)) if rew.size else float("nan")
    gap = concentrated_vs_ew_gap(returns, turnover_limit=turnover_limit)
    snr = float(gap / (reward_std + 1e-8)) if np.isfinite(gap) else float("nan")
    out: dict[str, float | str | bool] = {
        "reward_std": reward_std,
        "reward_concentrated_vs_ew_gap": gap,
        "reward_signal_to_noise": snr,
        "reward_unlearnable": bool(np.isfinite(snr) and snr < 1.0),
    }
    if out["reward_unlearnable"]:
        out["reward_noise_warning"] = (
            "reward_concentrated_vs_ew_gap < 1*reward_std: "
            "max feasible tilt may be unlearnable under residual noise"
        )
    return out
