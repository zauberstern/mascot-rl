"""Lag-1 sleeping Variable-Share over calm/turb copies of each expert.

Wake signal is turbulent[t-1] (operational seal mask). Same-day turb[t] and
R[t] never affect W[t]. Desk return uses sum of calm+turb mass per base expert.
"""
from __future__ import annotations

import numpy as np

from src.eval.expert_losses import expanding_unit_interval, log_wealth_loss


def expand_sleeping_returns(
    R: np.ndarray,
    turb_lag1: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build (T, 2n) panel: calm copies then turb copies (names only).

    Returns are identical for both copies; wake is handled in the mixer.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("R must be (T, n)")
    t_len, n = R.shape
    names = [f"e{i}_calm" for i in range(n)] + [f"e{i}_turb" for i in range(n)]
    R_sleep = np.concatenate([R, R], axis=1)
    _ = turb_lag1  # wake applied in mixer
    return R_sleep, names


def variable_share_sleeping(
    returns: np.ndarray,
    turb: np.ndarray,
    *,
    alpha: float,
    eta: float = 0.5,
) -> np.ndarray:
    """Lag-1 sleeping Variable-Share; returns base weights (T, n) = calm+turb.

    turb[t] is operational; wake at t uses turb[t-1] (t=0: all awake).
    Sleeping coords: skip loss update (mass frozen). Share only among awake.
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    turb_arr = np.asarray(turb, dtype=bool).reshape(-1)
    t_len, n = R.shape
    if turb_arr.shape[0] != t_len:
        raise ValueError("turb length must match T")
    if n < 2:
        raise ValueError("need at least 2 experts")
    a = float(alpha)
    e = float(eta)
    # 2n copies: [0:n) calm, [n:2n) turb
    m = 2 * n
    w = np.ones(m, dtype=np.float64) / m
    hist_base = np.zeros((t_len, n), dtype=np.float64)

    ell = log_wealth_loss(R)
    # Expand losses for both copies (same ell); scale causally on stacked
    ell_stack = np.concatenate([ell, ell], axis=1)
    L01 = expanding_unit_interval(ell_stack)

    for t in range(t_len):
        hist_base[t] = w[:n] + w[n:]
        # Wake: t=0 all awake; else calm awake if not turb[t-1], turb if turb[t-1]
        if t == 0:
            awake = np.ones(m, dtype=bool)
        else:
            is_turb = bool(turb_arr[t - 1])
            awake = np.zeros(m, dtype=bool)
            if is_turb:
                awake[n:] = True
            else:
                awake[:n] = True

        lt = L01[t]
        wm = w.copy()
        # Hedge only awake
        awake_idx = np.where(awake)[0]
        sleep_idx = np.where(~awake)[0]
        if awake_idx.size > 0:
            wm[awake_idx] = w[awake_idx] * np.exp(-e * lt[awake_idx])
        # Variable-Share only among awake (Herbster Fig.1 on awake subset)
        if awake_idx.size >= 2:
            wa = wm[awake_idx]
            la = lt[awake_idx]
            one_m_a = 1.0 - a
            share_frac = 1.0 - np.power(one_m_a, la)
            keep = np.power(one_m_a, la)
            pool = float(np.dot(share_frac, wa))
            na = int(awake_idx.size)
            wa_new = keep * wa + (pool - share_frac * wa) / (na - 1)
            # Sleeping mass frozen
            sleep_mass = float(wm[sleep_idx].sum()) if sleep_idx.size else 0.0
            awake_mass = float(wa_new.sum())
            total = awake_mass + sleep_mass
            if total <= 0.0 or not np.isfinite(total):
                w = np.ones(m, dtype=np.float64) / m
            else:
                w = wm.copy()
                w[awake_idx] = wa_new
                # renormalize preserving relative sleep vs awake blocks
                w = w / total
        elif awake_idx.size == 1:
            # Single awake: all tradable mass on that copy; sleep frozen then renorm
            sleep_mass = float(wm[sleep_idx].sum()) if sleep_idx.size else 0.0
            wm[awake_idx[0]] = max(1.0 - sleep_mass, 0.0)
            s = float(wm.sum())
            w = wm / s if s > 0 else np.ones(m) / m
        else:
            # No awake (should not happen): leave frozen
            s = float(wm.sum())
            w = wm / s if s > 0 else np.ones(m) / m

    return hist_base
