"""Performance-sleeping Variable-Share: wake on trailing mean return > 0."""
from __future__ import annotations

import numpy as np

from src.eval.expert_losses import expanding_unit_interval, log_wealth_loss
from src.eval.onehot_leader import onehot


def performance_sleeping(
    returns: np.ndarray,
    *,
    alpha: float,
    lookback: int = 126,
    eta: float = 0.5,
    min_obs: int = 20,
    owl_index: int | None = None,
) -> np.ndarray:
    """VS among experts with trailing mean > 0; else Owl one-hot or uniform."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = R.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    a = float(alpha)
    e = float(eta)
    lb = int(lookback)
    w = np.ones(n, dtype=np.float64) / n
    hist = np.zeros((t_len, n), dtype=np.float64)
    L01 = expanding_unit_interval(log_wealth_loss(R))
    uniform = np.ones(n, dtype=np.float64) / n

    for t in range(t_len):
        hist[t] = w
        if t == 0:
            awake = np.ones(n, dtype=bool)
        else:
            start = max(0, t - lb)
            awake = np.zeros(n, dtype=bool)
            for i in range(n):
                col = R[start:t, i]
                finite = col[np.isfinite(col)]
                awake[i] = finite.size >= int(min_obs) and float(np.mean(finite)) > 0.0

        if not awake.any():
            if owl_index is not None and 0 <= int(owl_index) < n:
                w = onehot(n, int(owl_index))
            else:
                w = uniform.copy()
            continue

        lt = L01[t]
        wm = w.copy()
        awake_idx = np.where(awake)[0]
        sleep_idx = np.where(~awake)[0]
        wm[awake_idx] = w[awake_idx] * np.exp(-e * lt[awake_idx])
        if awake_idx.size >= 2:
            wa = wm[awake_idx]
            la = lt[awake_idx]
            one_m_a = 1.0 - a
            share_frac = 1.0 - np.power(one_m_a, la)
            keep = np.power(one_m_a, la)
            pool = float(np.dot(share_frac, wa))
            na = int(awake_idx.size)
            wa_new = keep * wa + (pool - share_frac * wa) / (na - 1)
            sleep_mass = float(wm[sleep_idx].sum()) if sleep_idx.size else 0.0
            awake_mass = float(wa_new.sum())
            total = awake_mass + sleep_mass
            if total <= 0.0 or not np.isfinite(total):
                w = uniform.copy()
            else:
                w = wm.copy()
                w[awake_idx] = wa_new
                w = w / total
        elif awake_idx.size == 1:
            sleep_mass = float(wm[sleep_idx].sum()) if sleep_idx.size else 0.0
            wm[awake_idx[0]] = max(1.0 - sleep_mass, 0.0)
            s = float(wm.sum())
            w = wm / s if s > 0 else uniform.copy()
        else:
            s = float(wm.sum())
            w = wm / s if s > 0 else uniform.copy()
    return hist
