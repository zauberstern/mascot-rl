"""Page-Hinkley one-hot switcher on log-wealth gap vs average (causal)."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.expert_losses import log_wealth_loss
from mascotrl.eval.onehot_leader import onehot


def page_hinkley_switch(
    returns: np.ndarray,
    names: list[str],
    *,
    delta: float = 1e-4,
    lam: float = 0.02,
) -> np.ndarray:
    """Emit W[t] from incumbent before seeing R[t]; then update PH and maybe switch."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = R.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    if len(names) != n:
        raise ValueError("names length must match n")
    W = np.zeros((t_len, n), dtype=np.float64)
    uniform = np.ones(n, dtype=np.float64) / n
    if t_len == 0:
        return W
    W[0] = uniform
    inc = names.index("owl") if "owl" in names else 0
    m = 0.0
    dlt = float(delta)
    thr = float(lam)
    for t in range(1, t_len):
        W[t] = onehot(n, inc)
        ell_t = log_wealth_loss(R[t : t + 1])[0]
        # positive x => incumbent worse than average (higher loss)
        x = float(ell_t[inc] - np.mean(ell_t))
        m = max(0.0, m + x - dlt)
        if m > thr:
            S = log_wealth_loss(R[: t + 1]).sum(axis=0)
            order = np.argsort(S)
            new_inc = int(order[0])
            if new_inc == inc and n > 1:
                new_inc = int(order[1])
            inc = new_inc
            m = 0.0
    return W
