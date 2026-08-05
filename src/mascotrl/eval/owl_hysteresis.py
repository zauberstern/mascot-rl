"""Owl-core hysteresis: specialists must clear a margin; return to Owl without margin."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.onehot_leader import onehot, trailing_sharpe


def owl_hysteresis(
    returns: np.ndarray,
    names: list[str],
    *,
    lookback: int = 126,
    margin: float = 0.25,
    min_obs: int = 20,
) -> np.ndarray:
    """Default Owl. Specialist takes book only if Sharpe_i > Sharpe_owl + margin."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = R.shape
    if "owl" not in names:
        raise ValueError("owl must be in names")
    if len(names) != n:
        raise ValueError("names length must match n")
    owl_i = names.index("owl")
    lb = int(lookback)
    m = float(margin)
    W = np.zeros((t_len, n), dtype=np.float64)
    uniform = np.ones(n, dtype=np.float64) / n
    incumbent = owl_i
    for t in range(t_len):
        if t < lb:
            W[t] = uniform
            continue
        scores = np.array(
            [
                trailing_sharpe(R[:, j], t - lb, t, min_obs=min_obs)
                for j in range(n)
            ],
            dtype=np.float64,
        )
        s_owl = scores[owl_i]
        if incumbent != owl_i:
            if (
                not np.isfinite(scores[incumbent])
                or not np.isfinite(s_owl)
                or float(scores[incumbent]) < float(s_owl)
            ):
                incumbent = owl_i
            W[t] = onehot(n, incumbent)
            continue
        spec = [j for j in range(n) if j != owl_i and np.isfinite(scores[j])]
        if spec and np.isfinite(s_owl):
            j_star = max(spec, key=lambda j: float(scores[j]))
            if float(scores[j_star]) > float(s_owl) + m:
                incumbent = j_star
        W[t] = onehot(n, incumbent)
    return W
