"""Statistics helpers for equity allocation book sections."""
from __future__ import annotations

from typing import Any

import numpy as np

def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _annualized_sharpe(r: np.ndarray, periods: int = 252) -> float:
    from mascotrl.eval.stats_rigor import annualized_sharpe

    return float(annualized_sharpe(np.asarray(r, dtype=np.float64)))



def _hac_ols_full(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Newey-West HAC OLS for every coefficient (intercept + factors).

    Same Bartlett-kernel sandwich methodology as
    ``src.eval.signal_gate.ff_alpha``, generalized to report every
    coefficient's t-stat (not only the intercept).
    """
    from mascotrl.eval.stats_inference import newey_west_lag

    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    yy, xx = yy[mask], xx[mask]
    n = int(yy.size)
    p = int(xx.shape[1]) + 1
    if n < p + 2:
        return {"n": n, "coef": None, "t_stat": None}
    design = np.column_stack([np.ones(n), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    l_bw = int(newey_west_lag(n))
    scores = design * resid[:, None]
    s_mat = scores.T @ scores
    for j in range(1, l_bw + 1):
        if j >= n:
            break
        w = 1.0 - j / (l_bw + 1.0)
        gamma_j = scores[j:].T @ scores[:-j]
        s_mat = s_mat + w * (gamma_j + gamma_j.T)
    cov = xtx_inv @ s_mat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 1e-12)
    return {"n": n, "coef": beta, "t_stat": t_stat, "lags": l_bw}

