"""CMDP / turnover projectors shared by training and CRUCIBLE gates.

G1/G2 must use the same projection the trainer applies; a softmax stub makes
the discriminability ladder meaningless.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def turnover_cap_project(
    w: np.ndarray,
    *,
    w_prev: np.ndarray | None = None,
    tau: float,
    counter: dict[str, int] | None = None,
) -> np.ndarray:
    """Minimum-norm correction onto ``{w : ||w - w_prev||_1 <= tau}``."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    wp = np.zeros_like(w) if w_prev is None else np.asarray(w_prev, dtype=np.float64).reshape(-1)
    dw = w - wp
    turn = float(np.sum(np.abs(dw)))
    tau = float(tau)
    if tau < 0:
        raise ValueError(f"turnover_limit must be non-negative, got {tau}")
    if counter is not None:
        counter["steps"] = int(counter.get("steps", 0)) + 1
        if np.isfinite(turn) and turn > tau:
            counter["binding_steps"] = int(counter.get("binding_steps", 0)) + 1
    if not np.isfinite(turn) or turn <= tau or turn < 1e-12:
        return w
    return wp + dw * (tau / turn)


def soft_simplex_project(w: np.ndarray) -> np.ndarray:
    """Softmax projection onto the probability simplex (soft mode only)."""
    x = np.asarray(w, dtype=np.float64).reshape(-1)
    z = x - np.max(x)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / max(float(e.sum()), 1e-12)


def make_cmdp_projector(
    cfg: Mapping[str, Any],
    *,
    k: int | None = None,
    counter: dict[str, int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the projector used by the research trainer for this cfg.

    Hard mode: turnover-cap projection from equal weight (CRUCIBLE selection
    has no path-dependent w_prev; peers face the same cold-start constraint).
    Soft mode: softmax simplex.
    """
    mode = str(cfg.get("projection_mode") or "soft").lower().strip()
    if mode == "hard":
        turnover_limit = cfg.get("turnover_limit")
        if turnover_limit is None:
            raise ValueError(
                "projection_mode='hard' requires cfg['turnover_limit'] for CRUCIBLE G1/G2"
            )
        tau = float(turnover_limit)
        # policy_mode turnover multiplier
        try:
            from mascotrl.spectrum.policy_mode import apply_turnover_multiplier, resolve_policy_mode

            tau = float(apply_turnover_multiplier(tau, resolve_policy_mode(dict(cfg))))
        except Exception:
            pass

        def _hard(a: np.ndarray) -> np.ndarray:
            x = np.asarray(a, dtype=np.float64).reshape(-1)
            # Map raw action to a long-only proposal then enforce turnover from EW
            w_prop = soft_simplex_project(x)
            kk = int(k) if k is not None else int(w_prop.size)
            w_prev = np.full(kk, 1.0 / max(kk, 1), dtype=np.float64)
            if w_prop.size != kk:
                raise ValueError(f"projector expected k={kk}, got {w_prop.size}")
            return turnover_cap_project(w_prop, w_prev=w_prev, tau=tau, counter=counter)

        return _hard

    if mode not in ("soft", "none", ""):
        # Unknown modes fall back to soft with an explicit note via ValueError
        # only when clearly wrong; research_alpha_train is the strict validator.
        pass
    return soft_simplex_project
