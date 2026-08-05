"""Andersen Quadratic-Exponential Heston path generators (NumPy reference).

Production RL rollouts use the C++ twin in ``src/engine/worlds.cpp``
(``heston_scheme`` = QE / QE-M). This module is the readable reference and
the unit-test oracle. QuantLib remains optional conformance only
(``QuadraticExponentialMartingale``), never a Burst runtime dependency.

Feller ``2*kappa*theta >= xi**2`` is reported, never forced: market fits
often violate it; QE-M is the correct response, not parameter surgery.
"""
from __future__ import annotations

import numpy as np

# Andersen (2008) switching threshold for quadratic vs exponential branch.
_PSI_C = 1.5
_GAMMA1 = 0.5
_GAMMA2 = 0.5


def feller_satisfied(*, kappa: float, theta: float, xi: float) -> bool:
    return float(2.0 * kappa * theta) >= float(xi * xi)


def feller_gap(*, kappa: float, theta: float, xi: float) -> float:
    """Positive => Feller holds; negative => violated (do not re-fit to force)."""
    return float(2.0 * kappa * theta - xi * xi)


def _qe_step_variance(
    v: np.ndarray,
    *,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    u: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """One Andersen QE variance step. ``u``~U(0,1), ``z``~N(0,1), shape (n_paths,)."""
    ekt = np.exp(-kappa * dt)
    m = theta + (v - theta) * ekt
    s2 = (
        v * (xi**2) * ekt * (1.0 - ekt) / kappa
        + theta * (xi**2) * (1.0 - ekt) ** 2 / (2.0 * kappa)
    )
    psi = s2 / np.maximum(m * m, 1e-16)

    # Quadratic branch
    inv = 2.0 / np.maximum(psi, 1e-16)
    b2 = inv - 1.0 + np.sqrt(np.maximum(inv, 0.0)) * np.sqrt(np.maximum(inv - 1.0, 0.0))
    a = m / (1.0 + b2)
    v_quad = a * (np.sqrt(np.maximum(b2, 0.0)) + z) ** 2

    # Exponential branch
    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / np.maximum(m, 1e-16)
    v_exp = np.where(
        u <= p,
        0.0,
        np.log(np.maximum((1.0 - p) / np.maximum(1.0 - u, 1e-16), 1e-16)) / beta,
    )

    return np.where(psi <= _PSI_C, v_quad, np.maximum(v_exp, 0.0))


def _martingale_k0(
    *,
    v: np.ndarray,
    v_next: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dt: float,
    use_martingale: bool,
) -> np.ndarray:
    """Andersen K0; with QE-M, replace K0 so E[S] matches the risk-neutral drift."""
    g1, g2 = _GAMMA1, _GAMMA2
    k0 = -rho * kappa * theta * dt / xi
    k1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = g1 * dt * (1.0 - rho * rho)
    k4 = g2 * dt * (1.0 - rho * rho)
    if not use_martingale:
        return np.full_like(v, k0), k1, k2, k3, k4

    # Moment-matching correction (Andersen 2008 §3.2.3 / Lord et al. QE-M).
    A = k2 + 0.5 * k4
    ekt = np.exp(-kappa * dt)
    m = theta + (v - theta) * ekt
    s2 = (
        v * (xi**2) * ekt * (1.0 - ekt) / kappa
        + theta * (xi**2) * (1.0 - ekt) ** 2 / (2.0 * kappa)
    )
    psi = s2 / np.maximum(m * m, 1e-16)

    b2 = 2.0 / psi - 1.0 + np.sqrt(np.maximum(2.0 / psi, 1e-16)) * np.sqrt(
        np.maximum(2.0 / psi - 1.0, 0.0)
    )
    a = m / (1.0 + b2)
    # Quadratic MGF piece
    denom = np.maximum(1.0 - 2.0 * A * a, 1e-12)
    M_quad = np.exp(A * b2 * a / denom) / np.sqrt(denom)

    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / np.maximum(m, 1e-16)
    M_exp = p + (1.0 - p) * beta / np.maximum(beta - A, 1e-12)

    M = np.where(psi <= _PSI_C, M_quad, M_exp)
    # Guard against non-finite MGF (rare under extreme A)
    M = np.where(np.isfinite(M) & (M > 1e-16), M, 1.0)
    k0_star = -(k1 + 0.5 * k3) * v - np.log(M)
    return k0_star, k1, k2, k3, k4


def simulate_heston_qe_m(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    spot0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rate: float = 0.0,
    div_q: float = 0.0,
    seed: int = 0,
    martingale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Andersen QE / QE-M paths.

    Returns
    -------
    spots : (n_paths, n_steps)
    variances : (n_paths, n_steps)  non-negative by construction
    """
    rng = np.random.default_rng(seed)
    log_s = np.full(n_paths, np.log(spot0), dtype=np.float64)
    v = np.full(n_paths, float(v0), dtype=np.float64)
    spots = np.empty((n_paths, n_steps), dtype=np.float64)
    vars_ = np.empty((n_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        spots[:, t] = np.exp(log_s)
        vars_[:, t] = np.maximum(v, 0.0)

        u = rng.random(n_paths)
        z_v = rng.standard_normal(n_paths)
        z_s = rng.standard_normal(n_paths)
        v_next = _qe_step_variance(
            v, kappa=kappa, theta=theta, xi=xi, dt=dt, u=u, z=z_v
        )
        v_next = np.maximum(v_next, 0.0)

        k0, k1, k2, k3, k4 = _martingale_k0(
            v=v,
            v_next=v_next,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            dt=dt,
            use_martingale=martingale,
        )
        vol_term = np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0))
        log_s = log_s + (rate - div_q) * dt + k0 + k1 * v + k2 * v_next + vol_term * z_s
        v = v_next

    return spots, vars_
