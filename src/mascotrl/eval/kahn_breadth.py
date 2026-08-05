"""Kahn IR / effective-breadth proxies (Phase 4 / R5 / Phase F)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def selection_breadth_metrics(
    returns: np.ndarray, selected: Sequence[int]
) -> dict[str, float]:
    """effective_breadth + N_eff_ENB for a column subset of returns."""
    idx = [int(i) for i in selected]
    sub = np.asarray(returns, dtype=np.float64)[:, idx]
    corr = np.corrcoef(np.nan_to_num(sub, nan=0.0), rowvar=False)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    return {
        "effective_breadth": float(effective_breadth(sub)),
        "n_eff_enb": float(effective_number_of_bets_entropy(corr)),
    }


def effective_breadth(returns: np.ndarray, *, floor: float = 1.0) -> float:
    """Correlation-adjusted breadth: N_eff ≈ 1ᵀR⁻¹1 style proxy via mean corr.

    For T×K return matrix, use N_eff = K / (1 + (K-1) ρ̄) with ρ̄ = mean off-diag corr.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1:
        return float("nan")
    k = int(x.shape[1])
    if k == 1:
        return 1.0
    # drop all-nan columns
    good = np.isfinite(x).any(axis=0)
    x = x[:, good]
    k = int(x.shape[1])
    if k < 2:
        return float(max(k, floor))
    c = np.corrcoef(np.nan_to_num(x, nan=0.0), rowvar=False)
    off = c[np.triu_indices(k, k=1)]
    rho = float(np.nanmean(off)) if off.size else 0.0
    rho = float(np.clip(rho, -0.999, 0.999))
    neff = k / (1.0 + (k - 1.0) * max(rho, 0.0))
    return float(max(neff, floor))


def effective_number_of_bets_entropy(corr: np.ndarray) -> float:
    """Entropy-based effective number of bets (ENB) from a correlation matrix.

    N_eff_ENB = exp(-sum_i p_i ln p_i) with p_i = λ_i / sum_j λ_j over eigenvalues
    of the correlation matrix (non-negative spectrum after clipping numerics).
    """
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1:
        return float("nan")
    # Symmetrize and clip tiny negative eigenvalues from float noise.
    c = 0.5 * (c + c.T)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    # Degenerate / constant columns → singular corr; fail soft.
    if not np.isfinite(c).all():
        return float("nan")
    try:
        evals = np.linalg.eigvalsh(c)
    except np.linalg.LinAlgError:
        return float("nan")
    evals = np.clip(evals, 0.0, None)
    total = float(evals.sum())
    if total <= 0.0 or not np.isfinite(total):
        return float("nan")
    p = evals / total
    # Drop exact zeros for entropy; they contribute 0 to -p ln p.
    p = p[p > 0.0]
    if p.size == 0:
        return float("nan")
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def kahn_ir_components(
    ic: float,
    n_eff: float,
    g_refresh: float,
    tc: float,
) -> dict[str, float]:
    """Fundamental law pieces: IR ≈ IC × √(g · N_eff) × TC with BR = g · N_eff."""
    ic_f = float(ic)
    n_f = float(n_eff)
    g_f = float(g_refresh)
    tc_f = float(tc)
    br = g_f * n_f
    if not np.isfinite(br) or br < 0.0:
        pred = float("nan")
    else:
        pred = ic_f * float(np.sqrt(br)) * tc_f
    return {
        "ic": ic_f,
        "n_eff": n_f,
        "g_refresh": g_f,
        "tc": tc_f,
        "breadth": float(br),
        "predicted_ir": float(pred),
    }


def signal_refresh_rate(
    alpha_panel: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Estimate signal refresh g from cross-sectional AR(1) of forecasts.

    For each date t, take corr(α_t, α_{t-1}) across names; γ̄ = mean of those
    correlations. With γ = exp(-g / periods_per_year), return
    g = -ln(γ̄) · periods_per_year.
    """
    a = np.asarray(alpha_panel, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        return float("nan")
    gammas: list[float] = []
    for t in range(1, a.shape[0]):
        x = a[t - 1]
        y = a[t]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            continue
        c = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(c):
            gammas.append(float(c))
    if not gammas:
        return float("nan")
    gamma = float(np.mean(gammas))
    gamma = float(np.clip(gamma, 1e-6, 1.0 - 1e-12))
    g = -np.log(gamma) * float(periods_per_year)
    return float(g)


def turnover_normalized_mean(pnls: np.ndarray, turnovers: np.ndarray) -> float:
    """Mean PnL / mean turnover (refuse division by ~0)."""
    p = np.asarray(pnls, dtype=np.float64)
    t = np.asarray(turnovers, dtype=np.float64)
    mt = float(np.nanmean(t)) if t.size else 0.0
    if not np.isfinite(mt) or abs(mt) < 1e-12:
        return float("nan")
    return float(np.nanmean(p) / mt)


def _corr_from_returns(returns: np.ndarray) -> np.ndarray | None:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1:
        return None
    good = np.isfinite(x).any(axis=0)
    x = x[:, good]
    if x.shape[1] < 1:
        return None
    if x.shape[1] == 1:
        return np.ones((1, 1), dtype=np.float64)
    return np.corrcoef(np.nan_to_num(x, nan=0.0), rowvar=False)


def kahn_pack(
    returns: np.ndarray,
    pnls: np.ndarray,
    turnovers: np.ndarray,
    *,
    ic_after_cost: float | None = None,
    factor_alpha_positive: bool = False,
    saturation_flag: bool = False,
    k: int | None = None,
    projection_ceiling: int = 50,
    g_refresh: float | None = None,
    tc: float | None = None,
    alpha_panel: np.ndarray | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Publication metrics + refuse K-scale when IC/alpha/saturation block.

    When a real (T, K) return panel is passed, also reports ``n_eff_enb`` and
    optional fundamental-law ``predicted_ir`` (not the refused-zero stub).
    """
    ret = np.asarray(returns, dtype=np.float64)
    neff = effective_breadth(ret)
    corr = _corr_from_returns(ret)
    n_eff_enb = (
        effective_number_of_bets_entropy(corr) if corr is not None else float("nan")
    )
    tn = turnover_normalized_mean(pnls, turnovers)
    ic = float(ic_after_cost) if ic_after_cost is not None else float("nan")
    k_use = int(k if k is not None else (ret.shape[1] if ret.ndim == 2 else 0))
    scale_ok = (
        bool(factor_alpha_positive)
        and not bool(saturation_flag)
        and (not np.isfinite(ic) or ic > 0)
        and k_use <= int(projection_ceiling)
    )

    g_use = g_refresh
    if g_use is None and alpha_panel is not None:
        g_use = signal_refresh_rate(alpha_panel, periods_per_year=periods_per_year)
    tc_use = float(tc) if tc is not None else float("nan")

    pack: dict[str, Any] = {
        "effective_breadth": neff,
        "n_eff_enb": float(n_eff_enb),
        "turnover_normalized_mean": tn,
        "ic_after_cost": ic,
        "k": k_use,
        "projection_ceiling": int(projection_ceiling),
        "k_scale_claim_allowed": bool(scale_ok),
        "note": "IR≈IC×√breadth×TC; correlated slots ≪ K. Refuse scale while G2≤0 or τ glued.",
    }

    # Real panel: expose IR components when inputs allow; never stamp refuse-zero.
    real_panel = (
        ret.ndim == 2
        and ret.shape[0] > 1
        and ret.shape[1] > 0
        and np.isfinite(ret).any()
        and float(np.nanstd(ret)) > 0.0
    )
    if real_panel:
        pack["status"] = "ok"
        n_for_ir = float(n_eff_enb) if np.isfinite(n_eff_enb) else float(neff)
        if g_use is not None and np.isfinite(float(g_use)) and np.isfinite(ic):
            comps = kahn_ir_components(
                ic=ic,
                n_eff=n_for_ir,
                g_refresh=float(g_use),
                tc=tc_use if np.isfinite(tc_use) else 1.0,
            )
            pack.update(
                {
                    "g_refresh": comps["g_refresh"],
                    "tc": comps["tc"],
                    "breadth": comps["breadth"],
                    "predicted_ir": comps["predicted_ir"],
                }
            )
        elif g_use is not None:
            pack["g_refresh"] = float(g_use)
    else:
        pack["status"] = "refused_until_panel_returns"
        # Refuse is non-claim: never advertise K-scale / breadth wins without a panel.
        pack["k_scale_claim_allowed"] = False

    return pack
