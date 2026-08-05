"""Spectrum objective factory + score-function episode weights for actor path."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from mascotrl.policy.risk_objective import RiskObjective

# Modes whose primary gradient is dense per-step reward (not batch episode weights).
_DENSE_REWARD_MODES = frozenset(
    {
        "mikkila_asym",
        "differential_sharpe",
        "mtm_pnl",
        "sdr_composite",
    }
)

# Modes that use score-function episode weights when objective_primary.
_EPISODE_WEIGHT_MODES = frozenset(
    {
        "mean_std_cao",
        "meanvar_kolm",
        "cvar_ru",
        "cvar",
        "entropic_oce",
        "entropic",
        "smse",
        "rsqp",
    }
)

# Declared path when the mode is selected as primary (dense / episode).
OBJECTIVE_GRADIENT_PATH: dict[str, str] = {
    "mikkila_asym": "dense_reward",
    "differential_sharpe": "dense_reward",
    "mtm_pnl": "dense_reward",
    "sdr_composite": "dense_reward",
    "mean_std_cao": "episode_weight",
    "meanvar_kolm": "episode_weight",
    "cvar_ru": "episode_weight",
    "cvar": "episode_weight",
    "entropic_oce": "episode_weight",
    "entropic": "episode_weight",
    "smse": "episode_weight",
    "rsqp": "episode_weight",
}

_OBJECTIVE_ALIASES = {
    "cvar": "cvar_ru",
    "entropic": "entropic_oce",
    "residual_pnl": "mtm_pnl",
}


def mikkila_asym_reward(pnl: float, *, xi: float = 1.0) -> float:
    """Mikkilä 2023 Eq. (8) asymmetric per-step reward: ``r_t = PnL_t - xi*|PnL_t|``.

    Card ``papers.mikkila_2023_drl_option_trading.c6``: penalizes losses
    ``xi`` times harder than an equal-sized gain is rewarded (xi in {1,2,3}
    in the paper). ``pnl`` here is the env's raw per-step after-cost return,
    the equity-allocation analogue of the paper's option-hedge step PnL;
    this is an estimand adaptation (stk_ret cadence, not option-hedge PnL),
    not a literal replication of the paper's protocol.
    """
    x = float(pnl)
    return x - float(xi) * abs(x)


def sdr_composite_reward(
    pnl: float,
    *,
    bench_pnl: float = 0.0,
    beta: float = 1.0,
    weights: Mapping[str, float] | None = None,
    ann_factor: float = 252.0,
) -> float:
    """Srivastava 2025 composite: R_ann - sigma_down + D_ret + Treynor blend.

    Per-step proxy on daily pnl with YAML ``reward_weights`` keys
    w_ann, w_down, w_diff, w_treynor (defaults 1,1,1,1).
    """
    w = dict(weights or {})
    w1 = float(w.get("w_ann", 1.0))
    w2 = float(w.get("w_down", 1.0))
    w3 = float(w.get("w_diff", 1.0))
    w4 = float(w.get("w_treynor", 1.0))
    x = float(pnl)
    r_ann = x * float(ann_factor)
    sigma_down = max(0.0, -x)
    d_ret = x - float(bench_pnl)
    treynor = d_ret / max(float(beta), 1e-8)
    return w1 * r_ann - w2 * sigma_down + w3 * d_ret + w4 * treynor


def objective_gradient_path_for(mode: str, objective_primary: bool) -> str:
    """Stamp helper: dense_reward | episode_weight | critic_only."""
    m = str(mode)
    if m in _DENSE_REWARD_MODES:
        return "dense_reward"
    if bool(objective_primary) and m in _EPISODE_WEIGHT_MODES:
        return "episode_weight"
    return "critic_only"


def episode_weights(
    mode: str,
    G: torch.Tensor,
    *,
    cao_c: float = 1.5,
    kappa: float = 1.0,
    alpha: float = 0.95,
    lam: float = 1.0,
    zeta: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Score-function episode weights w for surrogate L = mean(w * log π).

    G is episode (or batch-proxy) returns; higher is better. Weights are shaped
    for *minimizing* the risk / mean-dispersion functional via Adam.
    """
    g = G.reshape(-1)
    m = _OBJECTIVE_ALIASES.get(str(mode), str(mode))

    if m == "mtm_pnl":
        return -g

    if m == "mean_std_cao":
        gbar = g.mean()
        sigma = g.std(unbiased=False).clamp_min(1e-8)
        c = float(cao_c)
        return (-1.0 - c * gbar / sigma) * g + (c / (2.0 * sigma)) * g.pow(2)

    if m == "meanvar_kolm":
        gbar = g.mean()
        k = float(kappa)
        return (-1.0 - k * gbar) * g + (k / 2.0) * g.pow(2)

    if m == "cvar_ru":
        # Rockafellar-Uryasev: weight the worst (1-α) tail of losses L=-G.
        # Using the empirical top-k set (not a soft quantile that can leave
        # the entire batch with zero weight when VaR lands on the max).
        n = max(int(g.numel()), 1)
        k = max(1, int(math.ceil((1.0 - float(alpha)) * n)))
        if zeta is None:
            losses = -g.detach()
            topk_idx = torch.topk(losses, k=k, largest=True).indices
            w = torch.zeros_like(g)
            w[topk_idx] = 1.0 / ((1.0 - float(alpha)) * float(n))
            return w
        z = (
            zeta
            if torch.is_tensor(zeta)
            else torch.as_tensor(zeta, device=g.device, dtype=g.dtype)
        )
        return (1.0 / (1.0 - float(alpha))) * F.relu(-g - z)

    if m == "entropic_oce":
        lam_f = float(lam)
        n = g.numel()
        # w = exp(-λG) / (λ * mean(exp(-λG))) via logsumexp
        log_mean_exp = torch.logsumexp(-lam_f * g, dim=0) - math.log(n)
        return torch.exp(-lam_f * g - math.log(lam_f) - log_mean_exp)

    if m == "smse":
        return F.relu(-g).pow(2)

    if m == "rsqp":
        pos2 = F.relu(-g).pow(2)
        rho = torch.sqrt(pos2.mean().clamp_min(1e-12))
        return pos2 / (2.0 * rho)

    raise ValueError(f"no episode weights for mode={mode!r}")


def resolve_objective_mode(cfg: dict[str, Any], *, default: str = "none") -> str:
    """Resolve the spectrum ``objective`` id from cfg (shared by HAPPO and
    the research PPO path so both read the same nine-way axis)."""
    risk = dict(cfg.get("risk") or {})
    mode = str(risk.get("mode", default) or default).strip()
    raw_obj = cfg.get("objective")
    if raw_obj is not None and str(raw_obj).strip():
        mode = str(raw_obj).strip()
        cand = _OBJECTIVE_ALIASES.get(mode, mode)
        try:
            from mascotrl.spectrum.registry import allowed_ids, validate_choice

            if cand in allowed_ids("objective"):
                mode = validate_choice("objective", cand)
            else:
                mode = cand
        except Exception:
            mode = cand
    return mode


def build_risk_objective(cfg: dict[str, Any]) -> RiskObjective | None:
    """Build RiskObjective from spectrum ``objective`` + optional ``risk`` overlay."""
    risk = dict(cfg.get("risk") or {})
    objective_primary = bool(cfg.get("objective_primary", False))
    mode = resolve_objective_mode(cfg)
    coef = float(risk.get("coef", 0.0))
    if mode.lower() in {"", "none"} and coef <= 0.0 and not objective_primary:
        return None
    if mode.lower() in {"", "none"}:
        return None

    return RiskObjective(
        mode=mode,
        alpha=float(risk.get("alpha", cfg.get("alpha", 0.95))),
        lam=float(risk.get("lambda", risk.get("lam", cfg.get("lam", 1.0)))),
        coef=coef,
        zeta_lr=float(risk.get("zeta_lr", 3e-3)),
        cao_c=float(risk.get("cao_c", cfg.get("cao_c", 1.5))),
        kappa=float(risk.get("kappa", cfg.get("kappa", 1.0))),
        objective_primary=objective_primary,
    )
