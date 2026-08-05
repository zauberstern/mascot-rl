"""C7: gate1, gate2, gate3 -- one shared module for every promotion gate the
eq allocation campaign and spectrum sweep both need, so cells stay comparable
and no caller invents its own bar.

gate1 -- break-even spread multiplier (cost realism): unchanged definition
already used in ``scripts/run_cpcv_campaign.py``, factored out here.
gate2 -- positive-edge gate: the policy's realized period returns must show a
significantly positive Newey-West (FF4 + Pastor-Stambaugh) alpha, not merely
a positive raw Sharpe that factor exposure alone can produce.
gate3 -- same-fold comparison: the policy must beat a named peer panel's
Sharpe when every Sharpe is computed on the identical CPCV OOS test windows
(the caller is responsible for that "same fold" guarantee; this module only
compares the numbers it is handed).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

GATE1_MIN_BREAK_EVEN = 0.25
GATE2_MIN_T_STAT = 2.0


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def compute_gate1(cost_ladder: Mapping[str, Any]) -> dict[str, Any]:
    """Break-even spread multiplier gate.

    ``cost_ladder`` is whatever :func:`src.eval.cost_ladder.run_cost_ladder`
    (or an equivalent fill-ladder computation) returned; only
    ``break_even_spread_multiplier`` and ``cost_source`` are read.
    """
    be = cost_ladder.get("break_even_spread_multiplier")
    ok = _finite(be) and float(be) >= GATE1_MIN_BREAK_EVEN
    return {
        "break_even_spread_multiplier": be,
        "min_required": GATE1_MIN_BREAK_EVEN,
        "cost_source": cost_ladder.get("cost_source"),
        "pass": bool(ok),
        "decision": "continue_positive_framing" if ok else "pivot_negative_economic_framing",
    }


def compute_gate2(
    policy_returns: np.ndarray,
    factors: np.ndarray,
    *,
    min_t_stat: float = GATE2_MIN_T_STAT,
    factor_names: list[str] | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Factor-alpha gate via Newey-West HAC time-series regression.

    Regresses daily excess returns on the supplied factor panel (typically
    MKT-RF, SMB, HML, RMW, CMA, UMD, PS_VWF when available). Passes when
    annualized alpha is positive and the HAC t-stat clears ``min_t_stat``.
    """
    from src.eval.signal_gate import ff_alpha

    yy = np.asarray(policy_returns, dtype=np.float64).reshape(-1)
    xx = np.asarray(factors, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    stats = ff_alpha(yy, xx)
    t_stat = stats.get("t_stat")
    alpha = stats.get("alpha")
    alpha_ann = (
        float(alpha) * float(periods_per_year)
        if _finite(alpha)
        else float("nan")
    )
    positive_edge = bool(
        _finite(t_stat)
        and float(t_stat) >= float(min_t_stat)
        and _finite(alpha)
        and float(alpha) > 0.0
    )
    # Optional factor loadings for eval diagnostics (OLS, not HAC).
    loadings: dict[str, float] = {}
    try:
        mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
        y_m = yy[mask]
        x_m = xx[mask]
        if y_m.size >= xx.shape[1] + 2:
            design = np.column_stack([np.ones(y_m.size), x_m])
            beta, *_ = np.linalg.lstsq(design, y_m, rcond=None)
            names = factor_names or [f"f{i}" for i in range(x_m.shape[1])]
            for i, name in enumerate(names[: x_m.shape[1]]):
                loadings[str(name)] = float(beta[i + 1])
    except Exception:  # noqa: BLE001
        loadings = {}
    return {
        "alpha": alpha,
        "alpha_annualized": alpha_ann,
        "t_stat": t_stat,
        "n": stats.get("n"),
        "lags": stats.get("lags"),
        "min_t_stat": float(min_t_stat),
        "n_factors": int(xx.shape[1]) if xx.ndim == 2 else 1,
        "factor_loadings": loadings,
        "positive_edge": positive_edge,
        "pass": positive_edge,
        "decision": (
            "continue_positive_framing" if positive_edge else "pivot_negative_economic_framing"
        ),
    }


def compute_gate3(
    policy_sharpe: float,
    baseline_sharpes: Mapping[str, float],
    *,
    require_beat_all: bool = False,
) -> dict[str, Any]:
    """Same-fold policy-vs-peers gate.

    ``require_beat_all=False`` (default) passes when the policy beats the
    single best-performing named baseline (the standard "beat the best
    peer" promotion bar); ``True`` requires beating every named baseline.

    OLPS stub names (CORN/BNN/ONS/Anticor/CWMR/RMR EG fallbacks) are excluded
    from ``n_baselines`` / ``n_beaten`` so EG clones do not inflate peer
    diversity. See :func:`src.eval.olps.olps_stub_names`.
    """
    from src.eval.olps import filter_olps_stubs_from_peers

    filtered = filter_olps_stubs_from_peers(baseline_sharpes)
    baselines = {
        str(name): {"sharpe": float(v)} for name, v in filtered.items() if _finite(v)
    }
    beats = {name: bool(float(policy_sharpe) > b["sharpe"]) for name, b in baselines.items()}
    n_beaten = sum(beats.values())
    best_baseline = max(baselines, key=lambda n: baselines[n]["sharpe"]) if baselines else None
    if require_beat_all:
        ok = bool(baselines) and n_beaten == len(baselines)
    else:
        ok = bool(baselines) and best_baseline is not None and bool(
            float(policy_sharpe) > baselines[best_baseline]["sharpe"]
        )
    edge = (
        float(policy_sharpe) - baselines[best_baseline]["sharpe"]
        if (best_baseline is not None and _finite(policy_sharpe))
        else None
    )
    return {
        "policy_sharpe": float(policy_sharpe) if _finite(policy_sharpe) else None,
        "baselines": baselines,
        "beats": beats,
        "n_baselines": len(baselines),
        "n_beaten": n_beaten,
        "best_baseline": best_baseline,
        "edge_vs_best_baseline": edge,
        "require_beat_all": require_beat_all,
        "pass": bool(ok),
        "decision": "continue_positive_framing" if ok else "pivot_negative_economic_framing",
    }
