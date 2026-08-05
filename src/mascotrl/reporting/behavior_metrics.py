"""Measured policy behaviour vector (CRUCIBLE Part E.2 / E.3).

All 23 measures are scale-free or unit-stamped. Interpretation only; never
feeds capital gates.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:
    from src.data.crucible import SLEEVE_IDS as _CRUCIBLE_SLEEVES
except Exception:  # pragma: no cover - import guard for isolated unit tests
    _CRUCIBLE_SLEEVES = (
        "trend",
        "reversal",
        "carry",
        "defensive",
        "lottery",
        "illiquid",
        "core",
    )

SLEEVE_IDS: tuple[str, ...] = tuple(_CRUCIBLE_SLEEVES)

REGIME_IDS: tuple[str, ...] = ("calm", "inflationary", "crisis")


def spectrum_foil_sleeve_matrix(k: int) -> np.ndarray:
    """Deterministic (K, 7) primary-sleeve assignment for spectrum behaviour export.

    Used when the full CRUCIBLE membership pipeline is unavailable (eq lake
    cells). Rotating primary sleeves give non-zero active tilts whenever the
    policy leaves equal weight - measurement geometry, not a CRSP claim.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    mat = np.zeros((k, len(SLEEVE_IDS)), dtype=np.float64)
    n_sleeves = len(SLEEVE_IDS)
    for i in range(k):
        mat[i, i % n_sleeves] = 1.0
    return mat

BEHAVIOUR_MEASURE_IDS: tuple[str, ...] = (
    "hhi_mean",
    "n_eff_mean",
    "max_weight_mean",
    "l1_vs_ew_mean",
    "turnover_mean",
    "turnover_cap_binding_frac",
    "action_entropy_mean",
    "weight_autocorr_lag1",
    "tilt_autocorr_lag21",
    "holding_period_days",
    "rotation_rate",
    "tilt_trend",
    "tilt_reversal",
    "tilt_carry",
    "tilt_defensive",
    "tilt_lottery",
    "tilt_illiquid",
    "tilt_core",
    "downside_capture",
    "upside_capture",
    "return_skew",
    "max_drawdown",
    "cvar_05",
    # Holdings / RBSA / regime-delta extensions (composition scoring).
    "exposure_size",
    "exposure_value",
    "exposure_momentum",
    "exposure_quality",
    "exposure_low_vol",
    "sector_hhi",
    "rbsa_r_squared",
    "delta_hhi_regime",
    "delta_turnover_regime",
    "delta_defensive_regime",
    "delta_quality_regime",
    "semantic_rotation_rate",
    "semantic_pc1_mean",
    "semantic_pc2_mean",
    "semantic_pc3_mean",
    "support_size_mean",
    "support_jaccard_lag1",
    "support_exit_rate",
    "support_reentry_rate",
    "style_agreement_cosine",
)

# Confirmatory AA feature space. Do not append semantic/support/style ids.
COMPOSITION_MEASURE_IDS: tuple[str, ...] = (
    "hhi_mean",
    "n_eff_mean",
    "max_weight_mean",
    "l1_vs_ew_mean",
    "turnover_mean",
    "turnover_cap_binding_frac",
    "action_entropy_mean",
    "weight_autocorr_lag1",
    "tilt_autocorr_lag21",
    "holding_period_days",
    "rotation_rate",
    "tilt_trend",
    "tilt_reversal",
    "tilt_carry",
    "tilt_defensive",
    "tilt_lottery",
    "tilt_illiquid",
    "tilt_core",
    "downside_capture",
    "upside_capture",
    "return_skew",
    "max_drawdown",
    "cvar_05",
    "exposure_size",
    "exposure_value",
    "exposure_momentum",
    "exposure_quality",
    "exposure_low_vol",
    "sector_hhi",
    "rbsa_r_squared",
    "delta_hhi_regime",
    "delta_turnover_regime",
    "delta_defensive_regime",
    "delta_quality_regime",
)

_EPS = 1e-12
SUPPORT_EPS = 1e-8


def _as_w(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if w.ndim != 2:
        raise ValueError(f"weights must be (T, K); got shape {w.shape}")
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def _sleeve_matrix(sleeve_matrix: np.ndarray | None, k: int) -> np.ndarray:
    if sleeve_matrix is None:
        return np.zeros((k, len(SLEEVE_IDS)), dtype=np.float64)
    s = np.asarray(sleeve_matrix, dtype=np.float64)
    if s.shape != (k, len(SLEEVE_IDS)):
        raise ValueError(
            f"sleeve_matrix must be (K, {len(SLEEVE_IDS)}) = ({k}, {len(SLEEVE_IDS)}); "
            f"got {s.shape}"
        )
    return s


def sleeve_tilt_series(weights: np.ndarray, sleeve_matrix: np.ndarray) -> np.ndarray:
    """Active sleeve tilts over time.

    Formula: ``tilt_s(t) = sum_i S_{i s} w_{t i} - n_s / K`` with
    ``n_s = sum_i S_{i s}``.

    Unit: weight fraction (dimensionless active share in sleeve s).
    """
    w = _as_w(weights)
    t, k = w.shape
    s = _sleeve_matrix(sleeve_matrix, k)
    n_s = s.sum(axis=0)
    port_s = w @ s  # (T, 7)
    return port_s - (n_s / float(max(k, 1)))[None, :]


def measure_hhi_mean(weights: np.ndarray) -> float:
    """Mean Herfindahl-Hirschman index of portfolio weights.

    Formula: ``mean_t sum_i w_{t i}^2``.

    Unit: dimensionless (1/K for equal weight; 1 for single-name).
    """
    w = _as_w(weights)
    return float(np.mean(np.sum(w * w, axis=1)))


def measure_n_eff_mean(weights: np.ndarray) -> float:
    """Mean effective number of names ``1 / HHI_t``.

    Formula: ``mean_t 1 / sum_i w_{t i}^2``.

    Unit: count of names (effective).
    """
    w = _as_w(weights)
    hhi = np.sum(w * w, axis=1)
    return float(np.mean(1.0 / np.maximum(hhi, _EPS)))


def measure_max_weight_mean(weights: np.ndarray) -> float:
    """Mean maximum name weight.

    Formula: ``mean_t max_i w_{t i}``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return float(np.mean(np.max(w, axis=1)))


def measure_l1_vs_ew_mean(weights: np.ndarray) -> float:
    """Mean L1 distance from equal weight (active-share proxy).

    Formula: ``mean_t sum_i |w_{t i} - 1/K|``.

    Unit: weight fraction (L1).
    """
    w = _as_w(weights)
    k = max(w.shape[1], 1)
    ew = 1.0 / float(k)
    return float(np.mean(np.sum(np.abs(w - ew), axis=1)))


def step_turnover(weights: np.ndarray) -> np.ndarray:
    """Per-step one-way turnover ``0.5 * ||w_t - w_{t-1}||_1`` (length T-1)."""
    w = _as_w(weights)
    if w.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    return 0.5 * np.sum(np.abs(np.diff(w, axis=0)), axis=1)


def measure_turnover_mean(weights: np.ndarray) -> float:
    """Mean one-way turnover.

    Formula: ``mean_{t>=1} 0.5 * sum_i |w_{t i} - w_{t-1,i}|``.

    Unit: fraction of NAV traded one-way per step.
    """
    to = step_turnover(weights)
    if to.size == 0:
        return float("nan")
    return float(np.mean(to))


def _support_sets(weights: np.ndarray, *, eps: float = SUPPORT_EPS) -> list[set[int]]:
    w = _as_w(weights)
    sets: list[set[int]] = []
    for t in range(w.shape[0]):
        idx = np.flatnonzero(w[t] > float(eps))
        sets.append(set(int(i) for i in idx.tolist()))
    return sets


def measure_support_size_mean(weights: np.ndarray) -> float:
    """Mean count of names with weight strictly above SUPPORT_EPS.

    Formula: ``mean_t |{i : w_{t i} > eps}|``.

    Unit: count of names.
    """
    sets = _support_sets(weights)
    if not sets:
        return float("nan")
    return float(np.mean([len(s) for s in sets]))


def measure_support_jaccard_lag1(weights: np.ndarray) -> float:
    """Mean Jaccard overlap of successive exact-nonzero supports.

    Formula: ``mean_{t>=1} |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|`` (skip t if both empty).

    Unit: dimensionless in [0, 1].
    """
    sets = _support_sets(weights)
    if len(sets) < 2:
        return float("nan")
    vals: list[float] = []
    for t in range(1, len(sets)):
        a = sets[t - 1]
        b = sets[t]
        union = a | b
        if not union:
            continue
        vals.append(float(len(a & b)) / float(len(union)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def measure_support_exit_rate(weights: np.ndarray) -> float:
    """Mean fraction of previous support that is dropped.

    Formula: ``mean_{t>=1} |S_{t-1} \\ S_t| / max(|S_{t-1}|, 1)``.

    Unit: fraction in [0, 1].
    """
    sets = _support_sets(weights)
    if len(sets) < 2:
        return float("nan")
    vals: list[float] = []
    for t in range(1, len(sets)):
        prev = sets[t - 1]
        cur = sets[t]
        vals.append(float(len(prev - cur)) / float(max(len(prev), 1)))
    return float(np.mean(vals))


def measure_support_reentry_rate(weights: np.ndarray) -> float:
    """Fraction of ever-held names that exit and later re-enter.

    A name re-enters if there exist times a < b < c with i in S_a, i not in S_b, i in S_c.

    Unit: fraction of ever-held names in [0, 1].
    """
    sets = _support_sets(weights)
    ever: set[int] = set()
    for s in sets:
        ever |= s
    if not ever:
        return float("nan")
    if len(sets) < 2:
        return float("nan")
    n_re = 0
    for i in ever:
        seen = False
        exited = False
        reentered = False
        for s in sets:
            if i in s:
                if exited:
                    reentered = True
                    break
                seen = True
            elif seen:
                exited = True
        if reentered:
            n_re += 1
    return float(n_re) / float(len(ever))


def measure_turnover_cap_binding_frac(
    weights: np.ndarray, *, turnover_cap: float | None = None
) -> float:
    """Fraction of steps where the turnover cap binds.

    Formula: ``mean_t 1{0.5||Δw_t||_1 >= turnover_cap}``.

    Unit: fraction of steps in [0, 1].
    """
    if turnover_cap is None or not np.isfinite(turnover_cap):
        return float("nan")
    to = step_turnover(weights)
    if to.size == 0:
        return float("nan")
    return float(np.mean(to >= float(turnover_cap) - 1e-15))


def measure_action_entropy_mean(weights: np.ndarray) -> float:
    """Mean Shannon entropy of the weight vector.

    Formula: ``mean_t -sum_i w_{t i} log w_{t i}`` with ``0 log 0 := 0``.

    Unit: nats.
    """
    w = _as_w(weights)
    ents = []
    for row in w:
        pos = row[row > 0.0]
        if pos.size == 0:
            ents.append(0.0)
        else:
            ents.append(float(-np.sum(pos * np.log(pos))))
    return float(np.mean(ents)) if ents else float("nan")


def _lag_autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= lag + 1:
        return float("nan")
    a = x[lag:]
    b = x[:-lag]
    if a.std() < _EPS or b.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def measure_weight_autocorr_lag1(weights: np.ndarray) -> float:
    """Mean across names of lag-1 weight autocorrelation.

    Formula: ``mean_i corr(w_{· i}[1:], w_{· i}[:-1])``.

    Unit: correlation in [-1, 1].
    """
    w = _as_w(weights)
    vals = [_lag_autocorr(w[:, i], 1) for i in range(w.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def measure_tilt_autocorr_lag21(
    weights: np.ndarray, sleeve_matrix: np.ndarray | None = None
) -> float:
    """Mean across sleeves of lag-21 tilt autocorrelation.

    Formula: ``mean_s corr(tilt_s[21:], tilt_s[:-21])``.

    Unit: correlation in [-1, 1].
    """
    w = _as_w(weights)
    tilts = sleeve_tilt_series(w, _sleeve_matrix(sleeve_matrix, w.shape[1]))
    vals = [_lag_autocorr(tilts[:, j], 21) for j in range(tilts.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def measure_holding_period_days(weights: np.ndarray) -> float:
    """Implied holding period from mean turnover.

    Formula: ``1 / turnover_mean``.

    Unit: days (under daily rebalance cadence).
    """
    to = measure_turnover_mean(weights)
    if not np.isfinite(to) or to <= _EPS:
        return float("nan")
    return float(1.0 / to)


def measure_rotation_rate(
    weights: np.ndarray, sleeve_matrix: np.ndarray | None = None
) -> float:
    """Mean monthly L1 change of the sleeve tilt vector.

    Formula: ``mean_t ||tilt_t - tilt_{t-21}||_1`` over available month steps.

    Unit: weight-fraction L1 per month.
    """
    w = _as_w(weights)
    tilts = sleeve_tilt_series(w, _sleeve_matrix(sleeve_matrix, w.shape[1]))
    if tilts.shape[0] <= 21:
        return float("nan")
    d = np.sum(np.abs(tilts[21:] - tilts[:-21]), axis=1)
    return float(np.mean(d))


def _tilt_mean(weights: np.ndarray, sleeve_matrix: np.ndarray, sleeve: str) -> float:
    tilts = sleeve_tilt_series(weights, sleeve_matrix)
    j = list(SLEEVE_IDS).index(sleeve)
    return float(np.mean(tilts[:, j]))


def measure_tilt_trend(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the trend sleeve.

    Formula: ``mean_t (sum_{i in trend} w_{t i} - n_trend / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "trend")


def measure_tilt_reversal(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the reversal sleeve.

    Formula: ``mean_t (sum_{i in reversal} w_{t i} - n_reversal / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "reversal")


def measure_tilt_carry(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the carry sleeve.

    Formula: ``mean_t (sum_{i in carry} w_{t i} - n_carry / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "carry")


def measure_tilt_defensive(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the defensive sleeve.

    Formula: ``mean_t (sum_{i in defensive} w_{t i} - n_defensive / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "defensive")


def measure_tilt_lottery(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the lottery sleeve.

    Formula: ``mean_t (sum_{i in lottery} w_{t i} - n_lottery / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "lottery")


def measure_tilt_illiquid(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the illiquid sleeve.

    Formula: ``mean_t (sum_{i in illiquid} w_{t i} - n_illiquid / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "illiquid")


def measure_tilt_core(weights: np.ndarray, sleeve_matrix: np.ndarray | None = None) -> float:
    """Time-mean active weight in the core sleeve.

    Formula: ``mean_t (sum_{i in core} w_{t i} - n_core / K)``.

    Unit: weight fraction.
    """
    w = _as_w(weights)
    return _tilt_mean(w, _sleeve_matrix(sleeve_matrix, w.shape[1]), "core")


def _portfolio_and_ew_returns(
    weights: np.ndarray, asset_returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    w = _as_w(weights)
    r = np.asarray(asset_returns, dtype=np.float64)
    if r.shape != w.shape:
        raise ValueError(f"asset_returns shape {r.shape} must match weights {w.shape}")
    r = np.nan_to_num(r, nan=0.0)
    r_p = np.sum(w * r, axis=1)
    r_ew = np.mean(r, axis=1)
    return r_p, r_ew


def measure_downside_capture(
    weights: np.ndarray, asset_returns: np.ndarray | None = None
) -> float:
    """Downside capture versus equal weight.

    Formula: ``sum(r_p | r_ew < 0) / sum(r_ew | r_ew < 0)``.

    Unit: ratio (dimensionless).
    """
    if asset_returns is None:
        return float("nan")
    r_p, r_ew = _portfolio_and_ew_returns(weights, asset_returns)
    mask = r_ew < 0.0
    if not np.any(mask) or abs(float(r_ew[mask].sum())) < _EPS:
        return float("nan")
    return float(r_p[mask].sum() / r_ew[mask].sum())


def measure_upside_capture(
    weights: np.ndarray, asset_returns: np.ndarray | None = None
) -> float:
    """Upside capture versus equal weight.

    Formula: ``sum(r_p | r_ew > 0) / sum(r_ew | r_ew > 0)``.

    Unit: ratio (dimensionless).
    """
    if asset_returns is None:
        return float("nan")
    r_p, r_ew = _portfolio_and_ew_returns(weights, asset_returns)
    mask = r_ew > 0.0
    if not np.any(mask) or abs(float(r_ew[mask].sum())) < _EPS:
        return float("nan")
    return float(r_p[mask].sum() / r_ew[mask].sum())


def measure_return_skew(
    weights: np.ndarray, asset_returns: np.ndarray | None = None
) -> float:
    """Skewness of portfolio returns.

    Formula: ``E[(r - mu)^3] / sigma^3`` (population moments).

    Unit: dimensionless skewness.
    """
    if asset_returns is None:
        return float("nan")
    r_p, _ = _portfolio_and_ew_returns(weights, asset_returns)
    if r_p.size < 3:
        return float("nan")
    mu = float(r_p.mean())
    sig = float(r_p.std(ddof=0))
    if sig < _EPS:
        return float("nan")
    return float(np.mean((r_p - mu) ** 3) / (sig**3))


def measure_max_drawdown(
    weights: np.ndarray, asset_returns: np.ndarray | None = None
) -> float:
    """Maximum drawdown of the portfolio wealth path.

    Formula: ``max_t (1 - wealth_t / peak_t)`` with ``wealth = cumprod(1+r)``.

    Unit: fraction of peak wealth lost.
    """
    if asset_returns is None:
        return float("nan")
    r_p, _ = _portfolio_and_ew_returns(weights, asset_returns)
    wealth = np.cumprod(1.0 + r_p)
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / np.maximum(peak, _EPS)
    return float(np.max(dd)) if dd.size else float("nan")


def measure_cvar_05(
    weights: np.ndarray, asset_returns: np.ndarray | None = None
) -> float:
    """Historical 5% CVaR (expected shortfall) of portfolio returns.

    Formula: ``mean{ r_p | r_p <= Q_0.05(r_p) }``.

    Unit: return per step.
    """
    if asset_returns is None:
        return float("nan")
    r_p, _ = _portfolio_and_ew_returns(weights, asset_returns)
    if r_p.size < 1:
        return float("nan")
    q = float(np.quantile(r_p, 0.05))
    tail = r_p[r_p <= q]
    if tail.size == 0:
        return float("nan")
    return float(tail.mean())


def compute_behaviour_vector(
    weights: np.ndarray,
    *,
    asset_returns: np.ndarray | None = None,
    sleeve_matrix: np.ndarray | None = None,
    turnover_cap: float | None = None,
) -> dict[str, float]:
    """Compute all ids in BEHAVIOUR_MEASURE_IDS (path measures plus NaN-filled layered slots)."""
    w = _as_w(weights)
    s = _sleeve_matrix(sleeve_matrix, w.shape[1])
    out: dict[str, float] = {
        "hhi_mean": measure_hhi_mean(w),
        "n_eff_mean": measure_n_eff_mean(w),
        "max_weight_mean": measure_max_weight_mean(w),
        "l1_vs_ew_mean": measure_l1_vs_ew_mean(w),
        "turnover_mean": measure_turnover_mean(w),
        "turnover_cap_binding_frac": measure_turnover_cap_binding_frac(
            w, turnover_cap=turnover_cap
        ),
        "action_entropy_mean": measure_action_entropy_mean(w),
        "weight_autocorr_lag1": measure_weight_autocorr_lag1(w),
        "tilt_autocorr_lag21": measure_tilt_autocorr_lag21(w, s),
        "holding_period_days": measure_holding_period_days(w),
        "rotation_rate": measure_rotation_rate(w, s),
        "tilt_trend": measure_tilt_trend(w, s),
        "tilt_reversal": measure_tilt_reversal(w, s),
        "tilt_carry": measure_tilt_carry(w, s),
        "tilt_defensive": measure_tilt_defensive(w, s),
        "tilt_lottery": measure_tilt_lottery(w, s),
        "tilt_illiquid": measure_tilt_illiquid(w, s),
        "tilt_core": measure_tilt_core(w, s),
        "downside_capture": measure_downside_capture(w, asset_returns),
        "upside_capture": measure_upside_capture(w, asset_returns),
        "return_skew": measure_return_skew(w, asset_returns),
        "max_drawdown": measure_max_drawdown(w, asset_returns),
        "cvar_05": measure_cvar_05(w, asset_returns),
        "support_size_mean": measure_support_size_mean(w),
        "support_jaccard_lag1": measure_support_jaccard_lag1(w),
        "support_exit_rate": measure_support_exit_rate(w),
        "support_reentry_rate": measure_support_reentry_rate(w),
        # Extended measures filled by refresh / build_policy_behavior layers.
        "exposure_size": float("nan"),
        "exposure_value": float("nan"),
        "exposure_momentum": float("nan"),
        "exposure_quality": float("nan"),
        "exposure_low_vol": float("nan"),
        "sector_hhi": float("nan"),
        "rbsa_r_squared": float("nan"),
        "delta_hhi_regime": float("nan"),
        "delta_turnover_regime": float("nan"),
        "delta_defensive_regime": float("nan"),
        "delta_quality_regime": float("nan"),
        "semantic_rotation_rate": float("nan"),
        "semantic_pc1_mean": float("nan"),
        "semantic_pc2_mean": float("nan"),
        "semantic_pc3_mean": float("nan"),
        "style_agreement_cosine": float("nan"),
    }
    return out


def _nan_measures() -> dict[str, float]:
    return {m: float("nan") for m in BEHAVIOUR_MEASURE_IDS}


def regime_conditional_behaviour(
    weights: np.ndarray,
    *,
    regimes: Sequence[str] | np.ndarray,
    asset_returns: np.ndarray | None = None,
    sleeve_matrix: np.ndarray | None = None,
    turnover_cap: float | None = None,
) -> dict[str, dict[str, float]]:
    """Behaviour measures conditioned on ``{calm, inflationary, crisis}``.

    Each regime dict includes the 23 measures plus ``n_days``. Empty regimes
    yield NaN measures and ``n_days = 0``.
    """
    w = _as_w(weights)
    reg = np.asarray(list(regimes), dtype=object).reshape(-1)
    if reg.size != w.shape[0]:
        raise ValueError(
            f"regimes length {reg.size} must match weight path T={w.shape[0]}"
        )
    s = _sleeve_matrix(sleeve_matrix, w.shape[1])
    out: dict[str, dict[str, float]] = {}
    for rid in REGIME_IDS:
        mask = reg == rid
        n_days = int(mask.sum())
        if n_days == 0:
            row = _nan_measures()
            row["n_days"] = 0.0
            out[rid] = row
            continue
        r_sub = None if asset_returns is None else np.asarray(asset_returns)[mask]
        row = compute_behaviour_vector(
            w[mask],
            asset_returns=r_sub,
            sleeve_matrix=s,
            turnover_cap=turnover_cap,
        )
        row["n_days"] = float(n_days)
        out[rid] = row
    return out


def ols_with_se(y: np.ndarray, X: np.ndarray) -> dict[str, np.ndarray]:
    """Classical OLS with iid standard errors (intercept included)."""
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(X, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    yy, xx = yy[mask], xx[mask]
    n = int(yy.size)
    design = np.column_stack([np.ones(n), xx])
    p = design.shape[1]
    if n < p + 1:
        nan = np.full(p, np.nan)
        return {"coef": nan, "se": nan, "tstat": nan, "n": n}
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    dof = max(n - p, 1)
    sigma2 = float(np.dot(resid, resid) / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0.0, None))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > _EPS)
    return {"coef": beta, "se": se, "tstat": tstat, "n": n}


def newey_west_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lags: int = 21,
) -> dict[str, Any]:
    """OLS with Bartlett-kernel Newey-West HAC standard errors.

    Design includes an intercept. ``lags`` defaults to 21 (one trading month)
    as specified for macro tilt sensitivity.
    """
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(X, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    yy, xx = yy[mask], xx[mask]
    n = int(yy.size)
    design = np.column_stack([np.ones(n), xx])
    p = int(design.shape[1])
    if n < p + 2:
        nan = np.full(p, np.nan)
        return {"coef": nan, "se": nan, "tstat": nan, "n": n, "lags": int(lags)}
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    l_bw = int(max(0, lags))
    scores = design * resid[:, None]
    s_mat = scores.T @ scores
    for j in range(1, l_bw + 1):
        if j >= n:
            break
        wj = 1.0 - j / (l_bw + 1.0)
        gamma_j = scores[j:].T @ scores[:-j]
        s_mat = s_mat + wj * (gamma_j + gamma_j.T)
    cov = xtx_inv @ s_mat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > _EPS)
    return {"coef": beta, "se": se, "tstat": tstat, "n": n, "lags": l_bw}


def macro_tilt_sensitivity(
    weights: np.ndarray,
    *,
    sleeve_matrix: np.ndarray,
    vix_z: np.ndarray,
    hy_oas_z: np.ndarray,
    term_spread: np.ndarray,
    lags: int = 21,
    epu_z: np.ndarray | None = None,
    gpri_z: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Causal-lag macro regression of sleeve tilts with Newey-West SEs.

    For each sleeve s::

        tilt_s(t) = a + b * vix_z(t-1) + c * hy_oas_z(t-1)
                    + d * term_spread(t-1) [+ epu_z(t-1) + gpri_z(t-1)] + e_t

    Optional ``epu_z`` / ``gpri_z`` widen B3 when present; omitted keeps the
    three-regressor path numerically identical. Never invents zeros for missing
    extras. Returns ``{sleeve: {regressor: {coef, se, tstat}}}``.
    """
    w = _as_w(weights)
    s = _sleeve_matrix(sleeve_matrix, w.shape[1])
    tilts = sleeve_tilt_series(w, s)
    vix = np.asarray(vix_z, dtype=np.float64).reshape(-1)
    hy = np.asarray(hy_oas_z, dtype=np.float64).reshape(-1)
    term = np.asarray(term_spread, dtype=np.float64).reshape(-1)
    t = w.shape[0]
    if min(vix.size, hy.size, term.size) < t:
        raise ValueError("macro series shorter than weight path")
    y_all = tilts[1:]
    parts = [vix[: t - 1], hy[: t - 1], term[: t - 1]]
    regressors: list[str] = ["vix_z", "hy_oas_z", "term_spread"]
    if epu_z is not None:
        epu = np.asarray(epu_z, dtype=np.float64).reshape(-1)
        if epu.size < t:
            raise ValueError("epu_z shorter than weight path")
        parts.append(epu[: t - 1])
        regressors.append("epu_z")
    if gpri_z is not None:
        gpri = np.asarray(gpri_z, dtype=np.float64).reshape(-1)
        if gpri.size < t:
            raise ValueError("gpri_z shorter than weight path")
        parts.append(gpri[: t - 1])
        regressors.append("gpri_z")
    X = np.column_stack(parts)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for j, sleeve in enumerate(SLEEVE_IDS):
        fit = newey_west_ols(y_all[:, j], X, lags=lags)
        coef, se, tstat = fit["coef"], fit["se"], fit["tstat"]
        sleeve_row: dict[str, dict[str, float]] = {
            "intercept": {
                "coef": float(coef[0]),
                "se": float(se[0]),
                "tstat": float(tstat[0]),
            }
        }
        for k, name in enumerate(regressors):
            sleeve_row[name] = {
                "coef": float(coef[k + 1]),
                "se": float(se[k + 1]),
                "tstat": float(tstat[k + 1]),
            }
        out[sleeve] = sleeve_row
    return out


def regime_tilt_variances(
    weights: np.ndarray,
    *,
    regimes: Sequence[str] | np.ndarray,
    sleeve_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Across-regime and within-regime variance of the 7-d tilt vector."""
    w = _as_w(weights)
    reg = np.asarray(list(regimes), dtype=object).reshape(-1)
    tilts = sleeve_tilt_series(w, _sleeve_matrix(sleeve_matrix, w.shape[1]))
    means = []
    within = []
    for rid in REGIME_IDS:
        mask = reg == rid
        if not np.any(mask):
            continue
        block = tilts[mask]
        means.append(block.mean(axis=0))
        within.append(float(np.mean(np.var(block, axis=0, ddof=0))))
    if len(means) >= 2:
        across = float(np.mean(np.var(np.stack(means, axis=0), axis=0, ddof=0)))
    else:
        across = float("nan")
    within_mean = float(np.mean(within)) if within else float("nan")
    return {
        "across_regime_tilt_variance": across,
        "within_regime_tilt_variance": within_mean,
    }


def regime_behaviour_deltas(
    by_regime: Mapping[str, Mapping[str, float]],
    *,
    quality_by_regime: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Crisis-minus-calm deltas for concentration / turnover / defensive / quality."""
    calm = dict(by_regime.get("calm") or {})
    crisis = dict(by_regime.get("crisis") or {})

    def _delta(key: str) -> float:
        a = calm.get(key)
        b = crisis.get(key)
        try:
            af = float(a) if a is not None else float("nan")
            bf = float(b) if b is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")
        if not (np.isfinite(af) and np.isfinite(bf)):
            return float("nan")
        return float(bf - af)

    out = {
        "delta_hhi_regime": _delta("hhi_mean"),
        "delta_turnover_regime": _delta("turnover_mean"),
        "delta_defensive_regime": _delta("tilt_defensive"),
        "delta_quality_regime": float("nan"),
    }
    if quality_by_regime:
        try:
            qc = float(quality_by_regime.get("calm", float("nan")))
            qr = float(quality_by_regime.get("crisis", float("nan")))
            if np.isfinite(qc) and np.isfinite(qr):
                out["delta_quality_regime"] = float(qr - qc)
        except (TypeError, ValueError):
            pass
    return out


def turbulence_regimes_from_returns(
    asset_returns: np.ndarray | None,
    *,
    existing: Sequence[str] | np.ndarray | None = None,
    macro_cols: np.ndarray | None = None,
    crisis_mask: np.ndarray | None = None,
    overlay_mode: str = "markov",
    hmm_window: int = 252 * 3,
    hmm_step: int = 21,
) -> np.ndarray | None:
    """Merge Kritzman turbulence into a 3-state regime array.

    Only promotes ``calm`` → ``crisis`` on turbulent days. Leaves
    ``inflationary`` (and existing ``crisis``) unchanged so Layer-4 inflation
    cells stay interpretable.

    Default ``overlay_mode="markov"`` uses filtered P>0.5 hard labels on daily
    turbulence. Pass ``crisis_mask`` to avoid a second Markov fit (scorecard).
    ``overlay_mode="q75"`` keeps the Skulls expanding-quantile path.
    """
    if asset_returns is None:
        return np.asarray(list(existing), dtype=object) if existing is not None else None
    from src.eval.turbulence import classify_regime, turbulence_index

    r = np.asarray(asset_returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < r.shape[1] + 2:
        return np.asarray(list(existing), dtype=object) if existing is not None else None
    t = r.shape[0]

    if crisis_mask is not None:
        mask = np.asarray(crisis_mask, dtype=bool).reshape(-1)
        if mask.size != t:
            raise ValueError(
                f"crisis_mask length {mask.size} != asset_returns T={t}"
            )
    elif overlay_mode == "q75":
        try:
            turb = turbulence_index(r, macro_cols=macro_cols)
            mask = classify_regime(turb)
        except Exception:
            return (
                np.asarray(list(existing), dtype=object)
                if existing is not None
                else None
            )
    elif overlay_mode == "markov":
        try:
            from src.eval.walk_forward_hmm import walk_forward_markov_filter

            turb = turbulence_index(r, macro_cols=macro_cols)
            filt = walk_forward_markov_filter(
                turb,
                window=int(hmm_window),
                step=int(hmm_step),
                k_regimes=2,
                growing=False,
            )
            mask = np.asarray(filt["hard"], dtype=np.int32) == 1
        except Exception:
            return (
                np.asarray(list(existing), dtype=object)
                if existing is not None
                else None
            )
    else:
        raise ValueError(
            f"overlay_mode must be 'markov' or 'q75'; got {overlay_mode!r}"
        )

    if existing is not None:
        reg = np.asarray(list(existing), dtype=object).reshape(-1)
        if reg.size != t:
            reg = np.full(t, "calm", dtype=object)
    else:
        reg = np.full(t, "calm", dtype=object)
    for i, flag in enumerate(mask):
        if flag and str(reg[i]) == "calm":
            # Option (a): only promote calm→crisis; preserve inflationary taxonomy.
            reg[i] = "crisis"
        elif str(reg[i]) not in REGIME_IDS:
            reg[i] = "calm"
    return reg

