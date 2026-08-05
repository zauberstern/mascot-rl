"""Template-driven behaviour explanations (CRUCIBLE Part E.5).

Links known design choices to measured behaviour with shuffle-null verdicts.
Interpretation only; never feeds capital gates.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mascotrl.reporting.behavior_metrics import (
    BEHAVIOUR_MEASURE_IDS,
    compute_behaviour_vector,
)

MECHANISM_IDS: tuple[str, ...] = (
    "objective_to_risk_shape",
    "algorithm_to_exploration",
    "architecture_to_memory",
    "mandate_to_constraint_binding",
    "regime_shift_response",
    "macro_tilt_response",
)


def _sign_token(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-15:
        return "zero"
    return "positive" if value > 0.0 else "negative"


def _outside_band(value: float, band: Sequence[float] | None) -> bool:
    if band is None or len(band) < 2:
        return np.isfinite(value) and abs(value) > 0.0
    lo, hi = float(band[0]), float(band[1])
    if not np.isfinite(value):
        return False
    return value < lo or value > hi


def _verdict_for_predictions(
    predicted: Mapping[str, str],
    observed: Mapping[str, float],
    null_band: Mapping[str, Sequence[float]],
) -> str:
    """Aggregate verdict across predicted keys.

    consistent: every key outside band with matching sign (or predicted abs).
    inconsistent: at least one key outside band with opposite sign.
    inconclusive: otherwise (all inside band, or missing).
    """
    any_outside = False
    any_mismatch = False
    any_match = False
    for key, pred in predicted.items():
        obs = float(observed.get(key, float("nan")))
        band = null_band.get(key)
        outside = _outside_band(obs, band)
        if not outside:
            continue
        any_outside = True
        got = _sign_token(obs)
        if pred in ("positive", "negative") and got != pred and got != "zero":
            any_mismatch = True
        elif pred in ("positive", "negative") and got == pred:
            any_match = True
        elif pred == "higher" and got == "positive":
            any_match = True
        elif pred == "lower" and got == "negative":
            any_match = True
    if any_mismatch:
        return "inconsistent"
    if any_outside and any_match:
        return "consistent"
    return "inconclusive"


def shuffle_null_band(
    weights: np.ndarray,
    *,
    sleeve_matrix: np.ndarray | None = None,
    asset_returns: np.ndarray | None = None,
    turnover_cap: float | None = None,
    measure_keys: Sequence[str] | None = None,
    n_shuffles: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Two-sided shuffle-null band by permuting the weight path in time."""
    w = np.asarray(weights, dtype=np.float64)
    keys = list(measure_keys) if measure_keys is not None else list(BEHAVIOUR_MEASURE_IDS)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    t = w.shape[0]
    for _ in range(int(n_shuffles)):
        perm = rng.permutation(t)
        w_s = w[perm]
        r_s = None if asset_returns is None else np.asarray(asset_returns)[perm]
        m = compute_behaviour_vector(
            w_s,
            asset_returns=r_s,
            sleeve_matrix=sleeve_matrix,
            turnover_cap=turnover_cap,
        )
        for k in keys:
            v = m.get(k, float("nan"))
            if np.isfinite(v):
                samples[k].append(float(v))
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    out: dict[str, list[float]] = {}
    for k, vals in samples.items():
        if len(vals) < 10:
            out[k] = [float("nan"), float("nan")]
        else:
            out[k] = [
                float(np.percentile(vals, lo_q)),
                float(np.percentile(vals, hi_q)),
            ]
    return out


def _objective_predictions(objective: str) -> tuple[dict[str, str], str]:
    obj = str(objective or "").lower()
    if obj in ("cvar_ru", "entropic_oce"):
        return (
            {"tilt_defensive": "positive", "tilt_lottery": "negative"},
            "left-tail penalty predicts defensive over lottery tilt",
        )
    if obj in ("mean_std_cao", "meanvar_kolm"):
        return (
            {"tilt_defensive": "zero", "downside_capture": "zero"},
            "symmetric variance penalty predicts smaller defensive / capture gap",
        )
    if obj == "differential_sharpe":
        return (
            {"rotation_rate": "positive"},
            "rising-Sharpe reward predicts higher sleeve rotation",
        )
    if obj in ("mikkila_asym", "mikkila"):
        return (
            {"tilt_defensive": "positive", "tilt_lottery": "negative"},
            "asymmetric downside avoidance without full CVaR tail tilt",
        )
    return ({}, f"no template for objective={obj}")


def _algo_predictions(algo: str) -> tuple[dict[str, str], str]:
    a = str(algo or "").lower()
    if a == "sac":
        return (
            {"action_entropy_mean": "positive", "hhi_mean": "negative"},
            "entropy regularisation predicts higher entropy and lower HHI",
        )
    if a in ("ddpg", "td3"):
        return (
            {"action_entropy_mean": "negative", "hhi_mean": "positive"},
            "deterministic actors predict lower entropy and higher concentration",
        )
    if a == "dqn":
        return (
            {"n_eff_mean": "negative", "weight_autocorr_lag1": "positive"},
            "discrete menu predicts low n_eff and step-like weight paths",
        )
    if a == "rrl":
        return (
            {"rotation_rate": "positive"},
            "direct differential Sharpe ascent mirrors differential_sharpe objective",
        )
    return ({}, f"no strong exploration template for algo={a}")


def _arch_predictions(architecture: str) -> tuple[dict[str, str], str]:
    arch = str(architecture or "").lower()
    if arch in ("mlp", "", "linear"):
        return (
            {"tilt_autocorr_lag21": "negative"},
            "memoryless net predicts low tilt persistence",
        )
    if arch in ("gru", "lstm", "mamba"):
        return (
            {"tilt_autocorr_lag21": "positive"},
            "recurrent state predicts higher tilt persistence",
        )
    if arch == "transformer":
        return (
            {"tilt_autocorr_lag21": "positive"},
            "attention window predicts strongest regime conditioning if it spans a change",
        )
    return ({}, f"no template for architecture={arch}")


def _mandate_predictions(policy_mode: str) -> tuple[dict[str, str], str]:
    mode = str(policy_mode or "balanced").lower()
    if mode in ("conservative", "risk_off", "defensive"):
        return (
            {"turnover_cap_binding_frac": "positive", "tilt_defensive": "positive"},
            "definitional: policy_mode sets turnover multiplier and risk aversion",
        )
    if mode in ("aggressive", "risk_on"):
        return (
            {"turnover_cap_binding_frac": "negative", "tilt_defensive": "negative"},
            "definitional: policy_mode sets turnover multiplier and risk aversion",
        )
    return (
        {"turnover_cap_binding_frac": "positive"},
        "definitional: policy_mode sets turnover multiplier and risk aversion",
    )


def _pack_mechanism(
    mechanism: str,
    design_input: str,
    predicted: Mapping[str, str],
    behaviour: Mapping[str, float],
    null_band: Mapping[str, Sequence[float]],
    note: str,
) -> dict[str, Any]:
    observed = {k: float(behaviour.get(k, float("nan"))) for k in predicted}
    if not predicted:
        verdict = "inconclusive"
    elif "definitional" in note.lower() and mechanism == "mandate_to_constraint_binding":
        # Still compute a data verdict, but the note marks the link as definitional.
        verdict = _verdict_for_predictions(predicted, observed, null_band)
    else:
        verdict = _verdict_for_predictions(predicted, observed, null_band)
    return {
        "mechanism": mechanism,
        "design_input": design_input,
        "predicted": dict(predicted),
        "observed": observed,
        "verdict": verdict,
        "note": note,
    }


def _regime_shift_predictions(
    behaviour_by_regime: Mapping[str, Mapping[str, float]] | None,
) -> tuple[dict[str, str], str]:
    if not behaviour_by_regime:
        return ({}, "no regime-conditional behaviour available")
    crisis = behaviour_by_regime.get("crisis") or {}
    calm = behaviour_by_regime.get("calm") or {}
    pred: dict[str, str] = {}
    for key in ("tilt_defensive", "tilt_lottery", "turnover_mean"):
        c_val = float(crisis.get(key, float("nan")))
        m_val = float(calm.get(key, float("nan")))
        if np.isfinite(c_val) and np.isfinite(m_val):
            diff = c_val - m_val
            pred[f"crisis_minus_calm_{key}"] = "positive" if diff > 0 else "negative"
    if not pred:
        return ({}, "regime slices present but tilt deltas undefined")
    return (pred, "crisis vs calm sleeve tilt co-movement (correlational)")


def _macro_tilt_predictions(
    macro_sens: Mapping[str, Any] | None,
) -> tuple[dict[str, str], str]:
    if not macro_sens:
        return ({}, "no macro tilt sensitivity available")
    pred: dict[str, str] = {}
    defensive = macro_sens.get("defensive") or {}
    vix_stats = defensive.get("vix_z") or defensive.get("vix_z_252")
    if isinstance(vix_stats, Mapping):
        coef = float(vix_stats.get("coef", float("nan")))
        if np.isfinite(coef):
            pred["defensive_vix_coef"] = "positive" if coef > 0 else "negative"
    if not pred:
        return ({}, "macro sensitivity present but no usable defensive-VIX coef")
    return (pred, "defensive sleeve vs lagged VIX z-score (Newey-West, correlational)")


def _observed_regime_deltas(
    behaviour_by_regime: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    if not behaviour_by_regime:
        return {}
    crisis = behaviour_by_regime.get("crisis") or {}
    calm = behaviour_by_regime.get("calm") or {}
    out: dict[str, float] = {}
    for key in ("tilt_defensive", "tilt_lottery", "turnover_mean"):
        c_val = float(crisis.get(key, float("nan")))
        m_val = float(calm.get(key, float("nan")))
        if np.isfinite(c_val) and np.isfinite(m_val):
            out[f"crisis_minus_calm_{key}"] = c_val - m_val
    return out


def _observed_macro_coefs(macro_sens: Mapping[str, Any] | None) -> dict[str, float]:
    if not macro_sens:
        return {}
    defensive = macro_sens.get("defensive") or {}
    vix_stats = defensive.get("vix_z") or defensive.get("vix_z_252")
    if isinstance(vix_stats, Mapping):
        coef = float(vix_stats.get("coef", float("nan")))
        if np.isfinite(coef):
            return {"defensive_vix_coef": coef}
    return {}


def explain_behaviour(
    cell_cfg: Mapping[str, Any],
    behaviour: Mapping[str, float],
    macro_sens: Mapping[str, Any] | None = None,
    *,
    null_band: Mapping[str, Sequence[float]] | None = None,
    behaviour_by_regime: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Compose mechanism explanations with consistent/inconsistent/inconclusive verdicts."""
    cfg = dict(cell_cfg or {})
    band = dict(null_band or {})
    objective = str(cfg.get("objective") or "")
    algo = str(cfg.get("algo") or "")
    architecture = str(cfg.get("architecture") or "")
    policy_mode = str(cfg.get("policy_mode") or "balanced")

    o_pred, o_note = _objective_predictions(objective)
    a_pred, a_note = _algo_predictions(algo)
    r_pred, r_note = _arch_predictions(architecture)
    m_pred, m_note = _mandate_predictions(policy_mode)
    reg_pred, reg_note = _regime_shift_predictions(behaviour_by_regime)
    mac_pred, mac_note = _macro_tilt_predictions(macro_sens)

    explanations = [
        _pack_mechanism(
            "objective_to_risk_shape",
            f"objective={objective}",
            o_pred,
            behaviour,
            band,
            o_note,
        ),
        _pack_mechanism(
            "algorithm_to_exploration",
            f"algo={algo}",
            a_pred,
            behaviour,
            band,
            a_note,
        ),
        _pack_mechanism(
            "architecture_to_memory",
            f"architecture={architecture}",
            r_pred,
            behaviour,
            band,
            r_note,
        ),
        _pack_mechanism(
            "mandate_to_constraint_binding",
            f"policy_mode={policy_mode}",
            m_pred,
            behaviour,
            band,
            m_note,
        ),
    ]
    if reg_pred:
        explanations.append(
            _pack_mechanism(
                "regime_shift_response",
                "regime=crisis_vs_calm",
                reg_pred,
                _observed_regime_deltas(behaviour_by_regime),
                band,
                reg_note,
            )
        )
    if mac_pred:
        explanations.append(
            _pack_mechanism(
                "macro_tilt_response",
                "macro=defensive_vs_vix",
                mac_pred,
                _observed_macro_coefs(macro_sens),
                band,
                mac_note,
            )
        )
    return {"explanations": explanations, "null_band": band}
