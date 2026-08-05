"""Part E.5: mechanism explanations with shuffle-null verdicts."""
from __future__ import annotations

import numpy as np

from mascotrl.reporting.behavior_explain import explain_behaviour, shuffle_null_band


def _cvar_cfg(**extra):
    cfg = {
        "objective": "cvar_ru",
        "algo": "ppo",
        "architecture": "mlp",
        "policy_mode": "balanced",
    }
    cfg.update(extra)
    return cfg


def test_cvar_defensive_tilt_outside_band_consistent():
    behaviour = {
        "tilt_defensive": 0.20,
        "tilt_lottery": -0.15,
        "action_entropy_mean": 1.0,
        "hhi_mean": 0.2,
        "n_eff_mean": 4.0,
        "weight_autocorr_lag1": 0.5,
        "tilt_autocorr_lag21": 0.1,
        "rotation_rate": 0.1,
        "turnover_cap_binding_frac": 0.2,
        "downside_capture": 0.8,
    }
    # Null band tightly around zero so 0.20 is outside
    null_band = {
        "tilt_defensive": [-0.02, 0.02],
        "tilt_lottery": [-0.02, 0.02],
    }
    out = explain_behaviour(_cvar_cfg(), behaviour, macro_sens={}, null_band=null_band)
    mech = next(m for m in out["explanations"] if m["mechanism"] == "objective_to_risk_shape")
    assert mech["verdict"] == "consistent"
    assert mech["predicted"]["tilt_defensive"] == "positive"
    assert mech["predicted"]["tilt_lottery"] == "negative"


def test_cvar_lottery_tilt_outside_band_inconsistent():
    behaviour = {
        "tilt_defensive": -0.01,
        "tilt_lottery": 0.25,
        "action_entropy_mean": 1.0,
        "hhi_mean": 0.2,
        "n_eff_mean": 4.0,
        "weight_autocorr_lag1": 0.5,
        "tilt_autocorr_lag21": 0.1,
        "rotation_rate": 0.1,
        "turnover_cap_binding_frac": 0.2,
        "downside_capture": 1.0,
    }
    null_band = {
        "tilt_defensive": [-0.05, 0.05],
        "tilt_lottery": [-0.02, 0.02],
    }
    out = explain_behaviour(_cvar_cfg(), behaviour, macro_sens={}, null_band=null_band)
    mech = next(m for m in out["explanations"] if m["mechanism"] == "objective_to_risk_shape")
    # lottery predicted negative but observed positive outside band
    assert mech["verdict"] == "inconsistent"


def test_tilt_inside_null_band_inconclusive():
    behaviour = {
        "tilt_defensive": 0.005,
        "tilt_lottery": -0.004,
        "action_entropy_mean": 1.0,
        "hhi_mean": 0.2,
        "n_eff_mean": 4.0,
        "weight_autocorr_lag1": 0.5,
        "tilt_autocorr_lag21": 0.1,
        "rotation_rate": 0.1,
        "turnover_cap_binding_frac": 0.2,
        "downside_capture": 1.0,
    }
    null_band = {
        "tilt_defensive": [-0.05, 0.05],
        "tilt_lottery": [-0.05, 0.05],
    }
    out = explain_behaviour(_cvar_cfg(), behaviour, macro_sens={}, null_band=null_band)
    mech = next(m for m in out["explanations"] if m["mechanism"] == "objective_to_risk_shape")
    assert mech["verdict"] == "inconclusive"


def test_policy_mode_mechanism_labelled_definitional():
    behaviour = {
        "tilt_defensive": 0.0,
        "tilt_lottery": 0.0,
        "action_entropy_mean": 1.0,
        "hhi_mean": 0.2,
        "n_eff_mean": 4.0,
        "weight_autocorr_lag1": 0.5,
        "tilt_autocorr_lag21": 0.1,
        "rotation_rate": 0.1,
        "turnover_cap_binding_frac": 0.5,
        "downside_capture": 1.0,
    }
    out = explain_behaviour(
        _cvar_cfg(policy_mode="conservative"),
        behaviour,
        macro_sens={},
        null_band={},
    )
    mech = next(m for m in out["explanations"] if m["mechanism"] == "mandate_to_constraint_binding")
    assert "definitional" in mech["note"].lower()
    assert len(out["explanations"]) == 4


def test_shuffle_null_band_returns_lo_hi():
    W = np.full((40, 4), 0.25)
    W[::5, 0] = 0.7
    W[::5, 1:] = 0.1
    S = np.eye(4, 7)
    band = shuffle_null_band(
        W,
        sleeve_matrix=S,
        measure_keys=["tilt_defensive", "hhi_mean"],
        n_shuffles=50,
        seed=0,
    )
    assert "tilt_defensive" in band
    lo, hi = band["tilt_defensive"]
    assert lo <= hi
