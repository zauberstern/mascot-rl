"""Measured trader-archetype diagnostics (CRUCIBLE Part E.4 / E.6).

Interpretation aid only. Never feeds capital gates. Archetype labels come from
scored behaviour measures, not from the algo string.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.reporting.behavior_explain import explain_behaviour, shuffle_null_band
from src.reporting.behavior_metrics import (
    BEHAVIOUR_MEASURE_IDS,
    SLEEVE_IDS,
    compute_behaviour_vector,
    macro_tilt_sensitivity,
    regime_conditional_behaviour,
    regime_tilt_variances,
    sleeve_tilt_series,
)

# Frozen a priori (Part E.4). Do not tune to make labels nicer.
# Reduced to 6 archetypes (5 named + mixed/balanced): carry_harvester,
# liquidity_provider, and index_hugger dropped — they never separated in PICK
# (sleeve-context measures unavailable in spectrum cells).
ARCHETYPE_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "trend_follower": {
        "tilt_trend": 0.5,
        "tilt_autocorr_lag21": 0.2,
        "holding_period_days": 0.2,
        "tilt_reversal": -0.1,
    },
    "contrarian": {
        "tilt_reversal": 0.5,
        "turnover_mean": 0.3,
        "tilt_trend": -0.2,
    },
    "risk_manager": {
        "tilt_defensive": 0.4,
        "neg_downside_capture": 0.3,
        "neg_max_weight_mean": 0.2,
        "b_defensive_vix": 0.1,
    },
    "speculator": {
        "tilt_lottery": 0.4,
        "hhi_mean": 0.3,
        "max_weight_mean": 0.2,
        "upside_capture": 0.1,
    },
    "tactical_rotator": {
        "rotation_rate": 0.5,
        "across_regime_tilt_variance": 0.3,
        "within_regime_tilt_variance": -0.2,
    },
}

ARCHETYPE_IDS: tuple[str, ...] = tuple(ARCHETYPE_SCORE_WEIGHTS.keys())
ARCHETYPE_MARGIN: float = 0.25
_NEG_PREFIX = "neg_"

# Pre-registered designed personality.
# Keys are 3-tuples (objective, algo, weight_head) or 4-tuples
# (objective, algo, weight_head, mandate_preset). Used for designed-vs-observed
# alignment scoring (FIGURE_WRITEUP Ch.9.4 / MOTIVATION.md Section C).
PERSONALITY_DESIGN_MAP: dict[tuple[str, ...], str] = {
    # Softmax / tanh_l1 controls (legacy).
    ("mean_std_cao", "ppo", "softmax"): "mixed",
    ("cvar_ru", "cppo", "softmax"): "risk_manager",
    ("cvar_ru", "ppo", "softmax"): "risk_manager",
    ("differential_sharpe", "ppo", "softmax"): "trend_follower",
    ("mtm_pnl", "ddpg", "tanh_l1"): "speculator",
    ("mtm_pnl", "ppo", "softmax"): "speculator",
    ("mean_std_cao", "happo", "softmax"): "tactical_rotator",
    ("meanvar_kolm", "ppo", "softmax"): "contrarian",
    # Sparse-tilt treatment (Scenario A Ch.9.4 designed personalities).
    ("differential_sharpe", "ppo", "sparse_tilt"): "trend_follower",
    ("meanvar_kolm", "ppo", "sparse_tilt"): "contrarian",
    ("cvar_ru", "ppo", "sparse_tilt"): "risk_manager",
    ("cvar_ru", "cppo", "sparse_tilt"): "risk_manager",
    ("mtm_pnl", "ppo", "sparse_tilt"): "speculator",
    ("mtm_pnl", "ddpg", "sparse_tilt"): "speculator",
    ("mtm_pnl", "sac", "sparse_tilt"): "speculator",
    ("mtm_pnl", "td3", "sparse_tilt"): "speculator",
    ("mtm_pnl", "mcpg", "sparse_tilt"): "speculator",
    ("mean_std_cao", "happo", "sparse_tilt"): "tactical_rotator",
    ("mean_std_cao", "ppo", "sparse_tilt"): "mixed",
    ("entropic_oce", "ppo", "sparse_tilt"): "contrarian",
    ("sdr_composite", "ppo", "sparse_tilt"): "trend_follower",
    ("rsqp", "ppo", "sparse_tilt"): "risk_manager",
    ("smse", "ppo", "sparse_tilt"): "risk_manager",
    ("mikkila_asym", "ppo", "sparse_tilt"): "speculator",
    ("mikkila_asym", "ddpg", "sparse_tilt"): "speculator",
    ("mikkila_asym", "sac", "sparse_tilt"): "speculator",
    ("mikkila_asym", "td3", "sparse_tilt"): "speculator",
    # Mandate-preset Hummingbird proxy (Ch.9.4 tactical rotator under sparse tilt).
    ("mean_std_cao", "ppo", "sparse_tilt", "archetype_carry"): "tactical_rotator",
    ("mean_std_cao", "ppo", "sparse_tilt", "archetype_crisis"): "tactical_rotator",
    ("mean_std_cao", "ppo", "sparse_tilt", "archetype_inflation"): "tactical_rotator",
}


def normalize_weight_head(raw: str) -> str:
    """Map config policy_mode / weight_head strings onto design-map heads."""
    s = str(raw or "softmax").lower().strip()
    if "sparse" in s or s == "tilt":
        return "sparse_tilt"
    if "tanh" in s:
        return "tanh_l1"
    if "softmax" in s:
        return "softmax"
    if "dirichlet" in s:
        return "dirichlet"
    if "entmax" in s:
        return "entmax"
    if "discrete" in s:
        return "discrete"
    return s


def designed_personality(
    *,
    objective: str,
    algo: str,
    weight_head: str = "softmax",
    mandate_preset: str = "",
) -> str:
    """Look up the pre-registered designed personality for a config triple/quad."""
    head = normalize_weight_head(weight_head)
    obj = str(objective or "").lower()
    alg = str(algo or "").lower()
    mandate = str(mandate_preset or "").lower().strip()
    if mandate.startswith("pm-"):
        mandate = mandate[len("pm-") :]
    key4 = (obj, alg, head, mandate)
    key3 = (obj, alg, head)
    if mandate and key4 in PERSONALITY_DESIGN_MAP:
        return PERSONALITY_DESIGN_MAP[key4]
    return PERSONALITY_DESIGN_MAP.get(key3, "mixed")


def compute_personality_alignment(
    designed: str,
    observed: Mapping[str, Any],
    *,
    attribution: Mapping[str, Any] | None = None,
    alignment_threshold: float = 0.4,
) -> dict[str, Any]:
    """Quantify designed vs observed personality alignment.

    ``alignment_score`` is Jaccard overlap of the designed archetype's scored
    measures (from ARCHETYPE_SCORE_WEIGHTS) against the observed top-5
    behaviour measures by absolute score contribution. Divergences are
    reported honestly, not buried.
    """
    obs_primary = str(
        observed.get("archetype_primary")
        or observed.get("assignment")
        or observed.get("archetype")
        or "mixed"
    )
    designed_l = str(designed or "mixed").lower()
    observed_l = obs_primary.lower()
    designed_measures = set(
        (ARCHETYPE_SCORE_WEIGHTS.get(designed_l) or {}).keys()
    )
    # Strip neg_ prefix for overlap comparison with raw measure ids.
    designed_raw = {
        m[len(_NEG_PREFIX) :] if m.startswith(_NEG_PREFIX) else m
        for m in designed_measures
    }
    scores = dict(observed.get("archetype_scores") or observed.get("scores") or {})
    if not scores and "behaviour" in observed:
        scores = dict((observed.get("behaviour") or {}).get("scores") or {})
    # Top-5 observed measures by absolute score weight contribution if available;
    # otherwise use primary archetype's own measure set.
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        observed_raw = set()
        for arch, _ in ranked[:1]:
            for m in (ARCHETYPE_SCORE_WEIGHTS.get(str(arch)) or {}):
                observed_raw.add(
                    m[len(_NEG_PREFIX) :] if m.startswith(_NEG_PREFIX) else m
                )
        # Also include top measures from the observed primary.
        for m in (ARCHETYPE_SCORE_WEIGHTS.get(observed_l) or {}):
            observed_raw.add(
                m[len(_NEG_PREFIX) :] if m.startswith(_NEG_PREFIX) else m
            )
    else:
        observed_raw = {
            m[len(_NEG_PREFIX) :] if m.startswith(_NEG_PREFIX) else m
            for m in (ARCHETYPE_SCORE_WEIGHTS.get(observed_l) or {})
        }
    if not designed_raw and not observed_raw:
        jaccard = 1.0 if designed_l == observed_l else 0.0
    else:
        inter = designed_raw & observed_raw
        union = designed_raw | observed_raw
        jaccard = float(len(inter) / len(union)) if union else 0.0
    match = designed_l == observed_l
    if match:
        divergence = ""
    else:
        divergence = (
            f"designed={designed_l} observed={observed_l} "
            f"jaccard={jaccard:.3f}"
        )
    top_attr = []
    if attribution:
        top_attr = list(
            (attribution.get("top_groups") or [])[:5]
        )
    return {
        "designed_personality": designed_l,
        "observed_personality": observed_l,
        "match": bool(match),
        "alignment_score": float(jaccard),
        "alignment_pass": bool(jaccard >= float(alignment_threshold) or match),
        "divergence_explanation": divergence,
        "top_attributed_groups": top_attr,
        "alignment_threshold": float(alignment_threshold),
    }


def _build_data_availability(
    *,
    regimes: Sequence[str] | np.ndarray | None,
    vix_z: np.ndarray | None,
    hy_oas_z: np.ndarray | None,
    term_spread: np.ndarray | None,
    sleeve_matrix: np.ndarray | None,
    sensitivities: Mapping[str, float] | None,
    by_regime: Mapping[str, Any],
    macro_sens: Mapping[str, Any],
    tilt_series: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Stamp which optional behaviour inputs were available at export time."""
    missing: dict[str, str] = {}
    has_regimes = regimes is not None and len(np.asarray(regimes).reshape(-1)) > 0
    has_macro_series = (
        vix_z is not None
        and hy_oas_z is not None
        and term_spread is not None
        and bool(macro_sens)
    )
    has_sleeves = sleeve_matrix is not None and any(
        len(list(v)) > 0 for v in tilt_series.values()
    )
    has_sens = bool(sensitivities)
    if not has_regimes:
        missing["regimes"] = "regime_labels_not_provided"
    elif not by_regime:
        missing["regimes"] = "regime_conditional_behaviour_empty"
    if not has_macro_series:
        if vix_z is None or hy_oas_z is None or term_spread is None:
            missing["macro"] = "macro_series_not_provided"
        else:
            missing["macro"] = "macro_tilt_sensitivity_empty"
    if sleeve_matrix is None:
        missing["sleeves"] = "sleeve_matrix_not_provided"
    elif not has_sleeves:
        missing["sleeves"] = "sleeve_tilt_series_empty"
    if not has_sens:
        missing["sensitivities"] = "policy_sensitivities_not_computed"
    return {
        "regimes": bool(by_regime),
        "macro": bool(macro_sens),
        "sleeves": has_sleeves,
        "sensitivities": has_sens,
        "missing_reason": missing,
    }


def weight_concentration(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    k = max(w.shape[1], 1)
    ew = np.full(k, 1.0 / k)
    hhi = np.sum(np.square(np.nan_to_num(w, nan=0.0)), axis=1)
    l1 = np.sum(np.abs(np.nan_to_num(w, nan=0.0) - ew), axis=1)
    mx = np.max(np.nan_to_num(w, nan=0.0), axis=1)
    return {
        "hhi_mean": float(np.mean(hhi)),
        "l1_vs_ew_mean": float(np.mean(l1)),
        "max_weight_mean": float(np.mean(mx)),
    }


def signal_weight_sensitivity(
    agent: Any,
    obs: np.ndarray,
    *,
    channel_index: int,
    eps: float = 1e-3,
) -> float:
    """Finite-difference L1 change in deterministic weights w.r.t. one obs channel."""
    import torch

    x = np.asarray(obs, dtype=np.float64).reshape(1, -1).copy()
    if channel_index < 0 or channel_index >= x.shape[1]:
        raise ValueError(f"channel_index {channel_index} out of range for obs dim {x.shape[1]}")
    with torch.no_grad():
        base = agent.act(torch.as_tensor(x, dtype=torch.float32), deterministic=True)
        w0 = base.detach().cpu().numpy().reshape(-1)
        x2 = x.copy()
        x2[0, channel_index] += float(eps)
        pert = agent.act(torch.as_tensor(x2, dtype=torch.float32), deterministic=True)
        w1 = pert.detach().cpu().numpy().reshape(-1)
    return float(np.sum(np.abs(w1 - w0)) / max(abs(float(eps)), 1e-12))


def _feature_value(row: Mapping[str, float], key: str) -> float:
    if key.startswith(_NEG_PREFIX):
        base = key[len(_NEG_PREFIX) :]
        v = float(row.get(base, float("nan")))
        return -v if np.isfinite(v) else float("nan")
    return float(row.get(key, float("nan")))


def _zscore_panel(
    rows: Sequence[Mapping[str, float]], keys: Sequence[str]
) -> list[dict[str, float]]:
    n = len(rows)
    zrows: list[dict[str, float]] = [dict() for _ in range(n)]
    for key in keys:
        vals = np.asarray([_feature_value(r, key) for r in rows], dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            for i in range(n):
                zrows[i][key] = 0.0
            continue
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=0))
        if sd < 1e-12:
            for i in range(n):
                zrows[i][key] = 0.0
        else:
            for i in range(n):
                v = vals[i]
                zrows[i][key] = float((v - mu) / sd) if np.isfinite(v) else 0.0
    return zrows


def score_archetypes(
    behaviour_rows: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    """Cross-cell z-scored archetype scores. Relative within the provided panel."""
    feature_keys: list[str] = []
    for weights in ARCHETYPE_SCORE_WEIGHTS.values():
        for k in weights:
            if k not in feature_keys:
                feature_keys.append(k)
    zrows = _zscore_panel(behaviour_rows, feature_keys)
    out: list[dict[str, float]] = []
    for z in zrows:
        scores: dict[str, float] = {}
        for arch, weights in ARCHETYPE_SCORE_WEIGHTS.items():
            s = 0.0
            for feat, w in weights.items():
                s += float(w) * float(z.get(feat, 0.0))
            scores[arch] = float(s)
        out.append(scores)
    return out


def assign_archetype(scores: Mapping[str, float]) -> dict[str, Any]:
    """Argmax with 0.25 margin; else primary ``mixed`` and report top two."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ordered:
        return {
            "archetype_primary": "mixed",
            "archetype_runner_up": "",
            "archetype_margin": float("nan"),
        }
    primary, pscore = ordered[0]
    runner, rscore = ordered[1] if len(ordered) > 1 else ("", pscore)
    margin = float(pscore - rscore) if runner else float("inf")
    if runner and margin < ARCHETYPE_MARGIN:
        return {
            "archetype_primary": "mixed",
            "archetype_runner_up": runner,
            "archetype_margin": margin,
            "archetype_top": primary,
        }
    return {
        "archetype_primary": primary,
        "archetype_runner_up": runner,
        "archetype_margin": margin,
    }


def _enrich_scoring_row(
    behaviour: Mapping[str, float],
    *,
    macro_sens: Mapping[str, Any] | None = None,
    regime_vars: Mapping[str, float] | None = None,
) -> dict[str, float]:
    row = {k: float(behaviour.get(k, float("nan"))) for k in BEHAVIOUR_MEASURE_IDS}
    if regime_vars:
        row["across_regime_tilt_variance"] = float(
            regime_vars.get("across_regime_tilt_variance", float("nan"))
        )
        row["within_regime_tilt_variance"] = float(
            regime_vars.get("within_regime_tilt_variance", float("nan"))
        )
    else:
        row.setdefault("across_regime_tilt_variance", float("nan"))
        row.setdefault("within_regime_tilt_variance", float("nan"))
    b_vix = float("nan")
    if macro_sens and "defensive" in macro_sens:
        b_vix = float((macro_sens["defensive"].get("vix_z") or {}).get("coef", float("nan")))
    row["b_defensive_vix"] = b_vix
    return row


def build_policy_behavior(
    *,
    algo: str = "",
    weights: np.ndarray | None = None,
    turnovers: Sequence[float] | None = None,
    entropies: Sequence[float] | None = None,
    sensitivities: Mapping[str, float] | None = None,
    extras: Mapping[str, Any] | None = None,
    cell_id: str = "",
    arm: str = "",
    architecture: str = "",
    objective: str = "",
    train_world: str = "",
    policy_mode: str = "",
    universe_fingerprint: str = "",
    asset_returns: np.ndarray | None = None,
    sleeve_matrix: np.ndarray | None = None,
    regimes: Sequence[str] | np.ndarray | None = None,
    vix_z: np.ndarray | None = None,
    hy_oas_z: np.ndarray | None = None,
    term_spread: np.ndarray | None = None,
    epu_z: np.ndarray | None = None,
    gpri_z: np.ndarray | None = None,
    turnover_cap: float | None = None,
    cell_cfg: Mapping[str, Any] | None = None,
    behaviour_panel: Sequence[Mapping[str, float]] | None = None,
    n_null_shuffles: int = 500,
    null_seed: int = 0,
    composition: Mapping[str, Any] | None = None,
    rbsa: Mapping[str, Any] | None = None,
    exposures: Mapping[str, float] | None = None,
    regime_deltas: Mapping[str, float] | None = None,
    semantic_tilt: Mapping[str, Any] | None = None,
    style_agreement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble ``policy_behavior.json`` schema_version 2 (interpretation only)."""
    from src.reporting.behavior_metrics import (
        regime_behaviour_deltas,
        turbulence_regimes_from_returns,
    )

    algo_key = str(algo or (cell_cfg or {}).get("algo") or "").lower()
    cfg = {
        "objective": objective or (cell_cfg or {}).get("objective") or "",
        "algo": algo_key,
        "architecture": architecture or (cell_cfg or {}).get("architecture") or "",
        "policy_mode": policy_mode or (cell_cfg or {}).get("policy_mode") or "balanced",
    }
    if cell_cfg:
        for k, v in cell_cfg.items():
            cfg.setdefault(k, v)

    behaviour: dict[str, float]
    by_regime: dict[str, Any] = {}
    tilt_series: dict[str, list[float]] = {s: [] for s in SLEEVE_IDS}
    macro_sens: dict[str, Any] = {}
    null_band: dict[str, list[float]] = {}
    regime_vars = {
        "across_regime_tilt_variance": float("nan"),
        "within_regime_tilt_variance": float("nan"),
    }

    # Prefer turbulence crisis labels when asset returns exist.
    overlay_mode = str((extras or {}).get("overlay_mode") or "markov")
    regimes_eff = turbulence_regimes_from_returns(
        asset_returns, existing=regimes, overlay_mode=overlay_mode
    )

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        behaviour = compute_behaviour_vector(
            w,
            asset_returns=asset_returns,
            sleeve_matrix=sleeve_matrix,
            turnover_cap=turnover_cap,
        )
        if regimes_eff is not None:
            by_regime = regime_conditional_behaviour(
                w,
                regimes=regimes_eff,
                asset_returns=asset_returns,
                sleeve_matrix=sleeve_matrix,
                turnover_cap=turnover_cap,
            )
            regime_vars = regime_tilt_variances(
                w, regimes=regimes_eff, sleeve_matrix=sleeve_matrix
            )
        if sleeve_matrix is not None:
            tilts = sleeve_tilt_series(w, sleeve_matrix)
            for j, sid in enumerate(SLEEVE_IDS):
                tilt_series[sid] = [float(x) for x in tilts[:, j]]
        if (
            sleeve_matrix is not None
            and vix_z is not None
            and hy_oas_z is not None
            and term_spread is not None
        ):
            macro_sens = macro_tilt_sensitivity(
                w,
                sleeve_matrix=sleeve_matrix,
                vix_z=vix_z,
                hy_oas_z=hy_oas_z,
                term_spread=term_spread,
                epu_z=epu_z,
                gpri_z=gpri_z,
            )
        null_band = shuffle_null_band(
            w,
            sleeve_matrix=sleeve_matrix,
            asset_returns=asset_returns,
            turnover_cap=turnover_cap,
            n_shuffles=int(n_null_shuffles),
            seed=int(null_seed),
        )
    else:
        behaviour = {m: float("nan") for m in BEHAVIOUR_MEASURE_IDS}
        if turnovers is not None:
            t = np.asarray(turnovers, dtype=np.float64).reshape(-1)
            behaviour["turnover_mean"] = float(np.nanmean(t)) if t.size else float("nan")
        if entropies is not None:
            e = np.asarray(entropies, dtype=np.float64).reshape(-1)
            behaviour["action_entropy_mean"] = (
                float(np.nanmean(e)) if e.size else float("nan")
            )

    # Merge optional measurement layers into the behaviour vector.
    if exposures:
        for key in (
            "exposure_size",
            "exposure_value",
            "exposure_momentum",
            "exposure_quality",
            "exposure_low_vol",
            "sector_hhi",
        ):
            if key in exposures and exposures[key] is not None:
                try:
                    behaviour[key] = float(exposures[key])
                except (TypeError, ValueError):
                    pass
    if rbsa and rbsa.get("rbsa_r_squared") is not None:
        try:
            behaviour["rbsa_r_squared"] = float(rbsa["rbsa_r_squared"])
        except (TypeError, ValueError):
            pass
    deltas = dict(regime_deltas or {})
    if not deltas and by_regime:
        deltas = regime_behaviour_deltas(by_regime)
    for key, val in deltas.items():
        try:
            behaviour[key] = float(val)
        except (TypeError, ValueError):
            behaviour[key] = float("nan")

    if semantic_tilt:
        for key in (
            "semantic_rotation_rate",
            "semantic_pc1_mean",
            "semantic_pc2_mean",
            "semantic_pc3_mean",
        ):
            if key in semantic_tilt and semantic_tilt[key] is not None:
                try:
                    behaviour[key] = float(semantic_tilt[key])
                except (TypeError, ValueError):
                    behaviour[key] = float("nan")
    if style_agreement and style_agreement.get("style_agreement_cosine") is not None:
        try:
            behaviour["style_agreement_cosine"] = float(
                style_agreement["style_agreement_cosine"]
            )
        except (TypeError, ValueError):
            behaviour["style_agreement_cosine"] = float("nan")

    score_row = _enrich_scoring_row(
        behaviour, macro_sens=macro_sens, regime_vars=regime_vars
    )
    panel = list(behaviour_panel) if behaviour_panel else [score_row]
    # Ensure the focal cell is first if panel was supplied without it
    if behaviour_panel is not None:
        panel = [score_row, *list(behaviour_panel)]
    scores_list = score_archetypes(panel)
    scores = scores_list[0]
    decision = assign_archetype(scores)

        # Composition overlay (panel re-score injects this; live Burst may omit).
    comp_block = dict(composition or {})
    if comp_block.get("archetype_composition"):
        primary = str(
            comp_block.get("archetype_primary")
            or max(
                comp_block["archetype_composition"],
                key=comp_block["archetype_composition"].get,
            )
        )
        confidence = float(
            comp_block.get("archetype_confidence")
            or comp_block["archetype_composition"].get(primary, float("nan"))
        )
        decision = {
            "archetype_primary": primary,
            "archetype_runner_up": "",
            "archetype_margin": confidence,
        }

    # Softmax collapse exception: L1 vs EW canary for exponential-head outliers.
    # Prefer weight_head over policy_mode: campaign YAMLs use policy_mode=single|multi
    # while the projection family lives in weight_head (softmax / sparse_tilt / ...).
    head_raw = (
        (cell_cfg or {}).get("weight_head")
        or cfg.get("weight_head")
        or ""
    )
    head_norm = normalize_weight_head(str(head_raw)) if head_raw else ""
    if head_norm in ("", "single", "multi", "balanced", "long_only"):
        # Unit tests historically pass the head via policy_mode=sparse_tilt|softmax.
        head_norm = normalize_weight_head(
            str(policy_mode or cfg.get("policy_mode") or "softmax")
        )
    if head_norm in ("", "single", "multi", "balanced", "long_only"):
        # Last resort: stem / cell_id tokens (landed Burst artifacts).
        stem = str(cell_id or "").lower()
        if "sparse_tilt" in stem:
            head_norm = "sparse_tilt"
        elif "tanh_l1" in stem:
            head_norm = "tanh_l1"
        elif "dirichlet" in stem:
            head_norm = "dirichlet"
        elif "softmax" in stem:
            head_norm = "softmax"
        else:
            head_norm = "softmax"
    cfg["weight_head"] = head_norm

    try:
        l1_vs_ew = float(behaviour.get("l1_vs_ew_mean", float("nan")))
    except (TypeError, ValueError):
        l1_vs_ew = float("nan")
    if head_norm == "softmax" and np.isfinite(l1_vs_ew) and l1_vs_ew > 0.25:
        behaviour["softmax_collapse_exception"] = True
        behaviour["softmax_escape_note"] = (
            f"algo={algo_key} produces larger logit scale under softmax "
            f"(l1_vs_ew={l1_vs_ew:.4f}>0.25)"
        )

    # Designed-vs-observed personality alignment (Scenario A A2 gate).
    # Mandate presets are stored as policy_mode=archetype_* in RC6 YAMLs
    # (not a separate mandate_preset key).
    mandate_raw = str(
        (cell_cfg or {}).get("mandate_preset")
        or (cell_cfg or {}).get("policy_mandate")
        or ""
    )
    if not mandate_raw:
        pm = str(
            (cell_cfg or {}).get("policy_mode") or cfg.get("policy_mode") or ""
        ).lower()
        if pm.startswith("archetype_"):
            mandate_raw = pm
    designed = designed_personality(
        objective=str(cfg.get("objective") or ""),
        algo=algo_key,
        weight_head=head_norm,
        mandate_preset=mandate_raw,
    )
    alignment = compute_personality_alignment(
        designed,
        {
            "archetype_primary": decision["archetype_primary"],
            "archetype_scores": scores,
            "behaviour": behaviour,
        },
    )

    explained = explain_behaviour(
        cfg,
        behaviour,
        macro_sens,
        null_band=null_band,
        behaviour_by_regime=by_regime,
    )

    out: dict[str, Any] = {
        "schema_version": 2,
        "cell_id": str(cell_id),
        "arm": str(arm),
        "algo": algo_key,
        "architecture": str(cfg.get("architecture") or ""),
        "objective": str(cfg.get("objective") or ""),
        "train_world": str(train_world),
        "policy_mode": str(cfg.get("policy_mode") or ""),
        "universe_fingerprint": str(universe_fingerprint),
        "interpretation_only": True,
        "feeds_capital_gates": False,
        "behaviour": behaviour,
        "behaviour_by_regime": by_regime,
        "sleeve_tilt_series": tilt_series,
        "macro_tilt_sensitivity": macro_sens,
        "archetype_scores": scores,
        "archetype_primary": decision["archetype_primary"],
        "archetype_runner_up": decision.get("archetype_runner_up", ""),
        "archetype_margin": float(decision.get("archetype_margin", float("nan"))),
        # One-release compat for report macros / preflight that still read archetype.name
        "archetype": {
            "name": str(decision["archetype_primary"]),
            "runner_up": str(decision.get("archetype_runner_up") or ""),
            "margin": float(decision.get("archetype_margin", float("nan"))),
        },
        "alignment_pass": bool(alignment["alignment_pass"]),
        "alignment_score": float(alignment["alignment_score"]),
        "designed_personality": str(alignment["designed_personality"]),
        "observed_personality": str(alignment["observed_personality"]),
        "alignment_divergence": str(alignment.get("divergence_explanation") or ""),
        "explanations": explained["explanations"],
        "null_band": null_band,
    }
    if comp_block.get("archetype_composition"):
        out["archetype_composition"] = dict(comp_block["archetype_composition"])
        out["archetype_confidence"] = float(
            comp_block.get("archetype_confidence")
            or out["archetype_margin"]
        )
    if rbsa:
        out["rbsa_loadings"] = list(rbsa.get("rbsa_loadings") or [])
        out["rbsa_r_squared"] = behaviour.get("rbsa_r_squared", float("nan"))
        if rbsa.get("factor_names"):
            out["rbsa_factor_names"] = list(rbsa["factor_names"])
    if sensitivities:
        out["signal_sensitivities"] = {str(k): float(v) for k, v in sensitivities.items()}
    if extras:
        out["extras"] = dict(extras)
    # Backward-compatible concentration block for legacy figure code.
    if weights is not None:
        out["concentration"] = weight_concentration(weights)
    out["data_availability"] = _build_data_availability(
        regimes=regimes_eff if regimes_eff is not None else regimes,
        vix_z=vix_z,
        hy_oas_z=hy_oas_z,
        term_spread=term_spread,
        sleeve_matrix=sleeve_matrix,
        sensitivities=sensitivities,
        by_regime=by_regime,
        macro_sens=macro_sens,
        tilt_series=tilt_series,
    )
    if semantic_tilt:
        out["semantic_tilt"] = {
            "text_asof": semantic_tilt.get("text_asof") or "2026-08-18",
            "text_pit": False,
            "embed_backend": semantic_tilt.get("embed_backend"),
            "data_availability_reason": semantic_tilt.get("data_availability_reason")
            or "",
        }
    if style_agreement:
        out["style_agreement"] = {
            "cosine": style_agreement.get("style_agreement_cosine"),
            "disagreement_flag": bool(style_agreement.get("style_disagreement_flag")),
            "reason": style_agreement.get("reason") or "",
            "holdings_style_vec": list(style_agreement.get("holdings_style_vec") or []),
            "rbsa_style_vec": list(style_agreement.get("rbsa_style_vec") or []),
        }
    return out


def extract_crucible_behaviour_inputs(
    *,
    results: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pull sleeve_matrix / membership / fingerprint from campaign crucible blobs.

    Lookup order: ``results["crucible"]``, ``results["dynamic_universe"]["crucible"]``,
    then ``cfg["_crucible_result"]``.
    """
    results = results or {}
    cfg = cfg or {}
    block: Mapping[str, Any] = {}
    for candidate in (
        results.get("crucible"),
        (results.get("dynamic_universe") or {}).get("crucible"),
        cfg.get("_crucible_result"),
    ):
        if isinstance(candidate, Mapping) and candidate:
            block = candidate
            break

    sleeve_matrix = None
    raw = block.get("sleeve_matrix")
    if raw is not None:
        sleeve_matrix = np.asarray(raw, dtype=np.float64)

    fingerprint = (
        block.get("fingerprint")
        or cfg.get("_crucible_universe_fingerprint")
        or ""
    )
    return {
        "sleeve_matrix": sleeve_matrix,
        "sleeve_membership": block.get("sleeve_membership"),
        "sleeve_primary": block.get("sleeve_primary"),
        "universe_fingerprint": str(fingerprint or ""),
        "crucible_block_found": bool(block),
    }


def load_behaviour_macro_context(
    dates: Sequence,
    *,
    lake_root: Path | str,
    lake_subdir: str = "macro/fioracle",
    min_history_days: int = 756,
) -> dict[str, Any]:
    """Align fioracle regimes + macro regressors to ``dates``.

    Graceful when the lake is missing: returns ``regimes`` / macro series as
    ``None`` and stamps ``status`` so the campaign can persist the skip reason
    without inventing tilts.
    """
    import pandas as pd

    from src.data.fioracle_macro import (
        build_fioracle_feature_frame,
        load_fioracle_macro,
    )
    from src.data.regime_labels import label_regimes

    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    n = len(idx)
    status: dict[str, Any] = {
        "status": "ok",
        "n_dates": int(n),
        "lake_subdir": str(lake_subdir),
        "reason": "",
    }
    empty = {
        "regimes": None,
        "vix_z": None,
        "hy_oas_z": None,
        "term_spread": None,
        "epu_z": None,
        "gpri_z": None,
        "status": status,
        "regime_meta": None,
    }
    if n == 0:
        status["status"] = "empty_dates"
        status["reason"] = "no evaluation dates"
        return empty

    try:
        # Pad start so expanding-window regime labels have history.
        pad_days = int(min_history_days) + 400
        start = (idx.min() - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
        end = idx.max().strftime("%Y-%m-%d")
        levels = load_fioracle_macro(
            lake_root=lake_root,
            start_date=start,
            end_date=end,
            lake_subdir=lake_subdir,
        )
        feats = build_fioracle_feature_frame(levels)
        labels, meta = label_regimes(feats, min_history_days=int(min_history_days))
        # Align to eval calendar; causal ffill only (no bfill into the past).
        labels_al = labels.reindex(idx).ffill()
        regimes = labels_al.fillna("calm").astype(str).to_numpy(dtype=object)

        def _col(name: str) -> np.ndarray:
            if name not in feats.columns:
                return np.full(n, np.nan, dtype=np.float64)
            return feats[name].reindex(idx).to_numpy(dtype=np.float64)

        vix = _col("vix_z_252")
        hy = _col("hy_oas_z_252")
        term = _col("term_spread_level")
        epu = _col("epu_z_252")
        gpri = _col("gpri_z_252")
        if not (
            np.isfinite(vix).any() and np.isfinite(hy).any() and np.isfinite(term).any()
        ):
            status["status"] = "insufficient_macro"
            status["reason"] = "aligned fioracle series all-NaN on eval dates"
            return {
                "regimes": regimes,
                "vix_z": None,
                "hy_oas_z": None,
                "term_spread": None,
                "epu_z": None,
                "gpri_z": None,
                "status": status,
                "regime_meta": meta,
            }
        status["status"] = "ok"
        return {
            "regimes": regimes,
            "vix_z": vix,
            "hy_oas_z": hy,
            "term_spread": term,
            # Optional B3 wideners: pass only when finite coverage exists.
            "epu_z": epu if np.isfinite(epu).any() else None,
            "gpri_z": gpri if np.isfinite(gpri).any() else None,
            "status": status,
            "regime_meta": meta,
        }
    except FileNotFoundError as exc:
        status["status"] = "missing_lake"
        status["reason"] = str(exc)[:300]
        return empty
    except Exception as exc:  # noqa: BLE001 - campaign must not abort on macro
        status["status"] = "error"
        status["reason"] = str(exc)[:300]
        return empty


def pack_policy_behavior_campaign_record(
    behavior: Mapping[str, Any],
    *,
    path: str | Path,
    figures: Sequence[str] | None = None,
    macro_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the full v2 payload plus path/figures (not archetype-only)."""
    out = dict(behavior)
    out["path"] = str(path)
    if figures is not None:
        out["figures"] = list(figures)
    if macro_status is not None:
        extras = dict(out.get("extras") or {})
        extras["macro_context_status"] = dict(macro_status)
        out["extras"] = extras
    return out


def _json_safe(obj: Any) -> Any:
    """Convert non-finite floats to None for strict JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def validate_policy_behavior_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Schema-check a ``*_policy_behavior.json`` artifact (interpretation only)."""
    errors: list[str] = []
    required = (
        "schema_version",
        "interpretation_only",
        "feeds_capital_gates",
        "archetype_scores",
        "archetype_primary",
        "behaviour",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing_key:{key}")
    if payload.get("schema_version") not in (2, "2"):
        errors.append(f"schema_version:{payload.get('schema_version')!r}")
    if payload.get("interpretation_only") is not True:
        errors.append("interpretation_only_must_be_true")
    if payload.get("feeds_capital_gates") is not False:
        errors.append("feeds_capital_gates_must_be_false")
    scores = payload.get("archetype_scores")
    if not isinstance(scores, Mapping) or not scores:
        errors.append("archetype_scores_missing")
    return {"ok": not errors, "errors": errors}


def write_policy_behavior(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(dict(payload))
    p.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n")
    return p


def plot_archetype_figures(
    payload: Mapping[str, Any],
    out_dir: str | Path,
) -> list[str]:
    """Emit a small set of interpretive plots from a behavior payload."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.reporting.book_style import C_ACCENT, C_BLUE, C_NAVY, C_NEG, C_POS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    conc = payload.get("concentration") or {}
    if not conc and payload.get("behaviour"):
        b = payload["behaviour"]
        conc = {
            "hhi_mean": b.get("hhi_mean"),
            "l1_vs_ew_mean": b.get("l1_vs_ew_mean"),
            "max_weight_mean": b.get("max_weight_mean"),
        }
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["HHI", "L1 vs EW", "max w"]
    vals = [
        float(conc.get("hhi_mean", float("nan"))),
        float(conc.get("l1_vs_ew_mean", float("nan"))),
        float(conc.get("max_weight_mean", float("nan"))),
    ]
    ax.bar(labels, vals, color=C_NAVY)
    ax.set_title(f"Concentration ({payload.get('algo')})")
    fig.tight_layout()
    p = out_dir / "archetype_concentration.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    scores = payload.get("archetype_scores") or {}
    if scores:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = list(scores.keys())
        ax.barh(names, [float(scores[n]) for n in names], color=C_BLUE)
        ax.set_xlabel("Score")
        ax.set_title(
            f"Archetype scores (primary={payload.get('archetype_primary')})"
        )
        fig.tight_layout()
        p = out_dir / "archetype_scores.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    sens = payload.get("signal_sensitivities") or {}
    if sens:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = list(sens.keys())
        ax.barh(names, [float(sens[n]) for n in names], color=C_ACCENT)
        ax.set_xlabel("d||w||1 / d channel")
        ax.set_title("Signal weight sensitivity")
        fig.tight_layout()
        p = out_dir / "archetype_signal_sensitivity.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    extras = payload.get("extras") or {}
    ent = extras.get("entropy_series")
    turn = extras.get("turnover_series")
    if ent is not None and turn is not None:
        e = np.asarray(ent, dtype=float).reshape(-1)
        t = np.asarray(turn, dtype=float).reshape(-1)
        n = min(e.size, t.size)
        if n > 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(t[:n], e[:n], s=12, alpha=0.7, color=C_BLUE)
            ax.set_xlabel("Turnover")
            ax.set_ylabel("Action entropy")
            ax.set_title("Entropy vs turnover")
            fig.tight_layout()
            p = out_dir / "archetype_entropy_vs_turnover.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            written.append(str(p))

    seed_sharpes = extras.get("seed_sharpes")
    if seed_sharpes is not None:
        s = np.asarray(seed_sharpes, dtype=float).reshape(-1)
        if s.size:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(np.arange(s.size), s, color=C_POS)
            ax.set_xlabel("Seed")
            ax.set_ylabel("Sharpe")
            ax.set_title("CPCV seed Sharpe spectrum")
            fig.tight_layout()
            p = out_dir / "archetype_seed_spectrum.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            written.append(str(p))

    slacks = extras.get("cmdp_slack_series")
    if slacks is not None:
        s = np.asarray(slacks, dtype=float).reshape(-1)
        if s.size:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(s, color=C_NEG, lw=1.2)
            ax.set_xlabel("Evaluation step")
            ax.set_ylabel("CMDP slack")
            ax.set_title("CMDP constraint slack")
            fig.tight_layout()
            p = out_dir / "archetype_cmdp_slack.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            written.append(str(p))

    return written
