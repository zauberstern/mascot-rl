"""Interpretability layer for archetype analysis (interpretation only).

Channel-group occlusion attribution and shallow decision-tree distillation.
Never feeds capital gates.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.reporting.behavior_metrics import (
    BEHAVIOUR_MEASURE_IDS,
    SLEEVE_IDS,
    compute_behaviour_vector,
    sleeve_tilt_series,
)

HEADLINE_MEASURES: tuple[str, ...] = (
    "turnover_mean",
    "hhi_mean",
    "tilt_trend",
    "tilt_defensive",
    "rotation_rate",
)

_CAUSAL_BANNED = re.compile(r"\b(causes?|because|leads to)\b", re.IGNORECASE)


def build_channel_groups_from_names(names: Sequence[str]) -> dict[str, list[int]]:
    """Map feature group names to flat obs channel indices (asset 0, latest step)."""
    from src.features.groups import FEATURE_GROUPS, _KELLY_PREFIX, _MACRO_PREFIXES

    name_list = list(names)
    groups: dict[str, list[int]] = {}
    assigned: set[int] = set()

    for group_name, channels in FEATURE_GROUPS.items():
        if not channels:
            continue
        idxs = [name_list.index(c) for c in channels if c in name_list]
        if idxs:
            groups[group_name] = idxs
            assigned.update(idxs)

    kelly = [i for i, n in enumerate(name_list) if str(n).startswith(_KELLY_PREFIX)]
    if kelly:
        groups["kelly_images"] = kelly
        assigned.update(kelly)

    macro = [
        i
        for i, n in enumerate(name_list)
        if any(str(n).startswith(p) for p in _MACRO_PREFIXES)
    ]
    if macro:
        groups["macro"] = macro
        assigned.update(macro)

    portfolio = [
        i
        for i, n in enumerate(name_list)
        if str(n) in ("w_prev", "days_held", "cum_cost", "w_base")
    ]
    if portfolio:
        groups["portfolio_state"] = portfolio
        assigned.update(portfolio)

    other = [i for i in range(len(name_list)) if i not in assigned]
    if other:
        groups["other"] = other
    return groups


def _aggregate_obs_features(
    obs_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Cross-sectional mean/std per named channel from (T, obs_dim) flat obs."""
    T, dim = obs_matrix.shape
    n_names = len(feature_names)
    if n_names == 0:
        return np.zeros((T, 0), dtype=np.float64)
    per_asset = max(dim // max(n_names, 1), 1)
    feats: list[np.ndarray] = []
    for i, name in enumerate(feature_names):
        start = i * per_asset
        end = min(start + per_asset, dim)
        if start >= dim:
            feats.append(np.zeros(T, dtype=np.float64))
            continue
        block = obs_matrix[:, start:end]
        feats.append(np.nanmean(block, axis=1))
        feats.append(np.nanstd(block, axis=1))
    return np.column_stack(feats)


def _weights_from_policy_fn(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    obs_matrix: np.ndarray,
) -> np.ndarray:
    weights = []
    for t in range(obs_matrix.shape[0]):
        w = policy_fn(obs_matrix[t])
        weights.append(np.asarray(w, dtype=np.float64).reshape(-1))
    return np.vstack(weights)


def _permute_group_in_obs(
    obs_matrix: np.ndarray,
    group_indices: Sequence[int],
    perm: np.ndarray,
) -> np.ndarray:
    """Permute group channels in time, preserving cross-asset structure."""
    corrupted = obs_matrix.copy()
    T = obs_matrix.shape[0]
    if len(group_indices) == 0 or T < 2:
        return corrupted
    perm = np.asarray(perm, dtype=int)
    for idx in group_indices:
        if 0 <= idx < obs_matrix.shape[1]:
            corrupted[:, idx] = obs_matrix[perm, idx]
    return corrupted


def channel_group_attribution(
    *,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    obs_matrix: np.ndarray,
    channel_groups: Mapping[str, list[int]],
    sleeve_matrix: np.ndarray | None = None,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Grouped occlusion attribution with shuffle-null materiality bands."""
    obs_matrix = np.asarray(obs_matrix, dtype=np.float64)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(1, -1)
    baseline_w = _weights_from_policy_fn(policy_fn, obs_matrix)
    baseline_beh = compute_behaviour_vector(
        baseline_w, sleeve_matrix=sleeve_matrix
    )
    T = obs_matrix.shape[0]
    group_results: dict[str, Any] = {}

    for gname, gidx in channel_groups.items():
        if not gidx:
            continue
        group_rng = np.random.default_rng(seed + hash(gname) % 10000)
        perm = group_rng.permutation(T)
        corrupted_obs = _permute_group_in_obs(obs_matrix, gidx, perm)
        corrupted_w = _weights_from_policy_fn(policy_fn, corrupted_obs)
        l1_delta = float(np.mean(np.sum(np.abs(corrupted_w - baseline_w), axis=1)))
        corrupted_beh = compute_behaviour_vector(
            corrupted_w, sleeve_matrix=sleeve_matrix
        )
        measure_deltas = {
            m: float(corrupted_beh.get(m, float("nan")))
            - float(baseline_beh.get(m, float("nan")))
            for m in HEADLINE_MEASURES
        }
        null_samples: list[float] = []
        dummy_size = len(gidx)
        all_idx = list(range(obs_matrix.shape[1]))
        for _ in range(int(n_shuffles)):
            if dummy_size >= len(all_idx):
                null_perm = group_rng.permutation(T)
                null_obs = _permute_group_in_obs(obs_matrix, all_idx, null_perm)
            else:
                dummy_idx = list(group_rng.choice(all_idx, size=dummy_size, replace=False))
                null_perm = group_rng.permutation(T)
                null_obs = _permute_group_in_obs(obs_matrix, dummy_idx, null_perm)
            null_w = _weights_from_policy_fn(policy_fn, null_obs)
            null_samples.append(
                float(np.mean(np.sum(np.abs(null_w - baseline_w), axis=1)))
            )
        null_p975 = float(np.percentile(null_samples, 97.5)) if null_samples else 0.0
        group_results[gname] = {
            "l1_delta": l1_delta,
            "measure_deltas": measure_deltas,
            "null_p975": null_p975,
            "material": l1_delta > null_p975,
        }

    top_groups = sorted(
        group_results,
        key=lambda k: group_results[k]["l1_delta"],
        reverse=True,
    )
    return {
        "groups": group_results,
        "top_groups": top_groups,
    }


def _weight_path_features(weights: np.ndarray) -> np.ndarray:
    """Autoregressive features from weight path when obs is unavailable."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    T, K = w.shape
    feats: list[np.ndarray] = []
    lag1 = np.vstack([w[0:1], w[:-1]])
    feats.append(lag1)
    hhi = np.sum(np.square(w), axis=1, keepdims=True)
    feats.append(hhi)
    l1_ew = np.sum(np.abs(w - 1.0 / max(K, 1)), axis=1, keepdims=True)
    feats.append(l1_ew)
    to = np.zeros((T, 1), dtype=np.float64)
    if T > 1:
        to[1:] = np.sum(np.abs(np.diff(w, axis=0)), axis=1, keepdims=True) / 2.0
    feats.append(to)
    return np.hstack(feats)


def distill_policy_tree(
    *,
    obs: np.ndarray | None,
    weights: np.ndarray,
    sleeve_matrix: np.ndarray,
    feature_names: Sequence[str],
    max_depth: int = 4,
    fidelity_gate: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Post-hoc depth-limited tree distillation with OOS R2 fidelity gate."""
    from sklearn.tree import DecisionTreeRegressor, export_text

    weights = np.asarray(weights, dtype=np.float64)
    sleeve_matrix = np.asarray(sleeve_matrix, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    T = weights.shape[0]
    if T < 8:
        return {
            "target": "sleeve_tilts",
            "r2_oos": float("nan"),
            "depth": 0,
            "n_leaves": 0,
            "rules_text": "",
            "distillable": False,
            "reason": "insufficient_timesteps",
        }

    if obs is not None and np.asarray(obs).size > 0:
        obs_arr = np.asarray(obs, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        T = min(T, obs_arr.shape[0])
        X = _aggregate_obs_features(obs_arr[:T], feature_names)
        agg_names = [
            f"{n}_{stat}" for n in feature_names for stat in ("mean", "std")
        ]
    else:
        X = _weight_path_features(weights[:T])
        agg_names = [f"w_feat_{i}" for i in range(X.shape[1])]
    tilts = sleeve_tilt_series(weights[:T], sleeve_matrix)
    y_turnover = np.concatenate(
        [
            np.sum(np.abs(np.diff(weights[:T], axis=0)), axis=1) / 2.0,
            [0.0],
        ]
    )
    targets = {sid: tilts[:, j] for j, sid in enumerate(SLEEVE_IDS)}
    targets["turnover"] = y_turnover

    split = max(int(T * 0.75), 1)
    X_train, X_test = X[:split], X[split:]
    if X_test.shape[0] < 2:
        return {
            "target": "sleeve_tilts",
            "r2_oos": float("nan"),
            "depth": 0,
            "n_leaves": 0,
            "rules_text": "",
            "distillable": False,
            "reason": "insufficient_oos_split",
        }

    best: dict[str, Any] = {
        "target": "",
        "r2_oos": float("-inf"),
        "depth": 0,
        "n_leaves": 0,
        "rules_text": "",
        "distillable": False,
    }
    for target_name, y_full in targets.items():
        y_train = y_full[:split]
        y_test = y_full[split:]
        tree = DecisionTreeRegressor(max_depth=int(max_depth), random_state=int(seed))
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        if np.isfinite(r2) and r2 > best["r2_oos"]:
            rules = export_text(
                tree,
                feature_names=agg_names[: X.shape[1]],
                max_depth=int(max_depth),
            )
            best = {
                "target": target_name,
                "r2_oos": float(r2),
                "depth": int(tree.get_depth()),
                "n_leaves": int(tree.get_n_leaves()),
                "rules_text": rules if r2 >= fidelity_gate else "",
                "distillable": bool(r2 >= fidelity_gate),
            }
    if not np.isfinite(best["r2_oos"]):
        best["distillable"] = False
        best["rules_text"] = ""
    return best


def explanation_quality_metrics(
    attribution: Mapping[str, Any],
    *,
    faithfulness_correlation: float | None = None,
    stability_score: float | None = None,
) -> dict[str, float]:
    """Score explanation quality: faithfulness, stability, sparsity.

    Faithfulness and stability may be supplied by the caller (feature-flip /
    perturbation loops). Sparsity is always derived from attribution mass.
    """
    groups = dict(attribution.get("groups") or {})
    deltas = {
        g: abs(float((meta or {}).get("l1_delta", 0.0) or 0.0))
        for g, meta in groups.items()
    }
    total = float(sum(deltas.values()))
    if total > 1e-12:
        shares = {g: v / total for g, v in deltas.items()}
        sparsity = float(sum(1 for v in shares.values() if v > 0.05) / max(len(shares), 1))
        cum = 0.0
        n_dom = 0
        for v in sorted(shares.values(), reverse=True):
            cum += v
            n_dom += 1
            if cum >= 0.80:
                break
    else:
        sparsity = float("nan")
        n_dom = 0
    return {
        "faithfulness_correlation": float(
            faithfulness_correlation
            if faithfulness_correlation is not None
            else float("nan")
        ),
        "stability_score": float(
            stability_score if stability_score is not None else float("nan")
        ),
        "explanation_sparsity": sparsity,
        "n_dominant_channels": float(n_dom),
    }


def compute_faithfulness_correlation(
    *,
    policy_fn,
    obs_matrix: np.ndarray,
    channel_groups: Mapping[str, list[int]],
    attribution: Mapping[str, Any],
) -> float:
    """Pearson r between attribution rank and output-change rank (feature flip)."""
    obs_matrix = np.asarray(obs_matrix, dtype=np.float64)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(1, -1)
    baseline_w = _weights_from_policy_fn(policy_fn, obs_matrix)
    groups = dict(attribution.get("groups") or {})
    attr_vals: list[float] = []
    flip_vals: list[float] = []
    for gname, gidx in channel_groups.items():
        if not gidx:
            continue
        attr_vals.append(float((groups.get(gname) or {}).get("l1_delta", 0.0) or 0.0))
        flipped = obs_matrix.copy()
        for idx in gidx:
            if 0 <= idx < flipped.shape[1]:
                col = flipped[:, idx]
                finite = np.isfinite(col)
                fill = float(np.mean(col[finite])) if np.any(finite) else 0.0
                flipped[:, idx] = fill
        flipped_w = _weights_from_policy_fn(policy_fn, flipped)
        flip_vals.append(
            float(np.mean(np.sum(np.abs(flipped_w - baseline_w), axis=1)))
        )
    if len(attr_vals) < 2:
        return float("nan")
    a = np.asarray(attr_vals, dtype=np.float64)
    b = np.asarray(flip_vals, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_stability_score(
    *,
    policy_fn,
    obs_matrix: np.ndarray,
    channel_groups: Mapping[str, list[int]],
    n_perturb: int = 20,
    sigma: float = 0.01,
    seed: int = 0,
) -> float:
    """Mean pairwise rank correlation of top-5 attributed groups under noise."""
    from scipy.stats import spearmanr

    obs_matrix = np.asarray(obs_matrix, dtype=np.float64)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(1, -1)
    rng = np.random.default_rng(seed)
    ranks: list[list[str]] = []
    for _ in range(int(n_perturb)):
        noise = rng.normal(0.0, float(sigma), size=obs_matrix.shape)
        noisy = obs_matrix + noise
        attr = channel_group_attribution(
            policy_fn=policy_fn,
            obs_matrix=noisy,
            channel_groups=channel_groups,
            n_shuffles=5,
            seed=int(rng.integers(0, 1_000_000)),
        )
        top = list(attr.get("top_groups") or [])[:5]
        ranks.append(top)
    if len(ranks) < 2:
        return float("nan")
    # Encode ranks as position maps over the union of group names.
    all_names = sorted({g for r in ranks for g in r} | set(channel_groups.keys()))
    vecs = []
    for r in ranks:
        pos = {g: i for i, g in enumerate(r)}
        vecs.append([float(pos.get(g, len(r) + 1)) for g in all_names])
    cors: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            rho, _ = spearmanr(vecs[i], vecs[j])
            if np.isfinite(rho):
                cors.append(float(rho))
    return float(np.mean(cors)) if cors else float("nan")


def build_interpretability_artifact(
    *,
    cell_id: str,
    attribution: Mapping[str, Any],
    distillation: Mapping[str, Any],
    mechanism_cards: Sequence[Mapping[str, Any]],
    data_availability: Mapping[str, Any] | None = None,
    quality_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble schema v1 interpretability JSON."""
    qm = dict(quality_metrics) if quality_metrics else explanation_quality_metrics(attribution)
    return {
        "schema_version": 1,
        "cell_id": str(cell_id),
        "interpretation_only": True,
        "feeds_capital_gates": False,
        "attribution": dict(attribution),
        "distillation": dict(distillation),
        "mechanism_cards": list(mechanism_cards),
        "data_availability": dict(data_availability or {}),
        "quality_metrics": qm,
    }


def mechanism_cards_from_behavior(behavior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract regime/macro mechanism cards from a policy_behavior payload."""
    cards: list[dict[str, Any]] = []
    for exp in behavior.get("explanations") or []:
        if exp.get("mechanism") in ("regime_shift_response", "macro_tilt_response"):
            cards.append(dict(exp))
    return cards


def prose_safe(text: str) -> bool:
    """Return True if text contains no banned causal phrasing."""
    return _CAUSAL_BANNED.search(text or "") is None
