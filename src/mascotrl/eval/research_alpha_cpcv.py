"""Short research CPCV: dry-run schema + real numpy-panel campaign."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from mascotrl.eval.collapse_guard import collapse_guard
from mascotrl.eval.cpcv import CPCVConfig, CPCVFold, residual_equity_cpcv_config, run_cpcv
from mascotrl.eval.cpcv_lib import run_cpcv_lib
from mascotrl.eval.friction import FrictionSpec, friction_spec_from_cfg
from mascotrl.eval.parity_harness import estimand_hash, score_equal_weight
from mascotrl.eval.policy_diagnostics import summarize_policy_diagnostics
from mascotrl.eval.research_alpha_baselines import (
    policy_beats_random,
    research_baselines_from_returns,
)
from mascotrl.eval.research_alpha_train import (
    SYNTHETIC_TRAIN_WORLDS,
    _discover_latest_checkpoint,
    build_research_hist_env,
    synthetic_train_panel,
    train_objective_equals_claim_metric,
    train_research_hist,
)
from mascotrl.eval.residualization import (
    ResidualizerState,
    fit_ff4_residualizer,
    freeze_residualizer,
)
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.research_alpha_cpcv")
from mascotrl.eval.stats_rigor import annualized_sharpe
from mascotrl.eval.yaml_honesty import track_copy
from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha
from mascotrl.reporting.research_alpha_router import resolve_research_primary_train


def _fill_ladder_specs(base_fric: FrictionSpec) -> dict[str, FrictionSpec]:
    """Build mid/pct75/worst friction specs for the cost-fill ladder.

    Equity fill stress must vary ``equity_bps``. ``om_touch_enabled`` alone must
    not divert to ``om_touch_spread_multiplier``: it scales option drag only, so
    on the eq arm mid/pct75/worst collapse and gate1 break-even becomes NaN.
    """
    eq_bps = float(base_fric.equity_bps)
    if eq_bps > 0.0:
        return {
            "mid": replace(base_fric, equity_bps=eq_bps * 0.5),
            "pct75": base_fric,
            "worst": replace(base_fric, equity_bps=eq_bps * 2.0),
        }
    mult = float(base_fric.om_touch_spread_multiplier)
    return {
        "mid": replace(base_fric, om_touch_spread_multiplier=mult * 0.5),
        "pct75": base_fric,
        "worst": replace(base_fric, om_touch_spread_multiplier=mult * 2.0),
    }


def _training_policy_diagnostics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect entropy series and exploration scale from fold training output."""
    entropies: list[float] = []
    log_stds: list[float] = []
    adv_vars: list[float] = []
    logit_stds: list[float] = []
    snrs: list[float] = []
    for fold in folds:
        for point in fold.get("learning_curve") or []:
            value = point.get("entropy") if isinstance(point, Mapping) else None
            if value is not None and np.isfinite(float(value)):
                entropies.append(float(value))
        stats = fold.get("train_stats") or {}
        entropy = stats.get("entropy")
        if entropy is not None and np.isfinite(float(entropy)):
            entropies.append(float(entropy))
        log_std = stats.get("log_std_mean")
        if log_std is not None and np.isfinite(float(log_std)):
            log_stds.append(float(log_std))
        adv = stats.get("advantage_variance")
        if adv is not None and np.isfinite(float(adv)):
            adv_vars.append(float(adv))
        lx = stats.get("logit_xsec_std")
        if lx is not None and np.isfinite(float(lx)):
            logit_stds.append(float(lx))
        snr = stats.get("reward_signal_to_noise")
        if snr is not None and np.isfinite(float(snr)):
            snrs.append(float(snr))
    return {
        "entropies": entropies,
        "log_std_mean": float(np.mean(log_stds)) if log_stds else None,
        "advantage_variance_mean": float(np.mean(adv_vars)) if adv_vars else None,
        "logit_xsec_std_mean": float(np.mean(logit_stds)) if logit_stds else None,
        "reward_signal_to_noise_mean": float(np.mean(snrs)) if snrs else None,
    }


def _signal_sensitivities(train_out: Mapping[str, Any]) -> dict[str, float]:
    """Measure named feature-channel sensitivity on the trained fold policy."""
    from mascotrl.reporting.policy_behavior import signal_weight_sensitivity

    env = train_out.get("env")
    agent = train_out.get("agent")
    builder = getattr(env, "feature_builder", None)
    names = list(getattr(builder, "names", ()) or ())
    if env is None or agent is None or not names:
        return {}
    obs, _ = env.reset()
    per_asset = int(getattr(builder, "obs_channels_per_asset", len(names) + 3))
    seq_len = int(getattr(builder, "seq_len", 1))
    latest_offset = max(0, seq_len - 1) * per_asset
    preferred = [n for n in names if "mfis" in str(n).lower() or "iv" in str(n).lower()]
    selected = preferred[:8] or names[:3]
    return {
        str(name): signal_weight_sensitivity(
            agent,
            np.asarray(obs, dtype=np.float64),
            channel_index=latest_offset + names.index(name),
        )
        for name in selected
    }


def _score_long_baseline_on_cpcv_tests(
    *,
    dates: Sequence[pd.Timestamp],
    returns: np.ndarray,
    factors: np.ndarray,
    folds: Sequence[CPCVFold],
    friction: FrictionSpec,
    cadence: str,
    rebalance_mask: np.ndarray | None,
    periods: float = 252.0,
) -> dict[str, Any]:
    """Score the long sleeve only on CPCV test slices with fold-local betas."""
    pnl_parts: list[np.ndarray] = []
    n_test_rows = 0
    for fold in folds:
        train_idx = _indices_for_windows(dates, fold.train_windows)
        test_idx = _indices_for_windows(dates, fold.test_windows)
        if train_idx.size < 2 or test_idx.size < 2:
            continue
        residualizer = freeze_residualizer(
            fit_ff4_residualizer(
                np.nanmean(returns[train_idx], axis=1),
                factors[train_idx],
                fold_id=f"long_baseline_fold_{fold.fold_id}",
            ),
            f"long_baseline_fold_{fold.fold_id}",
        )
        fold_mask = rebalance_mask[test_idx] if rebalance_mask is not None else None
        scored = score_equal_weight(
            returns[test_idx],
            factors=factors[test_idx],
            friction=friction,
            residualizer=residualizer,
            rebalance_mask=fold_mask,
            cadence=cadence,
        )
        pnl_parts.append(np.asarray(scored["total_net"], dtype=np.float64).reshape(-1))
        n_test_rows += int(test_idx.size)
    pnl = np.concatenate(pnl_parts) if pnl_parts else np.asarray([], dtype=np.float64)
    return {
        "sharpe": annualized_sharpe(pnl, periods=float(periods))
        if pnl.size
        else float("nan"),
        "n_test_rows": n_test_rows,
        "n_scored_rows": int(pnl.size),
    }


def _train_agent_for_fold(
    cfg: Mapping[str, Any],
    rets: np.ndarray,
    fac: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """C5: dispatch fold training on the ``train_world`` axis.

    ``historical`` (default) trains directly on the real panel's train
    window, unchanged from before. The five synthetic worlds replace the
    *training* panel with a Layer-1-generated one of matching (rows, K)
    while eval is untouched (``fold_runner`` always scores ``test_idx``
    against the real historical panel), so every ``train_world`` tests OOS
    on the same CPCV folds and the estimand hash stays comparable across
    cells. ``hybrid_pretrain_finetune`` genuinely pretrains the same agent
    on synthetic data then finetunes it on the real historical train
    window, replacing the prior silent collapse to pure historical.
    """
    train_world = str(cfg.get("train_world") or "historical").strip().lower()
    k = int(rets.shape[1])
    cfg_train = _slice_feature_extras(cfg, train_idx)
    if train_world == "historical":
        return train_research_hist(
            rets[train_idx], fac[train_idx], cfg_train, seed=seed, agent=None
        )
    if bool(cfg.get("use_equity_feature_cube", False)):
        # Synthetic panels have no cube surface. Hybrid is allowed because
        # pretrain forces cube off; finetune alone may keep cube=true.
        if train_world != "hybrid_pretrain_finetune":
            raise ValueError(
                f"train_world={train_world!r} has no synthetic feature-cube source; "
                "use_equity_feature_cube=true is only defined for "
                "train_world='historical' (or hybrid finetune phase)"
            )
    if train_world in SYNTHETIC_TRAIN_WORLDS:
        synth_rets, synth_fac = synthetic_train_panel(
            cfg, k=k, n_rows=int(train_idx.size), seed=seed, world=train_world
        )
        return train_research_hist(synth_rets, synth_fac, cfg_train, seed=seed, agent=None)
    if train_world == "hybrid_pretrain_finetune":
        pretrain_world = str(cfg.get("hybrid_pretrain_world") or "rbergomi").lower()
        synth_rets, synth_fac = synthetic_train_panel(
            cfg, k=k, n_rows=int(train_idx.size), seed=seed, world=pretrain_world
        )
        # Pretrain is synthetic: never attach a historical feature cube.
        cfg_pretrain = dict(cfg_train)
        cfg_pretrain["use_equity_feature_cube"] = False
        pretrain_out = train_research_hist(
            synth_rets, synth_fac, cfg_pretrain, seed=seed, agent=None
        )
        # Finetune keeps caller cube flag (historical panel). Raw-return NaNs
        # in inactive slots are zero-filled inside HistoricalArmEnv._obs.
        finetune_out = train_research_hist(
            rets[train_idx], fac[train_idx], cfg_train, seed=seed, agent=pretrain_out["agent"]
        )
        finetune_out["pretrain_stats"] = {
            "world": pretrain_world,
            "n_steps": int(pretrain_out.get("n_steps") or 0),
            "n_episodes": int(pretrain_out.get("n_episodes") or 0),
            "mean_reward": pretrain_out.get("mean_reward"),
        }
        return finetune_out
    raise ValueError(f"unhandled train_world={train_world!r}")  # validate_cfg already restricts range


def dry_run_research_alpha_cpcv(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Emit research CPCV artifact schema without running a full campaign.

    Refuses mid-only headlines. Toy Sharpes are NaN so
    ``stamp_research_positive_alpha`` stays false unless a real campaign fills them.
    """
    headline = str(cfg.get("headline_fill") or "")
    if headline != "pct75":
        raise ValueError(
            f"research CPCV requires headline_fill=pct75 (got {headline!r}); mid-only refused"
        )
    primary = resolve_research_primary_train(cfg)
    rng = np.random.default_rng(0)
    toy = rng.normal(0.0, 0.01, size=(64, 5))
    baselines = research_baselines_from_returns(toy, seed=0)
    fric_on = bool(cfg.get("om_touch_enabled", False)) or float(
        cfg.get("equity_bps", 0.0) or 0.0
    ) > 0.0
    return {
        "claim_tier": str(cfg.get("claim_tier") or "research"),
        "claim_category": str(cfg.get("claim_category") or ""),
        "claim_label_stem": str(cfg.get("claim_label_stem") or ""),
        "estimand_id": str(cfg.get("estimand_id") or ""),
        "primary_train": primary,
        "headline_fill": "pct75",
        "fill_ladder": {
            "mid": float("nan"),
            "pct75": float("nan"),
            "worst": float("nan"),
        },
        "path_summary": {"sharpe_mean": float("nan"), "n_paths": 0},
        "baselines": baselines,
        "random_baseline_sharpe": float(baselines["random"]["sharpe"]),
        "sign_lag_baseline_sharpe": float(baselines["sign_lag"]["sharpe"]),
        "long_baseline_sharpe": float(baselines["long"]["sharpe"]),
        "train_objective_equals_claim_metric": train_objective_equals_claim_metric(cfg),
        "friction_applied": fric_on,
        "spa_polarity": "policy_as_challenger",
        "dry_run": True,
    }


def _indices_for_windows(
    dates: Sequence[pd.Timestamp], windows: Sequence[Mapping[str, str]]
) -> np.ndarray:
    if not windows:
        return np.asarray([], dtype=int)
    out: list[int] = []
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        for i, d in enumerate(dates):
            ts = pd.Timestamp(d)
            if start <= ts <= end:
                out.append(i)
    return np.asarray(sorted(set(out)), dtype=int)


def _slice_feature_extras(cfg: Mapping[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """Copy cfg and row-slice panel extras to match a CPCV fold window."""
    out = track_copy(cfg)
    extras = dict(out.get("feature_extras") or {})
    if not extras or idx.size == 0:
        out["feature_extras"] = extras
    else:
        sliced: dict[str, Any] = {}
        for key, val in extras.items():
            if key in (
                "dollar_volume",
                "iv",
                "borrow",
                "fundamentals",
                "iv_surface",
                "macro",
                "ohlc",
                "microstructure",
                "fundamentals_pit",
                "sentiment",
                "option_flow",
                "jkp",
                "kelly_images",
            ):
                arr = np.asarray(val)
                if isinstance(val, dict):
                    # dict[str,(T,K)] surface signals: slice each panel.
                    sub: dict[str, Any] = {}
                    for sk, sv in val.items():
                        sa = np.asarray(sv)
                        if sa.ndim >= 1 and sa.shape[0] >= int(idx.max()) + 1:
                            sub[sk] = sa[idx]
                        else:
                            raise ValueError(
                                f"feature_extras[{key!r}][{sk!r}] T="
                                f"{sa.shape[0] if sa.ndim >= 1 else 'scalar'} "
                                f"mismatch fold idx max={int(idx.max())}"
                            )
                    sliced[key] = sub
                elif arr.ndim >= 1 and arr.shape[0] >= int(idx.max()) + 1:
                    sliced[key] = arr[idx]
                else:
                    raise ValueError(
                        f"feature_extras[{key!r}] T={arr.shape[0] if arr.ndim >= 1 else 'scalar'} "
                        f"mismatch fold idx max={int(idx.max())}"
                    )
            else:
                sliced[key] = val
        out["feature_extras"] = sliced
    # Slice rebalance mask to the fold window when present.
    mask = out.get("_rebalance_mask")
    if mask is not None and idx.size > 0:
        from mascotrl.eval.cadence import slice_rebalance_mask

        out["_rebalance_mask"] = slice_rebalance_mask(
            np.asarray(mask, dtype=bool), idx
        )
    # W4.2: slice the dynamic-universe slot validity mask (T, K) alongside
    # the rebalance mask so a CPCV fold's env sees the same row window as
    # its returns/factors/extras.
    slot_mask = out.get("_slot_valid_mask")
    if slot_mask is not None and idx.size > 0:
        m = np.asarray(slot_mask, dtype=bool)
        if m.ndim != 2:
            raise ValueError(
                f"_slot_valid_mask must be 2-D (T,K); got shape {m.shape!r}"
            )
        if int(idx.max()) >= m.shape[0]:
            raise ValueError(
                f"_slot_valid_mask T={m.shape[0]} mismatch fold idx max="
                f"{int(idx.max())}"
            )
        out["_slot_valid_mask"] = m[idx]
    marks = out.get("_om_marks")
    if isinstance(marks, dict) and idx.size > 0:
        sliced_marks: dict[str, Any] = {}
        for key, val in marks.items():
            arr = np.asarray(val)
            if arr.ndim == 2 and arr.shape[0] > int(idx.max()):
                sliced_marks[key] = arr[idx]
            else:
                sliced_marks[key] = val
        out["_om_marks"] = sliced_marks
    return out


def _roll_test_pnl(
    *,
    returns: np.ndarray,
    factors: np.ndarray,
    dates: Sequence[pd.Timestamp],
    idx: np.ndarray,
    agent: Any,
    cfg: Mapping[str, Any],
    friction: FrictionSpec,
    train_residualizer: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Roll the frozen agent OOS; return ``{date: {"total_net":..,"residual":..,"weights":..}}``.

    Both scorecard columns are always returned so downstream statistics can
    select the one that matches the peer they are being compared against
    (A2): the policy must never be scored on ``residual`` while a benchmark
    panel is scored on ``total_net``. ``weights`` (the post-fill executed
    weight vector) rides along so D2's per-strategy parquet export can
    persist the policy's holdings the same way it persists every benchmark's
    (via ``score_strategy``'s ``weights`` array).
    """
    from mascotrl.models.inference import roll_oos_with_agent

    return roll_oos_with_agent(
        returns=returns,
        factors=factors,
        dates=dates,
        idx=idx,
        agent=agent,
        cfg=cfg,
        friction=friction,
        train_residualizer=train_residualizer,
    )


def _split_scorecard(
    fold_pnl: dict[str, dict[str, float]], scorecard: str
) -> dict[str, float]:
    return {ds: float(v.get(scorecard, 0.0)) for ds, v in fold_pnl.items()}


def _jsonable_oos_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Serialize per-date OOS roll records for the CPCV resume manifest."""
    out: dict[str, dict[str, Any]] = {}
    for ds, rec in records.items():
        row: dict[str, Any] = {}
        for key, val in dict(rec).items():
            if key == "weights":
                row["weights"] = [float(x) for x in list(val or [])]
            elif isinstance(val, (float, int, np.floating, np.integer)):
                row[key] = float(val)
            else:
                row[key] = val
        out[str(ds)] = row
    return out


def _reconstruct_path0_aux_series(
    dates: Sequence[pd.Timestamp],
    fold_pnl: dict[int, dict[str, dict[str, Any]]],
    cpcv: CPCVConfig,
) -> dict[str, Any]:
    """D2: rebuild path 0's weights/turnover/cost/gross alongside its pnl.

    Mirrors :func:`src.eval.cpcv.reconstruct_paths`'s fold-per-group
    assignment walk (same ``group_bounds`` / ``assign_paths`` primitives) but
    pulls the auxiliary per-date fields ``_roll_test_pnl`` now records instead
    of a bare pnl float, so the eq allocation campaign can persist the
    policy's actual executed holdings the same way it persists every
    benchmark's (via ``score_strategy``'s ``weights`` array). Restricted to
    path 0 (one representative OOS walk) rather than every CPCV path, since
    the book only needs one policy holdings series, not a bundle for all
    ``C(n_splits, n_test_groups)`` paths.
    """
    from mascotrl.eval.cpcv import assign_paths, group_bounds

    bounds = group_bounds(list(dates), cpcv.n_splits)
    assignments = assign_paths(cpcv)
    if not assignments:
        return {"dates": [], "weights": [], "turnover": [], "cost": [], "gross": []}
    assignment = assignments[0]
    rows: list[tuple[str, list[float], float, float, float]] = []
    for g, fid in assignment:
        lo, hi = bounds[g]
        for i in range(lo, hi + 1):
            ds = str(pd.Timestamp(dates[i]).date())
            rec = (fold_pnl.get(fid) or {}).get(ds)
            if rec is None:
                continue
            rows.append(
                (
                    ds,
                    list(rec.get("weights") or []),
                    float(rec.get("turnover", 0.0)),
                    float(rec.get("cost", 0.0)),
                    float(rec.get("gross", 0.0)),
                )
            )
    rows.sort(key=lambda r: r[0])
    return {
        "dates": [r[0] for r in rows],
        "weights": [r[1] for r in rows],
        "turnover": [r[2] for r in rows],
        "cost": [r[3] for r in rows],
        "gross": [r[4] for r in rows],
    }


def run_policy_level_negative_control(
    dates: Sequence[pd.Timestamp],
    returns: np.ndarray,
    factors: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    cpcv: CPCVConfig,
    seed: int,
    clean_sharpe: float,
    signals: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Train and score one CPCV fold with name-permuted policy features."""
    from mascotrl.eval.cpcv import build_cpcv_folds
    from mascotrl.eval.negative_controls import (
        permute_signals_across_names,
        policy_level_negative_control_stamp,
    )
    from mascotrl.eval.stats_rigor import annualized_sharpe

    dates = list(dates)
    rets = np.asarray(returns, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    fold = next(
        (
            candidate
            for candidate in build_cpcv_folds(dates, cpcv)
            if _indices_for_windows(dates, candidate.train_windows).size >= 8
            and _indices_for_windows(dates, candidate.test_windows).size >= 3
        ),
        None,
    )
    if fold is None:
        raise RuntimeError("policy negative control has no usable CPCV fold")

    cfg_ctrl = track_copy(cfg)
    extras = dict(cfg_ctrl.get("feature_extras") or {})
    extras["iv_surface"] = permute_signals_across_names(signals, seed=int(seed))
    cfg_ctrl["feature_extras"] = extras
    cfg_ctrl["_fold_id"] = int(fold.fold_id)
    if cfg_ctrl.get("_checkpoint_dir"):
        cfg_ctrl["_checkpoint_dir"] = str(
            Path(str(cfg_ctrl["_checkpoint_dir"])) / "policy_negative_control"
        )
    if cfg_ctrl.get("_run_config_hash"):
        cfg_ctrl["_run_config_hash"] = f"{cfg_ctrl['_run_config_hash']}:policy-negative-control"
    cfg_ctrl.pop("_resume_checkpoint", None)

    train_idx = _indices_for_windows(dates, fold.train_windows)
    test_idx = _indices_for_windows(dates, fold.test_windows)
    fold_seed = int(seed) + int(fold.fold_id)
    train_out = _train_agent_for_fold(cfg_ctrl, rets, fac, train_idx, seed=fold_seed)
    train_resid = freeze_residualizer(
        fit_ff4_residualizer(
            np.nanmean(rets[train_idx], axis=1),
            fac[train_idx],
            fold_id=f"neg_ctrl_fold_{fold.fold_id}",
        ),
        f"neg_ctrl_fold_{fold.fold_id}",
    )
    pnl = _roll_test_pnl(
        returns=rets,
        factors=fac,
        dates=dates,
        idx=test_idx,
        agent=train_out["agent"],
        cfg=_slice_feature_extras(cfg_ctrl, test_idx),
        friction=friction_spec_from_cfg(cfg_ctrl),
        train_residualizer=train_resid,
    )
    control_sharpe = annualized_sharpe(
        np.asarray([row["total_net"] for row in pnl.values()], dtype=np.float64),
        periods=float(cfg_ctrl.get("_periods_per_year") or 252.0),
    )
    return policy_level_negative_control_stamp(
        control_sharpe=float(control_sharpe),
        clean_sharpe=float(clean_sharpe),
        seed=int(seed),
        fold_id=int(fold.fold_id),
    )


def run_research_alpha_cpcv(
    dates: Sequence[pd.Timestamp],
    returns: np.ndarray,
    factors: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    cpcv: CPCVConfig | None = None,
    seed: int = 0,
    panel_source: str = "optionmetrics",
    out_dir: Path | str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train single-agent hist PPO per CPCV fold; emit research stamp fields.

    ``out_dir``/``resume`` forward to :func:`run_cpcv`'s fold-level manifest
    cache (arm ``"eq_dii"``) so a crashed multi-hour campaign can resume
    without re-training already-completed folds. When ``out_dir`` is None
    (the pre-W3.1 default call sites / most tests), CPCV runs without a
    resume manifest exactly as before.
    """
    headline = str(cfg.get("headline_fill") or "")
    if headline != "pct75":
        raise ValueError(
            f"research CPCV requires headline_fill=pct75 (got {headline!r}); mid-only refused"
        )
    resolve_research_primary_train(cfg)
    dates = list(dates)
    rets = np.asarray(returns, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    if len(dates) != rets.shape[0]:
        raise ValueError("dates length must match returns T")
    cfg = track_copy(cfg)
    # Monthly/weekly cadence: stamp a panel-length rebalance mask so fold
    # slicing (_slice_feature_extras) and train_research_hist/build_research_hist_env
    # see the same schedule (Phase 2 fail-closed without dates|_rebalance_mask).
    cadence = str(cfg.get("rebalance_cadence") or "daily").lower()
    if cadence not in ("", "daily") and cfg.get("_rebalance_mask") is None:
        from mascotrl.eval.cadence import build_rebalance_mask

        cfg["_rebalance_mask"] = build_rebalance_mask(dates, cadence)
    # Stamp term-spread z for archetype_inflation risk-aversion scaling.
    from mascotrl.spectrum.policy_mode import (
        resolve_policy_mode,
        resolve_term_spread_z_for_train,
    )

    if resolve_policy_mode(cfg) == "archetype_inflation" and cfg.get(
        "term_spread_z", cfg.get("_term_spread_z_mean")
    ) is None:
        tsz = resolve_term_spread_z_for_train(cfg, dates=dates)
        if tsz is not None:
            cfg["_term_spread_z_mean"] = float(tsz)
    cpcv = cpcv or residual_equity_cpcv_config()

    base_fric = friction_spec_from_cfg(cfg)
    ladder_specs = _fill_ladder_specs(base_fric)

    # dict[fill_name][fold_id] -> {date: {"total_net":.., "residual":..}}
    fold_pnl_by_fill: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        name: {} for name in ladder_specs
    }
    fold_train_meta: dict[str, Any] = {
        "folds": [],
        "optimizer_steps_total": 0,
        "turnover_cap_projection_steps": 0,
        "turnover_cap_binding_steps": 0,
    }
    last_train_out: dict[str, Any] | None = None
    curves_dir = cfg.get("_learning_curves_dir")

    def fold_runner(fold: CPCVFold) -> dict[str, float]:
        nonlocal last_train_out
        arch = str(cfg.get("architecture") or "mlp").lower()
        if arch in ("lstm", "mamba", "gru", "transformer"):
            import gc

            gc.collect()
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        log.info("phase=cpcv_fold start fold_id=%s", fold.fold_id)
        cfg["_fold_id"] = int(fold.fold_id)
        train_idx = _indices_for_windows(dates, fold.train_windows)
        test_idx = _indices_for_windows(dates, fold.test_windows)
        if train_idx.size < 8 or test_idx.size < 3:
            for name in ladder_specs:
                fold_pnl_by_fill[name][int(fold.fold_id)] = {}
            log.info(
                "phase=cpcv_fold skip fold_id=%s train=%s test=%s",
                fold.fold_id,
                train_idx.size,
                test_idx.size,
            )
            return {}
        # A1: every fold trains a fresh agent. Warm-starting fold n+1 from
        # fold n let the fold-1 agent see parameters shaped by data that
        # overlapped its own test segment, breaking the CPCV
        # never-trained-on-its-test-segment guarantee. Lost budget is
        # recovered via a higher per-fold train_env_steps in config.
        # C5: train_world selects what panel that fresh agent trains on
        # (historical / one of 5 synthetic worlds / hybrid pretrain+finetune);
        # eval below always stays on the real historical test_idx slice.
        fold_seed = int(seed) + int(fold.fold_id)
        ckpt_dir = cfg.get("_checkpoint_dir")
        resume_ckpt = (
            _discover_latest_checkpoint(
                str(ckpt_dir),
                fold_seed,
                int(fold.fold_id),
                cfg.get("_run_config_hash"),
            )
            if ckpt_dir
            else None
        )
        if resume_ckpt is None:
            cfg.pop("_resume_checkpoint", None)
        else:
            cfg["_resume_checkpoint"] = str(resume_ckpt)
        train_out = _train_agent_for_fold(
            cfg, rets, fac, train_idx, seed=fold_seed
        )
        agent = train_out["agent"]
        last_train_out = train_out
        # Prune intermediate fold checkpoints once the fold train artifact is
        # in memory; keep only the latest .pt for crash recovery of *this*
        # fold until the seed JSON is written.
        ckpt_dir = cfg.get("_checkpoint_dir")
        if ckpt_dir:
            from mascotrl.eval.research_alpha_train import prune_fold_checkpoints

            prune_fold_checkpoints(
                Path(str(ckpt_dir)),
                keep_latest=1,
                fold_id=int(fold.fold_id),
                seed=int(fold_seed),
            )
        fold_train_meta["folds"].append(
            {
                "fold_id": int(fold.fold_id),
                "n_steps": int(train_out.get("n_steps") or 0),
                "n_episodes": int(train_out.get("n_episodes") or 0),
                "optimizer_steps": int(train_out.get("optimizer_steps") or 0),
                "turnover_cap_binding_fraction": float(
                    train_out.get("turnover_cap_binding_fraction") or 0.0
                ),
                "mean_reward": train_out.get("mean_reward"),
                "train_stats": train_out.get("train_stats") or {},
                "learning_curve": train_out.get("learning_curve") or [],
                "signal_sensitivities": _signal_sensitivities(train_out),
                "rl_backend": train_out.get("rl_backend"),
                "objective": train_out.get("objective"),
                "objective_gradient_path": train_out.get("objective_gradient_path"),
                "friction_applied": train_out.get("friction_applied"),
                "cost_in_decision_realized": train_out.get("cost_in_decision_realized"),
            }
        )
        for _prov_key in (
            "rl_backend",
            "objective",
            "objective_gradient_path",
            "friction_applied",
            "cost_in_decision_realized",
        ):
            if train_out.get(_prov_key) is not None:
                fold_train_meta[_prov_key] = train_out.get(_prov_key)
        fold_train_meta["optimizer_steps_total"] = int(
            fold_train_meta["optimizer_steps_total"]
        ) + int(train_out.get("optimizer_steps") or 0)
        fold_train_meta["turnover_cap_projection_steps"] += int(
            train_out.get("turnover_cap_projection_steps") or 0
        )
        fold_train_meta["turnover_cap_binding_steps"] += int(
            train_out.get("turnover_cap_binding_steps") or 0
        )
        if curves_dir:
            from mascotrl.eval.train_budget import write_learning_curve

            write_learning_curve(
                train_out.get("learning_curve") or [],
                Path(str(curves_dir)) / f"seed{seed}_fold{fold.fold_id}.json",
            )
        if arch in ("lstm", "mamba", "gru", "transformer"):
            import gc

            # Do not retain full learning curves or the train env in fold meta /
            # last_train_out; they pin multi-GB feature cubes across 15 CPCV folds.
            for meta in fold_train_meta["folds"]:
                if meta.get("fold_id") == int(fold.fold_id):
                    meta["learning_curve"] = []
                    break
            train_out.pop("env", None)
            train_out.pop("learning_curve", None)
            gc.collect()
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        train_resid = freeze_residualizer(
            fit_ff4_residualizer(
                np.nanmean(rets[train_idx], axis=1),
                fac[train_idx],
                fold_id=f"cpcv_fold_{fold.fold_id}",
            ),
            f"cpcv_fold_{fold.fold_id}",
        )
        for name, fric in ladder_specs.items():
            fold_pnl_by_fill[name][int(fold.fold_id)] = _roll_test_pnl(
                returns=rets,
                factors=fac,
                dates=dates,
                idx=test_idx,
                agent=agent,
                cfg=_slice_feature_extras(cfg, test_idx),
                friction=fric,
                train_residualizer=train_resid,
            )
        scorecard = _split_scorecard(
            fold_pnl_by_fill["pct75"][int(fold.fold_id)], "total_net"
        )
        if not scorecard:
            raise RuntimeError(
                f"CPCV fold {fold.fold_id} produced empty OOS pnl "
                f"(test_idx_size={int(np.asarray(test_idx).size)}); refuse silent hole"
            )
        log.info(
            "phase=cpcv_fold done fold_id=%s n_steps=%s",
            fold.fold_id,
            int(train_out.get("n_steps") or 0),
        )
        arch = str(cfg.get("architecture") or "mlp").lower()
        if arch in ("lstm", "mamba", "gru", "transformer"):
            import gc

            # last_train_out aliases train_out; drop agent so prior folds free.
            train_out.pop("agent", None)
            train_out.pop("env", None)
            del agent, train_out
            last_train_out = None
            gc.collect()
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        # Cache rich OOS records (weights/cost/turnover) so resume can rebuild
        # path-0 holdings for behaviour_export without retraining.
        from mascotrl.eval.cpcv import _CPCV_FOLD_AUX_KEY

        scorecard_out = dict(scorecard)
        scorecard_out[_CPCV_FOLD_AUX_KEY] = _jsonable_oos_records(
            fold_pnl_by_fill["pct75"][int(fold.fold_id)]
        )
        return scorecard_out

    # Note: run_cpcv's manifest only caches the pct75-rung pnl dict
    # fold_runner returns; a resumed (skipped) fold does not re-populate
    # fold_pnl_by_fill["mid"/"worst"] for that fold, so a crash-resumed run's
    # non-headline fill_ladder rungs can undercount resumed folds. The
    # headline pct75 rung (path_summary, capital gates) stays exact.
    periods = float(cfg.get("_periods_per_year") or 252.0)
    from mascotrl.eval.cpcv import stamp_reselect_purge_meta

    extra_purge_indices = None
    extra_purge_radius = None
    reselect_purge_meta: dict[str, Any] | None = None
    u_mask = cfg.get("_universe_reselect_mask")
    if u_mask is not None:
        reselect_purge_meta = stamp_reselect_purge_meta(
            dates,
            u_mask,
            purge_radius=int(getattr(cpcv, "purge_days", 21) or 21),
        )
        extra_purge_indices = list(reselect_purge_meta["reselect_indices"])
        extra_purge_radius = int(reselect_purge_meta["purge_radius"])
    from mascotrl.eval.cpcv_backend import resolve_use_purgedcv
    from mascotrl.eval.cpcv_lib import run_cpcv_lib

    _cpcv_runner = run_cpcv_lib if resolve_use_purgedcv(cfg) else run_cpcv
    if out_dir is not None:
        cpcv_art = _cpcv_runner(
            dates,
            fold_runner,
            cpcv,
            resume=resume,
            out_dir=out_dir,
            seed=seed,
            arm="eq_dii",
            periods=periods,
            extra_purge_indices=extra_purge_indices,
            extra_purge_radius=extra_purge_radius,
        )
    else:
        cpcv_art = _cpcv_runner(
            dates,
            fold_runner,
            cpcv,
            periods=periods,
            extra_purge_indices=extra_purge_indices,
            extra_purge_radius=extra_purge_radius,
        )
    if reselect_purge_meta is not None:
        cpcv_art["n_purged_at_reselect"] = int(
            reselect_purge_meta["n_purged_at_reselect"]
        )
    if cpcv_art.get("n_failed_folds"):
        # A10: a fold that raised leaves fold_pnl_by_fill missing entries
        # for that fold id; do not silently reconstruct a path with a hole.
        raise RuntimeError(
            "research alpha CPCV had failed folds (fail-closed): "
            f"{cpcv_art.get('failed_fold_ids')} reasons={cpcv_art.get('failure_reasons')}"
        )
    # D5: resume-skipped folds never re-enter fold_runner, so backfill the
    # headline pct75 rich records from the CPCV fold_aux cache.
    for fid, aux in dict(cpcv_art.get("fold_aux") or {}).items():
        if not isinstance(aux, Mapping):
            continue
        fold_pnl_by_fill["pct75"][int(fid)] = {
            str(ds): dict(rec) for ds, rec in aux.items() if isinstance(rec, Mapping)
        }
    from mascotrl.eval.cpcv import reconstruct_paths, summarize_paths, build_cpcv_folds
    from mascotrl.eval.cpcv_backend import resolve_use_purgedcv as _resolve_purgedcv_folds

    if _resolve_purgedcv_folds(cfg):
        from mascotrl.eval.cpcv_lib import build_cpcv_folds_lib

        folds = build_cpcv_folds_lib(
            dates,
            cpcv,
            extra_purge_indices=extra_purge_indices,
            extra_purge_radius=extra_purge_radius,
        )
    else:
        folds = build_cpcv_folds(
            dates,
            cpcv,
            extra_purge_indices=extra_purge_indices,
            extra_purge_radius=extra_purge_radius,
        )
    # A2: reconstruct CPCV paths per (fill rung x scorecard) so the policy
    # never ends up compared to a benchmark scored on the other column.
    fill_ladder: dict[str, float] = {}
    fill_ladder_residual: dict[str, float] = {}
    path_summaries: dict[str, Any] = {}
    path_summaries_residual: dict[str, Any] = {}
    paths_pct75: list[dict[str, Any]] = []
    paths_pct75_residual: list[dict[str, Any]] = []
    for name, fpnl in fold_pnl_by_fill.items():
        fpnl_tot = {fid: _split_scorecard(v, "total_net") for fid, v in fpnl.items()}
        fpnl_res = {fid: _split_scorecard(v, "residual") for fid, v in fpnl.items()}
        paths_tot = reconstruct_paths(dates, folds, fpnl_tot, cpcv, periods=periods)
        paths_res = reconstruct_paths(dates, folds, fpnl_res, cpcv, periods=periods)
        summary_tot = summarize_paths(paths_tot)
        summary_res = summarize_paths(paths_res)
        path_summaries[name] = summary_tot
        path_summaries_residual[name] = summary_res
        fill_ladder[name] = float(summary_tot.get("sharpe_mean", float("nan")))
        fill_ladder_residual[name] = float(summary_res.get("sharpe_mean", float("nan")))
        if name == "pct75":
            paths_pct75 = paths_tot
            paths_pct75_residual = paths_res

    # Gross baselines kept for continuity with pre-W1.2 seals.
    baselines_gross = research_baselines_from_returns(
        rets, seed=int(seed), periods=periods
    )
    long_sh_gross = float(baselines_gross["long"]["sharpe"])

    # Parity-matched baselines: same friction / factors / rebalance mask as
    # the policy estimand (gate3 equal_weight peer). Fail closed if the
    # resulting estimand hash diverges from the policy hash.
    policy_mask = cfg.get("_rebalance_mask")
    if policy_mask is not None:
        policy_mask = np.asarray(policy_mask, dtype=bool).reshape(-1)
    baselines = research_baselines_from_returns(
        rets,
        seed=int(seed),
        factors=fac,
        friction=base_fric,
        rebalance_mask=policy_mask,
        cadence=str(cfg.get("rebalance_cadence") or "daily"),
        periods=periods,
    )
    long_geometry = _score_long_baseline_on_cpcv_tests(
        dates=dates,
        returns=rets,
        factors=fac,
        folds=folds,
        friction=base_fric,
        cadence=str(cfg.get("rebalance_cadence") or "daily"),
        rebalance_mask=policy_mask,
        periods=periods,
    )
    baselines["long"]["sharpe"] = float(long_geometry["sharpe"])
    baselines["long"]["path_geometry"] = "concatenated_cpcv_test_indices"
    baselines["long"]["n_test_rows"] = int(long_geometry["n_test_rows"])
    random_sh = float(baselines["random"]["sharpe"])
    sign_sh = float(baselines["sign_lag"]["sharpe"])
    long_sh = float(baselines["long"]["sharpe"])
    # Headline scorecard is total_net (matches the parity harness / benchmark
    # panel headline), so policy_beats_* and SPA-vs-EW compare like for like.
    path_summary = path_summaries.get("pct75") or cpcv_art.get("path_summary") or {}
    path_summary_residual = path_summaries_residual.get("pct75") or {}
    policy_sh = float(path_summary.get("sharpe_mean", float("nan")))

    # A6: the policy carries an estimand_hash so require_uniform_estimand_hashes
    # can include it alongside benchmarks/OLPS/ceiling arms (previously the
    # policy was silently excluded from that fail-closed gate).
    policy_cadence = str(cfg.get("rebalance_cadence") or "daily")
    policy_universe = (
        cfg.get("_universe_secids")
        or cfg.get("universe_secids")
        or cfg.get("dii_secids")
    )
    policy_hash_total_net = estimand_hash(
        friction=base_fric,
        cadence=policy_cadence,
        scorecard="total_net",
        universe=policy_universe,
        rebalance_mask=policy_mask,
    )
    long_baseline_estimand_hash = estimand_hash(
        friction=base_fric,
        cadence=policy_cadence,
        scorecard="total_net",
        universe=policy_universe,
        rebalance_mask=policy_mask,
    )
    if long_baseline_estimand_hash != policy_hash_total_net:
        raise RuntimeError(
            "long baseline estimand_hash mismatch vs policy "
            f"({long_baseline_estimand_hash[:12]}… != {policy_hash_total_net[:12]}…); "
            "refuse apples-to-oranges policy_beats_long"
        )
    policy_hash_residual = estimand_hash(
        friction=base_fric,
        cadence=policy_cadence,
        scorecard="residual",
        universe=policy_universe,
        rebalance_mask=policy_mask,
        residualizer=ResidualizerState(
            fold_id="research", model="ff4", betas=np.zeros(4),
            factor_names=("mkt", "smb", "hml", "mom"),
        ),
    )

    # Turnover-cap honesty stamps (W1.3).
    proj_mode = str(cfg.get("projection_mode") or "soft")
    turnover_limit = cfg.get("turnover_limit")
    turnover_cap_enforced = proj_mode == "hard" and turnover_limit is not None
    turnover_projection_steps = int(
        fold_train_meta.get("turnover_cap_projection_steps") or 0
    )
    turnover_binding_steps = int(
        fold_train_meta.get("turnover_cap_binding_steps") or 0
    )
    turnover_cap_binding_fraction = (
        float(turnover_binding_steps / turnover_projection_steps)
        if turnover_projection_steps
        else 0.0
    )
    path0_aux = _reconstruct_path0_aux_series(
        dates, fold_pnl_by_fill["pct75"], cpcv
    )
    turnover_values = np.asarray(path0_aux.get("turnover") or [], dtype=np.float64)
    turnover_mean = (
        float(np.nanmean(turnover_values)) if turnover_values.size else float("nan")
    )

    art: dict[str, Any] = {
        "claim_tier": str(cfg.get("claim_tier") or "research"),
        "claim_category": str(cfg.get("claim_category") or ""),
        "claim_label_stem": str(cfg.get("claim_label_stem") or ""),
        "estimand_id": str(cfg.get("estimand_id") or ""),
        "primary_train": "historical_arm_env",
        "headline_fill": "pct75",
        "scorecard": "total_net",
        "estimand_hash": policy_hash_total_net,
        "estimand_hash_residual": policy_hash_residual,
        "fill_ladder": fill_ladder,
        "fill_ladder_residual": fill_ladder_residual,
        "path_summary": path_summary,
        "path_summary_residual": path_summary_residual,
        "path_summaries_by_fill": path_summaries,
        "path_summaries_by_fill_residual": path_summaries_residual,
        "baselines": baselines,
        "baselines_gross": baselines_gross,
        "random_baseline_sharpe": random_sh,
        "sign_lag_baseline_sharpe": sign_sh,
        "long_baseline_sharpe": long_sh,
        "long_baseline_sharpe_gross": long_sh_gross,
        "long_baseline_estimand_hash": long_baseline_estimand_hash,
        "projection_mode": proj_mode,
        "turnover_limit": turnover_limit,
        "turnover_cap_enforced": bool(turnover_cap_enforced),
        "turnover_cap_binding_fraction": turnover_cap_binding_fraction,
        "turnover_mean": turnover_mean,
        "train_objective_equals_claim_metric": train_objective_equals_claim_metric(cfg),
        "friction_applied": bool(
            (last_train_out or {}).get("friction_applied")
            if last_train_out is not None
            else fold_train_meta.get("friction_applied", False)
        ),
        "spa_polarity": "policy_as_challenger",
        "dry_run": False,
        "panel_source": str(panel_source),
        "use_purgedcv": bool(resolve_use_purgedcv(cfg)),
        "rl_backend": str(
            fold_train_meta.get("rl_backend")
            or (last_train_out or {}).get("rl_backend")
            or cfg.get("rl_backend")
            or "sb3"
        ),
        "objective": str(
            fold_train_meta.get("objective")
            or (last_train_out or {}).get("objective")
            or cfg.get("objective")
            or ""
        ),
        "objective_gradient_path": str(
            fold_train_meta.get("objective_gradient_path")
            or (last_train_out or {}).get("objective_gradient_path")
            or cfg.get("objective_gradient_path")
            or ""
        ),
        "cost_in_decision_realized": (
            fold_train_meta.get("cost_in_decision_realized")
            if "cost_in_decision_realized" in fold_train_meta
            else (last_train_out or {}).get("cost_in_decision_realized")
        ),
        "bootstrap_backend": str(cfg.get("bootstrap_backend") or "custom"),
        "policy_beats_random": policy_beats_random(policy_sh, random_sh),
        "policy_beats_sign_lag": policy_beats_random(policy_sh, sign_sh),
        "policy_beats_long": policy_beats_random(policy_sh, long_sh),
        "cpcv": {
            "n_folds": cpcv_art.get("n_folds"),
            "n_paths": (path_summary or {}).get("n_paths"),
            "config": {
                "n_splits": cpcv.n_splits,
                "n_test_groups": cpcv.n_test_groups,
                "purge_days": cpcv.purge_days,
                "embargo_days": cpcv.embargo_days,
            },
        },
        "paths": {
            str(p.get("path_id")): (
                {
                    "pnl": p.get("pnl"),
                    "sharpe": p.get("sharpe"),
                    **path0_aux,
                }
                if int(p.get("path_id", -1)) == 0
                else {"pnl": p.get("pnl"), "sharpe": p.get("sharpe"), "dates": p.get("dates")}
            )
            for p in paths_pct75
        },
        "paths_residual": {
            str(p.get("path_id")): {"pnl": p.get("pnl"), "sharpe": p.get("sharpe")}
            for p in paths_pct75_residual
        },
        "train_meta": fold_train_meta,
    }
    # Persist OOS panel returns aligned to path-0 dates for behaviour export
    # (sleeve tilts / capture ratios / skew). Without this, remote cells
    # starve archetype scoring of return-based measures.
    path0_dates = list(path0_aux.get("dates") or [])
    idxs: list[int] = []
    if path0_dates:
        date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        idxs = [
            date_index[pd.Timestamp(d)]
            for d in path0_dates
            if pd.Timestamp(d) in date_index
        ]
    if idxs:
        art["panel_returns"] = np.asarray(rets[idxs], dtype=np.float64).tolist()
    if "panel_returns" not in art:
        art["panel_returns"] = np.asarray(rets, dtype=np.float64).tolist()
    # Persist universe alignment for holdings-based exposures (composition scoring).
    art["universe_secids"] = [str(s) for s in (cfg.get("_universe_secids") or [])]
    eval_dates_src = path0_dates or list(dates)
    art["eval_dates"] = [str(pd.Timestamp(d).date()) for d in eval_dates_src]
    # Gate2 inputs: path-0 PnL series + factor rows aligned to the same dates.
    # Spectrum campaign `_gates_from_runner` looks for policy_returns + factors.
    path0_pnl = None
    for p in paths_pct75:
        if int(p.get("path_id", -1)) == 0:
            path0_pnl = p.get("pnl")
            break
    if path0_pnl is not None:
        art["policy_returns"] = list(path0_pnl)
        art["oos_returns"] = art["policy_returns"]
    if idxs:
        art["factors"] = np.asarray(fac[idxs], dtype=np.float64).tolist()
        art["oos_factors"] = art["factors"]
    elif fac.size:
        art["factors"] = np.asarray(fac, dtype=np.float64).tolist()
        art["oos_factors"] = art["factors"]
    factor_names = cfg.get("_factor_names") or cfg.get("factor_names")
    if factor_names:
        art["factor_names"] = list(factor_names)
    n_fac = int(np.asarray(art.get("factors") or [], dtype=np.float64).shape[-1]) if art.get("factors") else 0
    if n_fac:
        art["n_factors"] = n_fac
    # In-sample train metric for transfer_report (never echo OOS Sharpe).
    train_rews = [
        float(f.get("mean_reward"))
        for f in fold_train_meta.get("folds") or []
        if f.get("mean_reward") is not None
        and float(f.get("mean_reward")) == float(f.get("mean_reward"))
    ]
    if train_rews:
        art["train_fold_metric"] = float(np.mean(train_rews))
    # W2.1: path-0 carries the policy's realized holdings; when present,
    # attach concentration/collapse diagnostics so a densest-subgraph or
    # equal-weight-relabeled policy is caught in the artifact itself.
    path0 = art["paths"].get("0") or {}
    path0_weights = path0.get("weights")
    train_diag = _training_policy_diagnostics(fold_train_meta["folds"])
    art["training_diagnostics"] = {
        "log_std_mean": train_diag.get("log_std_mean"),
        "advantage_variance_mean": train_diag.get("advantage_variance_mean"),
        "logit_xsec_std_mean": train_diag.get("logit_xsec_std_mean"),
        "reward_signal_to_noise_mean": train_diag.get("reward_signal_to_noise_mean"),
    }
    if path0_weights:
        art["policy_diagnostics"] = summarize_policy_diagnostics(
            weights=np.asarray(path0_weights, dtype=np.float64),
            turnovers=path0.get("turnover"),
            entropies=train_diag["entropies"],
            log_std_mean=train_diag["log_std_mean"],
        )
        art["policy_diagnostics"]["advantage_variance_mean"] = train_diag.get(
            "advantage_variance_mean"
        )
        art["policy_diagnostics"]["logit_xsec_std_mean"] = train_diag.get(
            "logit_xsec_std_mean"
        )
        art["policy_diagnostics"]["reward_signal_to_noise_mean"] = train_diag.get(
            "reward_signal_to_noise_mean"
        )
        art["equal_weight_collapse_detected"] = bool(
            art["policy_diagnostics"]["equal_weight_collapse_guard"]["collapse_detected"]
        )
        sensitivity_values: dict[str, list[float]] = {}
        for fold in fold_train_meta["folds"]:
            for name, value in (fold.get("signal_sensitivities") or {}).items():
                sensitivity_values.setdefault(str(name), []).append(float(value))
        art["policy_sensitivities"] = {
            name: float(np.mean(values)) for name, values in sensitivity_values.items()
        }
    elif path0.get("turnover"):
        art["policy_diagnostics"] = {
            "collapse_guard": collapse_guard(path0["turnover"]),
            "advantage_variance_mean": train_diag.get("advantage_variance_mean"),
            "logit_xsec_std_mean": train_diag.get("logit_xsec_std_mean"),
            "reward_signal_to_noise_mean": train_diag.get(
                "reward_signal_to_noise_mean"
            ),
        }
    for stamp_key in ("pit_as_of", "factors_source", "om_marks_degraded"):
        if stamp_key in cfg:
            art[stamp_key] = cfg[stamp_key]
    # Fail closed on empty learning (campaign must actually train).
    min_total = int(cfg.get("min_optimizer_steps_total", 0) or 0)
    if min_total > 0:
        from mascotrl.eval.train_budget import assert_optimizer_step_floor

        assert_optimizer_step_floor(
            int(fold_train_meta.get("optimizer_steps_total") or 0),
            min_steps=min_total,
        )
    # Persist last-fold agent into the model zoo for user access.
    if last_train_out is not None and last_train_out.get("agent") is not None:
        try:
            from mascotrl.eval.research_alpha_train import _checkpoint_payload
            from mascotrl.models.registry import ModelCard, make_model_id, save_model_bundle

            agent = last_train_out["agent"]
            env = last_train_out.get("env")
            action_dim = int(getattr(env, "K", 0) or cfg.get("n_assets") or 0)
            obs_dim = 0
            if env is not None:
                try:
                    o, _ = env.reset()
                    obs_dim = int(np.asarray(o).reshape(-1).shape[0])
                except Exception:
                    obs_dim = 0
            payload = _checkpoint_payload(
                agent,
                cfg,
                seed=int(seed),
                episode=int(last_train_out.get("n_episodes") or 0),
                optimizer_steps=int(last_train_out.get("optimizer_steps") or 0),
            )
            if payload is not None and obs_dim > 0 and action_dim > 0:
                mid = make_model_id(
                    family="research_single_agent",
                    algo=str(cfg.get("algo") or "ppo"),
                    arm=str(
                        (cfg.get("arm") or {}).get("id")
                        or cfg.get("portfolio_arm")
                        or "eq"
                    ),
                    seed=int(seed),
                    run_config_hash=str(cfg.get("_run_config_hash") or ""),
                )
                card = ModelCard(
                    model_id=mid,
                    family="research_single_agent",
                    algo=str(cfg.get("algo") or "ppo"),
                    train_world=str(cfg.get("train_world") or ""),
                    architecture=str(
                        cfg.get("architecture") or cfg.get("temporal_backend") or ""
                    ),
                    objective=str(cfg.get("objective") or cfg.get("reward") or ""),
                    arm=str(
                        (cfg.get("arm") or {}).get("id")
                        or cfg.get("portfolio_arm")
                        or "eq"
                    ),
                    obs_dim=int(obs_dim),
                    action_dim=int(action_dim),
                    n_assets=int(action_dim),
                    seed=int(seed),
                    run_config_hash=str(cfg.get("_run_config_hash") or ""),
                    estimand_id=str(cfg.get("estimand_id") or ""),
                    sharpe_mean=float(
                        (path_summary or {}).get("sharpe_mean") or float("nan")
                    ),
                )
                save_model_bundle(payload, card)
                art["model_id"] = mid
        except Exception as e:
            art["model_zoo_error"] = str(e)[:300]
    return stamp_research_positive_alpha(art)
