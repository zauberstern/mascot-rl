"""Spectrum campaign: dry-run schema or real arm dispatch (opt / eq / mix)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from src.aws_burst.profiles import BUDGET_USD
from src.eval.equity_substrate import stamp_equity_obs_defaults

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_JSONL: Path | None = None

# Knobs that must invalidate resume caches when they change.
_SPECTRUM_RESUME_HASH_KEYS = (
    "algo",
    "architecture",
    "objective",
    "reward",
    "n_assets",
    "headline_fill",
    "train_env_steps",
    "train_epochs",
    "train_episodes",
    "projection_mode",
    "turnover_limit",
    "rl_backend",
    "cvar_alpha",
    "cvar_k_ratio",
    "cmdp",
    "reward_weights",
    "cpcv_n_splits",
    "cpcv_n_test_groups",
    "cpcv_purge_days",
    "cpcv_embargo_days",
    "use_equity_feature_cube",
    "use_surface_signals",
    "surface_obs_lane",
    "universe_arm",
    "feature_groups_exclude",
    "feature_channels_exclude",
    "primary_train",
    "train_world",
    "portfolio_arm",
    # Behavior-critical knobs previously omitted (Sweep C/H/I resume hazard).
    "weight_head",
    "weight_head_temperature",
    "weight_head_tilt_gain",
    "lr",
    "gamma",
    "policy_mode",
    "universe_arm",
    "objective_primary",
    "seeds",
    "scr_mix",
    "scr_beta",
    "entropy_coef",
    "gae_lambda",
    "container_digest",
)


def _spectrum_run_config_hash(cfg: dict[str, Any]) -> str:
    """Stable 16-char fingerprint for spectrum resume / checkpoint gating."""
    payload = {k: cfg.get(k) for k in _SPECTRUM_RESUME_HASH_KEYS}
    # Container provenance: prefer cfg stamp, else MASCOTRL_CONTAINER_DIGEST env.
    if payload.get("container_digest") is None:
        payload["container_digest"] = os.environ.get("MASCOTRL_CONTAINER_DIGEST")
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _prepare_spectrum_resume_dirs(
    cfg: dict[str, Any],
    cell_out_dir: Path | str,
) -> tuple[Path, Path, str]:
    """Create cpcv/ + ckpt/ under cell_out_dir; stamp cfg resume keys.

    Returns ``(cpcv_dir, ckpt_dir, run_config_hash)``.
    """
    root = Path(cell_out_dir)
    cpcv_dir = root / "cpcv"
    ckpt_dir = root / "ckpt"
    cpcv_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_hash = _spectrum_run_config_hash(cfg)
    cfg["_checkpoint_dir"] = str(ckpt_dir)
    cfg["_run_config_hash"] = run_hash
    return cpcv_dir, ckpt_dir, run_hash


def _log_event(kind: str, **fields: Any) -> None:
    global _JSONL
    if _JSONL is None:
        return
    row = {"kind": kind, "ts": time.time(), **fields}
    try:
        with _JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except OSError:
        pass


def _behaviour_context_for_cell(
    art: dict[str, Any],
    cfg: dict[str, Any],
    runner_art: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve sleeves, macro, and returns for rich policy_behavior export."""
    from src.data.paths import LAKE_ROOT
    from src.reporting.behavior_metrics import spectrum_foil_sleeve_matrix
    from src.reporting.policy_behavior import (
        extract_crucible_behaviour_inputs,
        load_behaviour_macro_context,
    )

    ra = runner_art or art.get("runner_artifact") or {}
    paths = ra.get("paths") or {}
    path0 = paths.get("0") or paths.get(0) or {}
    dates = list(path0.get("dates") or art.get("dates") or [])
    cruc = extract_crucible_behaviour_inputs(results=ra if isinstance(ra, dict) else {}, cfg=cfg)
    fioracle_cfg = dict((cfg.get("feature_extras") or {}).get("fioracle_macro") or {})
    macro = load_behaviour_macro_context(
        dates,
        lake_root=LAKE_ROOT,
        lake_subdir=str(fioracle_cfg.get("lake_subdir") or "macro/fioracle"),
    )
    asset_returns = None
    panel = (
        ra.get("panel_returns")
        or ra.get("returns_panel")
        or art.get("panel_returns")
        or art.get("returns_panel")
    )
    if panel is not None:
        asset_returns = np.asarray(panel, dtype=np.float64)
    # Rehydrate from lake when the runner artifact omitted panel_returns (RC3).
    if asset_returns is None and str(
        cfg.get("portfolio_arm") or cfg.get("arm") or art.get("arm") or ""
    ).lower() in ("eq", ""):
        try:
            from src.eval.equity_substrate import load_lake_dyn_hrp_panel

            k = int(cfg.get("n_assets") or art.get("n_assets") or 100)
            _dates, lake_rets, _fac, _meta = load_lake_dyn_hrp_panel(dict(cfg), k=k)
            asset_returns = np.asarray(lake_rets, dtype=np.float64)
            if not dates and _dates is not None:
                dates = list(_dates)
        except Exception:
            pass
    sleeve_matrix = cruc.get("sleeve_matrix")
    if sleeve_matrix is None:
        k_sleeves = None
        if asset_returns is not None and np.asarray(asset_returns).ndim == 2:
            k_sleeves = int(np.asarray(asset_returns).shape[1])
        elif path0.get("weights") is not None:
            w0 = np.asarray(path0.get("weights"), dtype=np.float64)
            if w0.ndim == 2:
                k_sleeves = int(w0.shape[1])
        if k_sleeves is None:
            k_sleeves = int(cfg.get("n_assets") or art.get("n_assets") or 0)
        if k_sleeves and k_sleeves > 0:
            sleeve_matrix = spectrum_foil_sleeve_matrix(k_sleeves)
    return {
        "dates": dates,
        "sleeve_matrix": sleeve_matrix,
        "universe_fingerprint": cruc.get("universe_fingerprint") or "",
        "regimes": macro.get("regimes"),
        "vix_z": macro.get("vix_z"),
        "hy_oas_z": macro.get("hy_oas_z"),
        "term_spread": macro.get("term_spread"),
        "epu_z": macro.get("epu_z"),
        "gpri_z": macro.get("gpri_z"),
        "asset_returns": asset_returns,
        "macro_status": macro.get("status"),
        "turnover_cap": cfg.get("turnover_limit") or cfg.get("turnover_cap"),
        "secids": [str(s) for s in (art.get("universe_secids") or cfg.get("_universe_secids") or [])],
        "eval_dates": [
            str(d)
            for d in (
                art.get("eval_dates")
                or dates
                or []
            )
        ],
    }


def refresh_behavior_exports(
    out_dir: Path,
    *,
    config_dir: Path | None = None,
    panel_rescore: bool = True,
) -> dict[str, Any]:
    """Re-emit ``*_policy_behavior.json`` from stored cell artifacts (no retrain).

    When ``panel_rescore`` is True (default), behaviour vectors are collected
    first and each cell is re-scored against the full peer panel so archetype
    z-scores are non-degenerate (RC5). Pass 2 also fits Archetypal Analysis
    composition across the panel so every cell gets an honest mixture
    (no catch-all residual bucket). Solo-cell remote exports always yield
    all-zero archetype scores.
    """
    from src.eval.rbsa import rbsa_from_artifact
    from src.eval.semantic_tilt import nan_semantic_tilt
    from src.reporting.archetypal_scoring import (
        bootstrap_ari,
        choose_k,
        composition_for_rows,
        rows_to_zmatrix,
        select_k_from_table,
    )
    from src.reporting.behavior_metrics import (
        BEHAVIOUR_MEASURE_IDS,
        COMPOSITION_MEASURE_IDS,
        compute_behaviour_vector,
        regime_behaviour_deltas,
        regime_conditional_behaviour,
        turbulence_regimes_from_returns,
    )
    from src.reporting.holdings_exposure import (
        holdings_exposures,
        load_characteristic_panel,
        nan_exposures,
    )
    from src.reporting.personality_probe import discriminability_probe
    from src.reporting.policy_behavior import (
        _enrich_scoring_row,
        build_policy_behavior,
        write_policy_behavior,
    )
    from src.reporting.style_agreement import style_agreement as fit_style_agreement

    config_dir = config_dir or (ROOT / "config" / "spectrum" / "cherrypick")
    summary: dict[str, Any] = {
        "refreshed": [],
        "skipped": [],
        "errors": [],
        "panel_rescore": bool(panel_rescore),
    }

    # Pass 1: load artifacts + compute raw behaviour rows for the peer panel.
    prepared: list[dict[str, Any]] = []
    for art_path in sorted(out_dir.glob("*.json")):
        if art_path.name.endswith("_policy_behavior.json"):
            continue
        if art_path.name in ("index.json", "campaign_manifest.json", "behavior_refresh_summary.json"):
            continue
        try:
            art = json.loads(art_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(art, dict):
            continue
        art_name = art_path.stem
        w = art.get("weights") or art.get("oos_weights")
        if w is None:
            paths = art.get("paths") or {}
            path0 = paths.get("0") or paths.get(0) or {}
            if isinstance(path0, dict):
                w = path0.get("weights")
        if w is None:
            runner_art0 = art.get("runner_artifact") or {}
            w = runner_art0.get("weights") or runner_art0.get("oos_weights")
            if w is None:
                rpaths = runner_art0.get("paths") or {}
                rp0 = rpaths.get("0") or rpaths.get(0) or {}
                if isinstance(rp0, dict):
                    w = rp0.get("weights")
        if w is None:
            summary["skipped"].append(
                {
                    "cell_id": art_name,
                    "reason": "weight_path_not_persisted",
                }
            )
            continue
        cfg_path = art.get("config_path")
        cfg_pre: dict[str, Any] = {}
        if cfg_path and Path(str(cfg_path)).is_file():
            cfg_pre = load_cell_yaml(Path(str(cfg_path)))
        else:
            matches = list(config_dir.rglob(f"{art_name}.yaml"))
            if matches:
                cfg_pre = load_cell_yaml(matches[0])
        ctx = _behaviour_context_for_cell(
            art,
            cfg_pre,
            art.get("runner_artifact"),
        )
        runner_art = art.get("runner_artifact") or {}
        sensitivities = art.get("policy_sensitivities") or runner_art.get(
            "policy_sensitivities"
        )
        prepared.append(
            {
                "art_name": art_name,
                "art": art,
                "cfg_pre": cfg_pre,
                "ctx": ctx,
                "weights": np.asarray(w),
                "sensitivities": sensitivities,
            }
        )

    score_rows: list[dict[str, float]] = []
    layer_cache: list[dict[str, Any]] = []
    if panel_rescore and len(prepared) >= 2:
        for item in prepared:
            behaviour = compute_behaviour_vector(
                item["weights"],
                asset_returns=item["ctx"].get("asset_returns"),
                sleeve_matrix=item["ctx"].get("sleeve_matrix"),
                turnover_cap=item["ctx"].get("turnover_cap"),
            )
            # Holdings exposures
            secids = list(item["ctx"].get("secids") or [])
            eval_dates = list(item["ctx"].get("eval_dates") or item["ctx"].get("dates") or [])
            exposures: dict[str, Any]
            if secids and eval_dates and item["weights"].ndim == 2 and item["weights"].shape[1] == len(secids):
                try:
                    from src.data.paths import LAKE_ROOT

                    panels = load_characteristic_panel(eval_dates, secids, lake_root=LAKE_ROOT)
                    exposures = holdings_exposures(item["weights"], panels)
                except Exception as exc:  # noqa: BLE001
                    exposures = nan_exposures(reason=f"holdings_failed:{exc}"[:120])
            else:
                exposures = nan_exposures(reason="secids_unavailable")
            for k, v in exposures.items():
                if k.startswith("exposure_") or k == "sector_hhi":
                    try:
                        behaviour[k] = float(v)
                    except (TypeError, ValueError):
                        behaviour[k] = float("nan")

            # RBSA
            rbsa = rbsa_from_artifact(item["art"])
            try:
                behaviour["rbsa_r_squared"] = float(rbsa.get("rbsa_r_squared", float("nan")))
            except (TypeError, ValueError):
                behaviour["rbsa_r_squared"] = float("nan")

            # Semantic tilt (static-asof text; interpretation only)
            semantic = nan_semantic_tilt("secids_unavailable")
            if (
                secids
                and item["weights"].ndim == 2
                and item["weights"].shape[1] == len(secids)
            ):
                try:
                    from src.data.paths import LAKE_ROOT
                    from src.eval.semantic_tilt import (
                        align_embeddings_to_secids,
                        embed_descriptions,
                        load_firm_text_map,
                        semantic_tilt_metrics,
                    )

                    text_map = load_firm_text_map(LAKE_ROOT)
                    blob = embed_descriptions(text_map.get("texts") or {})
                    E = align_embeddings_to_secids(blob, secids)
                    semantic = semantic_tilt_metrics(item["weights"], E)
                    semantic["embed_backend"] = blob.get("backend")
                except Exception as exc:  # noqa: BLE001
                    semantic = nan_semantic_tilt(f"semantic_failed:{exc}"[:120])
            for k in (
                "semantic_rotation_rate",
                "semantic_pc1_mean",
                "semantic_pc2_mean",
                "semantic_pc3_mean",
            ):
                try:
                    behaviour[k] = float(semantic.get(k, float("nan")))
                except (TypeError, ValueError):
                    behaviour[k] = float("nan")

            # Style agreement
            style = fit_style_agreement(exposures, rbsa)
            try:
                behaviour["style_agreement_cosine"] = float(
                    style.get("style_agreement_cosine", float("nan"))
                )
            except (TypeError, ValueError):
                behaviour["style_agreement_cosine"] = float("nan")

            # Regime deltas via turbulence
            regimes_eff = turbulence_regimes_from_returns(
                item["ctx"].get("asset_returns"),
                existing=item["ctx"].get("regimes"),
            )
            deltas: dict[str, float] = {
                "delta_hhi_regime": float("nan"),
                "delta_turnover_regime": float("nan"),
                "delta_defensive_regime": float("nan"),
                "delta_quality_regime": float("nan"),
            }
            if regimes_eff is not None:
                try:
                    by_reg = regime_conditional_behaviour(
                        item["weights"],
                        regimes=regimes_eff,
                        asset_returns=item["ctx"].get("asset_returns"),
                        sleeve_matrix=item["ctx"].get("sleeve_matrix"),
                        turnover_cap=item["ctx"].get("turnover_cap"),
                    )
                    deltas = regime_behaviour_deltas(by_reg)
                except Exception:
                    pass
            behaviour.update(deltas)

            score_rows.append(_enrich_scoring_row(behaviour))
            layer_cache.append(
                {
                    "exposures": exposures,
                    "rbsa": rbsa,
                    "regime_deltas": deltas,
                    "score_row": score_rows[-1],
                    "semantic": semantic,
                    "style": style,
                }
            )

    # Panel-wide composition (AA); empty when <2 cells.
    compositions: list[dict[str, Any]] = []
    if panel_rescore and len(score_rows) >= 2:
        feat_names = [
            m
            for m in COMPOSITION_MEASURE_IDS
            if any(np.isfinite(float(r.get(m, float("nan")))) for r in score_rows)
        ]
        compositions = composition_for_rows(
            score_rows, feature_names=feat_names or None, k=5, method="aa"
        )
        Xz = rows_to_zmatrix(score_rows, feat_names) if feat_names else np.zeros((0, 0))
        k_table = choose_k(Xz, ks=range(3, 9)) if Xz.shape[0] >= 6 else {}
        k_selected = int(select_k_from_table(k_table)) if k_table else 5
        stability = bootstrap_ari(Xz, k=5, n_boot=50, frac=0.8, seed=0)

        def _json_num(sv: Any) -> Any:
            if isinstance(sv, (bool, np.bool_)):
                return bool(sv)
            if isinstance(sv, (int, float, np.floating, np.integer)):
                fv = float(sv)
                return None if not np.isfinite(fv) else fv
            return sv

        summary["k_selection"] = {
            str(int(k)): {sk: _json_num(sv) for sk, sv in row.items()}
            for k, row in k_table.items()
        }
        summary["k_selected"] = int(k_selected)
        summary["k_used"] = 5
        summary["k_used_reason"] = "locked_five_named_archetypes"
        summary["composition_stability"] = {
            str(kk): _json_num(vv) for kk, vv in dict(stability).items()
        }
        summary["style_disagreement_cells"] = [
            item["art_name"]
            for item, layers in zip(prepared, layer_cache)
            if (layers.get("style") or {}).get("style_disagreement_flag")
        ]

    # Pass 2: write with peer panel + composition stamp.
    for idx, item in enumerate(prepared):
        peers = None
        if panel_rescore and score_rows:
            peers = [row for j, row in enumerate(score_rows) if j != idx]
        layers = layer_cache[idx] if idx < len(layer_cache) else {}
        comp = compositions[idx] if idx < len(compositions) else None
        try:
            beh = build_policy_behavior(
                cell_id=str(item["art"].get("spectrum_cell_id") or item["art_name"]),
                arm=str(item["art"].get("arm") or item["art"].get("portfolio_arm") or ""),
                algo=str(item["cfg_pre"].get("algo") or ""),
                architecture=str(item["cfg_pre"].get("architecture") or ""),
                objective=str(
                    item["cfg_pre"].get("objective") or item["cfg_pre"].get("reward") or ""
                ),
                train_world=str(item["cfg_pre"].get("train_world") or ""),
                policy_mode=str(item["cfg_pre"].get("policy_mode") or "balanced"),
                universe_fingerprint=str(item["ctx"].get("universe_fingerprint") or ""),
                cell_cfg=item["cfg_pre"],
                weights=item["weights"],
                turnovers=item["art"].get("turnovers"),
                sensitivities=item["sensitivities"],
                asset_returns=item["ctx"].get("asset_returns"),
                sleeve_matrix=item["ctx"].get("sleeve_matrix"),
                regimes=item["ctx"].get("regimes"),
                vix_z=item["ctx"].get("vix_z"),
                hy_oas_z=item["ctx"].get("hy_oas_z"),
                term_spread=item["ctx"].get("term_spread"),
                epu_z=item["ctx"].get("epu_z"),
                gpri_z=item["ctx"].get("gpri_z"),
                turnover_cap=item["ctx"].get("turnover_cap"),
                behaviour_panel=peers,
                n_null_shuffles=20,
                composition=comp,
                rbsa=layers.get("rbsa"),
                exposures=layers.get("exposures"),
                regime_deltas=layers.get("regime_deltas"),
                semantic_tilt=layers.get("semantic"),
                style_agreement=layers.get("style"),
            )
            beh_path = out_dir / f"{item['art_name']}_policy_behavior.json"
            write_policy_behavior(beh_path, beh)
            summary["refreshed"].append(str(beh_path))
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(
                {"cell_id": item["art_name"], "error": str(exc)[:200]}
            )

    # Discriminability probe across the panel feature matrix.
    if panel_rescore and len(score_rows) >= 4:
        feat_names = list(BEHAVIOUR_MEASURE_IDS)
        X = np.zeros((len(score_rows), len(feat_names)), dtype=np.float64)
        configs = []
        for i, item in enumerate(prepared[: len(score_rows)]):
            for j, f in enumerate(feat_names):
                try:
                    X[i, j] = float(score_rows[i].get(f, 0.0))
                except (TypeError, ValueError):
                    X[i, j] = 0.0
                if not np.isfinite(X[i, j]):
                    X[i, j] = 0.0
            configs.append(
                {
                    "weight_head": str(item["cfg_pre"].get("weight_head") or ""),
                    "objective": str(item["cfg_pre"].get("objective") or ""),
                }
            )
        # Z-score columns.
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xz = (X - mu) / sd
        summary["discriminability"] = discriminability_probe(Xz, configs)

    manifest_path = out_dir / "behavior_refresh_summary.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


from src.eval.collapse_guard import collapse_guard
from src.eval.transfer_report import build_transfer_report
from src.spectrum.cell_schema import validate_cell_cfg
from src.spectrum.yaml_loader import load_cell_yaml
from src.spectrum.registry import METRIC_ORIENTATION, metric_orientation, validate_cfg


def _aggregate_spectrum_seed_arts(seed_arts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-seed CPCV artifacts into a headline-compatible aggregate (B-SEED)."""
    if not seed_arts:
        raise ValueError("seed_arts must be non-empty")
    if len(seed_arts) == 1:
        out = dict(seed_arts[0])
        out["n_seeds"] = 1
        out["seed_results"] = [{"seed": seed_arts[0].get("seed", 0), **{
            k: seed_arts[0].get(k) for k in ("sharpe", "sharpe_mean", "mean_pl")
            if k in seed_arts[0]
        }}]
        if "sharpe_mean" not in out and "sharpe" in out:
            out["sharpe_mean"] = out["sharpe"]
        return out

    sharpes: list[float] = []
    for a in seed_arts:
        s = a.get("sharpe_mean", a.get("sharpe"))
        if s is not None and np.isfinite(float(s)):
            sharpes.append(float(s))
    out = dict(seed_arts[0])
    out["seed_results"] = seed_arts
    out["n_seeds"] = len(seed_arts)
    if sharpes:
        out["sharpe_mean"] = float(np.mean(sharpes))
        out["sharpe_std"] = float(np.std(sharpes, ddof=0))
        out["sharpe"] = out["sharpe_mean"]
    return out


def _claim_metric_and_orientation(cfg: dict, arm: str) -> tuple[str, str]:
    claim = str(cfg.get("claim_metric") or cfg.get("claim_label_stem") or "")
    if not claim:
        claim = "sharpe_mean"
    orient = str(cfg.get("metric_orientation") or metric_orientation(claim))
    if orient not in ("higher_better", "lower_better"):
        orient = METRIC_ORIENTATION.get(claim.lower(), "higher_better")
    return claim, orient


def _collapse_from_runner(runner_art: dict[str, Any] | None, *, dry_run: bool) -> dict:
    """Prefer real turnover series; never invent nanmean([]) = 0.0."""
    if dry_run or runner_art is None:
        return collapse_guard([0.05, 0.08, 0.04], action_l1=[0.5, 0.6, 0.4])

    turnovers = runner_art.get("turnovers")
    action_l1 = runner_art.get("action_l1")
    turnover_source = "turnovers"
    if turnovers is None:
        paths = runner_art.get("paths") or {}
        p0 = paths.get("0") or paths.get(0) or {}
        if isinstance(p0, dict) and p0.get("turnover") is not None:
            turnovers = p0.get("turnover")
            turnover_source = "paths.0.turnover"
        else:
            pol = runner_art.get("policy_diagnostics") or {}
            tm = pol.get("turnover_mean")
            if tm is None:
                tm = runner_art.get("turnover_mean")
            if tm is not None and float(tm) == float(tm):
                turnovers = [float(tm)]
                turnover_source = "turnover_mean"
            else:
                turnovers = None
                turnover_source = "missing"
    if turnovers is None:
        out = collapse_guard([float("nan")], action_l1=action_l1)
        out["turnover_source"] = turnover_source
        out["mean_turnover"] = float("nan")
        return out
    out = collapse_guard(turnovers, action_l1=action_l1)
    out["turnover_source"] = turnover_source
    return out


def _transfer_from_runner(
    *,
    cfg: dict,
    resolved: dict,
    arm: str,
    runner_art: dict[str, Any] | None,
    dry_run: bool,
) -> dict:
    claim, orient = _claim_metric_and_orientation(cfg, arm)
    train_world = resolved["train_world"]
    real_ref = train_world == "historical" or bool(
        (runner_art or {}).get("real_reference_arm_present")
    )
    if dry_run or runner_art is None:
        return build_transfer_report(
            train_metric=float("nan"),
            eval_metric=float("nan"),
            train_world=train_world,
            eval_world="optionmetrics",
            real_reference_arm_present=real_ref,
            claim_metric=claim,
            metric_orientation=orient,
        )

    path_sum = runner_art.get("path_summary") or {}
    pol = runner_art.get("policy") or {}
    train_m = runner_art.get("train_metric")
    eval_m = runner_art.get("eval_metric")
    if eval_m is None:
        eval_m = path_sum.get("sharpe_mean")
    if eval_m is None:
        eval_m = pol.get(claim) or pol.get("cao_y") or pol.get("mean_pl")
    # Never echo OOS into train_metric — use train_fold_metric or NaN.
    if train_m is None:
        train_m = runner_art.get("train_fold_metric")
    real_ref_metric = None
    if train_world != "historical" and eval_m is not None:
        real_ref_metric = float(eval_m)
    return build_transfer_report(
        train_metric=float(train_m) if train_m is not None else float("nan"),
        eval_metric=float(eval_m) if eval_m is not None else float("nan"),
        train_world=train_world,
        eval_world="optionmetrics",
        real_reference_arm_present=real_ref,
        real_reference_metric=real_ref_metric,
        claim_metric=claim,
        metric_orientation=orient,
    )


def _resolve_arm(cfg: dict) -> str:
    raw = cfg.get("arm") or cfg.get("portfolio_arm") or cfg.get("spectrum_arm")
    if isinstance(raw, dict):
        raw = raw.get("id") or raw.get("name")
    arm = str(raw or "opt").lower().strip()
    if arm in ("hedge", "hedge_mdp", "deep_hedge"):
        # Tier-1 hedge-MDP deleted; fall back to allocator opt arm.
        return "opt"
    if arm in ("opt", "eq", "mix"):
        return arm
    return "opt"


def _toy_research_panel(n_days: int = 64, k: int = 8, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n_days, k))
    factors = rng.normal(0.0, 0.005, size=(n_days, 4))
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    return dates, rets, factors


def _hoist_runner_weights(runner_art: dict[str, Any] | None) -> dict[str, Any]:
    """Pull OOS weight path + diagnostics out of nested runner artifacts.

    Research CPCV stores path-0 holdings under ``paths["0"]["weights"]``.
    Without this hoist, behaviour export sees no weights and writes all-NaN
    measures (archetype narrative starves). HAPPO smoke has no weight path;
    callers must stamp ``behaviour_export: unavailable`` in that case.
    """
    out: dict[str, Any] = {}
    if not isinstance(runner_art, dict):
        return out
    weights = runner_art.get("weights") or runner_art.get("oos_weights")
    if weights is None:
        paths = runner_art.get("paths") or {}
        path0 = paths.get("0") or paths.get(0) or {}
        if isinstance(path0, dict):
            weights = path0.get("weights")
            if runner_art.get("turnovers") is None and path0.get("turnover") is not None:
                out["turnovers"] = path0.get("turnover")
    if weights is not None:
        try:
            w_arr = np.asarray(weights, dtype=np.float64)
            if w_arr.size > 0:
                # list-of-lists so cell JSON stays reloadable
                out["weights"] = w_arr.tolist()
                out["weights_by_seed_available"] = bool(
                    runner_art.get("n_seeds", 1) and int(runner_art.get("n_seeds") or 1) > 1
                )
        except (TypeError, ValueError):
            pass
    if "turnovers" not in out and runner_art.get("turnovers") is not None:
        out["turnovers"] = runner_art.get("turnovers")
    if runner_art.get("dates") is not None:
        out["dates"] = runner_art.get("dates")
    elif isinstance(path0 := ((runner_art.get("paths") or {}).get("0") or {}), dict):
        if path0.get("dates"):
            out["dates"] = path0.get("dates")
    if runner_art.get("panel_returns") is not None:
        out["panel_returns"] = runner_art.get("panel_returns")
    if runner_art.get("policy_sensitivities") is not None:
        out["policy_sensitivities"] = runner_art.get("policy_sensitivities")
    if isinstance(runner_art.get("coordination_proxies"), dict):
        out.setdefault("coordination_proxies", runner_art["coordination_proxies"])
    diag = runner_art.get("policy_diagnostics") or runner_art.get("training_diagnostics")
    if diag is None and isinstance(runner_art.get("happo_trainer_stats"), dict):
        diag = {"happo_trainer_stats": runner_art["happo_trainer_stats"]}
    if isinstance(diag, dict) and diag:
        out["training_diagnostics"] = diag
    return out


def resolve_spectrum_budget(cfg: dict) -> dict[str, Any]:
    """Shared train budget for HAPPO/PPO parity (Part D.4).

    When ``claim_tier == dispatch_only``, HAPPO stays smoke-scale and is
    excluded from OFAT comparison tables. Smoke caps are applied *inside*
    this budget so cell artifacts cannot claim full-scale episode counts.

    Narrative / ``happo_full_budget`` cells opt out of the smoke cap so the
    multi-agent spine can run the configured episode budget (Phase 2 lock).
    """
    claim_tier = str(cfg.get("claim_tier") or "research").strip().lower()
    protocol_tier = str(cfg.get("protocol_tier") or "").strip().lower()
    algo = str(cfg.get("algo") or cfg.get("policy_algo") or "").lower().strip()
    full_budget = bool(cfg.get("happo_full_budget", False)) or protocol_tier == "narrative"
    dispatch_only = (
        not full_budget
        and (
            claim_tier == "dispatch_only"
            or bool(cfg.get("happo_dispatch_only", False))
            or algo == "happo"
        )
    )
    if dispatch_only:
        claim_tier = "dispatch_only"
    elif full_budget and claim_tier == "dispatch_only":
        claim_tier = "research"
    seeds_raw = cfg.get("seeds")
    if isinstance(seeds_raw, (list, tuple)) and seeds_raw:
        seeds = [int(s) for s in seeds_raw]
    else:
        seeds = [int(cfg.get("seed", 0) or 0)]
    n_episodes = int(
        cfg.get("train_episodes")
        or cfg.get("train_epochs")
        or cfg.get("spectrum_happo_episodes")
        or 1
    )
    horizon = int(
        cfg.get("horizon")
        or cfg.get("train_horizon")
        or cfg.get("n_steps")
        or cfg.get("spectrum_happo_horizon")
        or 0
    )
    train_env_steps = int(cfg.get("train_env_steps", 0) or 0)
    # Honesty stamp: when YAML declares train_env_steps, reflect that in the
    # budget instead of silently advertising n_episodes=1 (trainer expands
    # episodes from steps via research_alpha_train; artifacts must match).
    if train_env_steps > 0 and not (
        cfg.get("train_episodes") or cfg.get("spectrum_happo_episodes")
    ):
        # Conservative lower-bound episode count assuming ~252 trading days.
        # Exact fold length is known only after panel load; this stamp is
        # metadata honesty, not a second training schedule.
        n_episodes = max(n_episodes, max(1, int(train_env_steps // 252)))
    if dispatch_only:
        # Explicit smoke scale (never silent full-scale promotion).
        n_episodes = min(max(1, n_episodes), 2)
        if horizon <= 0:
            horizon = 6
        else:
            horizon = min(horizon, 6)
        train_env_steps = min(train_env_steps, 500) if train_env_steps > 0 else 0
    return {
        "n_episodes": max(1, n_episodes),
        "horizon": max(0, horizon),
        "train_env_steps": max(0, train_env_steps),
        "seeds": seeds,
        "cpcv_n_splits": int(cfg.get("cpcv_n_splits", 8) or 8),
        "cpcv_n_test_groups": int(cfg.get("cpcv_n_test_groups", 3) or 3),
        "dispatch_only": bool(dispatch_only),
        "claim_tier": claim_tier,
    }


def discover_spectrum_configs(
    config_dir: Path,
    *,
    config_glob: str = "**/*.yaml",
) -> list[Path]:
    """D.7: recursive YAML discovery (eq_ofat_priority / remaining included).

    Protocol ``fullgrid/`` is excluded unless ``config_dir`` itself is the
    fullgrid directory (run with ``--config-dir config/spectrum/fullgrid``).
    """
    pattern = str(config_glob or "**/*.yaml")
    # Path.rglob does not take **/ prefixes the same way; use glob for both.
    if pattern.startswith("/"):
        raise ValueError("config-glob must be relative")
    root = config_dir.resolve()
    paths = sorted({p.resolve() for p in config_dir.glob(pattern) if p.suffix in (".yaml", ".yml")})
    if not paths and pattern == "**/*.yaml":
        paths = sorted({p.resolve() for p in config_dir.rglob("*.yaml")})
    if root.name != "fullgrid":
        paths = [p for p in paths if "fullgrid" not in p.relative_to(root).parts]
    return paths


def _try_om_research_panel(cfg: dict, k: int):
    try:
        from scripts.run_research_alpha_cpcv import _try_load_om_panel
    except Exception:
        return None
    try:
        return _try_load_om_panel(cfg, k)
    except ValueError:
        raise
    except Exception:
        return None


def _apply_spectrum_universe_arm(
    cfg: dict,
    *,
    dates,
    rets,
) -> None:
    """Apply ``universe_arm`` foil masks on the spectrum path.

    ``dyn_hrp`` / absent: no mask (all slots tradable).
    ``dyn_crucible``: deterministic rotating slot dropout foil so Sweep I cells
    differ from the base universe without requiring the full eq_alloc CRUCIBLE
    CRSP pipeline (disclosed foil; see SPECTRUM_CHERRYPICK.md Sweep I).
    """
    ua = str(cfg.get("universe_arm") or "dyn_hrp").strip().lower()
    if ua in ("", "dyn_hrp", "none", "off"):
        cfg["_universe_arm_applied"] = ua or "dyn_hrp"
        return
    if ua != "dyn_crucible":
        raise ValueError(
            f"unsupported universe_arm={ua!r} on spectrum path; "
            "accepted={'dyn_hrp','dyn_crucible'}"
        )
    import numpy as np

    arr = np.asarray(rets, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("dyn_crucible foil requires 2-D returns")
    T, K = arr.shape
    mask = np.ones((T, K), dtype=bool)
    for t in range(T):
        q = t // 63
        for k in range(K):
            if (k + q) % 5 == 0:
                mask[t, k] = False
    # Keep at least one slot active every day.
    for t in range(T):
        if not mask[t].any():
            mask[t, t % K] = True
    cfg["_slot_valid_mask"] = mask
    cfg["_universe_arm_applied"] = "dyn_crucible_spectrum_foil"


def _run_research_arm(
    cfg: dict,
    arm: str,
    *,
    allow_toy_panel: bool = False,
    no_dry_run: bool = False,
    cell_out_dir: Path | str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from src.eval.cpcv import CPCVConfig
        from src.eval.research_alpha_cpcv import (
            dry_run_research_alpha_cpcv,
            run_research_alpha_cpcv,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"research_import_failed: {exc}"

    cfg_local = dict(cfg)
    cfg_local.setdefault("claim_tier", "research")
    if arm in ("eq", "mix", "opt"):
        cfg_local.setdefault("portfolio_arm", arm)

    cpcv_out_dir: Path | None = None
    resume_enabled = False
    if cell_out_dir is not None:
        cpcv_out_dir, _ckpt_dir, _run_hash = _prepare_spectrum_resume_dirs(
            cfg_local, cell_out_dir
        )
        resume_enabled = True

    # D.4: HAPPO shares budget resolution with PPO; multi-agent trainer
    # lives in _run_happo_arm, not a separate run_cell branch.
    resolved = validate_cfg(cfg_local)
    budget = resolve_spectrum_budget(cfg_local)
    if resolved.get("algo") == "happo":
        cfg_local["claim_tier"] = budget["claim_tier"]
        cfg_local["spectrum_happo_episodes"] = int(budget["n_episodes"])
        if int(budget["horizon"]) > 0:
            cfg_local["spectrum_happo_horizon"] = int(budget["horizon"])
        if budget["dispatch_only"]:
            art, err = _run_happo_arm(cfg_local, arm)
        else:
            from src.eval.research_happo_cpcv import run_happo_cpcv

            art, err = run_happo_cpcv(
                cfg_local,
                arm,
                budget=budget,
                allow_toy_panel=allow_toy_panel,
                no_dry_run=no_dry_run,
                out_dir=cpcv_out_dir,
                resume=resume_enabled,
            )
        if art is not None:
            art["spectrum_budget"] = budget
            art["claim_tier"] = budget["claim_tier"]
            art["n_episodes"] = int(budget["n_episodes"])
            art["horizon"] = int(budget["horizon"])
            if budget["dispatch_only"]:
                # Dispatch proof only: never comparison / promotion evidence.
                art["real_reference_arm_present"] = False
                art["spectrum_run_note"] = (
                    "happo_dispatch_only_smoke; excluded from OFAT comparison; "
                    "chapter-7 compute reason in config/preregistration/HAPPO_TIMING.md"
                )
        return art, err

    cfg_local.setdefault("headline_fill", "pct75")
    k = int(cfg_local.get("n_assets", 8) or 8)
    from src.eval.equity_substrate import (
        attach_equity_obs_substrate,
        load_lake_dyn_hrp_panel,
        resolve_substrate_secids,
        stamp_equity_obs_defaults,
    )

    # Parity: cube + geometry_lite for all historical arms (RC6: opt/mix too).
    if str(cfg_local.get("train_world") or "historical").lower() == "historical":
        stamp_equity_obs_defaults(cfg_local)

    print(f"phase=panel_load start k={k}", flush=True)
    panel_source = "optionmetrics"
    panel_meta: dict[str, Any] = {}
    # Equity historical cells: lake sp500_sec + real dyn_hrp (H0 substrate).
    # Opt/mix keep the Arctic OM path (deferred arms).
    if (
        str(arm).lower() == "eq"
        and str(cfg_local.get("train_world") or "historical").lower() == "historical"
        and not bool(cfg_local.get("force_om_panel", False))
    ):
        try:
            dates, rets, factors, panel_meta = load_lake_dyn_hrp_panel(
                cfg_local, k=k
            )
            panel_source = str(panel_meta.get("panel_source") or "lake_sp500_sec")
            panel = (dates, rets, factors)
        except Exception as exc:  # noqa: BLE001
            if no_dry_run and not allow_toy_panel:
                raise RuntimeError(
                    f"lake dyn_hrp panel unavailable for eq arm: {exc}"
                ) from exc
            panel = None
            cfg_local.setdefault("_panel_load_errors", []).append(str(exc)[:300])
    else:
        panel = _try_om_research_panel(cfg_local, k)
        panel_source = "optionmetrics"

    if panel is None:
        # D.5: refuse silent toy fallback under --no-dry-run unless flagged.
        if no_dry_run and not allow_toy_panel:
            raise RuntimeError(
                "research panel unavailable; refusing toy panel "
                "under --no-dry-run (pass --allow-toy-panel to override; "
                "artifact will stamp toy_panel=true)"
            )
        panel = _toy_research_panel(n_days=64, k=k, seed=0)
        panel_source = "toy"
    dates, rets, factors = panel
    print(
        f"phase=panel_load done source={panel_source} "
        f"n_days={len(dates)} n_assets={rets.shape[1] if hasattr(rets, 'shape') else k}",
        flush=True,
    )
    # Always apply universe_arm foils (including lake_sp500_sec). Skipping
    # lake panels previously left Sweep I dyn_crucible identical to dyn_hrp.
    _apply_spectrum_universe_arm(cfg_local, dates=dates, rets=rets)

    # Observation substrate: dollar_volume + geometry_lite iv_surface.
    # Feature-net extras stay OFF by default (G0); opt in via
    # use_feature_net_extras=true (FEATNET wave).
    if bool(cfg_local.get("use_equity_feature_cube", False)):
        try:
            attach_equity_obs_substrate(
                cfg_local,
                dates=dates,
                rets=rets,
                secids=resolve_substrate_secids(
                    cfg_local, panel_source=panel_source, k=int(rets.shape[1])
                ),
                slots_rows=cfg_local.get("_slots_rows"),
                dollar_volume=panel_meta.get("dollar_volume"),
                fail_closed_surface=bool(cfg_local.get("use_surface_signals", False))
                and panel_source != "toy",
            )
            if panel_meta:
                cfg_local["_substrate_meta"] = {
                    k_: panel_meta[k_]
                    for k_ in (
                        "panel_source",
                        "universe_arm",
                        "fingerprint_size",
                        "universe_fingerprint",
                        "universe_fingerprint_kind",
                        "k",
                        "n_days",
                    )
                    if k_ in panel_meta
                }
        except Exception as exc:  # noqa: BLE001
            if panel_source != "toy" and bool(
                cfg_local.get("use_equity_feature_cube", False)
            ):
                raise
            cfg_local.setdefault("_feature_net_errors", [])
            cfg_local["_feature_net_errors"].append(str(exc)[:300])

    if bool(cfg_local.get("use_feature_net_extras", False)):
        from src.eval.equity_substrate import (
            apply_feature_net_extras_if_enabled,
            stamp_lake_universe_secids_for_featnet,
        )

        try:
            # Physics/OM train panels do not stamp lake secids; FEATNET still
            # needs them to attach feature-net extras (fail-closed otherwise).
            stamp_lake_universe_secids_for_featnet(
                cfg_local, k=int(rets.shape[1])
            )
            apply_feature_net_extras_if_enabled(
                cfg_local,
                dates=dates,
                secids=resolve_substrate_secids(
                    cfg_local, panel_source=panel_source, k=int(rets.shape[1])
                ),
                panel_source=panel_source,
                slots_rows=cfg_local.get("_slots_rows"),
            )
        except Exception as exc:  # noqa: BLE001
            if panel_source != "toy":
                raise
            cfg_local.setdefault("_feature_net_errors", [])
            cfg_local["_feature_net_errors"].append(str(exc)[:300])

    cpcv = CPCVConfig(
        n_splits=int(budget["cpcv_n_splits"]),
        n_test_groups=int(budget["cpcv_n_test_groups"]),
        purge_days=int(cfg_local.get("cpcv_purge_days", 21) or 21),
        embargo_days=int(cfg_local.get("cpcv_embargo_days", 21) or 21),
    )
    # Toy panels cannot support heavy CPCV geometry; shrink for smoke.
    if panel_source == "toy":
        cpcv = CPCVConfig(
            n_splits=min(cpcv.n_splits, 3),
            n_test_groups=1,
            purge_days=0,
            embargo_days=0,
        )

    try:
        seed_arts: list[dict[str, Any]] = []
        seed_errs: list[str] = []
        for seed in list(budget["seeds"]):
            print(
                f"phase=cpcv_train start seed={seed} "
                f"n_splits={cpcv.n_splits} n_test_groups={cpcv.n_test_groups}",
                flush=True,
            )
            try:
                art_s = run_research_alpha_cpcv(
                    dates,
                    rets,
                    factors,
                    cfg_local,
                    cpcv=cpcv,
                    seed=int(seed),
                    panel_source=panel_source,
                    out_dir=cpcv_out_dir,
                    resume=resume_enabled,
                )
                art_s["seed"] = int(seed)
                seed_arts.append(art_s)
                print(f"phase=cpcv_train done seed={seed}", flush=True)
            except Exception as exc_s:  # noqa: BLE001
                seed_errs.append(f"seed={seed}: {exc_s}")
                print(f"phase=cpcv_train failed seed={seed} err={exc_s}", flush=True)
        if not seed_arts:
            raise RuntimeError("; ".join(seed_errs) or "no_seed_artifacts")
        art = _aggregate_spectrum_seed_arts(seed_arts)
        art["spectrum_budget"] = budget
        art.setdefault("claim_tier", budget.get("claim_tier") or cfg_local.get("claim_tier"))
        art.setdefault("n_episodes", int(budget["n_episodes"]))
        art.setdefault("horizon", int(budget["horizon"]))
        if panel_source == "toy":
            art["spectrum_run_note"] = "om_unavailable_toy_panel"
            art["toy_panel"] = True
        if seed_errs:
            art["spectrum_seed_errors"] = seed_errs
        # Propagate feature-net degrade so run_cell can hoist into the artifact.
        if cfg_local.get("_feature_net_errors"):
            art["_feature_net_errors"] = list(cfg_local["_feature_net_errors"])
        if cfg_local.get("_substrate_meta"):
            art["substrate_meta"] = dict(cfg_local["_substrate_meta"])
        return art, None
    except Exception as exc:  # noqa: BLE001
        try:
            art = dry_run_research_alpha_cpcv(cfg_local)
            return art, f"research_cpcv_failed_dry_schema: {exc}"
        except Exception as exc2:  # noqa: BLE001
            return None, f"research_failed: {exc}; dry_also_failed: {exc2}"


def _lagged_w_prev_actions(actions: list[Any]) -> Any:
    """Build (T, K) lagged executed weights for HAPPO TrainBatch."""
    import torch

    if not actions:
        raise ValueError("actions must be non-empty")
    k = int(actions[0].shape[-1])
    device = actions[0].device
    dtype = actions[0].dtype
    rows = [torch.zeros(1, k, device=device, dtype=dtype)]
    for action in actions[:-1]:
        rows.append(action.detach())
    return torch.cat(rows, dim=0)


def _run_happo_arm(cfg: dict, arm: str) -> tuple[dict[str, Any] | None, str | None]:
    """C4: algo='happo' dispatches to the real multi-agent HAPPO trainer.

    Reuses the same minimal-e2e primitives as the local CPCV smoke helpers
    (``get_surface_tensor`` / ``build_feature_extractor`` /
    ``build_happo_engine`` / ``CMDPEnv`` / ``HAPPOTrainer``). Screening /
    dispatch-only cells stay smoke-scale; narrative / ``happo_full_budget``
    cells route to :func:`run_happo_cpcv` instead.
    """
    try:
        import torch

        from src.env.cmdp_env import CMDPEnv
        from src.plugins.registry import build_feature_extractor, build_happo_engine
        from src.plugins.resolve import resolve_plugins
        from src.policy.trainer import HAPPOTrainer, TrainBatch
        from src.simulator import get_surface_tensor
    except Exception as exc:  # noqa: BLE001
        return None, f"happo_import_failed: {exc}"

    cfg_local = dict(cfg)
    cfg_local.setdefault("n_assets", 4)
    cfg_local.setdefault("n_paths", 4)
    cfg_local.setdefault("n_steps", 24)
    cfg_local.setdefault("n_strikes", 11)
    cfg_local.setdefault("n_maturities", 3)
    cfg_local.setdefault("hurst_exponent", 0.1)
    cfg_local.setdefault("d_model", 32)
    cfg_local.setdefault("d_state", 8)
    cfg_local.setdefault("macro_dim", 8)
    cfg_local.setdefault("turnover_limit", 0.25)
    cfg_local.setdefault("use_gpu", False)
    cfg_local.setdefault("train_world", "rbergomi")
    horizon = int(cfg_local.get("spectrum_happo_horizon", 6) or 6)
    n_episodes = int(cfg_local.get("spectrum_happo_episodes", 2) or 2)

    try:
        torch.manual_seed(int(cfg_local.get("seed", 0) or 0))
        K = int(cfg_local["n_assets"])
        train_world = str(cfg_local.get("train_world") or "rbergomi").lower().strip()
        # C6/C8 + Wave 5: tape-primary worlds cannot feed get_surface_tensor
        # (strike/maturity smile). Use the equity_panel_to_cmdp_tensors bridge
        # for eq always, and for opt/mix when train_world is historical.
        use_panel_bridge = arm == "eq" or train_world in {
            "historical",
            "optionmetrics",
        }
        panel_source = train_world
        if use_panel_bridge:
            from src.env.equity_cmdp_bridge import equity_panel_to_cmdp_tensors

            n_days = int(cfg_local.get("n_steps", 24) or 24) + 1
            panel = None
            # Eq historical: lake sp500_sec + dyn_hrp (H0 / spectrum parity).
            if (
                str(arm).lower() == "eq"
                and train_world == "historical"
                and not bool(cfg_local.get("force_om_panel", False))
            ):
                try:
                    from src.eval.equity_substrate import (
                        load_lake_dyn_hrp_panel,
                        stamp_equity_obs_defaults,
                    )

                    stamp_equity_obs_defaults(cfg_local)
                    _d, eq_full, _f, meta = load_lake_dyn_hrp_panel(cfg_local, k=K)
                    eq_rets = np.asarray(eq_full, dtype=np.float64)[:n_days, :K]
                    if eq_rets.shape[0] < n_days:
                        pad = np.zeros(
                            (n_days - eq_rets.shape[0], K), dtype=eq_rets.dtype
                        )
                        eq_rets = np.concatenate([eq_rets, pad], axis=0)
                    panel_source = str(meta.get("panel_source") or "lake_sp500_sec")
                    panel = (_d, eq_rets, _f)
                except Exception:  # noqa: BLE001
                    panel = None
            if panel is None:
                panel = _try_om_research_panel(cfg_local, K)
                if panel is None:
                    _, eq_rets, _ = _toy_research_panel(
                        n_days=n_days, k=K, seed=int(cfg_local.get("seed", 0) or 0)
                    )
                    panel_source = "toy"
                else:
                    _dates, eq_rets, _factors = panel
                    eq_rets = np.asarray(eq_rets, dtype=np.float64)[:n_days, :K]
                    if eq_rets.shape[0] < n_days:
                        pad = np.zeros(
                            (n_days - eq_rets.shape[0], K), dtype=eq_rets.dtype
                        )
                        eq_rets = np.concatenate([eq_rets, pad], axis=0)
                    panel_source = "optionmetrics"
            else:
                _dates, eq_rets, _factors = panel
            bridge = equity_panel_to_cmdp_tensors(eq_rets)
            surfaces = bridge["surfaces"]
        else:
            surfaces = get_surface_tensor(cfg_local)
        d_model = int(cfg_local["d_model"])
        macro_dim = int(cfg_local["macro_dim"])
        plugins = resolve_plugins(cfg_local)
        fe = build_feature_extractor(
            K, d_model, d_state=cfg_local.get("d_state", 8), cfg=cfg_local, plugins=plugins
        )
        policy = build_happo_engine(K, d_model, macro_dim, cfg=cfg_local, plugins=plugins)
        exec_spread = float(cfg_local.get("execution_spread_bps", 0.0) or 0.0)
        exec_impact = float(cfg_local.get("execution_impact_coef", 0.0) or 0.0)
        if bool(cfg_local.get("cost_in_decision", False)):
            if exec_spread <= 0.0 and exec_impact <= 0.0:
                return None, "cost_in_decision_requires_nonzero_friction"
        macro_series = None
        try:
            from src.data.paths import CANONICAL_LAKE
            from src.spectrum.happo_macro_inject import maybe_load_happo_macro

            usb_root = cfg_local.get("usb_lake_root") or CANONICAL_LAKE
            t_surf = int(surfaces.shape[0]) if hasattr(surfaces, "shape") else None
            macro_series = maybe_load_happo_macro(
                cfg_local, usb_root, n_rows=t_surf
            )
        except Exception:
            macro_series = None
        env = CMDPEnv(
            surfaces=surfaces,
            feature_extractor=fe,
            policy=policy,
            d_model=d_model,
            macro_dim=macro_dim,
            use_gpu=False,
            execution_spread_bps=exec_spread,
            execution_impact_coef=exec_impact,
            macro_series=macro_series,
        )
        trainer = HAPPOTrainer(policy, use_compile=False)

        ep_rewards: list[float] = []
        turnovers: list[float] = []
        action_l1: list[float] = []
        last_stats: dict = {}
        for ep in range(max(1, n_episodes)):
            path = ep % max(1, int(cfg_local["n_paths"]))
            obs = env.reset(path=path)
            w_prev = torch.zeros(1, K)
            enriched, macro, deltas, actions, log_probs, values, rewards, dones, raw_actions = (
                [] for _ in range(9)
            )
            for _ in range(max(1, horizon)):
                vol_scale = float(obs.info.get("atm_vol", 0.2))
                w, lp, v, w_raw = policy.act_stochastic(
                    obs.enriched, obs.macro, w_prev, obs.deltas, vol_scale=vol_scale
                )
                nxt = env.step(w.detach())
                turnovers.append(float((w.detach() - w_prev).abs().sum()))
                action_l1.append(float(w.detach().abs().sum()))
                enriched.append(obs.enriched.detach())
                macro.append(obs.macro.detach())
                deltas.append(obs.deltas.detach())
                actions.append(w.detach())
                log_probs.append(lp.detach())
                values.append(v.detach().reshape(-1))
                rewards.append(nxt.reward.detach().reshape(-1))
                dones.append(torch.tensor([float(nxt.done)]))
                raw_actions.append(w_raw.detach())
                ep_rewards.append(float(nxt.reward.item()))
                w_prev = w.detach()
                obs = nxt
                if nxt.done:
                    break
            if not rewards:
                continue
            batch = TrainBatch(
                enriched=torch.cat(enriched, dim=0),
                macro=torch.cat(macro, dim=0),
                w_prev=_lagged_w_prev_actions(actions),
                deltas=torch.cat(deltas, dim=0),
                actions=torch.cat(actions, dim=0),
                log_probs=torch.cat(log_probs, dim=0),
                values=torch.cat(values, dim=0),
                rewards=torch.cat(rewards, dim=0),
                dones=torch.cat(dones, dim=0),
                raw_actions=torch.cat(raw_actions, dim=0),
            )
            last_stats = trainer.update(batch, epochs=1)

        if not ep_rewards:
            return None, "happo_run_produced_no_transitions"
        arr = np.asarray(ep_rewards, dtype=np.float64)
        sharpe = float(arr.mean() / (arr.std(ddof=0) + 1e-12) * np.sqrt(252.0))
        claim_tier = str(cfg_local.get("claim_tier") or "research").strip().lower()
        dispatch_only = claim_tier == "dispatch_only" or bool(
            cfg_local.get("happo_dispatch_only", False)
        )
        art: dict[str, Any] = {
            "path_summary": {"sharpe_mean": sharpe},
            "policy": {"sharpe_mean": sharpe, "mean_pl": float(arr.mean())},
            "train_metric": sharpe,
            "eval_metric": sharpe,
            # Wave 5: smoke dispatch must not look like a real reference arm.
            "real_reference_arm_present": not dispatch_only,
            "turnovers": turnovers,
            "action_l1": action_l1,
            "panel_source": panel_source,
            "n_paths": int(cfg_local["n_paths"]),
            "n_episodes": int(n_episodes),
            "horizon": int(horizon),
            "claim_tier": "dispatch_only" if dispatch_only else claim_tier,
            "claim_metric": "sharpe_mean",
            "happo_trainer_stats": {
                k: v for k, v in last_stats.items() if isinstance(v, (int, float))
            },
        }
        if panel_source == "toy":
            art["toy_panel"] = True
        return art, None
    except Exception as exc:  # noqa: BLE001
        return None, f"happo_run_failed: {exc}"


def _gates_from_runner(
    runner_art: dict[str, Any] | None,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """D.6: compute gate1/2/3 when real runner results exist."""
    from src.eval.spectrum_gates import compute_gate1, compute_gate2, compute_gate3

    if dry_run or runner_art is None:
        return {
            "gate1": {"pass": None, "skipped": True, "reason": "dry_run_or_missing"},
            "gate2": {"pass": None, "skipped": True, "reason": "dry_run_or_missing"},
            "gate3": {"pass": None, "skipped": True, "reason": "dry_run_or_missing"},
        }
    gates: dict[str, Any] = {}
    try:
        cost_ladder = runner_art.get("cost_ladder") or {}
        if cost_ladder:
            gates["gate1"] = compute_gate1(cost_ladder)
        else:
            gates["gate1"] = {
                "pass": None,
                "skipped": True,
                "reason": "no_cost_ladder",
            }
    except Exception as exc:  # noqa: BLE001
        gates["gate1"] = {"pass": False, "error": str(exc)[:300]}
    try:
        rets = (
            runner_art.get("policy_returns")
            or runner_art.get("oos_returns")
        )
        fac = runner_art.get("factors") or runner_art.get("oos_factors")
        # Belt-and-suspenders: if only path-0 pnl is nested under paths, lift it.
        if rets is None:
            path0 = (runner_art.get("paths") or {}).get("0") or {}
            if path0.get("pnl") is not None:
                rets = path0["pnl"]
        if rets is not None and fac is not None:
            factor_names = runner_art.get("factor_names")
            gates["gate2"] = compute_gate2(
                np.asarray(rets, dtype=np.float64),
                np.asarray(fac, dtype=np.float64),
                factor_names=list(factor_names) if factor_names else None,
            )
        else:
            gates["gate2"] = {
                "pass": None,
                "skipped": True,
                "reason": "no_policy_returns_or_factors",
            }
    except Exception as exc:  # noqa: BLE001
        gates["gate2"] = {"pass": False, "error": str(exc)[:300]}
    try:
        path_sum = runner_art.get("path_summary") or {}
        pol = runner_art.get("policy") or {}
        policy_sharpe = path_sum.get("sharpe_mean")
        if policy_sharpe is None:
            policy_sharpe = pol.get("sharpe_mean")
        baselines = runner_art.get("baselines") or runner_art.get("baseline_sharpes") or {}
        if isinstance(baselines, dict) and baselines and policy_sharpe is not None:
            # baselines may be nested {name: {sharpe: ...}}
            flat: dict[str, float] = {}
            for name, val in baselines.items():
                if isinstance(val, dict) and "sharpe" in val:
                    flat[str(name)] = float(val["sharpe"])
                elif isinstance(val, (int, float)):
                    flat[str(name)] = float(val)
            gates["gate3"] = compute_gate3(float(policy_sharpe), flat)
        else:
            gates["gate3"] = {
                "pass": None,
                "skipped": True,
                "reason": "no_baselines_or_policy_sharpe",
            }
    except Exception as exc:  # noqa: BLE001
        gates["gate3"] = {"pass": False, "error": str(exc)[:300]}
    return gates


def run_cell(
    cfg_path: Path,
    *,
    dry_run: bool = True,
    allow_toy_panel: bool = False,
    cell_out_dir: Path | str | None = None,
    strict: bool = False,
) -> dict:
    cfg = load_cell_yaml(cfg_path)
    resolved = validate_cfg(cfg)
    cell_id = str(cfg.get("spectrum_cell_id") or cfg_path.stem)
    arm = _resolve_arm(cfg)
    budget = resolve_spectrum_budget(cfg)

    runner_art: dict[str, Any] | None = None
    fallback_reason: str | None = None
    effective_dry = bool(dry_run)

    if not dry_run:
        # D.4: no algo==happo special case here; _run_research_arm routes HAPPO.
        runner_art, fallback_reason = _run_research_arm(
            cfg,
            arm,
            allow_toy_panel=allow_toy_panel,
            no_dry_run=True,
            cell_out_dir=cell_out_dir,
        )
        if runner_art is None:
            effective_dry = True
            fallback_reason = fallback_reason or "runner_returned_none"

    transfer = _transfer_from_runner(
        cfg=cfg,
        resolved=resolved,
        arm=arm,
        runner_art=runner_art,
        dry_run=effective_dry,
    )
    collapse = _collapse_from_runner(runner_art, dry_run=effective_dry)
    gates = _gates_from_runner(runner_art, dry_run=effective_dry)

    real_ref = bool(transfer.get("real_reference_arm_present"))
    # Dry-run (incl. forced fallback) never promotes.
    # Wave 5: dispatch_only / smoke HAPPO can never look full-scale-promotable.
    promotable = bool(
        (not effective_dry)
        and real_ref
        and collapse.get("ok")
        and not budget.get("dispatch_only")
        and str(budget.get("claim_tier") or "").lower() != "dispatch_only"
    )

    out: dict[str, Any] = {
        "spectrum_cell_id": cell_id,
        "spectrum_axis": str(cfg.get("spectrum_axis") or "none"),
        "arm": arm,
        "config_path": str(cfg_path),
        "resolved": resolved,
        "claim_tier": budget.get("claim_tier") or cfg.get("claim_tier") or "research",
        "n_episodes": int(budget.get("n_episodes") or 0),
        "horizon": int(budget.get("horizon") or 0),
        "spectrum_budget": budget,
        "cost_in_decision": bool(cfg.get("cost_in_decision", False)),
        "dry_run": bool(effective_dry),
        "transfer_report": transfer,
        "collapse_guard": collapse,
        "gate1": gates.get("gate1"),
        "gate2": gates.get("gate2"),
        "gate3": gates.get("gate3"),
        "promotable": promotable,
    }
    if runner_art is not None and runner_art.get("toy_panel"):
        out["toy_panel"] = True
    if runner_art is not None and runner_art.get("substrate_meta"):
        out["substrate_meta"] = runner_art["substrate_meta"]
    if runner_art is not None:
        out["runner_artifact"] = {
            k: runner_art[k]
            for k in (
                "path_summary",
                "policy",
                "baselines",
                "panel_source",
                "spectrum_run_note",
                "n_paths",
                "n_episodes",
                "horizon",
                "claim_metric",
                "toy_panel",
                "spectrum_budget",
                "claim_tier",
                "paths",
                "dates",
                "panel_returns",
                "policy_sensitivities",
                "coordination_proxies",
                "happo_trainer_stats",
            )
            if k in runner_art
        }
        # Per-regime Sharpe keys when the runner stamped them.
        for key in (
            "sharpe_calm",
            "sharpe_inflationary",
            "sharpe_crisis",
            "n_days_calm",
            "n_days_inflationary",
            "n_days_crisis",
            "per_regime_sharpe",
        ):
            if key in runner_art:
                out[key] = runner_art[key]
        # Hoist path-0 weights so behaviour export is not all-NaN.
        hoisted = _hoist_runner_weights(runner_art)
        out.update(hoisted)
        if "weights" not in out:
            # HAPPO smoke (and failed hoists) cannot feed archetype clustering.
            out["behaviour_export"] = "unavailable"
            out["behaviour_export_reason"] = "no_oos_weight_path"
        else:
            out["behaviour_export"] = "ready"
        if str(resolved.get("algo") or cfg.get("algo") or "").lower() == "happo":
            out["protocol_tier"] = str(
                cfg.get("protocol_tier") or budget.get("claim_tier") or "research"
            )
            out["happo_full_budget"] = bool(
                cfg.get("happo_full_budget")
                or str(cfg.get("protocol_tier") or "").lower() == "narrative"
            )
            out["dispatch_only"] = bool(budget.get("dispatch_only"))
    if fallback_reason:
        out["fallback_reason"] = fallback_reason
        out["dry_run"] = True
        out["promotable"] = False
    # Hoist feature-net / seed degrade signals into the top-level artifact.
    feat_errs = []
    seed_errs = []
    if runner_art is not None:
        feat_errs = list(
            runner_art.get("_feature_net_errors")
            or runner_art.get("feature_net_errors")
            or []
        )
        seed_errs = list(runner_art.get("spectrum_seed_errors") or [])
    if feat_errs:
        out["feature_net_errors"] = feat_errs
    if seed_errs:
        out["spectrum_seed_errors"] = seed_errs
    # Strict (AWS default): any degrade path cannot promote and is flagged.
    if strict and (
        fallback_reason
        or feat_errs
        or seed_errs
        or bool(out.get("dry_run"))
    ):
        out["promotable"] = False
        out["strict_degraded"] = True
        out["strict"] = True
    elif strict:
        out["strict"] = True
    out.setdefault("protocol_tier", str(cfg.get("protocol_tier") or budget.get("claim_tier") or "research"))
    out.setdefault("grid_kind", str(cfg.get("grid_kind") or cfg_path.parent.name))
    print(
        f"phase=artifact_write cell={cell_id} dry_run={out.get('dry_run')} "
        f"promotable={out.get('promotable')} strict_degraded={out.get('strict_degraded')}",
        flush=True,
    )
    if os.environ.get("MASCOTRL_COMPUTE_HOST") == "remote":
        try:
            from src.reporting.provenance_stamp import build_provenance_stamp

            stamp = build_provenance_stamp(
                container_digest=os.environ.get("MASCOTRL_CONTAINER_DIGEST"),
                compute_host="remote",
                universe_fingerprint=(
                    str(runner_art.get("universe_fingerprint") or "")
                    if runner_art
                    else None
                ) or None,
                universe_fingerprint_kind=(
                    str(runner_art.get("universe_fingerprint_kind") or "")
                    if runner_art
                    else None
                ) or None,
                crucible_fingerprint=(
                    str(runner_art.get("crucible_fingerprint") or "")
                    if runner_art
                    else None
                ) or None,
            )
            out.update(stamp)
        except Exception as exc:  # noqa: BLE001
            out["provenance_stamp_error"] = str(exc)[:300]
    return out


def _count_happo_cells(config_dir: Path, config_glob: str = "**/*.yaml") -> int:
    n = 0
    for path in discover_spectrum_configs(config_dir, config_glob=config_glob):
        try:
            cfg = load_cell_yaml(path)
        except Exception:
            continue
        if str(cfg.get("algo") or "").lower().strip() == "happo":
            n += 1
    return n


def _project_happo_hours(
    *,
    elapsed_s: float,
    n_happo_cells: int,
    n_seeds: int,
) -> float:
    """projected_hours = probe_elapsed_s * n_happo_cells * n_seeds / 3600."""
    return float(elapsed_s) * max(1, int(n_happo_cells)) * max(1, int(n_seeds)) / 3600.0


def _budget_hours_limit() -> float | None:
    raw = (os.environ.get("MASCOTRL_SPECTRUM_BUDGET_HOURS") or "").strip()
    if not raw:
        return None
    try:
        hours = float(raw)
    except ValueError as exc:
        raise SystemExit(
            f"MASCOTRL_SPECTRUM_BUDGET_HOURS={raw!r} is not a float"
        ) from exc
    if hours <= 0:
        raise SystemExit("MASCOTRL_SPECTRUM_BUDGET_HOURS must be > 0")
    return hours


def _peak_rss_mb() -> float:
    try:
        import resource

        # Linux ru_maxrss is kilobytes.
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def main() -> None:
    global _JSONL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "config" / "spectrum",
    )
    p.add_argument(
        "--config-glob",
        type=str,
        default="**/*.yaml",
        help="Relative glob under --config-dir (default: recursive **/*.yaml).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "spectrum",
    )
    # Default True for safety; --no-dry-run enables real dispatch.
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--no-dry-run", action="store_true")
    p.add_argument(
        "--allow-toy-panel",
        action="store_true",
        help="Permit toy Gaussian panel under --no-dry-run; stamps toy_panel=true.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail-closed: any feature_net_errors / spectrum_seed_errors / "
            "fallback_reason marks promotable=False and strict_degraded=True."
        ),
    )
    p.add_argument(
        "--refresh-behavior",
        action="store_true",
        help=(
            "Re-emit *_policy_behavior.json from stored artifacts without retraining; "
            "exit after refresh."
        ),
    )
    p.add_argument(
        "--timing-probe",
        type=str,
        default=None,
        help="Run exactly one cell (id or path), write timing JSON, exit.",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on discovered cells (after glob; for dry-run smoke).",
    )
    args = p.parse_args()
    if args.refresh_behavior:
        summary = refresh_behavior_exports(
            args.out_dir,
            config_dir=args.config_dir,
        )
        print(
            f"refresh_behavior: refreshed={len(summary['refreshed'])} "
            f"skipped={len(summary['skipped'])} errors={len(summary['errors'])}"
        )
        return
    dry = not bool(args.no_dry_run)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _JSONL = out_dir / "campaign.jsonl"
    _log_event("spectrum_start", dry_run=dry, out_dir=str(out_dir))
    budget_h = _budget_hours_limit()
    t0 = time.perf_counter()
    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    paths = discover_spectrum_configs(args.config_dir, config_glob=args.config_glob)
    if args.max_cells is not None:
        if args.max_cells < 0:
            raise SystemExit("--max-cells must be >= 0")
        paths = paths[: int(args.max_cells)]

    if args.timing_probe:
        probe = str(args.timing_probe)
        probe_path = Path(probe)
        if not probe_path.is_file():
            matches = [p for p in paths if p.stem == probe or p.name == probe]
            if not matches:
                raise SystemExit(f"--timing-probe cell not found: {probe!r}")
            probe_path = matches[0]
        rss0 = _peak_rss_mb()
        t_probe = time.perf_counter()
        art = run_cell(
            probe_path,
            dry_run=dry,
            allow_toy_panel=bool(args.allow_toy_panel),
            cell_out_dir=out_dir,
            strict=bool(args.strict),
        )
        elapsed_s = time.perf_counter() - t_probe
        cfg_probe = load_cell_yaml(probe_path)
        budget = art.get("spectrum_budget") or resolve_spectrum_budget(cfg_probe)
        n_seeds = len(budget.get("seeds") or [0])
        n_happo = _count_happo_cells(args.config_dir, config_glob=args.config_glob)
        projected_hours = _project_happo_hours(
            elapsed_s=elapsed_s,
            n_happo_cells=n_happo,
            n_seeds=n_seeds,
        )
        probe_payload = {
            "cell_id": art.get("spectrum_cell_id"),
            "config_path": str(probe_path),
            "elapsed_s": float(elapsed_s),
            "peak_rss_mb": float(max(_peak_rss_mb(), rss0)),
            "n_episodes": int(budget.get("n_episodes") or 0),
            "horizon": int(budget.get("horizon") or 0),
            "n_folds": int(budget.get("cpcv_n_splits") or 0),
            "n_seeds": int(n_seeds),
            "n_happo_cells": int(n_happo),
            "projected_hours": float(projected_hours),
            "claim_tier": art.get("claim_tier") or budget.get("claim_tier"),
            "dry_run": bool(art.get("dry_run")),
        }
        dest = out_dir / f"timing_probe_{art.get('spectrum_cell_id')}.json"
        # Also write under the plan's logs/artifacts/spectrum path convention.
        dest.write_text(
            json.dumps(probe_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {dest}")
        print(
            f"projected_hours={projected_hours:.4f} "
            f"(elapsed_s={elapsed_s:.3f} * n_happo={n_happo} * n_seeds={n_seeds} / 3600)"
        )
        return

    for path in paths:
        if budget_h is not None:
            elapsed_h = (time.perf_counter() - t0) / 3600.0
            if elapsed_h >= budget_h:
                skipped.append(
                    {
                        "path": str(path),
                        "spectrum_cell_id": path.stem,
                        "reason": "budget_exhausted",
                        "elapsed_h": elapsed_h,
                        "budget_h": budget_h,
                    }
                )
                continue
        cfg_pre = load_cell_yaml(path)
        validate_cell_cfg(cfg_pre, path=str(path))
        art = run_cell(
            path,
            dry_run=dry,
            allow_toy_panel=bool(args.allow_toy_panel),
            cell_out_dir=out_dir,
            strict=bool(args.strict),
        )
        try:
            rel = path.resolve().relative_to(args.config_dir.resolve())
        except ValueError:
            rel = Path(path.name)
        if len(rel.parts) > 1:
            art_name = "__".join(rel.with_suffix("").parts)
            art["artifact_id"] = art_name
            art["config_relpath"] = str(rel)
        else:
            art_name = str(art["spectrum_cell_id"])
        dest = out_dir / f"{art_name}.json"
        dest.write_text(json.dumps(art, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        # B-BEH: export policy_behavior for fullgrid / cherrypick when weights exist.
        grid_kind = str(art.get("grid_kind") or path.parent.name)
        export_grids = ("fullgrid", "cherrypick", "cherrypick_deskorg", "cherrypick_narrative")
        should_export = (
            grid_kind in export_grids
            or any(
                g in str(path)
                for g in (
                    "fullgrid",
                    "cherrypick",
                    "deskorg",
                )
            )
        )
        if should_export and art.get("behaviour_export") != "unavailable":
            try:
                from src.reporting.decision_trace import (
                    build_decision_trace_rows,
                    write_decision_trace,
                )
                from src.reporting.policy_behavior import (
                    build_policy_behavior,
                    write_policy_behavior,
                )
                from src.reporting.training_telemetry import (
                    training_rows_from_diagnostics,
                    write_training_jsonl,
                )

                w = art.get("weights") or art.get("oos_weights")
                if w is None:
                    art["behaviour_export"] = "unavailable"
                    art["behaviour_export_reason"] = "no_weights_after_hoist"
                else:
                    ctx = _behaviour_context_for_cell(
                        art,
                        cfg_pre,
                        art.get("runner_artifact"),
                    )
                    runner_art = art.get("runner_artifact") or {}
                    sensitivities = (
                        art.get("policy_sensitivities")
                        or runner_art.get("policy_sensitivities")
                    )
                    beh = build_policy_behavior(
                        cell_id=str(art.get("spectrum_cell_id") or art_name),
                        arm=str(art.get("arm") or art.get("portfolio_arm") or ""),
                        algo=str(cfg_pre.get("algo") or ""),
                        architecture=str(cfg_pre.get("architecture") or ""),
                        objective=str(cfg_pre.get("objective") or cfg_pre.get("reward") or ""),
                        train_world=str(cfg_pre.get("train_world") or ""),
                        policy_mode=str(cfg_pre.get("policy_mode") or "balanced"),
                        universe_fingerprint=str(ctx.get("universe_fingerprint") or ""),
                        cell_cfg=cfg_pre,
                        weights=np.asarray(w),
                        turnovers=art.get("turnovers"),
                        sensitivities=sensitivities,
                        asset_returns=ctx.get("asset_returns"),
                        sleeve_matrix=ctx.get("sleeve_matrix"),
                        regimes=ctx.get("regimes"),
                        vix_z=ctx.get("vix_z"),
                        hy_oas_z=ctx.get("hy_oas_z"),
                        term_spread=ctx.get("term_spread"),
                        epu_z=ctx.get("epu_z"),
                        gpri_z=ctx.get("gpri_z"),
                        turnover_cap=ctx.get("turnover_cap"),
                    )
                    beh_path = out_dir / f"{art_name}_policy_behavior.json"
                    write_policy_behavior(beh_path, beh)
                    art["policy_behavior_path"] = str(beh_path)
                    art["behaviour_export"] = "written"
                    trace_rows = build_decision_trace_rows(
                        dates=ctx.get("dates") or list(range(np.asarray(w).shape[0])),
                        weights=np.asarray(w),
                        turnovers=art.get("turnovers"),
                        sleeve_matrix=ctx.get("sleeve_matrix"),
                        regimes=ctx.get("regimes"),
                        turnover_cap=ctx.get("turnover_cap"),
                    )
                    trace_path = out_dir / f"{art_name}_decision_trace.jsonl"
                    write_decision_trace(trace_path, trace_rows)
                    art["decision_trace_path"] = str(trace_path)
                    train_rows = training_rows_from_diagnostics(
                        art.get("training_diagnostics")
                        or (art.get("runner_artifact") or {}).get("training_diagnostics"),
                        cell_id=str(art.get("spectrum_cell_id") or art_name),
                    )
                    if train_rows:
                        train_path = out_dir / f"{art_name}_training.jsonl"
                        write_training_jsonl(train_path, train_rows)
                        art["training_telemetry_path"] = str(train_path)
                    # Desk-org companion for multi/HAPPO narrative cells.
                    if str(cfg_pre.get("algo") or "").lower() == "happo" or str(
                        cfg_pre.get("agent") or ""
                    ).lower() == "multi":
                        from src.reporting.deskorg import (
                            build_deskorg_artifact,
                            projection_slacks_from_path0,
                            write_deskorg,
                        )

                        path0 = (
                            (art.get("runner_artifact") or {}).get("paths") or {}
                        ).get("0") or {}
                        desk = build_deskorg_artifact(
                            cell_id=str(art.get("spectrum_cell_id") or art_name),
                            claim_tier=str(
                                art.get("claim_tier")
                                or cfg_pre.get("claim_tier")
                                or "narrative"
                            ),
                            behaviour_path=art.get("policy_behavior_path"),
                            decision_trace_path=art.get("decision_trace_path"),
                            coordination_proxies=art.get("coordination_proxies")
                            or (art.get("runner_artifact") or {}).get(
                                "coordination_proxies"
                            )
                            or (art.get("training_diagnostics") or {}).get(
                                "happo_trainer_stats"
                            ),
                            projection_slacks=projection_slacks_from_path0(path0),
                            training_telemetry_path=art.get("training_telemetry_path"),
                            eval_protocol=str(
                                (art.get("runner_artifact") or {}).get("eval_protocol")
                                or "combinatorial_purged_cv"
                            ),
                        )
                        desk_path = out_dir / f"{art_name}_deskorg.json"
                        write_deskorg(desk_path, desk)
                        art["deskorg_path"] = str(desk_path)
                dest.write_text(
                    json.dumps(art, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                art["policy_behavior_error"] = str(exc)[:300]
                _log_event(
                    "behaviour_export_error",
                    cell_id=str(art.get("spectrum_cell_id") or art_name),
                    error=str(exc)[:200],
                )
        elif should_export and art.get("behaviour_export") == "unavailable":
            # Honest disclosure: HAPPO smoke / missing weights skip clustering.
            dest.write_text(
                json.dumps(art, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        # Trial ledger append (B-DSR family size).
        try:
            from src.eval.pbo_appendix import append_trial_ledger_entry

            append_trial_ledger_entry(
                ROOT / "logs" / "trial_ledger.json",
                source="spectrum_campaign",
                trial_id=str(art.get("spectrum_cell_id") or art_name),
                sharpe=(
                    float(art["sharpe_mean"])
                    if art.get("sharpe_mean") is not None
                    else None
                ),
                status="ok" if not art.get("dry_run") else "dry_run",
                extra={
                    "artifact_path": str(dest),
                    "protocol_tier": art.get("protocol_tier"),
                    "n_seeds": art.get("n_seeds")
                    or len((art.get("spectrum_budget") or {}).get("seeds") or [0]),
                    "grid_kind": art.get("grid_kind") or "ofat",
                },
            )
        except Exception as exc:  # noqa: BLE001
            art["trial_ledger_error"] = str(exc)[:200]
        cells.append(art)
        _log_event(
            "cell_done",
            cell_id=str(art.get("spectrum_cell_id") or art_name),
            promotable=bool(art.get("promotable")),
            dry_run=bool(art.get("dry_run")),
            behaviour_export=art.get("behaviour_export"),
        )
        print(
            f"wrote {dest} promotable={art['promotable']} dry_run={art['dry_run']}"
        )
    if skipped:
        skip_path = out_dir / "SKIPPED.md"
        lines = [
            "# Spectrum cells skipped",
            "",
            f"Budget: `MASCOTRL_SPECTRUM_BUDGET_HOURS={budget_h}`",
            "",
            "| cell | reason | elapsed_h |",
            "|---|---|---|",
        ]
        for row in skipped:
            lines.append(
                f"| `{row['spectrum_cell_id']}` | {row['reason']} | "
                f"{float(row['elapsed_h']):.6f} |"
            )
        skip_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {skip_path} n_skipped={len(skipped)}")
    index = {
        "n_cells": len(cells),
        "cells": [c["spectrum_cell_id"] for c in cells],
        "dry_run": dry,
        "budget_hours": budget_h,
        "budget_exhausted": bool(skipped),
        "n_skipped": len(skipped),
        "skipped_cells": [s["spectrum_cell_id"] for s in skipped],
        "config_glob": args.config_glob,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
