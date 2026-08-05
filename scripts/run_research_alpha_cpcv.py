#!/usr/bin/env python3
"""Research CPCV runner: dry-run schema or numpy/OM short CPCV campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mascotrl.eval.cpcv import CPCVConfig
from mascotrl.eval.research_alpha_cpcv import (
    dry_run_research_alpha_cpcv,
    run_research_alpha_cpcv,
)
from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha

_OM_DEFERRED_ALLOWED_CLAIM_TIERS = frozenset(
    {
        "",
        "screening",
        "research",
        "dispatch_only",  # HAPPO screening smoke stamp
    }
)


def _refuse_om_deferred_without_rematerialize(cfg: dict, arm_id: str) -> None:
    """Fail closed on elevated claim_tier until OM panel is rematerialized."""
    if str(arm_id) not in ("opt", "mix"):
        return
    tier = str(cfg.get("claim_tier") or "").strip().lower()
    if tier not in _OM_DEFERRED_ALLOWED_CLAIM_TIERS:
        raise ValueError(
            f"om_deferred_path_refuses_claim_tier_{tier!r}: opt/mix arms with "
            "as_of=None require rematerialize (oos_force_rematerialize: true) "
            "before claim_tier above screening/research/dispatch_only"
        )


def _stamp_om_marks_honesty(cfg: dict, marks: dict | None) -> None:
    if isinstance(marks, dict) and marks.get("om_marks_degraded"):
        cfg["om_marks_degraded"] = True


def _toy_campaign_panel(n_days: int = 126, k: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n_days, k))
    factors = rng.normal(0.0, 0.005, size=(n_days, 4))
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    return dates, rets, factors


def _try_load_om_panel(cfg: dict, k: int):
    """Best-effort Arctic load; returns None if unavailable."""
    try:
        from mascotrl.data.arctic_store import ArcticStateStore
        from mascotrl.data.oos_panel import extract_om_marks, label_matrix, load_oos_panel
        from mascotrl.arms.training import resolve_claim_label_stem, resolve_portfolio_arm
    except Exception:
        return None
    try:
        import os

        arctic_path = cfg.get("arctic_db_path") or os.environ.get("MASCOTRL_ARCTIC_DIR")
        lake_base = cfg.get("lake_base_dir") or os.environ.get("MASCOTRL_LAKE_BASE")
        k_i = int(k)
        if cfg.get("arctic_library"):
            lib = str(cfg.get("arctic_library"))
        elif k_i <= 50:
            lib = "hyper_volanet_features"
        elif k_i <= 75:
            lib = "hyper_volanet_features_eq75"
        else:
            lib = "hyper_volanet_features_eq100"
        store = ArcticStateStore(
            db_path=arctic_path,
            library_name=lib,
        )
        start = str(cfg.get("hist_panel_start", cfg.get("is_hist_start", "2019-01-01")))
        end = str(cfg.get("oos_end", "2024-12-31"))
        as_of = None
        panel, _secids = load_oos_panel(store, start=start, end=end, as_of=as_of)
        cfg["pit_as_of"] = panel.attrs.get("pit_as_of", as_of)
        arm = resolve_portfolio_arm(cfg)
        _refuse_om_deferred_without_rematerialize(cfg, str(arm.id))
        if arm.id == "mix":
            n_opt = int(arm.option_slots)
            n_eq = int(arm.equity_slots)
            opt_rets = label_matrix(panel, n_opt, stem=str(arm.option_label_stem))
            eq_rets = label_matrix(panel, n_eq, stem=str(arm.equity_label_stem))
            rets = np.concatenate([opt_rets, eq_rets], axis=1)
            cfg["claim_label_stem"] = (
                f"{arm.option_label_stem}|{arm.equity_label_stem}"
            )
            try:
                from mascotrl.data.slot_mask import coverage_masks_for_arm

                cfg["_slot_valid_mask"] = coverage_masks_for_arm(
                    arm=arm,
                    atm=None,
                    option_labels=opt_rets,
                    equity_labels=eq_rets,
                    spot=None,
                )
            except Exception:
                pass
            cfg["_om_marks"] = extract_om_marks(
                panel,
                n_opt=n_opt,
                n_eq=n_eq,
                label_stem=str(arm.option_label_stem),
            )
            _stamp_om_marks_honesty(cfg, cfg.get("_om_marks"))
        else:
            stem = resolve_claim_label_stem(cfg)
            rets = label_matrix(panel, k_i, stem=stem)
            cfg["claim_label_stem"] = stem
            if arm.id == "opt":
                cfg["_om_marks"] = extract_om_marks(
                    panel, n_opt=k_i, n_eq=0, label_stem=stem
                )
                _stamp_om_marks_honesty(cfg, cfg.get("_om_marks"))
            else:
                cfg.pop("_om_marks", None)
        dates = list(panel.index)
        # Factors: zeros if FF4 unavailable (residualizer still runs).
        factors = np.zeros((len(dates), 4), dtype=np.float64)
        factors_source = "zeros"
        try:
            from mascotrl.eval.equity_factors import _try_load_ff_panel

            ff = _try_load_ff_panel(lake_base)
            if ff is not None:
                aligned = ff.reindex(panel.index).fillna(0.0)
                cols = [c for c in aligned.columns][:4]
                if cols:
                    factors = aligned[cols].to_numpy(dtype=np.float64)
                    if factors.shape[1] < 4:
                        pad = np.zeros((factors.shape[0], 4 - factors.shape[1]))
                        factors = np.concatenate([factors, pad], axis=1)
                    factors_source = "ff4"
        except Exception:
            pass
        cfg["factors_source"] = factors_source
        return dates, rets, factors
    except ValueError:
        raise
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "workflows" / "research_alpha_om_hist.yaml",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "research_alpha" / "cpcv_research_v0.json",
    )
    p.add_argument("--dry-run", action="store_true", help="Schema-only; no CPCV")
    p.add_argument(
        "--toy-panel",
        action="store_true",
        help="Force synthetic panel (unit/smoke campaign)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    if args.dry_run:
        art = dry_run_research_alpha_cpcv(cfg)
        art = stamp_research_positive_alpha(art)
    else:
        k = int(cfg.get("n_assets") or 8)
        loaded = None if args.toy_panel else _try_load_om_panel(cfg, k)
        if loaded is None:
            dates, rets, fac = _toy_campaign_panel(
                n_days=int(cfg.get("research_cpcv_days", 126)),
                k=min(k, 8),
                seed=int(args.seed),
            )
            cfg = {**cfg, "n_assets": int(rets.shape[1])}
            panel_source = "toy"
        else:
            dates, rets, fac = loaded
            panel_source = "optionmetrics"
        # Default short geometry; allow YAML override.
        cpcv = CPCVConfig(
            n_splits=int(cfg.get("cpcv_n_splits", 6)),
            n_test_groups=int(cfg.get("cpcv_n_test_groups", 2)),
            purge_days=int(cfg.get("cpcv_purge_days", 21)),
            embargo_days=int(cfg.get("cpcv_embargo_days", 21)),
        )
        # Toy/smoke: shrink folds for wall-clock.
        if panel_source == "toy" and not cfg.get("cpcv_full_geometry"):
            cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
        art = run_research_alpha_cpcv(
            dates, rets, fac, cfg, cpcv=cpcv, seed=int(args.seed), panel_source=panel_source
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} research_positive_alpha={art.get('research_positive_alpha')} "
        f"sharpe_mean={(art.get('path_summary') or {}).get('sharpe_mean')} "
        f"beats_random={art.get('policy_beats_random')}"
    )


if __name__ == "__main__":
    main()
