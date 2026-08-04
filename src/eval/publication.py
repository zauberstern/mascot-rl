"""Publication artifacts: baselines, DSR, regimes, ablations, limitations, plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.arctic_store import ArcticStateStore
from src.data.oos_panel import (
    SIGNALS_SYMBOL,
    label_matrix,
    load_oos_panel,
    wide_feature_matrix,
)
from src.eval.baselines import run_baseline_suite_on_panel
from src.eval.gate_ladder import refuse_alpha_stamp, run_gate_ladder
from src.eval.orientation_benchmarks import run_and_attach_orientation_benchmarks
from src.eval.stats_inference import (
    cscv_pbo_from_paths,
    hac_mean_tstat,
    hac_sharpe_se,
    romano_wolf_stepdown,
)
from src.eval.stats_rigor import deflated_sharpe_ratio, regime_performance_table
from src.logging_utils import get_logger
from src.reporting.capital_gates import KNOWN_UNMODELED_RISKS, PROJECTION_K_CEILING
from src.reporting.claim_language import SPA_DO_NOT_CLAIM

log = get_logger("mascotrl.eval.publication")

# Alpha v2 Step 26: frozen SPA rival families (opt + eq transparent baselines
# plus the bakeoff best single-agent RL slot). Immutable tuple on purpose.
SPA_FAMILIES: tuple[str, ...] = (
    "no_trade",
    "rv_iv_rank",
    "ridge",
    "equal_weight_factor_blend",
    "best_single_agent_rl",
)

SPA_N_BOOT = 10_000
SPA_MEAN_BLOCK_MONTHS = 6
HAC_ALPHA_LAGS_MONTHLY = 3

# Re-export for train/backfill call sites.
__all__ = (
    "SPA_FAMILIES",
    "attach_publication_stats",
    "compute_dsr",
    "compute_hac_alpha",
    "compute_pbo",
    "estimate_n_trials",
    "executed_trial_count",
    "hansen_spa_with_roles",
    "plot_publication_figures",
    "refuse_alpha_stamp",
    "run_and_attach_baselines",
    "run_and_attach_orientation_benchmarks",
    "run_gate_ladder",
    "run_spa",
    "spa_role_assignment",
    "write_limitations_section",
)


def _ledger_trial_rows(trial_ledger: Any) -> list[Any]:
    """Normalize trial-ledger shapes (list, MappingProxy rows, or JSON blob)."""
    if trial_ledger is None:
        return []
    if isinstance(trial_ledger, Sequence) and not isinstance(trial_ledger, (str, bytes)):
        return list(trial_ledger)
    if isinstance(trial_ledger, Mapping):
        rows = trial_ledger.get("trials") or trial_ledger.get("rows") or []
        return list(rows)
    return []


def trial_ledger_n(trial_ledger: Any) -> int:
    """Auditable N for DSR: count of registered trials in the ledger."""
    return len(_ledger_trial_rows(trial_ledger))


def compute_hac_alpha(
    returns: Sequence[float],
    factors: Any = None,
    *,
    lags: int = HAC_ALPHA_LAGS_MONTHLY,
) -> dict[str, Any]:
    """Newey-West HAC t on residual (or raw) returns; Alpha v2 G5 uses lags=3."""
    del factors  # residual series is precomputed by the caller for G5.
    out = hac_mean_tstat(returns, lags=int(lags))
    out["lags_requested"] = int(lags)
    out["hac_t"] = out.get("t_hac")
    return out


def compute_dsr(
    returns: Sequence[float],
    trial_ledger: Any,
    *,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Deflated Sharpe with N taken from the trial ledger (Bailey & LdP)."""
    n = int(n_trials) if n_trials is not None else max(1, trial_ledger_n(trial_ledger))
    out = deflated_sharpe_ratio(returns, n_trials=n)
    out["n_trials_source"] = "trial_ledger" if n_trials is None else "explicit"
    out["n_trials_ledger"] = trial_ledger_n(trial_ledger)
    return out


def compute_pbo(
    trial_matrix: Sequence[Sequence[float]] | Sequence[float],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """CSCV PBO on path/trial return matrix (or Sharpe vector fallback)."""
    if not trial_matrix:
        return {"pbo": float("nan"), "reason": "empty trial_matrix"}
    first = trial_matrix[0]
    if isinstance(first, (int, float, np.floating)):
        from src.eval.pbo_appendix import probability_of_backtest_overfitting

        return probability_of_backtest_overfitting(
            [float(x) for x in trial_matrix], seed=int(seed)
        )
    paths = [list(p) for p in trial_matrix if p]
    return cscv_pbo_from_paths(paths, seed=int(seed))


def run_spa(
    candidate: Sequence[float],
    baselines: Mapping[str, Sequence[float]],
    *,
    reps: int = SPA_N_BOOT,
    mean_block_months: int = SPA_MEAN_BLOCK_MONTHS,
    seed: int = 0,
) -> dict[str, Any]:
    """Hansen SPA: candidate superiority vs preregistered baseline family.

    Each baseline is treated as the benchmark with the candidate as the sole
    rival. Reported ``spa_p`` is the *max* consistent p-value across baselines
    (must clear G5's p<=0.05 against the strongest / hardest baseline). Low p
    means the candidate significantly beats that baseline.
    """
    from src.eval.stats_rigor import hansen_spa_test

    if not baselines:
        return {
            "ok": False,
            "reason": "spa_rivals_insufficient",
            "n_economic_rivals": 0,
            "spa_p": float("nan"),
            "spa_families": list(SPA_FAMILIES),
        }
    per: dict[str, Any] = {}
    pvals: list[float] = []
    for name, series in baselines.items():
        spa = hansen_spa_test(
            list(series),
            {"candidate": list(candidate)},
            n_boot=int(reps),
            block_mean=max(1, int(mean_block_months)),
            seed=int(seed),
        )
        per[str(name)] = spa
        if spa.get("ok") and spa.get("pvalue_consistent") is not None:
            pvals.append(float(spa["pvalue_consistent"]))
    from src.eval.alpha_gates import NONSENSE_PEERS

    n_econ = len([k for k in baselines if k not in NONSENSE_PEERS])
    if not pvals:
        return {
            "ok": False,
            "reason": "spa_rivals_insufficient",
            "n_economic_rivals": n_econ,
            "spa_p": float("nan"),
            "per_baseline": per,
            "spa_families": list(SPA_FAMILIES),
        }
    spa_p = float(max(pvals))
    return {
        "ok": True,
        "spa_p": spa_p,
        "pvalue_consistent": spa_p,
        "n_boot": int(reps),
        "mean_block_months": int(mean_block_months),
        "n_economic_rivals": n_econ,
        "rival_names": sorted(baselines.keys()),
        "per_baseline": per,
        "spa_families": list(SPA_FAMILIES),
        "interpretation": (
            "Candidate is the rival; each baseline is a benchmark. Low spa_p "
            "rejects the null that the candidate is not superior."
        ),
    }


def spa_role_assignment(*, happo_as_claimant: bool = False) -> dict[str, Any]:
    """Declare SPA roles for honesty locks / claim-path wiring.

    Historical default: HAPPO is the **benchmark** (null = no rival beats HAPPO).
    Claim path: HAPPO is the **claimant** / rival vs each panel member as
    benchmark (null = HAPPO is not superior to that baseline).
    """
    if happo_as_claimant:
        return {
            "happo_as_claimant": True,
            "claimant": "happo",
            "benchmark_role": "panel",
            "hansen_benchmark": "panel_member",
            "hansen_rival": "happo",
            "interpretation": (
                "HAPPO is the claimant. Each panel member is a benchmark; "
                "low spa_p rejects the null that HAPPO is not superior."
            ),
            "do_not_claim": SPA_DO_NOT_CLAIM,
        }
    return {
        "happo_as_claimant": False,
        "claimant": None,
        "benchmark_role": "happo",
        "hansen_benchmark": "happo",
        "hansen_rival": "panel",
        "interpretation": (
            "HAPPO is the benchmark. Low pvalue_consistent rejects the null that "
            "no rival beats HAPPO (i.e. some rival is superior). High p "
            "(≈1) means we do not find a superior rival — not proof of alpha."
        ),
        "do_not_claim": SPA_DO_NOT_CLAIM,
    }


def hansen_spa_with_roles(
    happo_pnls: Sequence[float],
    rivals: Mapping[str, Sequence[float]],
    *,
    happo_as_claimant: bool = False,
    n_boot: int = 499,
    block_mean: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Hansen SPA with explicit role assignment.

    ``happo_as_claimant=False`` (default): HAPPO = benchmark, panel = rivals.
    ``happo_as_claimant=True``: claim path via ``run_spa`` (HAPPO rival vs each
    baseline as benchmark; reported spa_p is the max consistent p).
    """
    roles = spa_role_assignment(happo_as_claimant=happo_as_claimant)
    if happo_as_claimant:
        spa = run_spa(
            happo_pnls,
            rivals,
            reps=int(n_boot),
            mean_block_months=int(block_mean),
            seed=int(seed),
        )
    else:
        from src.eval.stats_rigor import hansen_spa_test

        spa = hansen_spa_test(
            list(happo_pnls),
            {k: list(v) for k, v in rivals.items()},
            n_boot=int(n_boot),
            block_mean=int(block_mean),
            seed=int(seed),
        )
    spa.update(roles)
    return spa


def executed_trial_count(report: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """
    Number of configurations actually executed, from the trial ledger.

    Bailey and Lopez de Prado (2014) define N as the number of independent
    trials involved in *selecting* the reported strategy. A combinatorial
    upper bound is not that number: it can be far above the true search (making
    DSR uninformative) or below it (making DSR anti-conservative), and either
    way it is not auditable. This reads the ledger of executed trials instead.
    """
    ledger = report.get("trial_ledger") or {}
    trials = _ledger_trial_rows(ledger)
    by_source: dict[str, int] = {}
    for t in trials:
        if not isinstance(t, Mapping):
            by_source["unknown"] = by_source.get("unknown", 0) + 1
            continue
        src = str(t.get("source") or t.get("baseline") or "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    # CPCV paths are additional evaluated configurations of the same strategy.
    cpcv = report.get("cpcv") or {}
    n_paths = int(((cpcv.get("path_summary") or {}).get("n_paths") or 0))
    if n_paths:
        by_source["cpcv_paths"] = n_paths
    n = int(sum(by_source.values()))
    return n, {
        "n_trials": n,
        "by_source": by_source,
        "source": "executed_trial_ledger",
        "auditable": True,
        "citation": "Bailey and Lopez de Prado (2014, JPM 40(5))",
    }


def estimate_n_trials(
    report: dict[str, Any], cfg: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """
    Trial count for DSR.

    Prefers the **executed** trial ledger (auditable, and what Bailey and Lopez
    de Prado actually define N to be). Falls back to the conservative
    combinatorial bound only when no ledger is present, and labels the result so
    the report never presents a proxy as a measured count. An explicit
    ``publication_n_trials`` override still wins.
    """
    cfg = cfg or {}
    if cfg.get("publication_n_trials") is None:
        n_exec, exec_meta = executed_trial_count(report)
        if n_exec > 0:
            exec_meta["formula"] = "count(executed trials in ledger) + cpcv paths"
            return max(1, n_exec), exec_meta
    n_seeds = len(
        [
            x
            for x in str(cfg.get("eval_seeds", "0,1,2,3,4")).split(",")
            if str(x).strip()
        ]
    )
    n_folds_cfg = max(
        1,
        int(
            (report.get("nested_wfo") or {}).get("n_folds")
            or cfg.get("publication_planned_folds", 5)
        ),
    )
    n_ablate = int(cfg.get("publication_ablation_variants", 5))
    n_hp = int(cfg.get("nested_wfo_finetune_passes", 3))
    n_base = int(cfg.get("publication_baseline_count", 3))
    breakdown = {
        "n_seeds": n_seeds,
        "n_folds": n_folds_cfg,
        "n_ablation_variants": n_ablate,
        "n_hp_passes": n_hp,
        "n_baselines": n_base,
        "formula": "max(explicit, seeds×folds×ablations×hp + baselines)",
        "source_kind": "combinatorial_upper_bound",
        "auditable": False,
        "caveat": (
            "Combinatorial bound, not the executed search. Present only because "
            "no trial ledger was available on this report."
        ),
    }
    computed = n_seeds * n_folds_cfg * n_ablate * n_hp + n_base
    if cfg.get("publication_n_trials") is not None:
        n = max(1, int(cfg["publication_n_trials"]))
        breakdown["source"] = "publication_n_trials override"
    else:
        n = max(1, computed)
        breakdown["source"] = "computed"
    breakdown["n_trials"] = n
    breakdown["computed_raw"] = computed
    return n, breakdown


def attach_publication_stats(
    report: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach DSR / PSR / regimes / bootstrap CIs / SPA from historical series."""
    from src.eval.stats_rigor import (
        block_bootstrap_metric_ci,
    )

    cfg = cfg or {}
    ho = report.get("historical_oos") or {}
    hi = report.get("historical_is") or {}
    cal = report.get("historical_calendar") or {}
    # Prefer historical_calendar: train_happo strips IS pnls from historical_is
    # to keep the report compact, which previously left dates=IS+OOS but
    # pnls=OOS-only (length mismatch → empty/corrupt regime table, silent GFC gap).
    dates_is = list(cal.get("is_dates") or hi.get("dates") or [])
    dates_oos = list(cal.get("oos_dates") or ho.get("dates") or [])
    pnls_h = list(cal.get("is_pnls") or [])
    if not pnls_h:
        raw_h = hi.get("pnls") or {}
        pnls_h = list(raw_h.get("happo") or []) if isinstance(raw_h, dict) else []
    pnls_o = list(cal.get("oos_pnls") or [])
    if not pnls_o:
        raw_o = ho.get("pnls") or {}
        pnls_o = list(raw_o.get("happo") or []) if isinstance(raw_o, dict) else []
    dates = dates_is + dates_oos
    pnls = pnls_h + pnls_o
    turns_h = (
        (hi.get("turnovers") or {}).get("happo")
        if isinstance(hi.get("turnovers"), dict)
        else []
    ) or []
    turns_o = (
        (ho.get("turnovers") or {}).get("happo")
        if isinstance(ho.get("turnovers"), dict)
        else []
    ) or []
    turns = list(turns_h) + list(turns_o)
    # Align lengths: never pass mismatched date/PnL vectors into regime math.
    if dates and pnls and len(dates) != len(pnls):
        log.warning(
            "regime series length mismatch dates=%d pnls=%d — truncating to min",
            len(dates),
            len(pnls),
        )
        n = min(len(dates), len(pnls))
        dates, pnls = dates[:n], pnls[:n]
        turns = turns[:n] if turns else turns
    # Drop non-finite days (poisoned IS walks) before DSR / regimes.
    if dates and pnls:
        keep = [i for i, p in enumerate(pnls) if np.isfinite(p)]
        dropped = len(pnls) - len(keep)
        if dropped:
            log.warning(
                "dropping %d non-finite hist PnL days before DSR/regimes (kept=%d)",
                dropped,
                len(keep),
            )
        dates = [dates[i] for i in keep]
        pnls = [float(pnls[i]) for i in keep]
        if turns and len(turns) >= max(keep or [0]) + 1:
            turns = [turns[i] for i in keep if i < len(turns)]
    if not pnls:
        pnls = list((ho.get("pnls") or {}).get("happo") or [])
        dates = list(ho.get("dates") or [])
        turns = (
            (ho.get("turnovers") or {}).get("happo")
            if isinstance(ho.get("turnovers"), dict)
            else None
        )
        if dates and pnls:
            keep = [i for i, p in enumerate(pnls) if np.isfinite(p)]
            dates = [dates[i] for i in keep]
            pnls = [float(pnls[i]) for i in keep]

    n_trials, breakdown = estimate_n_trials(report, cfg)
    report["n_trials_breakdown"] = breakdown
    trial_sharpes = []
    for row in (report.get("multiseed_oos") or {}).get("rows") or []:
        if row.get("sharpe") is not None:
            trial_sharpes.append(float(row["sharpe"]))
    for fold in (report.get("nested_wfo") or {}).get("folds") or []:
        sh = (fold.get("summary") or {}).get("happo", {}).get("sharpe")
        if sh is not None:
            trial_sharpes.append(float(sh))
        elif fold.get("test_sharpe") is not None:
            trial_sharpes.append(float(fold["test_sharpe"]))

    # Dual-series DSR / bootstrap: never conflate OOS point Sharpe with pooled CIs.
    dsr_pooled = deflated_sharpe_ratio(
        pnls,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes or None,
        n_trials_breakdown=breakdown,
    )
    dsr_pooled["series"] = "pooled_is_oos"
    dsr_pooled["n_obs"] = int(len(pnls))
    dsr_pooled["role"] = "appendix"
    dsr_pooled["caveat"] = (
        "Pooled series includes in-sample days; reported as an appendix only. "
        "The headline deflated Sharpe is the out-of-sample series."
    )
    report["deflated_sharpe_pooled"] = dsr_pooled

    pnls_oos_only = [float(x) for x in pnls_o if np.isfinite(x)] if pnls_o else list(
        (ho.get("pnls") or {}).get("happo") or []
    )
    pnls_oos_only = [float(x) for x in pnls_oos_only if np.isfinite(x)]
    dsr_oos = deflated_sharpe_ratio(
        pnls_oos_only,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes or None,
        n_trials_breakdown=breakdown,
    )
    dsr_oos["series"] = "oos"
    dsr_oos["n_obs"] = int(len(pnls_oos_only))
    dsr_oos["role"] = "headline"
    report["deflated_sharpe_oos"] = dsr_oos
    # Headline alias points at the OOS series: an in-sample-contaminated
    # deflated Sharpe is not a claim a referee will accept.
    report["deflated_sharpe"] = dsr_oos
    report["deflated_sharpe_headline_series"] = "oos"

    # HAC inference on the OOS series (serial correlation from persistent
    # positions and overlapping hedges makes the iid t-stat overstated).
    report["hac_inference_oos"] = hac_sharpe_se(pnls_oos_only)
    report["hac_inference_pooled"] = hac_sharpe_se(pnls)
    # Alpha v2 G5: NW HAC t on residual returns with 3 monthly lags.
    residual_oos = list(
        (report.get("ensemble_residual_returns") or ho.get("residual_pnls") or [])
    )
    if residual_oos:
        report["hac_alpha"] = compute_hac_alpha(
            residual_oos, lags=HAC_ALPHA_LAGS_MONTHLY
        )
    elif pnls_oos_only:
        report["hac_alpha"] = compute_hac_alpha(
            pnls_oos_only, lags=HAC_ALPHA_LAGS_MONTHLY
        )

    report["regime_performance"] = regime_performance_table(dates, pnls, turns)
    if (report["regime_performance"].get("sanity") or {}).get("warnings"):
        for w in report["regime_performance"]["sanity"]["warnings"]:
            log.warning("REGIME_SANITY %s", w)

    boot_seed = int(cfg.get("seed", 42))
    from src.eval.arch_bootstrap import resolve_bootstrap_backend

    boot_backend = resolve_bootstrap_backend(cfg if isinstance(cfg, dict) else dict(cfg or {}))
    report["bootstrap_backend"] = boot_backend
    report["bootstrap_cis_pooled"] = {
        "series": "pooled_is_oos",
        "n_obs": int(len(pnls)),
        "backend": boot_backend,
        "sharpe": block_bootstrap_metric_ci(
            pnls, metric="sharpe", seed=boot_seed, backend=boot_backend
        ),
        "max_drawdown": block_bootstrap_metric_ci(
            pnls, metric="max_drawdown", seed=boot_seed + 1, backend=boot_backend
        ),
    }
    if turns:
        report["bootstrap_cis_pooled"]["turnover"] = block_bootstrap_metric_ci(
            turns, metric="mean_abs", seed=boot_seed + 2, backend=boot_backend
        )
    report["bootstrap_cis"] = report["bootstrap_cis_pooled"]  # back-compat
    report["bootstrap_cis_oos"] = {
        "series": "oos",
        "n_obs": int(len(pnls_oos_only)),
        "backend": boot_backend,
        "sharpe": block_bootstrap_metric_ci(
            pnls_oos_only, metric="sharpe", seed=boot_seed + 10, backend=boot_backend
        ),
        "max_drawdown": block_bootstrap_metric_ci(
            pnls_oos_only, metric="max_drawdown", seed=boot_seed + 11, backend=boot_backend
        ),
    }

    # SPA: HAPPO is the *benchmark*; rivals are rich BASELINE_NAMES only.
    # Zero/random are nonsense peers (critique 07).
    from src.eval.baselines import BASELINE_NAMES
    from src.eval.alpha_gates import NONSENSE_PEERS

    oos_pnls = dict(ho.get("pnls") or {})
    rivals: dict[str, Any] = {}
    for name in BASELINE_NAMES:
        xs = oos_pnls.get(name)
        if xs:
            rivals[name] = xs
    for name, xs in ((report.get("baselines") or {}).get("pnls") or {}).items():
        if xs and name not in rivals and name not in NONSENSE_PEERS:
            rivals[name] = xs
    # Gate3 same-fold rivals (EW / quintile / myopic MV / XGB / MLP, …).
    gate3 = report.get("gate3_same_fold") or report.get("gate3") or {}
    g3_body = gate3.get("gate3") if isinstance(gate3.get("gate3"), dict) else gate3
    for name, xs in ((g3_body.get("baseline_pnls") or g3_body.get("pnls") or {})).items():
        if xs and name not in rivals and name not in NONSENSE_PEERS:
            rivals[name] = xs
    for row in g3_body.get("baselines") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("id") or "")
        xs = row.get("pnl") or row.get("pnls") or row.get("returns")
        if name and xs and name not in rivals and name not in NONSENSE_PEERS:
            rivals[name] = xs

    n_economic = len(rivals)
    spa_meta = {
        "rival_names": sorted(rivals.keys()),
        "n_rivals": len(rivals),
        "n_economic_rivals": n_economic,
        "min_economic_rivals": 3,
        "preregistered": list(BASELINE_NAMES) + ["gate3_*"],
        "spa_families": list(SPA_FAMILIES),
    }
    residual_equity = bool(
        cfg.get("residual_equity_protocol")
        or cfg.get("alpha_v2")
        or report.get("residual_equity_protocol")
        or report.get("alpha_v2")
    )
    spa_n_boot = int(cfg.get("spa_n_boot", SPA_N_BOOT if residual_equity else 499))
    # Default remains HAPPO-as-benchmark (honesty lock). Opt into claim-path roles
    # via spa_happo_as_claimant / spa_claim_path.
    happo_as_claimant = bool(
        cfg.get("spa_happo_as_claimant") or cfg.get("spa_claim_path")
    )
    if oos_pnls.get("happo") and n_economic >= 3:
        spa = hansen_spa_with_roles(
            oos_pnls["happo"],
            rivals,
            happo_as_claimant=happo_as_claimant,
            n_boot=spa_n_boot,
            seed=int(cfg.get("seed", 42)),
        )
        spa.update(spa_meta)
        spa["do_not_claim"] = SPA_DO_NOT_CLAIM
        report["hansen_spa"] = spa
        # Residual-equity G5 framing: candidate superiority vs frozen SPA_FAMILIES.
        if residual_equity:
            family_baselines = {
                k: rivals[k] for k in SPA_FAMILIES if k in rivals and k != "best_single_agent_rl"
            }
            # Map aliases that may appear under industry names.
            alias = {
                "rv_iv_rank": ("goyal_saretto_hv_iv", "rv_iv_rank"),
                "equal_weight_factor_blend": ("equal_weight", "equal_weight_factor_blend"),
                "ridge": ("ridge",),
                "no_trade": ("no_trade", "zero"),
            }
            for fam, keys in alias.items():
                if fam in family_baselines:
                    continue
                for k in keys:
                    if k in rivals:
                        family_baselines[fam] = rivals[k]
                        break
            if family_baselines:
                report["residual_equity_spa"] = run_spa(
                    oos_pnls["happo"],
                    family_baselines,
                    reps=spa_n_boot,
                    mean_block_months=int(
                        cfg.get("spa_mean_block_months", SPA_MEAN_BLOCK_MONTHS)
                    ),
                    seed=int(cfg.get("seed", 42)),
                )
                report["hansen_spa"]["residual_equity_spa_p"] = report["residual_equity_spa"].get(
                    "spa_p"
                )
        report["romano_wolf"] = romano_wolf_stepdown(
            oos_pnls["happo"],
            rivals,
            seed=int(cfg.get("seed", 42)),
            alpha=float(cfg.get("romano_wolf_alpha", 0.05)),
        )
        from src.eval.stats_inference import white_reality_check

        report["white_reality_check"] = white_reality_check(
            oos_pnls["happo"],
            rivals,
            n_boot=spa_n_boot,
            seed=int(cfg.get("seed", 42)),
        )
        # Path-level RW on HAPPO − bench_j diffs when a panel is present.
        from src.eval.stats_inference import romano_wolf_over_panel

        panel_for_rw = dict(rivals)
        fold_panels = (report.get("cpcv") or {}).get("fold_benchmark_panels") or {}
        if fold_panels and not panel_for_rw:
            # Aggregate first available fold panel series if rivals empty.
            for _fid, art in fold_panels.items():
                pnls = (art or {}).get("pnls") or {}
                if pnls:
                    panel_for_rw = pnls
                    break
        if panel_for_rw:
            report["romano_wolf_panel"] = romano_wolf_over_panel(
                oos_pnls["happo"],
                panel_for_rw,
                seed=int(cfg.get("seed", 42)),
                alpha=float(cfg.get("romano_wolf_alpha", 0.05)),
            )
    elif oos_pnls.get("happo"):
        roles = spa_role_assignment(happo_as_claimant=happo_as_claimant)
        report["hansen_spa"] = {
            "ok": False,
            "reason": "spa_rivals_insufficient",
            "do_not_claim": SPA_DO_NOT_CLAIM,
            **roles,
            **spa_meta,
        }
        report["romano_wolf"] = {
            "ok": False,
            "reason": "spa_rivals_insufficient",
            **spa_meta,
        }
    else:
        roles = spa_role_assignment(happo_as_claimant=happo_as_claimant)
        report["hansen_spa"] = {
            "ok": False,
            "reason": "missing OOS series",
            **roles,
        }
        report["romano_wolf"] = {"ok": False, "reason": "missing OOS series"}

    # CSCV PBO (Bailey 2017) is separate from CPCV positive_path_rate.
    cpcv_paths = [
        p.get("pnl") or [] for p in ((report.get("cpcv") or {}).get("paths") or [])
    ]
    if len([p for p in cpcv_paths if p]) >= 2:
        report["cscv_pbo"] = cscv_pbo_from_paths(
            [p for p in cpcv_paths if p], seed=int(cfg.get("seed", 42))
        )
        report["cscv_pbo"]["not_positive_path_rate"] = True
        report["cscv_pbo"]["definition"] = (
            "PBO is the frequency the IS winner ranks below OOS median; "
            "it is not the fraction of negative CPCV paths."
        )
    pos_rate = ((report.get("cpcv") or {}).get("path_summary") or {}).get(
        "positive_path_rate"
    )
    if pos_rate is not None:
        report["cpcv_positive_path_rate"] = pos_rate
        report["pbo_vs_path_rate_fence"] = (
            "Do not equate cscv_pbo.pbo with cpcv positive_path_rate."
        )

    # Impact sensitivity sweep (robustness axis, not training).
    if bool(cfg.get("publication_impact_sweep", True)) and pnls and turns:
        from src.reporting.capital_gates import capacity_curve_from_daily

        sweep = []
        for coef in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
            curve = capacity_curve_from_daily(
                pnls,
                turns if turns else [0.15] * len(pnls),
                impact_coef=coef,
                spread_bps=float(cfg.get("capacity_spread_bps", 5.0)),
                base_notional=float(cfg.get("capacity_base_aum_usd", 50_000_000.0)),
            )
            row1 = next(
                (r for r in curve.get("rows") or [] if abs(r["aum_multiplier"] - 1.0) < 1e-9),
                None,
            )
            sweep.append(
                {
                    "execution_impact_coef": coef,
                    "sharpe_at_1x": None if row1 is None else row1.get("sharpe"),
                    "capacity_ceiling_multiplier": curve.get("capacity_ceiling_multiplier"),
                }
            )
        report["impact_sensitivity_sweep"] = sweep

    log.info(
        "DSR_pooled=%.3f DSR_oos=%.3f PSR=%.3f n_trials=%d significant_05=%s spa_p_c=%s",
        report["deflated_sharpe_pooled"].get("dsr", float("nan")),
        report["deflated_sharpe_oos"].get("dsr", float("nan")),
        report["deflated_sharpe"].get("psr", float("nan")),
        report["deflated_sharpe"].get("n_trials", 0),
        report["deflated_sharpe"].get("significant_05"),
        (report.get("hansen_spa") or {}).get("pvalue_consistent"),
    )
    return report


def run_and_attach_baselines(
    store: ArcticStateStore,
    report: dict[str, Any],
    *,
    start: str,
    end: str,
    seq_len: int,
    n_assets: int,
    turnover_limit: float,
    as_of=None,
    asset_names: list[str] | None = None,
    report_dir: Path | str | None = None,
    metrics_dir: Path | str | None = None,
    use_projection: bool = True,
    max_name_abs_weight: float = 5.0,
) -> dict[str, Any]:
    if SIGNALS_SYMBOL not in store.list_available_features():
        raise KeyError(SIGNALS_SYMBOL)
    panel, secids = load_oos_panel(store, start=start, end=end, as_of=as_of)
    if len(secids) < n_assets:
        raise RuntimeError(f"universe has {len(secids)} names; need {n_assets}")
    atm = wide_feature_matrix(panel, "atm_iv", n_assets)
    deltas = wide_feature_matrix(panel, "delta", n_assets)
    fwd = label_matrix(panel, n_assets)
    try:
        skew = wide_feature_matrix(panel, "skew_25d", n_assets)
    except KeyError:
        skew = None

    from src.data.oos_panel import load_universe_meta
    from src.eval.baselines import load_underlier_returns_matrix
    from src.reporting.portfolio_accounting import PortfolioAccountingLedger

    names = asset_names
    if names is None:
        meta = load_universe_meta(store)
        display = list(meta.get("display_names") or meta.get("tickers") or [])
        if len(display) >= n_assets:
            names = [str(display[i]) for i in range(n_assets)]
    ledger = PortfolioAccountingLedger(asset_names=names, num_assets=n_assets)

    use_secids = [int(s) for s in secids[:n_assets]]
    urets, umeta = load_underlier_returns_matrix(use_secids, panel.index.to_list())
    # Drop non-JSON lead array before attaching report; suite still receives it.
    suite = run_baseline_suite_on_panel(
        atm=atm,
        deltas_np=deltas,
        fwd=fwd,
        dates=panel.index.to_list(),
        seq_len=seq_len,
        turnover_limit=turnover_limit,
        ledger=ledger,
        phase="OOS_TEST",
        use_projection=bool(use_projection),
        max_name_abs_weight=float(max_name_abs_weight),
        skew=skew,
        underlier_rets=urets,
        underlier_meta=umeta,
    )
    report["baselines"] = suite

    out_report = Path(report_dir) if report_dir else None
    out_metrics = Path(metrics_dir) if metrics_dir else None
    if out_report is None and report.get("portfolio_ledger_excel"):
        # Co-locate with HAPPO ledger when paths already known.
        out_report = Path(str(report["portfolio_ledger_excel"])).parent
    if out_metrics is None and report.get("portfolio_ledger_parquet"):
        out_metrics = Path(str(report["portfolio_ledger_parquet"])).parent
    if out_report is not None:
        out_report.mkdir(parents=True, exist_ok=True)
        excel_path = out_report / "portfolio_allocations_baselines.xlsx"
        ledger.export_excel(excel_path)
        report["baseline_ledger_excel"] = str(excel_path)
    if out_metrics is not None:
        out_metrics.mkdir(parents=True, exist_ok=True)
        parquet_path = out_metrics / "portfolio_allocations_baselines.parquet"
        ledger.export_parquet(parquet_path)
        report["baseline_ledger_parquet"] = str(parquet_path)
    report["baseline_ledger_rows"] = len(ledger.records)

    # Edge vs strongest non-trivial baseline on mean PnL + alpha stamp.
    from src.eval.alpha_gates import apply_rich_baseline_alpha_gate

    happo_mu = float(
        ((report.get("historical_oos") or {}).get("summary") or {})
        .get("happo", {})
        .get("mean_pnl", float("nan"))
    )
    best_b = None
    best_mu = float("-inf")
    for name, sm in (suite.get("summary") or {}).items():
        if str(name) in ("zero", "random", "happo", "happo_gross"):
            continue
        mu = float(sm.get("mean_pnl", float("nan")))
        if np.isfinite(mu) and mu > best_mu:
            best_mu = mu
            best_b = name
    report["edge_vs_best_baseline"] = (
        float(happo_mu - best_mu) if best_b is not None and np.isfinite(happo_mu) else float("nan")
    )
    report["best_baseline"] = best_b
    apply_rich_baseline_alpha_gate(report)
    log.info(
        "baselines done best=%s edge_vs_best=%s alpha_found=%s ledger_rows=%d",
        report.get("best_baseline"),
        report.get("edge_vs_best_baseline"),
        report.get("alpha_found"),
        report.get("baseline_ledger_rows", 0),
    )
    return report


def write_limitations_section(run_dir: Path, report: dict[str, Any] | None = None) -> Path:
    """Honest limitations section for paper drafts (from capital hygiene artifacts)."""
    run_dir = Path(run_dir)
    path = run_dir / "LIMITATIONS.md"
    lines = [
        "# Limitations",
        "",
        "This section is generated from machine-readable risk disclosures so that",
        "paper drafts cannot silently omit capacity and modeling constraints.",
        "",
        f"- **Projection production ceiling:** K ≤ {PROJECTION_K_CEILING}.",
        "- **Primary training measure:** rBergomi + Dupire synthetic paths; capital",
        "  claims require PIT OptionMetrics nested walk-forward with stability gates.",
        "- **Nested WFO ≠ CPCV:** expanding fine-tune folds without purge/embargo",
        "  beyond fold cuts (AFML CPCV is literature counterpart only).",
        "- **Execution:** small spread/√turnover drag in training reward; live market",
        "  impact and venue fragmentation are not validated end-to-end.",
        "- **Ops:** kill-switch / reconcile / shadow stubs are research MVPs, not a",
        "  production OMS.",
        "- **Universe:** single OptionMetrics US equity-option lake (no Nasdaq/Europe",
        "  rematerialize yet); external validity is disclosed, not claimed.",
        "- **Regime coverage:** stress windows (2008 GFC / 2020 COVID / 2022 hike) are",
        "  reported only when the PIT hist panel overlaps the window; missing windows",
        "  are explicit `N/A — data unavailable`, never zero-filled Sharpes.",
        "",
        "## Known unmodeled risks",
        "",
    ]
    for row in KNOWN_UNMODELED_RISKS:
        lines.append(f"### `{row['id']}` ({row['severity']})")
        lines.append("")
        lines.append(row["summary"])
        lines.append("")
    if report:
        dsr = (
            report.get("deflated_sharpe_pooled")
            or report.get("deflated_sharpe")
            or {}
        )
        dsr_o = report.get("deflated_sharpe_oos") or {}
        spa = report.get("hansen_spa") or {}
        pbo = report.get("pbo_appendix") or {}
        ca = report.get("corporate_actions") or {}
        ho = report.get("historical_oos") or {}
        fr = ho.get("friction") or {}
        lines.extend(
            [
                "## Statistical caveats",
                "",
                f"- DSR **pooled** series={dsr.get('series', 'pooled_is_oos')} "
                f"DSR={dsr.get('dsr')} (n_trials={dsr.get('n_trials')}, "
                f"significant@5%={dsr.get('significant_05')}). "
                "Lead with this — not OOS point Sharpe alone.",
                f"- DSR **OOS** series={dsr_o.get('series', 'oos')} "
                f"DSR={dsr_o.get('dsr')} n_obs={dsr_o.get('n_obs')} "
                f"significant@5%={dsr_o.get('significant_05')}.",
                f"- Citation: {dsr.get('citation') or 'Bailey & López de Prado (2014) DSR'}",
                f"- SPA: benchmark_role=`{spa.get('benchmark_role', 'happo')}` — "
                f"null = no rival beats HAPPO. Do **not** claim: "
                f"{spa.get('do_not_claim') or 'SPA proves HAPPO alpha'}.",
                f"- PBO appendix={pbo.get('pbo')} "
                "(trial Sharpes / CSCV-style; nested WFO ≠ CPCV).",
                "",
                "## Friction / CA / borrow",
                "",
                f"- `friction_applied={ho.get('friction_applied')}`, "
                f"`friction_model={ho.get('friction_model') or fr.get('execution_drag_mode')}`.",
                f"- `name_borrow_coverage={fr.get('name_borrow_coverage', 0)}` — "
                "GC proxy only when coverage is 0; do not claim name-level locate.",
                f"- Corporate actions joined={ca.get('joined')}: "
                f"{ca.get('limitation') or 'see clearance no_corporate_actions'}.",
                "- Soft ruler = OptionMetrics near-ATM mid; orientation ≠ dollar overlay.",
                "",
            ]
        )
        pb = report.get("projection_k_benchmark") or {}
        if pb:
            lines.append(
                f"- Projection bench break_k={pb.get('wallclock_break_k_ms_per_sample_gt_50')} "
                f"(production_ceiling_k={pb.get('production_ceiling_k')})."
            )
            lines.append("")
        reg = (report.get("regime_performance") or {}).get("regimes") or []
        missing = [
            r for r in reg if r.get("available") is False or r.get("status") == "unavailable"
        ]
        if missing:
            lines.extend(["## Regime windows unavailable", ""])
            for r in missing:
                lines.append(
                    f"- **{r.get('label', r.get('id'))}:** "
                    f"{r.get('note') or 'N/A — data unavailable'}"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n")
    caveats = run_dir / "KNOWN_LIMITATIONS.md"
    if not caveats.is_file():
        caveats = run_dir / "RUN_CAVEATS.md"  # delta-hedged option allocator / attic runs
    if caveats.is_file():
        path.write_text(
            path.read_text()
            + "\n## Run-specific caveats\n\n"
            + caveats.read_text().strip()
            + "\n"
        )
    return path


def plot_publication_figures(report: dict[str, Any], out_dir: Path) -> list[str]:
    """Baseline equity overlay, ablation bars, regime bars — screen-first academic style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.reporting.book_style import (
        C_ACCENT,
        C_BLUE,
        C_GRAY,
        C_NAVY,
        C_NEG,
        C_POS,
        C_STEEL,
        C_ZERO,
        WIDTH_DOUBLE_IN,
        WIDTH_SINGLE_IN,
        apply_academic_rc,
        save_figure,
    )

    apply_academic_rc()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # --- 30: baseline cumulative PnL (strategy peers) ---
    ho = report.get("historical_oos") or {}
    base = report.get("baselines") or {}
    series = {}
    if (ho.get("pnls") or {}).get("happo"):
        series["happo"] = np.cumsum(np.asarray(ho["pnls"]["happo"], dtype=np.float64))
    for m, xs in (base.get("pnls") or {}).items():
        if xs:
            series[m] = np.cumsum(np.asarray(xs, dtype=np.float64))
    if series:
        fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE_IN, 4.2))
        colors = {
            "happo": C_NAVY,
            "short_vol_carry": C_ACCENT,
            "garch_vol_timing": C_BLUE,
            "heston_iv_momentum": C_STEEL,
            "goyal_saretto_hv_iv": C_POS,
            "iv_rank_timing": "#6B4C9A",
            "timed_long_gamma": "#C47B2C",
            "skew_risk_reversal": C_NEG,
            "cao_han_high_ivol": "#3D7A6A",
        }
        for name, eq in series.items():
            lw = 2.0 if name == "happo" else 1.3
            ax.plot(eq, label=name, color=colors.get(name, C_GRAY), lw=lw)
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_title("Historical OOS equity vs strategy peers (ATM-mid proxy)")
        ax.set_xlabel("Day index")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(loc="best", fontsize=7, ncol=2)
        p = out_dir / "30_baseline_equity.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

    # --- 30c: baseline Sharpe bars (readable when equity curves clutter) ---
    base_sm = (base.get("summary") or {}) if base else {}
    if base_sm or (ho.get("summary") or {}).get("happo"):
        fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE_IN, 3.8))
        labels: list[str] = []
        vals: list[float] = []
        cols: list[str] = []
        happo_sh = float(
            ((ho.get("summary") or {}).get("happo") or {}).get("sharpe", float("nan"))
        )
        labels.append("happo")
        vals.append(happo_sh)
        cols.append(C_NAVY)
        peer_colors = {
            "short_vol_carry": C_ACCENT,
            "garch_vol_timing": C_BLUE,
            "heston_iv_momentum": C_STEEL,
            "goyal_saretto_hv_iv": C_POS,
            "iv_rank_timing": "#6B4C9A",
            "timed_long_gamma": "#C47B2C",
            "skew_risk_reversal": C_NEG,
            "cao_han_high_ivol": "#3D7A6A",
        }
        for name in (base.get("modes") or list(base_sm.keys())):
            sm = base_sm.get(name) or {}
            if sm.get("unavailable"):
                continue
            labels.append(str(name))
            vals.append(float(sm.get("sharpe", float("nan"))))
            cols.append(peer_colors.get(name, C_GRAY))
        plot_vals = [0.0 if not np.isfinite(v) else v for v in vals]
        ax.bar(range(len(labels)), plot_vals, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_ylabel("OOS Sharpe (ATM-mid)")
        ax.set_title("Strategy peers — OOS Sharpe bars")
        p = out_dir / "30c_baseline_sharpe_bars.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

    # --- 30b: orientation Sharpe bars (cash mean_ann shown as note; equity Sharpe) ---
    orient = report.get("orientation_benchmarks") or {}
    lead = report.get("orientation_lead") or {}
    if orient.get("summary") or lead:
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 3.6))
        labels = ["HAPPO", "equity"]
        vals = [
            float(lead.get("happo_sharpe", float("nan"))),
            float(
                (orient.get("summary") or {})
                .get("equity_market", {})
                .get("sharpe", lead.get("equity_market_sharpe", float("nan")))
            ),
        ]
        cols = [C_NAVY, C_ACCENT]
        plot_vals = [0.0 if not np.isfinite(v) else v for v in vals]
        bars = ax.bar(labels, plot_vals, color=cols)
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                bar.set_height(0.0)
        cash_ann = float(lead.get("cash_rf_mean_ann", float("nan")))
        if np.isfinite(cash_ann):
            ax.set_title(f"Orientation Sharpes (cash mean_ann={cash_ann:.3f})")
        else:
            ax.set_title("Orientation: HAPPO vs equity Sharpe")
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_ylabel("Annualized Sharpe")
        p = out_dir / "30b_orientation_sharpe.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

    # --- 31: ablation sharpe bars ---
    abl = report.get("ablations") or {}
    rows = abl.get("rows") or []
    if rows:
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 4.0))
        labels = [r.get("spec", {}).get("id", "?") for r in rows]
        sharpes = [float(r.get("sharpe", float("nan"))) for r in rows]
        cols = [C_NAVY if lab == "full" else C_STEEL for lab in labels]
        ax.bar(range(len(labels)), sharpes, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_ylabel("OOS Sharpe")
        ax.set_title("Leave-one-component-out ablations")
        p = out_dir / "31_ablation_sharpe.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

        if any("positive_fold_rate" in r for r in rows):
            fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 4.0))
            rates = [float(r.get("positive_fold_rate", float("nan"))) for r in rows]
            ax.bar(range(len(labels)), rates, color=cols)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.axhline(0.7, color=C_ACCENT, ls="--", lw=1.0, label="gate 0.70")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("positive_fold_rate")
            ax.set_title("Ablation nested-fold positive rate")
            ax.legend(fontsize=8)
            p = out_dir / "32_ablation_pos_fold.png"
            save_figure(fig, p)
            plt.close(fig)
            written.append(str(p))

    # --- 33: regime sharpe ---
    reg = (report.get("regime_performance") or {}).get("regimes") or []
    if reg:
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 4.0))
        labs = []
        sh = []
        cols = []
        for r in reg:
            lab = str(r.get("label", r.get("id")))
            if r.get("available") is False or r.get("status") == "unavailable":
                labs.append(f"{lab}\n(N/A)")
                sh.append(float("nan"))
                cols.append(C_ZERO)
            else:
                labs.append(lab)
                v = float(r.get("sharpe", float("nan")))
                sh.append(v)
                cols.append(C_POS if (np.isfinite(v) and v > 0) else C_NEG)
        # matplotlib skips NaN bars — draw 0-height hatch markers for N/A instead.
        plot_vals = [0.0 if not np.isfinite(v) else v for v in sh]
        bars = ax.bar(range(len(labs)), plot_vals, color=cols)
        for bar, v in zip(bars, sh):
            if not np.isfinite(v):
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                bar.set_height(0.0)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.axhline(0.0, color=C_ZERO, lw=0.7)
        ax.set_ylabel("Sharpe")
        ax.set_title("Regime-sliced Sharpe (N/A = no panel overlap)")
        p = out_dir / "33_regime_sharpe.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

    # --- 34: DSR gauge-ish bar ---
    dsr = report.get("deflated_sharpe") or {}
    if dsr:
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 3.2))
        vals = [float(dsr.get("psr", float("nan"))), float(dsr.get("dsr", float("nan")))]
        ax.bar(["PSR", "DSR"], vals, color=[C_BLUE, C_NAVY])
        ax.axhline(0.95, color=C_ACCENT, ls="--", lw=1.0, label="0.95")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_title(
            f"Bailey–LdP rigor (n_trials={dsr.get('n_trials')})"
        )
        ax.legend(fontsize=8)
        p = out_dir / "34_psr_dsr.png"
        save_figure(fig, p)
        plt.close(fig)
        written.append(str(p))

    (out_dir / "publication_figures.json").write_text(
        json.dumps({"written": written}, indent=2) + "\n"
    )
    return written
