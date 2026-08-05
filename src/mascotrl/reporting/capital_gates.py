"""Capital-path hygiene: provenance, known gaps, capacity, capital gates."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Protocols that may appear on a capital-facing verdict line.
CAPITAL_GRADE_PROTOCOLS = frozenset(
    {
        "pit_optionmetrics_atm_is_oos",
        "pit_optionmetrics_nested_wfo",
        "pit_optionmetrics_nested_wfo_retrain",
    }
)

SYNTHETIC_PROTOCOLS = frozenset(
    {
        "synthetic_rbergomi_holdout",
        "synthetic_rbergomi_train",
    }
)

# Production CMDP projection ceiling (see scripts/bench_projection_k.py).
PROJECTION_K_CEILING = 50
PROJECTION_K_BREAK_HINT = 75


KNOWN_UNMODELED_RISKS: list[dict[str, str]] = [
    {
        "id": "sim_train_rbergomi",
        "severity": "high",
        "summary": (
            "Primary overnight training trajectories are rBergomi+Dupire synthetic "
            "surfaces; nested WFO retrain fine-tunes on OptionMetrics ATM marks."
        ),
    },
    {
        "id": "no_venue_fragmentation",
        "severity": "medium",
        "summary": "Single mid/mark model; no NBBO, auction, or venue routing costs.",
    },
    {
        "id": "no_corporate_actions",
        "severity": "medium",
        "summary": "Splits/dividends/M&A not simulated; hist panel assumes clean ATM mids.",
    },
    {
        "id": "borrow_funding_omitted",
        "severity": "high",
        "summary": (
            "Default status-quo reward omits stock-borrow / funding. Opt-in "
            "plugins.funding.enabled adds a documented GC-borrow proxy "
            "(see docs/PLUGIN_FEATURES.md); do not clear this ID unless the "
            "claiming run had funding enabled."
        ),
    },
    {
        "id": "execution_cost_in_train_reward",
        "severity": "info",
        "summary": (
            "Small linear spread + √turnover impact are added to R_t alongside hard CMDP "
            "τ/δ constraints (not a soft substitute for the projection). "
            "plugins.execution_drag_mode=vol_scaled scales drag by σ/σ_ref."
        ),
    },
    {
        "id": "single_book_k_limit",
        "severity": "medium",
        "summary": (
            f"PRODUCTION CEILING: K≤{PROJECTION_K_CEILING}. Exact cvxpylayers KKT is the "
            "architecture choice for the single-book research system; multi-book / K≫50 "
            "needs SCS/ADMM or approximate projection before any scale-up "
            "(plugins.projection_backend=admm|multibook_cvxpy — multibook is exact "
            "per book only, not joint). "
            "Published curve: scripts/bench_projection_k.py → logs/projection_k_benchmark.json "
            "and per-run report/projection_k_benchmark.json."
        ),
    },
    {
        "id": "single_universe_us_optionmetrics",
        "severity": "high",
        "summary": (
            "Empirical evidence is OptionMetrics US equity-option ATM marks on one "
            "copula-tail universe. No second-index (Nasdaq/Europe) panel is materialized "
            "in-lake; external validity requires a separate universe rematerialize + "
            "zero-shot transfer eval (scripts/eval_zero_shot.py) when additional "
            "OptionMetrics coverage exists."
        ),
    },
    {
        "id": "regime_window_requires_panel_overlap",
        "severity": "medium",
        "summary": (
            "Publication regime rows (2008 GFC / 2020 COVID / 2022 hike) require the "
            "PIT hist panel (hist_panel_start→oos_end) to overlap each window. Lake VIX "
            "and options exist from 2003; if a window still has zero days it must be "
            "reported as N/A — data unavailable (never a silent zero Sharpe)."
        ),
    },
    {
        "id": "lake_vix_duplicate_event_dates",
        "severity": "low",
        "summary": (
            "cboe_vix.parquet contained duplicate rows for 2003-09-22 and 2003-10-30 "
            "(same close, divergent OHLC highs — vendor restatement/correction pattern). "
            "DuckDB compute_macro_state ROW_NUMBER-dedupes by date; Arctic persist/read_ffill "
            "also drop duplicate event-time labels. A lake rebuild without those guards "
            "would again break PIT reindex."
        ),
    },
    {
        "id": "shadow_book_mvp_only",
        "severity": "high",
        "summary": (
            "No live paper shadow-book or reconcile tooling exists in this "
            "repository. Multi-week live-feed shadowing is an ops process, not "
            "completed by a single overnight train. This repo emits no "
            "capital-allocation claim fields."
        ),
    },
    {
        "id": "no_broker_oms",
        "severity": "critical",
        "summary": "No production broker/OMS adapter; kill-switch + reconcile are research MVPs.",
    },
    {
        "id": "tau_not_learnable",
        "severity": "info",
        "summary": (
            "Turnover limit τ is a declared CMDP constraint (config / CVXPY "
            "Parameter), not an nn.Parameter, by design: a free learnable "
            "budget would let the agent optimise away its own constraint. "
            "Optional tau_schedule anneals toward the hard cap; "
            "turnover_cap_binding_fraction measures whether the cap binds."
        ),
    },
]


def write_known_unmodeled_risks(run_dir: Path) -> Path:
    """Attach a capital-risk disclosure to every run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "KNOWN_UNMODELED_RISKS.md"
    lines = [
        "# Known unmodeled risks",
        "",
        "Lab Sharpe ≠ tradable Sharpe. This file is attached so capital decisions",
        "cannot ignore structural gaps in the research stack.",
        "",
        f"**Projection production ceiling:** K ≤ {PROJECTION_K_CEILING} "
        f"(do not scale past K≈{PROJECTION_K_BREAK_HINT} without solver redesign).",
        "",
    ]
    for row in KNOWN_UNMODELED_RISKS:
        lines.append(f"## `{row['id']}` ({row['severity']})")
        lines.append("")
        lines.append(row["summary"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    (run_dir / "known_unmodeled_risks.json").write_text(
        json.dumps(KNOWN_UNMODELED_RISKS, indent=2) + "\n"
    )
    return path


@dataclass
class ModelCard:
    run_label: str
    run_dir: str
    eval_protocol: str
    n_assets: int
    seed: int | None
    train_episodes: int | None
    architecture_notes: list[str] = field(default_factory=list)
    known_limits: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    train_distribution: str = "rbergomi_dupire"

    def write(self, run_dir: Path) -> Path:
        run_dir = Path(run_dir)
        path = run_dir / "MODEL_CARD.md"
        lines = [
            f"# Model card — `{self.run_label}`",
            "",
            f"- **run_dir:** `{self.run_dir}`",
            f"- **eval_protocol:** `{self.eval_protocol}`",
            f"- **train_distribution:** `{self.train_distribution}`",
            f"- **n_assets (K):** {self.n_assets}",
            f"- **seed:** {self.seed}",
            f"- **train_episodes:** {self.train_episodes}",
            "",
            "## Reproducibility fingerprint",
            "",
            "```json",
            json.dumps(self.fingerprint, indent=2, default=str),
            "```",
            "",
            "## Architecture notes",
            "",
        ]
        for n in self.architecture_notes:
            lines.append(f"- {n}")
        lines.extend(["", "## Known limits", ""])
        for n in self.known_limits:
            lines.append(f"- {n}")
        lines.extend(["", "## Headline metrics", "", "```json"])
        lines.append(json.dumps(self.metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")
        path.write_text("\n".join(lines) + "\n")
        (run_dir / "model_card.json").write_text(
            json.dumps(asdict(self), indent=2, default=str) + "\n"
        )
        return path


def collect_reproducibility_fingerprint(
    *,
    root: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Git commit, config hash, dependency pin snapshot for audit trails."""
    import hashlib
    import platform
    import subprocess

    root = Path(root or Path.cwd())
    fp: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        fp["torch"] = None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        fp["git_commit"] = out.decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        fp["git_dirty"] = bool(dirty != 0)
    except Exception:
        fp["git_commit"] = None
        fp["git_dirty"] = None
    pyproj = root / "pyproject.toml"
    if pyproj.is_file():
        fp["pyproject_sha256"] = hashlib.sha256(pyproj.read_bytes()).hexdigest()[:16]
    if cfg:
        blob = json.dumps(cfg, sort_keys=True, default=str).encode()
        fp["config_sha256"] = hashlib.sha256(blob).hexdigest()[:16]
        fp["seed"] = cfg.get("seed")
        fp["temporal_backend"] = cfg.get("temporal_backend", "mamba")
        fp["use_dhgnn"] = cfg.get("use_dhgnn", True)
        fp["use_projection"] = cfg.get("use_projection", True)
    return fp


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def arctic_provenance_ok(report: Mapping[str, Any]) -> bool:
    """True if claim path has knowledge-time as_of or single-write lake attestation."""
    if report.get("arctic_as_of") not in (None, "", "None"):
        return True
    if report.get("lake_revision_hash") not in (None, "", "None"):
        return True
    if bool(report.get("single_write_immutable_lake")) and report.get(
        "lake_checksum"
    ) not in (None, "", "None"):
        return True
    return False


def default_estimand_residuals(report: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Default residual disclosure for delta-hedged option allocator / CPCV claim packs."""
    report = report or {}
    funding_on = bool(
        ((report.get("plugins") or {}).get("funding") or {}).get("enabled")
    ) or bool(report.get("funding_enabled"))
    return {
        "american_residual": str(report.get("american_residual") or "disclosed"),
        "borrow_state": "enabled" if funding_on else "omitted",
        "eod_mid_sensitivity_id": str(
            report.get("eod_mid_sensitivity_id")
            or (report.get("eod_mid_sensitivity") or {}).get("artifact_id")
            or "none"
        ),
        "fill_model": str(report.get("fill_model") or "om_touch"),
    }


# @lat: [[invariants#Capital-grade]]
def assert_protocol_provenance(
    report: dict[str, Any],
    *,
    require_stability_gates: bool | None = None,
    wfo_positive_fold_rate_min: float | None = None,
    multiseed_sharpe_p05_min: float | None = None,
    adversarial_max_degradation: float | None = None,
    require_retrain_wfo: bool | None = None,
    require_factor_alpha: bool | None = None,
    require_after_cost: bool | None = None,
    min_break_even_multiplier: float | None = None,
) -> dict[str, Any]:
    """
    Hard-gate capital claims on protocol provenance **and** stability quality.

    Synthetic holdout can still be logged, but cannot silently become the claim.
    When stability gates are required, nested WFO / multi-seed / IV-stress must
    clear pre-committed thresholds (not merely be present as report fields).

    Two further gates reflect what this literature now requires: a positive
    factor-adjusted alpha (Goyal and Saretto 2024) and survival of a non-trivial
    share of the quoted half-spread (O'Donovan and Yu 2025).
    """
    protocol = str(report.get("eval_protocol") or "").strip()
    if not protocol:
        raise RuntimeError(
            "eval_protocol missing — refuse to write run_stats without provenance"
        )

    from mascotrl.reporting.claim_language import stamp_dh_option_allocator_claim_category
    from mascotrl.reporting.claim_stamps import apply_dh_option_allocator_claim_gates

    report = stamp_dh_option_allocator_claim_category(report)

    if require_stability_gates is None:
        require_stability_gates = bool(report.get("capital_gates_require_stability", True))
    if wfo_positive_fold_rate_min is None:
        wfo_positive_fold_rate_min = float(
            report.get("wfo_positive_fold_rate_min", 0.70)
        )
    if multiseed_sharpe_p05_min is None:
        multiseed_sharpe_p05_min = float(report.get("multiseed_sharpe_p05_min", 0.0))
    if adversarial_max_degradation is None:
        adversarial_max_degradation = float(
            report.get("adversarial_max_degradation", 0.50)
        )
    if require_retrain_wfo is None:
        require_retrain_wfo = bool(report.get("capital_gates_require_retrain_wfo", True))
    if require_factor_alpha is None:
        require_factor_alpha = bool(
            report.get("capital_gates_require_factor_alpha", True)
        )
    if require_after_cost is None:
        require_after_cost = bool(report.get("capital_gates_require_after_cost", True))
    if min_break_even_multiplier is None:
        min_break_even_multiplier = float(
            report.get("min_break_even_spread_multiplier", 0.25)
        )

    synthetic_alpha = report.get("alpha_found_synthetic_holdout")
    hist_alpha = None
    ho = report.get("historical_oos") or {}
    if isinstance(ho, dict) and "alpha_found_historical" in ho:
        hist_alpha = ho.get("alpha_found_historical")
    elif "alpha_found_historical" in report:
        hist_alpha = report.get("alpha_found_historical")

    failures: list[str] = []
    capital_ok = protocol in CAPITAL_GRADE_PROTOCOLS and hist_alpha is True
    if protocol not in CAPITAL_GRADE_PROTOCOLS:
        failures.append(f"protocol_not_capital_grade:{protocol}")
    if hist_alpha is not True:
        failures.append("historical_alpha_missing_or_false")

    nested = report.get("nested_wfo") or {}
    multiseed = report.get("multiseed_oos") or {}
    adv = report.get("adversarial_iv_stress") or {}

    if require_stability_gates:
        if require_retrain_wfo and protocol != "pit_optionmetrics_nested_wfo_retrain":
            # Allow claim only when retrain WFO ran (even if protocol tag lagged).
            if str(nested.get("mode") or "") != "retrain_per_fold":
                capital_ok = False
                failures.append("nested_wfo_retrain_required")

        pos_rate = nested.get("positive_fold_rate")
        if not _finite(pos_rate) or float(pos_rate) < float(wfo_positive_fold_rate_min):
            capital_ok = False
            failures.append(
                f"wfo_positive_fold_rate<{wfo_positive_fold_rate_min} (got {pos_rate})"
            )

        p05 = multiseed.get("sharpe_p05")
        if not _finite(p05) or float(p05) <= float(multiseed_sharpe_p05_min):
            capital_ok = False
            failures.append(
                f"multiseed_sharpe_p05<={multiseed_sharpe_p05_min} (got {p05})"
            )

        deg = adv.get("sharpe_degradation")
        fragile = bool(adv.get("fragile"))
        if fragile or (
            _finite(deg) and float(deg) > float(adversarial_max_degradation)
        ):
            capital_ok = False
            failures.append(
                f"adversarial_iv_fragile_or_deg>{adversarial_max_degradation} (got {deg})"
            )
        if not adv:
            capital_ok = False
            failures.append("adversarial_iv_stress_missing")

    # Factor-adjusted alpha gate. Goyal and Saretto (2024, RFS 38(6)) and the
    # Dallas Fed IPCA study (WP 2214) both find factor models absorb apparent
    # equity-option alpha, so beating zero/random cannot carry a capital claim.
    fa = (report.get("factor_alpha") or {}).get("alpha") or {}
    if require_factor_alpha:
        if not fa:
            capital_ok = False
            failures.append("factor_alpha_missing")
        elif not bool(fa.get("alpha_significant_05")):
            capital_ok = False
            failures.append(
                "factor_adjusted_alpha_not_significant "
                f"(t_hac={fa.get('alpha_t_hac')})"
            )

    # After-cost gate: the edge must survive a non-zero share of the quoted
    # half-spread. O'Donovan and Yu (2025) find no published delta-hedged option
    # sort survives full costs unmitigated, so a gross-only claim is untenable.
    ladder = report.get("cost_ladder") or {}
    if require_after_cost:
        be = ladder.get("break_even_spread_multiplier")
        if not ladder:
            capital_ok = False
            failures.append("cost_ladder_missing")
        elif not (_finite(be) and float(be) >= float(min_break_even_multiplier)):
            capital_ok = False
            failures.append(
                f"break_even_spread_multiplier<{min_break_even_multiplier} (got {be})"
            )

    # DSR / rival-Sharpe hygiene (Moody differential Sharpe; AFML DSR; AlphaStock).
    # Soft overnight reports previously allowed capital while DSR failed — block
    # that false comfort when fields are present. Random is not an alpha peer;
    # gate vs best rich BASELINE_NAMES (critique 07).
    require_dsr = bool(report.get("capital_gates_require_dsr", True))
    dsr = report.get("deflated_sharpe_oos") or report.get("deflated_sharpe_pooled") or {}
    if require_dsr and dsr:
        if dsr.get("significant_05") is False:
            capital_ok = False
            failures.append("dsr_not_significant_05")
    require_sharpe_vs_best_baseline = bool(
        report.get(
            "capital_gates_require_sharpe_vs_best_baseline",
            report.get("capital_gates_require_sharpe_vs_random", True),
        )
    )
    if require_sharpe_vs_best_baseline:
        from mascotrl.eval.alpha_gates import pick_best_baseline

        ho_sum = (ho.get("summary") or {}) if isinstance(ho, dict) else {}
        h_sh = (ho_sum.get("happo") or {}).get("sharpe")
        h_mu = (ho_sum.get("happo") or {}).get("mean_pnl")
        best_name, best_sm = pick_best_baseline(report.get("baselines"))
        if best_name is None or best_sm is None:
            capital_ok = False
            failures.append("best_baseline_missing")
        else:
            b_sh = best_sm.get("sharpe")
            b_mu = best_sm.get("mean_pnl")
            if _finite(h_sh) and _finite(b_sh) and float(h_sh) <= float(b_sh):
                capital_ok = False
                failures.append(
                    f"oos_sharpe_not_above_best_baseline "
                    f"(happo={h_sh}, {best_name}={b_sh})"
                )
            edge = report.get("edge_vs_best_baseline")
            if edge is None and _finite(h_mu) and _finite(b_mu):
                edge = float(h_mu) - float(b_mu)
            if _finite(edge) and float(edge) <= 0.0:
                capital_ok = False
                failures.append(
                    f"edge_vs_best_baseline_nonpositive (edge={edge}, best={best_name})"
                )
        if isinstance(ho, dict) and ho.get("sharpe_beats_best_baseline") is False:
            capital_ok = False
            if not any("best_baseline" in f for f in failures):
                failures.append("sharpe_beats_best_baseline_false")

    # Pack Gate1/2 + framing SoT (A1): soft overnight cannot override publication pack.
    require_pack = bool(report.get("capital_gates_require_pack_gates", True))
    pack = (
        report.get("publication_evidence_pack")
        or report.get("publication_evidence")
        or {}
    )
    if require_pack and isinstance(pack, dict) and pack:
        if pack.get("framing") == "pivot_negative_economic_framing":
            capital_ok = False
            failures.append("pack_pivot_negative_economic_framing")
        # Legacy pack field: refuse if an old artifact still claims True.
        if pack.get("capital_claim_allowed") is True:
            capital_ok = False
            failures.append("legacy_pack_capital_claim_true_refused")
        gates = pack.get("gates") or {}
        g1 = gates.get("gate1") if isinstance(gates, dict) else None
        if isinstance(g1, dict) and g1.get("pass") is False:
            capital_ok = False
            failures.append("pack_gate1_fail")
        g2h = gates.get("gate2_happo") if isinstance(gates, dict) else None
        if isinstance(g2h, dict) and g2h.get("pass") is False:
            capital_ok = False
            failures.append("pack_gate2_happo_fail")

    # Spectrum: shared actors are disclosed stamps, not capital refusals.
    prov = report.get("algorithm_provenance") or {}
    plugins = report.get("plugins") or {}
    actor_backend = str(
        prov.get("actor_backend") or plugins.get("actor_backend") or "modulelist"
    )
    if actor_backend in ("hypernet", "shared", "shared_mappo"):
        report["shared_actor_disclosed"] = True
        report["actor_backend"] = actor_backend
    proj = str(
        prov.get("projection_backend") or plugins.get("projection_backend") or "cvxpy"
    )
    if proj == "admm" and bool(prov.get("admm_ste", True)):
        capital_ok = False
        failures.append("admm_ste_gradient_not_exact")
    if bool(prov.get("teamtr_enabled", False)):
        capital_ok = False
        failures.append("teamtr_enabled_non_kuba")
    spread = float(prov.get("execution_spread_bps", report.get("execution_spread_bps", 0.0)) or 0.0)
    impact = float(
        prov.get("execution_impact_coef", report.get("execution_impact_coef", 0.0)) or 0.0
    )
    if spread > 0.0 or impact > 0.0:
        honest = bool(report.get("reward_shaping_ablation", False))
        cost_in = bool(report.get("cost_in_decision", False))
        collapse_ok = bool((report.get("collapse_guard") or {}).get("ok", False))
        if not honest and not (cost_in and collapse_ok):
            capital_ok = False
            failures.append("reward_shaping_enabled_in_training")
    if prov and not bool(prov.get("truncation_bootstrap", True)):
        capital_ok = False
        failures.append("truncation_bootstrap_disabled")

    # Knowledge-time / lake provenance (R7): claim-grade paths must attest as_of
    # or a single-write immutable lake with checksum.
    if protocol in CAPITAL_GRADE_PROTOCOLS or protocol == "combinatorial_purged_cv":
        if not arctic_provenance_ok(report):
            capital_ok = False
            failures.append("arctic_as_of_or_single_write_attestation_required")

    # Estimand residual disclosure on claim paths
    residuals = report.get("estimand_residuals") or {}
    if protocol in CAPITAL_GRADE_PROTOCOLS or protocol == "combinatorial_purged_cv":
        if not residuals:
            report["estimand_residuals"] = default_estimand_residuals(report)
            residuals = report["estimand_residuals"]
        if str(residuals.get("american_residual") or "") not in ("disclosed", "priced"):
            capital_ok = False
            failures.append("american_residual_undisclosed")
        if str(residuals.get("borrow_state") or "") not in ("omitted", "enabled"):
            capital_ok = False
            failures.append("borrow_state_missing")

    # delta-hedged option allocator estimand / OM-touch claim / transfer / arm lock (hire dh_ret_lagdelta).
    n_fail_before = len(failures)
    apply_dh_option_allocator_claim_gates(report, failures)
    if len(failures) > n_fail_before:
        capital_ok = False

    # Spectrum honesty: primary objective must reach the actor; promotion needs
    # transfer_report + collapse_guard; DSR trial count must include spectrum cells.
    ogp = str(report.get("objective_gradient_path") or "")
    if bool(report.get("objective_primary") or (report.get("risk") or {}).get("objective_primary")):
        if ogp in ("", "critic_only"):
            capital_ok = False
            failures.append("objective_primary_claimed_but_critic_only")
    if bool(report.get("spectrum_promotable") or report.get("spectrum_cell_id")):
        if not isinstance(report.get("transfer_report"), dict):
            capital_ok = False
            failures.append("spectrum_promotion_without_transfer_report")
        cg = report.get("collapse_guard")
        if not isinstance(cg, dict) or not bool(cg.get("ok", False)):
            # Missing collapse guard blocks promotion; present-but-failed already
            # reflected in cg.ok.
            if not isinstance(cg, dict):
                capital_ok = False
                failures.append("spectrum_promotion_without_collapse_guard")
    dsr = report.get("deflated_sharpe_oos") or report.get("deflated_sharpe") or {}
    breakdown = (dsr.get("n_trials_breakdown") if isinstance(dsr, dict) else None) or {}
    if report.get("spectrum_cell_id") and isinstance(dsr, dict):
        # B-DSR: accept either ``cells`` or legacy ``n_cells`` key.
        cell_count = breakdown.get("cells", breakdown.get("n_cells"))
        if not cell_count:
            capital_ok = False
            failures.append("dsr_trial_count_excludes_spectrum_cells")

    from mascotrl.eval.collapse_guard import assert_collapse_guard_ok
    from mascotrl.eval.gate_ladder import run_gate_ladder

    ladder = run_gate_ladder(dict(report.get("bundle") or {}))
    report["gate_ladder"] = ladder
    if not bool(ladder.get("pass")):
        capital_ok = False
        failures.append("gate_ladder_failed")

    cg = report.get("collapse_guard")
    if isinstance(cg, dict):
        try:
            assert_collapse_guard_ok(cg)
        except ValueError:
            capital_ok = False
            failures.append("collapse_guard_failed")

    report.pop("capital_claim_allowed", None)
    report.pop("tradable_claim_allowed", None)
    report.pop("capital_claim_nuked", None)
    report["protocol_gate"] = {
        "eval_protocol": protocol,
        "is_synthetic_protocol": protocol in SYNTHETIC_PROTOCOLS,
        "is_capital_grade_protocol": protocol in CAPITAL_GRADE_PROTOCOLS,
        "alpha_found_synthetic_holdout": synthetic_alpha,
        "alpha_found_historical": hist_alpha,
        "alpha_found_reported": report.get("alpha_found"),
        "require_stability_gates": require_stability_gates,
        "wfo_positive_fold_rate_min": wfo_positive_fold_rate_min,
        "multiseed_sharpe_p05_min": multiseed_sharpe_p05_min,
        "adversarial_max_degradation": adversarial_max_degradation,
        "nested_wfo_positive_fold_rate": nested.get("positive_fold_rate"),
        "multiseed_sharpe_p05": multiseed.get("sharpe_p05"),
        "adversarial_sharpe_degradation": adv.get("sharpe_degradation"),
        "require_factor_alpha": bool(require_factor_alpha),
        "factor_adjusted_alpha_significant": bool(fa.get("alpha_significant_05")),
        "factor_alpha_t_hac": fa.get("alpha_t_hac"),
        "require_after_cost": bool(require_after_cost),
        "min_break_even_multiplier": float(min_break_even_multiplier),
        "break_even_spread_multiplier": ladder.get("break_even_spread_multiplier"),
        "require_dsr": bool(report.get("capital_gates_require_dsr", True)),
        "dsr_significant_05": dsr.get("significant_05") if dsr else None,
        "require_sharpe_vs_random": False,
        "require_sharpe_vs_best_baseline": bool(
            report.get(
                "capital_gates_require_sharpe_vs_best_baseline",
                report.get("capital_gates_require_sharpe_vs_random", True),
            )
        ),
        "require_pack_gates": bool(
            report.get("capital_gates_require_pack_gates", True)
        ),
        "gate_failures": failures,
        "protocol_hygiene_ok": bool(capital_ok),
    }
    return report


def capacity_curve_from_daily(
    step_pnls: list[float] | np.ndarray,
    turnovers: list[float] | np.ndarray,
    *,
    base_notional: float = 1.0,
    multipliers: list[float] | None = None,
    impact_coef: float = 0.01,
    spread_bps: float = 5.0,
) -> dict[str, Any]:
    """
    Post-hoc capacity curve: scale gross exposure by AUM multiplier and subtract
    square-root impact + linear spread on turnover.
    """
    pnl = np.asarray(step_pnls, dtype=np.float64)
    turn = np.asarray(turnovers, dtype=np.float64)
    if pnl.size == 0:
        return {"multipliers": [], "rows": []}
    if turn.size != pnl.size:
        turn = np.resize(turn, pnl.size)
    multipliers = multipliers or [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    spread = float(spread_bps) / 1e4
    rows = []
    for m in multipliers:
        impact = float(impact_coef) * np.sqrt(np.maximum(m * turn, 0.0))
        spread_cost = spread * m * turn
        adj = m * pnl - impact - spread_cost
        mu = float(adj.mean())
        sd = float(adj.std(ddof=0)) + 1e-12
        rows.append(
            {
                "aum_multiplier": float(m),
                "notional": float(base_notional * m),
                "mean_pnl": mu,
                "sharpe": float(mu / sd * np.sqrt(252.0)),
                "hit_rate": float((adj > 0).mean()),
                "mean_impact_drag": float(impact.mean()),
                "mean_spread_drag": float(spread_cost.mean()),
                "pnl_sum": float(adj.sum()),
            }
        )
    base = next((r for r in rows if abs(r["aum_multiplier"] - 1.0) < 1e-12), rows[0])
    base_sh = base["sharpe"]
    ceiling = base["aum_multiplier"]
    for r in rows:
        if base_sh <= 0:
            if r["sharpe"] > 0:
                ceiling = r["aum_multiplier"]
        elif r["sharpe"] >= 0.5 * base_sh:
            ceiling = r["aum_multiplier"]
    return {
        "impact_coef": float(impact_coef),
        "spread_bps": float(spread_bps),
        "base_notional": float(base_notional),
        "base_aum_usd": float(base_notional),
        "capacity_ceiling_multiplier": float(ceiling),
        "capacity_ceiling_aum_usd": float(base_notional * ceiling),
        "rows": rows,
        "note": (
            "Multipliers scale gross book vs base_aum_usd. K≤50 production ceiling is "
            "orthogonal: at base_aum_usd with K names, per-name notional ≈ "
            f"{base_notional / max(1, 50):.0f} USD (illustrative at K=50)."
        ),
    }


def adversarial_iv_stress_summary(
    base_sharpe: float,
    stressed_sharpe: float,
    *,
    shock: float,
) -> dict[str, Any]:
    """Crowding/decay proxy: Sharpe drop under ATM IV adversarial shock."""
    deg = float("nan")
    if np.isfinite(base_sharpe) and abs(base_sharpe) > 1e-12:
        deg = float((base_sharpe - stressed_sharpe) / abs(base_sharpe))
    return {
        "iv_shock": float(shock),
        "base_sharpe": float(base_sharpe),
        "stressed_sharpe": float(stressed_sharpe),
        "sharpe_degradation": deg,
        "fragile": bool(np.isfinite(deg) and deg > 0.5),
    }
