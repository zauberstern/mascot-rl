"""HAC inference, Romano-Wolf stepdown, and path-based CSCV (W6)."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.stats_inference import (
    cscv_pbo_from_paths,
    hac_mean_tstat,
    hac_sharpe_se,
    newey_west_lag,
    romano_wolf_stepdown,
)
from src.eval.publication import estimate_n_trials, executed_trial_count


# ------------------------------------------------------------------------- HAC

def test_newey_west_lag_rule():
    # floor(4 * (n/100)^(2/9))
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1) == 0
    assert newey_west_lag(1000) >= 4


def test_hac_equals_iid_for_white_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000) * 0.01
    out = hac_mean_tstat(x)
    # No autocorrelation -> HAC and iid standard errors should be close.
    assert out["se_hac"] == pytest.approx(out["se_iid"], rel=0.20)


def test_hac_se_exceeds_iid_under_positive_autocorrelation():
    """Persistent positions inflate the true standard error; HAC must catch it."""
    rng = np.random.default_rng(1)
    n = 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    out = hac_mean_tstat(x)
    assert out["se_hac"] > out["se_iid"] * 1.5
    assert abs(out["t_hac"]) < abs(out["t_iid"])


def test_hac_reports_lag_rule_and_citation():
    out = hac_mean_tstat(np.random.default_rng(2).standard_normal(500))
    assert "Newey" in out["lag_rule"]
    assert "Newey and West" in out["citation"]
    assert out["kernel"] == "bartlett"


def test_hac_handles_degenerate_series():
    out = hac_mean_tstat([0.01, 0.01])
    assert out["n"] == 2
    assert np.isnan(out["se_hac"])


def test_hac_sharpe_reports_inflation_factor():
    rng = np.random.default_rng(3)
    n = 2000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    out = hac_sharpe_se(x + 0.05)
    assert out["hac_inflation_factor"] > 1.0
    assert np.isfinite(out["sharpe_annual"])
    assert abs(out["t_sharpe_hac"]) < abs(out["t_iid"])


def test_white_reality_check_rejects_clear_winner():
    from src.eval.stats_inference import white_reality_check

    rng = np.random.default_rng(10)
    n = 300
    bench = rng.standard_normal(n) * 0.01
    rivals = {"strong": bench + 0.008, "weak": bench - 0.005}
    out = white_reality_check(bench, rivals, n_boot=99, seed=0)
    assert out["ok"] is True
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["best_rival"] == "strong"


# ---------------------------------------------------------------- Romano-Wolf

def test_romano_wolf_rejects_a_clearly_superior_rival():
    rng = np.random.default_rng(4)
    n = 600
    bench = rng.standard_normal(n) * 0.01
    rivals = {
        "strong": bench + 0.01,            # deterministic outperformance
        "same": bench.copy(),
        "weak": bench - 0.01,
    }
    out = romano_wolf_stepdown(bench, rivals, n_boot=199, seed=0)
    assert "strong" in out["rejected"]
    assert "weak" not in out["rejected"]


def test_romano_wolf_rejects_nothing_when_no_rival_is_better():
    rng = np.random.default_rng(5)
    n = 600
    bench = rng.standard_normal(n) * 0.01
    rivals = {f"r{i}": bench - 0.005 for i in range(4)}
    out = romano_wolf_stepdown(bench, rivals, n_boot=199, seed=0)
    assert out["rejected"] == []


def test_romano_wolf_controls_familywise_error_with_many_null_rivals():
    """With all rivals null, familywise rejections must be rare."""
    rng = np.random.default_rng(6)
    n = 400
    bench = rng.standard_normal(n) * 0.01
    rivals = {f"r{i}": rng.standard_normal(n) * 0.01 for i in range(10)}
    out = romano_wolf_stepdown(bench, rivals, n_boot=199, seed=1, alpha=0.05)
    assert out["n_rivals"] == 10
    # Not a strict guarantee on one draw, but a blatant failure would reject many.
    assert out["n_rejected"] <= 2


def test_romano_wolf_reports_adjusted_pvalues_and_citation():
    rng = np.random.default_rng(7)
    bench = rng.standard_normal(300) * 0.01
    out = romano_wolf_stepdown(bench, {"a": bench + 0.005}, n_boot=99, seed=0)
    assert "Romano and Wolf" in out["citation"]
    assert all("p_adjusted" in r for r in out["results"])
    assert all(0.0 <= r["p_adjusted"] <= 1.0 for r in out["results"])


def test_romano_wolf_skips_misaligned_rivals():
    bench = np.zeros(100)
    out = romano_wolf_stepdown(bench, {"short": [0.0] * 50}, n_boot=49)
    assert "short" in out["skipped_misaligned"]


def test_romano_wolf_over_panel_uses_happo_minus_bench_diffs():
    from src.eval.stats_inference import romano_wolf_over_panel

    rng = np.random.default_rng(11)
    n = 200
    weak = rng.standard_normal(n) * 0.01
    happo = weak + 0.01
    panel = {
        "weak": weak,
        "same": happo.copy(),
    }
    out = romano_wolf_over_panel(happo, panel, n_boot=99, seed=0)
    assert out["protocol"] == "romano_wolf_over_panel"
    assert out["diff_definition"] == "HAPPO - bench_j"
    assert out["claimant"] == "happo"
    assert "weak" in out["rejected"]


# ------------------------------------------------------------------ CSCV paths

def test_cscv_pbo_low_for_genuinely_persistent_winner():
    rng = np.random.default_rng(8)
    T = 500
    # One path is genuinely better throughout; PBO should be low.
    paths = [rng.standard_normal(T) * 0.01 for _ in range(5)]
    paths[0] = paths[0] + 0.01
    out = cscv_pbo_from_paths(paths, seed=0)
    assert out["is_proxy"] is False
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo"] < 0.3


def test_cscv_pbo_high_when_ranking_is_pure_noise():
    rng = np.random.default_rng(9)
    paths = [rng.standard_normal(400) * 0.01 for _ in range(8)]
    out = cscv_pbo_from_paths(paths, seed=0)
    assert out["pbo"] > 0.2


def test_cscv_requires_multiple_paths():
    out = cscv_pbo_from_paths([np.zeros(100)])
    assert np.isnan(out["pbo"])
    assert "need" in out["reason"]


def test_cscv_runs_on_return_paths_not_trial_sharpes():
    out = cscv_pbo_from_paths(
        [np.random.default_rng(i).standard_normal(300) * 0.01 for i in range(4)]
    )
    assert out["protocol"] == "cscv_on_return_paths"
    assert out["n_obs"] == 300


# ------------------------------------------------------------ trial accounting

def test_executed_trial_count_reads_the_ledger():
    report = {
        "trial_ledger": {
            "trials": [
                {"source": "ablation", "id": "a1"},
                {"source": "ablation", "id": "a2"},
                {"source": "nested_fold", "id": "f0"},
            ]
        },
        "cpcv": {"path_summary": {"n_paths": 5}},
    }
    n, meta = executed_trial_count(report)
    assert n == 8
    assert meta["by_source"]["ablation"] == 2
    assert meta["by_source"]["cpcv_paths"] == 5
    assert meta["auditable"] is True


def test_estimate_n_trials_prefers_executed_ledger_over_proxy():
    report = {"trial_ledger": {"trials": [{"source": "ablation"}] * 7}}
    n, meta = estimate_n_trials(report, {})
    assert n == 7
    assert meta["source"] == "executed_trial_ledger"
    assert meta["auditable"] is True


def test_estimate_n_trials_labels_the_combinatorial_fallback_as_unauditable():
    n, meta = estimate_n_trials({}, {})
    assert n > 1
    assert meta["source_kind"] == "combinatorial_upper_bound"
    assert meta["auditable"] is False
    assert "not the executed search" in meta["caveat"]


def test_explicit_override_still_wins():
    report = {"trial_ledger": {"trials": [{"source": "ablation"}] * 7}}
    n, meta = estimate_n_trials(report, {"publication_n_trials": 999})
    assert n == 999
