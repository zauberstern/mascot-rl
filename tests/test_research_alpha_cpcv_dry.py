"""Research CPCV dry-run schema + baseline peers (Slices D/E)."""
from __future__ import annotations

import math

import numpy as np

from mascotrl.eval.research_alpha_baselines import (
    equal_weight_sharpe,
    random_baseline_sharpe,
    research_baselines_from_returns,
    sign_lag_return_sharpe,
)
from mascotrl.eval.research_alpha_cpcv import dry_run_research_alpha_cpcv
from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha


def test_baselines_finite_on_toy_panel() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=(80, 5))
    peers = research_baselines_from_returns(rets, seed=0)
    for name in ("random", "equal_weight", "sign_lag"):
        assert name in peers
        assert math.isfinite(float(peers[name]["sharpe"]))


def test_baseline_helpers_match_names() -> None:
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=(40, 3))
    assert math.isfinite(random_baseline_sharpe(rets, seed=1))
    assert math.isfinite(equal_weight_sharpe(rets))
    assert math.isfinite(sign_lag_return_sharpe(rets))


def test_dry_run_emits_schema_keys() -> None:
    art = dry_run_research_alpha_cpcv(
        {
            "claim_tier": "research",
            "headline_fill": "pct75",
            "om_touch_enabled": True,
            "primary_train": "historical_arm_env",
        }
    )
    for key in (
        "path_summary",
        "random_baseline_sharpe",
        "baselines",
        "fill_ladder",
        "headline_fill",
        "train_objective_equals_claim_metric",
        "friction_applied",
        "claim_tier",
        "spa_polarity",
    ):
        assert key in art, key
    assert art["headline_fill"] == "pct75"
    assert art["spa_polarity"] == "policy_as_challenger"
    assert "mid" in art["fill_ladder"] and "pct75" in art["fill_ladder"]


def test_dry_run_refuses_mid_only_headline() -> None:
    try:
        dry_run_research_alpha_cpcv({"headline_fill": "mid", "claim_tier": "research"})
        raised = False
    except ValueError as exc:
        raised = True
        assert "pct75" in str(exc).lower() or "mid" in str(exc).lower()
    assert raised


def test_dry_run_stamp_path_does_not_unlock_capital() -> None:
    art = dry_run_research_alpha_cpcv(
        {
            "claim_tier": "research",
            "headline_fill": "pct75",
            "primary_train": "historical_arm_env",
            "om_touch_enabled": True,
        }
    )
    stamped = stamp_research_positive_alpha(art)
    # Dry-run uses null/toy sharpes — must not false-green research seal.
    assert stamped["research_positive_alpha"] is False


def test_kill_when_policy_not_above_random() -> None:
    from mascotrl.eval.research_alpha_baselines import policy_beats_random

    assert policy_beats_random(0.1, 0.2) is False
    assert policy_beats_random(0.3, 0.2) is True
    assert policy_beats_random(float("nan"), 0.0) is False
