"""Part F: figure label registry."""
from __future__ import annotations

import os
import re

import pytest

from mascotrl.eval.benchmark_panel import BENCHMARK_PANEL_NAMES
from mascotrl.reporting.figures.labels import (
    ARCHETYPE_LABELS,
    AXIS_LABELS,
    METRIC_LABELS,
    SLEEVE_LABELS,
    STRATEGY_LABELS,
    human,
    is_snake_case,
)

_SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")


def test_cpcv_baseline_ids_have_strategy_label() -> None:
    for bid in ("long", "random", "sign_lag", "equal_weight"):
        assert bid in STRATEGY_LABELS, f"STRATEGY_LABELS missing CPCV baseline: {bid}"


def test_every_benchmark_panel_id_has_strategy_label() -> None:
    missing = [n for n in BENCHMARK_PANEL_NAMES if n not in STRATEGY_LABELS]
    assert not missing, f"STRATEGY_LABELS missing panel ids: {missing}"


def test_every_sleeve_and_archetype_has_label() -> None:
    from mascotrl.data.crucible import SLEEVE_IDS

    for s in SLEEVE_IDS:
        assert s in SLEEVE_LABELS
    assert set(ARCHETYPE_LABELS) >= {
        "trend_follower",
        "contrarian",
        "risk_manager",
        "speculator",
        "tactical_rotator",
        "mixed",
    }
    assert "carry_harvester" not in ARCHETYPE_LABELS
    assert "index_hugger" not in ARCHETYPE_LABELS
    assert "liquidity_provider" not in ARCHETYPE_LABELS


def test_human_raises_in_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASCOTRL_STRICT_LABELS", "1")
    with pytest.raises(KeyError, match="unmapped"):
        human("not_a_real_strategy_id_xyz", kind="strategy")


def test_human_soft_fallback_without_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASCOTRL_STRICT_LABELS", raising=False)
    lab = human("some_future_bench", kind="strategy")
    assert " " in lab or lab == "Some Future Bench"


def test_stem_short_label_under_25_chars() -> None:
    from mascotrl.reporting.figures.labels import figure_cell_label, stem_short_label

    cases = {
        "eq_K100_single_ppo_mlp_sparse_tilt_smse": "PPO sparse SMSE",
        "eq_K100_single_td3_mlp_softmax_mtm_pnl": "TD3 softmax profit",
        "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao": "PPO sparse cost-aware",
        "eq_K100_single_ppo_mlp_entmax_15_differential_sharpe": "PPO entmax DiffSharpe",
        "eq_K100_single_ppo_mlp_tanh_l1_meanvar_kolm": "PPO tanh-L1 mean-var",
    }
    for stem, want in cases.items():
        got = stem_short_label(stem)
        assert len(got) <= 25, (stem, got)
        assert got == want, (stem, got, want)
    assert figure_cell_label("eq_K100_single_ppo_mlp_sparse_tilt_smse") == "Cheetah"
    assert figure_cell_label("eq_K100_single_td3_mlp_softmax_mtm_pnl") == "Owl"


def test_parse_eval_dates_avoids_1970_epoch() -> None:
    from mascotrl.reporting.figures.labels import parse_eval_dates

    idx = parse_eval_dates(list(range(5)))
    assert str(idx[0].year) == "2014"
    assert all(ts.year >= 2014 for ts in idx)


def test_expert_display_maps_child_keys() -> None:
    from mascotrl.reporting.figures.labels import expert_display_name

    names = ["fox", "cheetah", "owl"]
    assert expert_display_name("child0", expert_names=names) == "Fox"
    assert expert_display_name("_child2", expert_names=names) == "Owl"
    assert expert_display_name("fox") == "Fox"
