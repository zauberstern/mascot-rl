"""Per-cell telemetry: decision trace + training jsonl (C5)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import json
from pathlib import Path

import numpy as np

from src.reporting.decision_trace import build_decision_trace_rows, write_decision_trace
from src.reporting.policy_behavior import build_policy_behavior
from src.reporting.training_telemetry import (
    training_rows_from_diagnostics,
    write_training_jsonl,
)


def test_rich_behaviour_has_tilts_and_macro() -> None:
    rng = np.random.default_rng(0)
    T, K = 60, 7
    S = np.eye(K, 7)
    W = np.full((T, K), 1.0 / K)
    vix = rng.standard_normal(T) * 0.5
    hy = rng.standard_normal(T) * 0.1
    term = rng.standard_normal(T) * 0.1
    R = rng.normal(0.0, 0.01, size=(T, K))
    beh = build_policy_behavior(
        algo="ppo",
        weights=W,
        asset_returns=R,
        sleeve_matrix=S,
        regimes=np.array(["calm"] * T),
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        n_null_shuffles=5,
    )
    assert beh["sleeve_tilt_series"]["trend"]
    assert beh["macro_tilt_sensitivity"]


def test_decision_trace_schema(tmp_path: Path) -> None:
    w = np.full((5, 4), 0.25)
    rows = build_decision_trace_rows(
        dates=[f"2019-01-0{i}" for i in range(1, 6)],
        weights=w,
        turnovers=[0.1, 0.12, 0.11, 0.09, 0.1],
        sleeve_matrix=np.eye(4, 7),
        turnover_cap=0.15,
    )
    path = write_decision_trace(tmp_path / "trace.jsonl", rows)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    row0 = json.loads(lines[0])
    assert "projected_w" in row0
    assert "turnover" in row0


def test_training_jsonl_schema(tmp_path: Path) -> None:
    rows = training_rows_from_diagnostics(
        {"policy_loss": 0.1, "entropy": 1.2, "approx_kl": 0.01},
        cell_id="cell_a",
    )
    path = write_training_jsonl(tmp_path / "train.jsonl", rows)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["cell_id"] == "cell_a"
    assert "entropy" in row


def test_training_keys_cover_cppo_happo_sb3_emitters() -> None:
    from src.reporting.training_telemetry import _TRAINING_KEYS, normalize_training_row

    required = {
        "policy_grad_norm",
        "value_grad_norm",
        "cvar_eta",
        "cvar_nu",
        "cvar_beta",
        "trajectory_cvar",
        "cvar_violation",
        "cmdp_lambda",
        "cmdp_j_c_violation_frac",
        "weight_entropy",
        "rl_backend",
        "exec_turnover",
        "optimizer_steps",
    }
    missing = required - set(_TRAINING_KEYS)
    assert not missing, missing
    row = normalize_training_row(
        {k: 1.0 for k in required} | {"rl_backend": "sb3"},
        cell_id="x",
    )
    assert row["rl_backend"] == "sb3"
    assert row["cvar_eta"] == pytest.approx(1.0, **FLOAT_TOL)


def test_training_keys_cover_reward_decomposition() -> None:
    from src.reporting.training_telemetry import _TRAINING_KEYS, normalize_training_row

    required = [
        "reward_return_term",
        "reward_cost_term",
        "reward_turnover_penalty",
        "reward_cvar_term",
        "reward_entropy_bonus",
        "reward_composite_total",
        "mean_reward",
        "grad_norm",
        "policy_grad_norm",
    ]
    missing = set(required) - set(_TRAINING_KEYS)
    assert not missing, missing
    row = normalize_training_row(
        {k: float(i) for i, k in enumerate(required)},
        cell_id="decomp",
    )
    assert row["reward_return_term"] == pytest.approx(0.0, **FLOAT_TOL)
    assert row["reward_composite_total"] == pytest.approx(5.0, **FLOAT_TOL)
    assert row["policy_grad_norm"] == pytest.approx(8.0, **FLOAT_TOL)


def test_reward_decomp_from_step_info() -> None:
    from src.reporting.training_telemetry import reward_decomp_from_step_info

    info = {"gross": 0.012, "cost": 0.0015, "turnover": 0.08, "borrow": 0.0002}
    decomp = reward_decomp_from_step_info(
        info,
        train_reward=0.0103,
        entropy_bonus=0.01,
        cvar_term=-0.002,
        turnover_penalty_coef=0.0,
    )
    assert decomp["reward_return_term"] == pytest.approx(0.012, **FLOAT_TOL)
    assert decomp["reward_cost_term"] == pytest.approx(-0.0015, **FLOAT_TOL)
    assert abs(decomp["reward_composite_total"] - 0.0103) < 1e-9
    assert decomp["reward_entropy_bonus"] == pytest.approx(0.01, **FLOAT_TOL)
    assert decomp["reward_cvar_term"] == pytest.approx(-0.002, **FLOAT_TOL)


def test_alias_grad_norm_to_policy_grad_norm() -> None:
    from src.reporting.training_telemetry import alias_grad_norm

    stats = {"grad_norm": 1.5, "policy_loss": 0.2}
    out = alias_grad_norm(stats)
    assert out["policy_grad_norm"] == pytest.approx(1.5, **FLOAT_TOL)
    assert out["grad_norm"] == pytest.approx(1.5, **FLOAT_TOL)
    # Idempotent when policy_grad_norm already set.
    already = {"policy_grad_norm": 2.0, "grad_norm": 1.0}
    assert alias_grad_norm(already)["policy_grad_norm"] == pytest.approx(2.0, **FLOAT_TOL)
