"""Part E.6 isolation: gates must not import behaviour metrics; v2 schema."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.reporting.policy_behavior import build_policy_behavior, write_policy_behavior

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "src" / "eval"

GATE_MODULES = (
    "gate_ladder.py",
    "alpha_gates.py",
    "spectrum_gates.py",
    "signal_gate.py",
    "collapse_guard.py",
    "yaml_honesty.py",
)

V2_REQUIRED_KEYS = {
    "schema_version",
    "cell_id",
    "arm",
    "algo",
    "architecture",
    "objective",
    "train_world",
    "policy_mode",
    "universe_fingerprint",
    "interpretation_only",
    "feeds_capital_gates",
    "behaviour",
    "behaviour_by_regime",
    "sleeve_tilt_series",
    "macro_tilt_sensitivity",
    "archetype_scores",
    "archetype_primary",
    "archetype_runner_up",
    "archetype_margin",
    "explanations",
    "null_band",
}


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_no_gate_module_imports_behaviour_metrics():
    for name in GATE_MODULES:
        path = EVAL / name
        assert path.is_file(), f"missing gate module {name}"
        imported = _imports_of(path)
        assert not any("behavior_metrics" in m for m in imported), (
            f"{name} imports behaviour_metrics: {imported}"
        )
        # also forbid reading the artifact by string constant scan of import graph
        text = path.read_text(encoding="utf-8")
        assert "behavior_metrics" not in text
        assert "policy_behavior.json" not in text


def test_policy_behavior_v2_schema(tmp_path):
    T, K = 20, 4
    W = np.full((T, K), 1.0 / K)
    S = np.eye(K, 7)
    regimes = np.array(["calm"] * 10 + ["crisis"] * 10, dtype=object)
    R = np.random.default_rng(0).normal(0.0, 0.01, size=(T, K))
    payload = build_policy_behavior(
        cell_id="cell0",
        arm="eq",
        algo="ppo",
        architecture="mlp",
        objective="mean_std_cao",
        train_world="hist",
        policy_mode="balanced",
        universe_fingerprint="deadbeef",
        weights=W,
        asset_returns=R,
        sleeve_matrix=S,
        regimes=regimes,
        cell_cfg={
            "objective": "mean_std_cao",
            "algo": "ppo",
            "architecture": "mlp",
            "policy_mode": "balanced",
        },
        behaviour_panel=None,
    )
    assert payload["schema_version"] == 2
    assert payload["interpretation_only"] is True
    assert payload["feeds_capital_gates"] is False
    missing = V2_REQUIRED_KEYS - set(payload)
    assert not missing, f"missing keys: {missing}"
    # Compat block may include archetype.name for report macros; primary remains authoritative.
    assert payload["archetype_primary"]
    compat = payload.get("archetype") or {}
    if compat:
        assert compat.get("name") == payload["archetype_primary"]
    assert len(payload["behaviour"]) == 43
    assert len(payload["archetype_scores"]) == 5
    assert len(payload["explanations"]) >= 4
    path = write_policy_behavior(tmp_path / "policy_behavior.json", payload)
    assert path.is_file()
