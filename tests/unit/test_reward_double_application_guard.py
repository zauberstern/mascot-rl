"""Reward-shaping double-application guard (workflow-routing regression lock)."""
from __future__ import annotations

import pytest

from src.eval.differential_sharpe import DifferentialSharpe
from src.spectrum import cell_schema


@pytest.mark.unit
def test_differential_sharpe_double_apply_diverges():
    """Feeding the same return stream through two nested DSR instances diverges.

    Proves that applying DifferentialSharpe twice is not idempotent; callers
    must guard against double application rather than assuming composition is safe.
    """
    stream = [0.01, -0.005, 0.02, 0.0, -0.01, 0.015, 0.008, -0.003]
    single = DifferentialSharpe(eta=0.01)
    nested = DifferentialSharpe(eta=0.01)
    outer = DifferentialSharpe(eta=0.01)
    single_out = [single.step(r) for r in stream]
    nested_out = [outer.step(nested.step(r)) for r in stream]
    assert single_out != nested_out
    diffs = [
        abs(a - b)
        for a, b in zip(single_out, nested_out)
        if a == a and b == b
    ]
    assert any(d > 1e-8 for d in diffs)


@pytest.mark.unit
def test_reward_shaping_ablation_flag_is_boolean_in_schema():
    """reward_shaping_ablation must remain a bool field in the cell schema."""
    spec = cell_schema.SCHEMA["reward_shaping_ablation"]
    assert spec.typ is bool
