"""Alpha v2 Block E Step 26: SPA rival families are frozen constants."""
from __future__ import annotations

import pytest

from src.eval.publication import SPA_FAMILIES


def test_spa_families_immutable_tuple_includes_best_single_agent_rl():
    assert isinstance(SPA_FAMILIES, tuple)
    assert "best_single_agent_rl" in SPA_FAMILIES
    # Locked transparent baselines (opt + eq) plus bakeoff winner slot.
    for name in (
        "no_trade",
        "rv_iv_rank",
        "ridge",
        "equal_weight_factor_blend",
        "best_single_agent_rl",
    ):
        assert name in SPA_FAMILIES

    with pytest.raises(TypeError):
        SPA_FAMILIES[0] = "hacked"  # type: ignore[index]

    with pytest.raises(AttributeError):
        SPA_FAMILIES.append("extra")  # type: ignore[attr-defined]
