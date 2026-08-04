"""WP-S2/S5: capacity probe and protocol tiers."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from src.spectrum.capacity_probe import probe_universe_capacity
from src.spectrum.protocol_tiers import apply_protocol_tier, resolve_protocol_tier


def test_capacity_refuses_400_when_pool_small() -> None:
    res = probe_universe_capacity(350, requested=(100, 200, 400))
    assert 400 in res.refused_k
    assert 100 in res.feasible_k
    assert 200 in res.feasible_k
    assert res.k_max == 200
    assert 400 not in res.k_axis


def test_capacity_accepts_all_when_pool_large() -> None:
    res = probe_universe_capacity(500, requested=(100, 200, 400))
    assert res.refused_k == ()
    assert res.k_axis == (100, 200, 400)
    assert res.k_max == 400


def test_protocol_tiers_share_cpcv_geometry() -> None:
    keys = ("cpcv_n_splits", "cpcv_n_test_groups", "cpcv_purge_days", "cpcv_embargo_days")
    specs = [resolve_protocol_tier(t) for t in ("confirmatory", "parity", "screening")]
    geo = {k: specs[0][k] for k in keys}
    for s in specs[1:]:
        for k in keys:
            assert s[k] == geo[k]
    assert specs[0]["n_seeds"] == 10
    assert specs[1]["n_seeds"] == 3
    assert specs[2]["n_seeds"] == 1
    assert specs[2]["train_env_steps"] == 25_000
    assert "capital_eligible" not in specs[0]


def test_apply_protocol_tier_stamps() -> None:
    cfg: dict = {"capital_eligible": True}
    apply_protocol_tier(cfg, "parity")
    assert cfg["protocol_tier"] == "parity"
    assert cfg["seeds"] == [0, 1, 2]
    assert "capital_eligible" not in cfg
    assert cfg["train_env_steps"] == 100_000
