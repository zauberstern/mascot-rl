"""A1: universe_arm resolution order is CLI > YAML > dyn_liquidity."""
from __future__ import annotations

from scripts.run_eq_alloc_campaign import (
    _campaign_config_fingerprint,
    resolve_universe_arm,
)


def test_yaml_only_resolves_to_dyn_liquidity() -> None:
    arm = resolve_universe_arm(cli_arm=None, cfg={"universe_arm": "dyn_liquidity"})
    assert arm == "dyn_liquidity"


def test_cli_overrides_yaml() -> None:
    arm = resolve_universe_arm(cli_arm="dyn_hrp", cfg={"universe_arm": "dyn_liquidity"})
    assert arm == "dyn_hrp"


def test_neither_resolves_to_dyn_liquidity() -> None:
    arm = resolve_universe_arm(cli_arm=None, cfg={})
    assert arm == "dyn_liquidity"


def test_fingerprint_matches_executed_arm() -> None:
    cfg = {"universe_arm": "dyn_liquidity", "cpcv_n_splits": 6}
    arm = resolve_universe_arm(cli_arm=None, cfg=cfg)
    cfg["universe_arm"] = arm
    fp = _campaign_config_fingerprint(cfg, realized_k=40)
    # Re-resolve must not change the fingerprint payload arm.
    assert cfg["universe_arm"] == "dyn_liquidity"
    assert "universe_arm" in fp or len(fp) == 16
    # Fingerprint string itself is a hex digest; assert arm is in hashed payload
    # by changing arm and seeing hash diverge.
    cfg2 = dict(cfg)
    cfg2["universe_arm"] = "dyn_hrp"
    fp2 = _campaign_config_fingerprint(cfg2, realized_k=40)
    assert fp != fp2
