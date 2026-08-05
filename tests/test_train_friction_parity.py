"""Step 5/6: hist OOS arm wiring + FrictionSpec train/OOS parity."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest
from tests.conftest import FLOAT_TOL
from mascotrl.eval.friction import FrictionSpec, assert_friction_parity, friction_spec_from_cfg, scale_friction

def test_friction_parity_ok_when_multiplier_differs():
    train = FrictionSpec(spec_id='v2_quote_touch', cost_multiplier=1.0)
    oos = scale_friction(train, 2.0)
    assert_friction_parity(train, oos)
    assert oos.cost_multiplier == pytest.approx(2.0, **FLOAT_TOL)

def test_friction_parity_fails_on_bps_mismatch():
    train = FrictionSpec(equity_bps=5.0)
    oos = FrictionSpec(equity_bps=10.0)
    with pytest.raises(AssertionError, match='equity_bps'):
        assert_friction_parity(train, oos)

def test_scale_friction_123():
    base = FrictionSpec()
    assert scale_friction(base, 1.0).cost_multiplier == pytest.approx(1.0, **FLOAT_TOL)
    assert scale_friction(base, 2.0).cost_multiplier == pytest.approx(2.0, **FLOAT_TOL)
    assert scale_friction(base, 3.0).cost_multiplier == pytest.approx(3.0, **FLOAT_TOL)

def test_preregistered_cost_ladder_includes_1_5x():
    """O'Donovan / refine: ladder 1.0 / 1.5 / 2.0 / 3.0 before seeing failures."""
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    for name in ('workflows/arm_options.yaml', 'workflows/happo_cmdp_mamba_k50.yaml'):
        p = root / 'config' / name
        if not p.is_file():
            continue
        cfg = yaml.safe_load(p.read_text())
        ladder = cfg.get('cost_multipliers')
        if ladder is None:
            continue
        assert 1.5 in [float(x) for x in ladder]
        base = friction_spec_from_cfg(cfg)
        assert scale_friction(base, 1.5).cost_multiplier == pytest.approx(1.5, **FLOAT_TOL)
        assert_friction_parity(base, scale_friction(base, 1.5))
        return
    base = FrictionSpec(spec_id='v2_quote_touch', cost_multiplier=1.0)
    assert scale_friction(base, 1.5).cost_multiplier == pytest.approx(1.5, **FLOAT_TOL)

def test_friction_spec_from_v2_cfg():
    cfg = {'friction_spec_id': 'v2_quote_touch', 'execution_spread_bps': 5.0, 'borrow_floor_bps_annual': 25.0, 'arm': {'friction_spec_id': 'v2_quote_touch'}}
    spec = friction_spec_from_cfg(cfg)
    assert spec.spec_id == 'v2_quote_touch'
    assert spec.borrow_floor_bps_annual == pytest.approx(25.0, **FLOAT_TOL)
