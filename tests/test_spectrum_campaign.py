"""TDD: spectrum campaign dry-run emits transfer + collapse fields."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_spectrum_budget_hours_skips_remaining(tmp_path: Path, monkeypatch) -> None:
    """MASCOTRL_SPECTRUM_BUDGET_HOURS stops the cell loop and records SKIPPED.md."""
    import os
    cfg_dir = tmp_path / 'cfgs'
    cfg_dir.mkdir()
    for name in ('a_cell.yaml', 'b_cell.yaml', 'c_cell.yaml'):
        (cfg_dir / name).write_text('algo: ppo\narchitecture: mlp\nobjective: mean_std_cao\ntrain_world: historical\nn_assets: 4\nclaim_label_stem: stk_ret\n', encoding='utf-8')
    out = tmp_path / 'spectrum'
    env = os.environ.copy()
    env['MASCOTRL_SPECTRUM_BUDGET_HOURS'] = '1e-9'
    env['PYTHONPATH'] = str(ROOT)
    cmd = [sys.executable, str(ROOT / 'scripts' / 'run_spectrum_campaign.py'), '--config-dir', str(cfg_dir), '--out-dir', str(out), '--dry-run']
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    skipped = (out / 'SKIPPED.md').read_text(encoding='utf-8')
    assert 'budget_exhausted' in skipped
    index = json.loads((out / 'index.json').read_text(encoding='utf-8'))
    assert index.get('budget_exhausted') is True
    assert int(index.get('n_skipped') or 0) >= 1

def test_run_happo_arm_historical_opt_uses_panel_bridge() -> None:
    """Wave 5: historical opt HAPPO must not crash on get_surface_tensor."""
    from scripts.run_spectrum_campaign import _run_happo_arm
    cfg = {'algo': 'happo', 'train_world': 'historical', 'n_assets': 4, 'n_steps': 16, 'd_model': 16, 'd_state': 8, 'macro_dim': 8, 'architecture': 'mlp', 'spectrum_happo_horizon': 4, 'spectrum_happo_episodes': 1, 'claim_tier': 'dispatch_only', 'seed': 0}
    art, err = _run_happo_arm(cfg, 'opt')
    assert err is None, err
    assert art is not None
    assert art['panel_source'] in ('toy', 'optionmetrics')
    assert art['claim_tier'] == 'dispatch_only'
    assert art['real_reference_arm_present'] is False
    assert art['n_episodes'] == 1

def test_run_happo_arm_eq_routes_through_equity_cmdp_bridge(monkeypatch) -> None:
    """C6/C8: arm='eq' happo cells must build surfaces via
    equity_panel_to_cmdp_tensors (a real equity return panel collapsed to a
    1x1 strike/maturity placeholder), not get_surface_tensor's option-smile
    generator, so train_world='historical' (unsupported by
    get_surface_tensor) works for the eq arm."""
    import mascotrl.eval.equity_substrate as es
    from scripts.run_spectrum_campaign import _run_happo_arm

    # Keep this unit test on the OM→toy bridge path; lake parity is covered
    # in tests/test_equity_substrate_parity.py.
    monkeypatch.setattr(
        es,
        "load_lake_dyn_hrp_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("lake_disabled_in_unit_test")),
    )
    cfg = {'algo': 'happo', 'train_world': 'historical', 'n_assets': 4, 'n_steps': 16, 'd_model': 16, 'd_state': 8, 'macro_dim': 8, 'architecture': 'mlp', 'temporal_backend': 'mlp', 'use_dhgnn': False, 'spatial_mode': 'none', 'spectrum_happo_horizon': 4, 'spectrum_happo_episodes': 2, 'seed': 0}
    art, err = _run_happo_arm(cfg, 'eq')
    assert err is None, err
    assert art is not None
    assert art['train_metric'] == art['train_metric']
    assert len(art['turnovers']) > 0
    assert art['panel_source'] in ('toy', 'optionmetrics')
