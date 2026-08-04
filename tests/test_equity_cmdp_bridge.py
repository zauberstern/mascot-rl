"""A12: production-entry-point coverage for equity_panel_to_cmdp_tensors."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from src.env.equity_cmdp_bridge import equity_panel_to_cmdp_tensors, should_route_eq_via_cmdp

def test_equity_panel_to_cmdp_tensors_shapes() -> None:
    rng = np.random.default_rng(0)
    t, k = (60, 5)
    rets = rng.normal(0.0003, 0.01, size=(t, k))
    out = equity_panel_to_cmdp_tensors(rets, spot0=100.0)
    assert set(out.keys()) == {'spot_paths', 'surfaces', 'returns'}
    assert out['spot_paths'].shape == (1, k, t)
    assert out['surfaces'].shape == (1, k, t, 1, 1)
    assert isinstance(out['spot_paths'], torch.Tensor)
    assert isinstance(out['surfaces'], torch.Tensor)

def test_equity_panel_to_cmdp_tensors_spot_path_compounds_returns() -> None:
    t, k = (5, 1)
    rets = np.array([[0.0], [0.1], [-0.05], [0.0], [0.02]])
    out = equity_panel_to_cmdp_tensors(rets, spot0=100.0)
    spots = out['spot_paths'][0, 0].numpy()
    expected = 100.0
    path = [expected]
    for r in rets[1:, 0]:
        expected *= 1.0 + r
        path.append(expected)
    assert spots == pytest.approx(np.array(path), rel=1e-05)

def test_equity_panel_to_cmdp_tensors_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        equity_panel_to_cmdp_tensors(np.zeros((3, 3, 3)))

def test_should_route_eq_via_cmdp_reads_flag() -> None:
    assert should_route_eq_via_cmdp({'route_eq_via_cmdp': True}) is True
    assert should_route_eq_via_cmdp({'route_eq_via_cmdp': False}) is False
    assert should_route_eq_via_cmdp({}) is False

def test_bridge_output_builds_a_real_cmdpenv_and_steps() -> None:
    """C6: the bridge's (surfaces, spot_paths) must be directly consumable by
    CMDPEnv -- the real HAPPO+CMDP spine, not a schema-only shape check --
    including a genuine HAPPOTrainer.update on the resulting transition."""
    from src.env.cmdp_env import CMDPEnv
    from src.features.extractor import AlphaFeatureExtractor
    from src.policy.happo import HAPPOEngine
    from src.policy.trainer import HAPPOTrainer, TrainBatch
    rng = np.random.default_rng(1)
    t, k = (40, 3)
    rets = rng.normal(0.0003, 0.01, size=(t, k))
    bridge = equity_panel_to_cmdp_tensors(rets, spot0=100.0)
    d_model, macro_dim = (16, 8)
    torch.manual_seed(0)
    fe = AlphaFeatureExtractor(k, d_model, d_state=8)
    policy = HAPPOEngine(k, d_model, macro_dim, 0.25)
    env = CMDPEnv(bridge['surfaces'], fe, policy, d_model, macro_dim, use_gpu=False, transition_source='historical', spot_paths=bridge['spot_paths'])
    obs = env.reset(path=0)
    w_prev = torch.zeros(1, k)
    w, lp, v, w_raw = policy.act_stochastic(obs.enriched, obs.macro, w_prev, obs.deltas)
    nxt = env.step(w.detach())
    assert torch.isfinite(nxt.reward).all()
    trainer = HAPPOTrainer(policy, use_compile=False)
    batch = TrainBatch(enriched=obs.enriched.detach(), macro=obs.macro.detach(), w_prev=w_prev, deltas=obs.deltas.detach(), actions=w.detach(), log_probs=lp.detach(), values=v.detach().reshape(-1), rewards=nxt.reward.detach().reshape(-1), dones=torch.tensor([float(nxt.done)]), raw_actions=w_raw.detach())
    stats = trainer.update(batch, epochs=1)
    assert np.isfinite(stats['policy_loss'])
    assert np.isfinite(stats['value_loss'])

def test_route_eq_via_cmdp_surfaces_have_single_strike_and_maturity() -> None:
    """The equity bridge collapses the option smile to a 1x1 placeholder;
    downstream CMDPEnv must accept that shape without special-casing."""
    rng = np.random.default_rng(2)
    t, k = (25, 2)
    rets = rng.normal(0.0, 0.01, size=(t, k))
    out = equity_panel_to_cmdp_tensors(rets)
    assert out['surfaces'].shape[-2:] == (1, 1)
