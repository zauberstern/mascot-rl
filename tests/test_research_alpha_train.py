"""TDD: research hist train loop on HistoricalArmEnv + PPO + FrictionSpec."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import numpy as np
from src.arms import ArmSpec
from src.eval.friction import assert_friction_parity
from src.eval.research_alpha_train import SYNTHETIC_TRAIN_WORLDS, build_research_hist_env, synthetic_train_panel, train_research_hist
from src.reporting.research_alpha_router import research_train_friction_pair

def _toy_panel(t: int=60, k: int=4, seed: int=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.01, size=(t, 4))
    return (rets, factors)

def test_build_research_hist_env_uses_friction_parity() -> None:
    rets, fac = _toy_panel()
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'om_touch_enabled': True, 'hedge_leg_spread_bps': 5.0, 'n_assets': 4, 'reward_shaping_ablation': True}
    env = build_research_hist_env(rets, fac, cfg)
    train, oos = research_train_friction_pair(cfg)
    assert_friction_parity(train, oos)
    assert env.friction.om_touch_enabled is True
    obs, info = env.reset()
    assert obs.shape[0] == 4

def test_build_research_hist_env_reads_slot_valid_mask_from_cfg() -> None:
    """W4.2: cfg['_slot_valid_mask'] threads into HistoricalArmEnv."""
    rets, fac = _toy_panel(t=20, k=4)
    mask = np.ones((20, 4), dtype=bool)
    mask[:, 3] = False
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'om_touch_enabled': True, 'hedge_leg_spread_bps': 5.0, 'n_assets': 4, '_slot_valid_mask': mask}
    env = build_research_hist_env(rets, fac, cfg)
    assert env.slot_valid_mask is not None
    np.testing.assert_array_equal(env.slot_valid_mask, mask)

def test_build_research_hist_env_refuses_slot_valid_mask_shape_mismatch() -> None:
    rets, fac = _toy_panel(t=20, k=4)
    bad_mask = np.ones((20, 3), dtype=bool)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'om_touch_enabled': True, 'n_assets': 4, '_slot_valid_mask': bad_mask}
    import pytest

    with pytest.raises(ValueError, match='_slot_valid_mask'):
        build_research_hist_env(rets, fac, cfg)

def test_train_research_hist_returns_metrics() -> None:
    rets, fac = _toy_panel(t=40, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'om_touch_enabled': True, 'hedge_leg_spread_bps': 5.0, 'n_assets': 3, 'lr': 0.0003, 'train_epochs': 2, 'policy': 'single_agent', 'projection_mode': 'soft'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out['primary_train'] == 'historical_arm_env'
    assert out['policy'] == 'single_agent'
    # Default residual train reward != total_net claim metric (honest stamp).
    assert out['train_objective_equals_claim_metric'] is False
    assert out['friction_applied'] is True
    assert out['n_steps'] > 0
    assert 'mean_reward' in out
    assert out['agent'] is not None

def test_train_refuses_non_hist_primary() -> None:
    rets, fac = _toy_panel(k=2)
    try:
        train_research_hist(rets, fac, {'primary_train': 'synth_cmdp', 'n_assets': 2}, seed=0)
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_soft_projection_is_identity_passthrough() -> None:
    rets, fac = _toy_panel(k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'om_touch_enabled': False, 'n_assets': 3, 'projection_mode': 'soft'}
    env = build_research_hist_env(rets, fac, cfg)
    assert env.project_fn is not None
    w = np.array([0.2, -0.1, 0.3])
    assert np.allclose(env.project_fn(w), w)

def test_train_research_hist_dense_reward_mikkila_asym() -> None:
    """C3: mikkila_asym is reachable as objective (dense_reward path) and
    reported back with the resolved gradient path."""
    rets, fac = _toy_panel(t=40, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'objective': 'mikkila_asym', 'mikkila_xi': 2.0}
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out['objective'] == 'mikkila_asym'
    assert out['objective_gradient_path'] == 'dense_reward'
    assert out['n_steps'] > 0

def test_train_research_hist_mcpg_softmax_uses_raw_logits() -> None:
    """Path A: MCPG under softmax must train on pre-head logits, not weights."""
    rets, fac = _toy_panel(t=40, k=3)
    # MCPG is on-policy single-shot (Phase 1); use train_updates_per_fold to
    # accumulate optimizer_steps rather than n_minibatches repeats.
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'n_minibatches': 4, 'train_episodes': 2, 'train_updates_per_fold': 4, 'min_optimizer_steps': 4, 'policy': 'single_agent', 'projection_mode': 'soft', 'algo': 'mcpg', 'weight_head': 'softmax', 'weight_head_temperature': 0.5, 'objective': 'differential_sharpe', 'objective_primary': True, 'use_equity_feature_cube': True, 'architecture': 'transformer', 'feature_seq_len': 4}
    out = train_research_hist(rets, fac, cfg, seed=0)
    from src.policy.single_agent import MCPGAgent
    assert isinstance(out['agent'], MCPGAgent)
    assert out['agent'].weight_head == 'softmax'
    assert float(out['agent'].weight_head_temperature) == pytest.approx(0.5, **FLOAT_TOL)
    assert out['n_steps'] > 0
    assert int(out['optimizer_steps']) >= 4
    assert out['train_stats'].get('on_policy_single_shot') is True

def test_train_research_hist_episode_weight_objective_reaches_ppo_update() -> None:
    """C3: an episode_weight objective (mean_std_cao) must actually reach
    the PPO update as a non-trivial sample_weight, not just be validated
    and dropped -- this is HAPPO's episode_weights mechanism ported to the
    single-agent research PPO path."""
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'train_episodes': 4, 'policy': 'single_agent', 'projection_mode': 'soft', 'objective': 'mean_std_cao', 'objective_primary': True}
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out['objective'] == 'mean_std_cao'
    assert out['objective_gradient_path'] == 'episode_weight'
    assert out['train_stats']['objective_gradient_path'] == 'episode_weight'

def test_train_research_hist_objective_none_is_unchanged_critic_only() -> None:
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'rl_backend': 'custom'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out['objective'] == 'none'
    assert out['objective_gradient_path'] == 'critic_only'

def test_train_research_hist_train_updates_per_fold_multiplies_optimizer_steps() -> None:
    """W2.3: train_updates_per_fold > 1 collects a fresh trajectory and
    trains again, looping -- not just a documented but unread config key."""
    rets, fac = _toy_panel(t=40, k=3)
    base_cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'n_minibatches': 1, 'train_episodes': 2, 'policy': 'single_agent', 'projection_mode': 'soft'}
    out1 = train_research_hist(rets, fac, dict(base_cfg, train_updates_per_fold=1), seed=0)
    out3 = train_research_hist(rets, fac, dict(base_cfg, train_updates_per_fold=3), seed=0)
    assert out3['train_updates_per_fold'] == 3
    assert out3['n_episodes'] == out1['n_episodes'] * 3
    assert out3['optimizer_steps'] > out1['optimizer_steps']
    assert len(out3['learning_curve']) == len(out1['learning_curve']) * 3

def test_train_research_hist_train_updates_per_fold_default_is_one() -> None:
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'rl_backend': 'custom'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out['train_updates_per_fold'] == 1

def test_train_research_hist_every_algo_axis_value_actually_trains() -> None:
    """C4: algo is resolved by validate_cfg AND consumed -- every
    registered algo id must reach a real agent.train_epoch call, not
    silently fall back to the PPO default."""
    import pytest
    from src.policy.single_agent import DDPGAgent, DQNAgent, MCPGAgent, PPOAgent, RRLAgent, SACAgent, TD3Agent
    expected = {'ppo': PPOAgent, 'sac': SACAgent, 'td3': TD3Agent, 'ddpg': DDPGAgent, 'mcpg': MCPGAgent, 'rrl': RRLAgent, 'dqn': DQNAgent}
    for algo, cls in expected.items():
        rets, fac = _toy_panel(t=30, k=3)
        cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'algo': algo, 'objective': 'mtm_pnl' if algo == 'rrl' else 'differential_sharpe', 'objective_primary': True, 'rl_backend': 'custom'}
        out = train_research_hist(rets, fac, cfg, seed=0)
        assert isinstance(out['agent'], cls), f'algo={algo} built wrong class'
        assert out['n_steps'] > 0
        assert np.isfinite(out['train_stats'].get('loss', float('nan')))

def test_train_research_hist_happo_not_a_valid_single_agent_algo() -> None:
    """algo='happo' is a real registry option but is dispatched at the
    sweep level (run_spectrum_campaign.py) to the multi-agent HAPPO
    trainer, not through this single-agent PPO/SAC/... path."""
    rets, fac = _toy_panel(k=2)
    try:
        train_research_hist(rets, fac, {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 2, 'algo': 'happo'}, seed=0)
        raised = False
    except (ValueError, KeyError):
        raised = True
    assert raised

def test_train_research_hist_architecture_non_mlp_with_non_ppo_algo_fails_closed() -> None:
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'policy': 'single_agent', 'projection_mode': 'soft', 'algo': 'sac', 'architecture': 'gru'}
    try:
        train_research_hist(rets, fac, cfg, seed=0)
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_train_research_hist_architecture_gru_routes_through_extractor() -> None:
    """C2: architecture != 'mlp' must actually swap the PPO body to
    AlphaFeatureExtractor's per-asset temporal backend, not be a documented
    but unread config key."""
    rets, fac = _toy_panel(t=40, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'use_equity_feature_cube': True, 'architecture': 'gru', 'rl_backend': 'custom'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    from src.policy.single_agent import _AssetTemporalActorCritic
    assert isinstance(out['agent'].net, _AssetTemporalActorCritic)
    assert out['agent'].net.extractor.temporal_backend == 'gru'
    assert out['n_steps'] > 0

def test_train_research_hist_surface_image_encoder_reaches_actor_critic() -> None:
    """B4: use_surface_image_encoder=true must actually attach a trainable
    SurfaceImageEncoder inside the PPO body via the equity feature cube's
    kelly_images extra, not be a documented but unread config key."""
    t, k = (30, 3)
    rets, fac = _toy_panel(t=t, k=k)
    rng = np.random.default_rng(0)
    kelly_images = rng.normal(0.2, 0.05, size=(t, k, 11, 34))
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': k, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'use_equity_feature_cube': True, 'architecture': 'gru', 'use_surface_image_encoder': True, 'feature_extras': {'kelly_images': kelly_images}, 'rl_backend': 'custom'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    from src.policy.single_agent import _AssetTemporalActorCritic
    net = out['agent'].net
    assert isinstance(net, _AssetTemporalActorCritic)
    assert net.use_surface_image_encoder is True
    assert net.image_channels == 11 * 34
    assert net.image_encoder is not None
    assert out['n_steps'] > 0

def test_train_research_hist_architecture_mlp_default_unchanged() -> None:
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'rl_backend': 'custom'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    from src.policy.single_agent import _ActorCritic
    assert isinstance(out['agent'].net, _ActorCritic)

def test_train_research_hist_architecture_non_mlp_without_feature_cube_raises() -> None:
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'architecture': 'transformer'}
    try:
        train_research_hist(rets, fac, cfg, seed=0)
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_train_research_hist_unknown_architecture_fails_closed() -> None:
    rets, fac = _toy_panel(k=2)
    try:
        train_research_hist(rets, fac, {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 2, 'architecture': 'not_a_real_backend'}, seed=0)
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_build_research_hist_env_aligns_eq_slots_to_panel_k() -> None:
    """YAML equity_slots=100 must not crash when coverage yields K=5."""
    rets, fac = _toy_panel(t=30, k=5)
    cfg = {'primary_train': 'historical_arm_env', 'n_assets': 100, 'arm': {'id': 'eq', 'option_slots': 0, 'equity_slots': 100, 'delta_mode': 'off'}, 'projection_mode': 'soft'}
    env = build_research_hist_env(rets, fac, cfg)
    assert env.K == 5
    assert int(env.arm.equity_slots) == 5
    assert int(env.arm.n_slots) == 5

def test_synthetic_train_panel_rejects_historical_and_unknown_world() -> None:
    for bad in ('historical', 'hybrid_pretrain_finetune', 'not_a_world'):
        try:
            synthetic_train_panel({'train_world': bad}, k=2, n_rows=10, seed=0)
            raised = False
        except ValueError:
            raised = True
        assert raised, f'world={bad!r} should have raised'

def test_synthetic_train_panel_rejects_nonpositive_k_or_rows() -> None:
    for k, n_rows in ((0, 10), (2, 0), (-1, 10)):
        try:
            synthetic_train_panel({'train_world': 'gbm'}, k=k, n_rows=n_rows, seed=0)
            raised = False
        except ValueError:
            raised = True
        assert raised, f'k={k} n_rows={n_rows} should have raised'

def test_train_research_hist_writes_intra_fold_checkpoint(tmp_path) -> None:
    """W3.2: setting _checkpoint_dir must persist a policy/optimizer/hash
    checkpoint (mirrors scripts/train_happo.py's post_train.pt pattern) that
    can be loaded back into a fresh agent of the same shape."""
    import torch
    rets, fac = _toy_panel(t=30, k=3)
    cfg = {'primary_train': 'historical_arm_env', 'portfolio_arm': 'eq', 'n_assets': 3, 'lr': 0.0003, 'train_epochs': 1, 'policy': 'single_agent', 'projection_mode': 'soft', 'rl_backend': 'custom', '_checkpoint_dir': str(tmp_path / 'ckpt'), '_fold_id': 2, '_run_config_hash': 'abc123'}
    out = train_research_hist(rets, fac, cfg, seed=0)
    ckpts = sorted((tmp_path / 'ckpt').glob('*.pt'))
    assert ckpts, 'expected at least the end-of-training checkpoint'
    blob = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
    assert blob['fold_id'] == 2
    assert blob['run_config_hash'] == 'abc123'
    assert blob['seed'] == 0
    assert blob['optimizer'] is not None
    assert set(blob['policy'].keys()) == set(out['agent'].net.state_dict().keys())

def test_ppo_agent_checkpoint_roundtrip(tmp_path) -> None:
    """W3.2: save/load roundtrip of a tiny PPOAgent state_dict via the same
    helpers train_research_hist uses (_save_checkpoint / _maybe_resume_checkpoint)."""
    import torch
    from src.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
    from src.policy.single_agent import make_single_agent
    src_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, hidden=8, rl_backend='custom')
    cfg_save = {'_checkpoint_dir': str(tmp_path / 'ckpt'), '_fold_id': 4, '_run_config_hash': 'hash_ok'}
    _save_checkpoint(src_agent, cfg_save, seed=0, episode=1, optimizer_steps=5)
    ckpts = sorted((tmp_path / 'ckpt').glob('*.pt'))
    assert len(ckpts) == 1
    blob = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    assert blob['fold_id'] == 4
    assert blob['seed'] == 0
    assert blob['episode'] == 1
    assert blob['optimizer_steps'] == 5
    assert blob['run_config_hash'] == 'hash_ok'
    assert blob['optimizer'] is not None
    dst_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, hidden=8, rl_backend='custom')
    src_state = {k: v.clone() for k, v in src_agent.net.state_dict().items()}
    any_diff = any((not torch.equal(v, src_state[k]) for k, v in dst_agent.net.state_dict().items()))
    assert any_diff, 'fresh agent must start from different (randomly initialized) weights'
    result = _maybe_resume_checkpoint(dst_agent, {'_resume_checkpoint': str(ckpts[0]), '_run_config_hash': 'hash_ok'})
    assert result is not None
    for k, v in dst_agent.net.state_dict().items():
        assert torch.equal(v, src_state[k]), f'resumed weight {k} did not match checkpoint'

def test_ppo_agent_checkpoint_refuses_hash_mismatch(tmp_path) -> None:
    """W3.2: a run_config_hash mismatch must fail closed instead of silently
    loading weights trained under a different config."""
    from src.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
    from src.policy.single_agent import make_single_agent
    src_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, hidden=8, rl_backend='custom')
    _save_checkpoint(src_agent, {'_checkpoint_dir': str(tmp_path / 'ckpt'), '_fold_id': 0, '_run_config_hash': 'hash_a'}, seed=0, episode=1, optimizer_steps=1)
    ckpt = next((tmp_path / 'ckpt').glob('*.pt'))
    dst_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, hidden=8, rl_backend='custom')
    try:
        _maybe_resume_checkpoint(dst_agent, {'_resume_checkpoint': str(ckpt), '_run_config_hash': 'hash_b'})
        raised = False
    except RuntimeError:
        raised = True
    assert raised, 'hash mismatch must refuse to resume'

def test_maybe_resume_checkpoint_missing_path_is_noop() -> None:
    from src.eval.research_alpha_train import _maybe_resume_checkpoint
    from src.policy.single_agent import make_single_agent
    agent = make_single_agent('ppo', obs_dim=4, action_dim=2, hidden=4, rl_backend='custom')
    assert _maybe_resume_checkpoint(agent, {'_resume_checkpoint': '/no/such/path.pt'}) is None
    assert _maybe_resume_checkpoint(agent, {}) is None


def test_sb3_ppo_checkpoint_roundtrip(tmp_path) -> None:
    """SB3 agents expose ``net`` so _save/_maybe_resume preserve policy keys."""
    import torch
    from src.eval.research_alpha_train import (
        _agent_policy_module,
        _maybe_resume_checkpoint,
        _save_checkpoint,
    )
    from src.policy.single_agent import make_single_agent

    src_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, lr=1e-3, rl_backend='sb3')
    assert _agent_policy_module(src_agent) is not None
    assert isinstance(src_agent.net, torch.nn.Module)

    cfg_save = {
        '_checkpoint_dir': str(tmp_path / 'ckpt'),
        '_fold_id': 1,
        '_run_config_hash': 'sb3_hash',
    }
    _save_checkpoint(src_agent, cfg_save, seed=0, episode=1, optimizer_steps=3)
    ckpts = sorted((tmp_path / 'ckpt').glob('*.pt'))
    assert len(ckpts) == 1
    blob = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    assert set(blob['policy'].keys()) == set(src_agent.net.state_dict().keys())

    dst_agent = make_single_agent('ppo', obs_dim=6, action_dim=3, lr=1e-3, rl_backend='sb3')
    src_state = {k: v.clone() for k, v in src_agent.net.state_dict().items()}
    any_diff = any(
        not torch.equal(v, src_state[k]) for k, v in dst_agent.net.state_dict().items()
    )
    assert any_diff, 'fresh SB3 agent must start from different weights'
    result = _maybe_resume_checkpoint(
        dst_agent,
        {'_resume_checkpoint': str(ckpts[0]), '_run_config_hash': 'sb3_hash'},
    )
    assert result is not None
    for k, v in dst_agent.net.state_dict().items():
        assert torch.equal(v, src_state[k]), f'resumed weight {k} did not match'


def test_train_research_hist_raises_on_nan_obs() -> None:
    """A-6: non-finite obs must fail closed before policy act."""
    import pytest
    from unittest.mock import patch

    from src.eval.research_alpha_train import build_research_hist_env, train_research_hist

    rets, fac = _toy_panel(t=20, k=3)
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": 3,
        "train_epochs": 1,
        "train_episodes": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "rl_backend": "custom",
    }
    env = build_research_hist_env(rets, fac, cfg)
    orig_reset = env.reset

    def reset_nan(*args, **kwargs):
        obs, info = orig_reset(*args, **kwargs)
        bad = np.full(int(np.asarray(obs).size), np.nan, dtype=np.float32)
        return bad, info

    env.reset = reset_nan

    with patch("src.eval.research_alpha_train.build_research_hist_env", return_value=env):
        with pytest.raises(ValueError, match="non-finite|NaN|inf"):
            train_research_hist(rets, fac, cfg, seed=0)


def test_sb3_agent_policy_module_for_ppo_sac_recurrent() -> None:
    """Checkpoint bridge must resolve net for SB3 PPO, SAC, and RecurrentPPO."""
    import pytest
    import torch
    from src.eval.research_alpha_train import _agent_policy_module
    from src.policy.sb3_adapter import make_sb3_agent
    from src.policy.single_agent import make_single_agent

    agents = [
        make_single_agent('ppo', obs_dim=8, action_dim=2, lr=1e-3, rl_backend='sb3'),
        make_single_agent('sac', obs_dim=8, action_dim=2, lr=1e-3, rl_backend='sb3'),
    ]
    pytest.importorskip('sb3_contrib')
    agents.append(
        make_sb3_agent('ppo_recurrent', obs_dim=8, action_dim=2, num_assets=2, seq_len=2)
    )
    for agent in agents:
        mod = _agent_policy_module(agent)
        assert mod is not None, f'_agent_policy_module None for {agent.name}'
        assert isinstance(mod, torch.nn.Module)
        assert isinstance(agent.net, torch.nn.Module)
