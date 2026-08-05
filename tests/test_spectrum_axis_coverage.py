"""Part D.9: every registry id maps to an implementing symbol or is refused."""
from __future__ import annotations
import importlib
import pytest
from mascotrl.spectrum.registry import AXES, all_options, get_option, validate_cfg
_IMPLEMENTATIONS: dict[str, dict[str, str]] = {'architecture': {'mlp': 'mascotrl.policy.bodies.MLPPolicyBody', 'gru': 'mascotrl.policy.bodies.AssetTemporalPolicyBody', 'lstm': 'mascotrl.policy.bodies.AssetTemporalPolicyBody', 'transformer': 'mascotrl.policy.bodies.AssetTemporalPolicyBody', 'mamba': 'mascotrl.policy.bodies.AssetTemporalPolicyBody'}, 'algo': {'ppo': 'mascotrl.policy.single_agent.PPOAgent', 'sac': 'mascotrl.policy.single_agent.SACAgent', 'td3': 'mascotrl.policy.single_agent.TD3Agent', 'ddpg': 'mascotrl.policy.single_agent.DDPGAgent', 'dqn': 'mascotrl.policy.single_agent.DQNAgent', 'mcpg': 'mascotrl.policy.single_agent.MCPGAgent', 'rrl': 'mascotrl.policy.single_agent.RRLAgent', 'happo': 'mascotrl.policy.trainer.HAPPOTrainer', 'cppo': 'mascotrl.policy.cppo.CPPOAgent'}, 'objective': {'mean_std_cao': 'mascotrl.policy.objective_factory.episode_weights', 'mtm_pnl': 'mascotrl.env.historical_env.HistoricalArmEnv', 'meanvar_kolm': 'mascotrl.policy.objective_factory.episode_weights', 'cvar_ru': 'mascotrl.policy.objective_factory.episode_weights', 'entropic_oce': 'mascotrl.policy.objective_factory.episode_weights', 'smse': 'mascotrl.policy.objective_factory.episode_weights', 'rsqp': 'mascotrl.policy.objective_factory.episode_weights', 'differential_sharpe': 'mascotrl.eval.differential_sharpe.DifferentialSharpe', 'mikkila_asym': 'mascotrl.policy.objective_factory.mikkila_asym_reward', 'sdr_composite': 'mascotrl.policy.objective_factory.sdr_composite_reward'}, 'train_world': {'historical': 'mascotrl.env.historical_env.HistoricalArmEnv', 'rbergomi': 'mascotrl.simulator.get_surface_tensor', 'gbm': 'mascotrl.simulator.get_surface_tensor', 'heston': 'mascotrl.simulator.get_surface_tensor', 'garch': 'mascotrl.simulator.get_surface_tensor', 'sabr': 'mascotrl.simulator.get_surface_tensor', 'hybrid_pretrain_finetune': 'mascotrl.eval.research_alpha_cpcv.run_research_alpha_cpcv'}, 'policy_mode': {'shared': 'mascotrl.spectrum.policy_mode.resolve_policy_mode', 'archetype_carry': 'mascotrl.spectrum.policy_mode.apply_turnover_multiplier', 'archetype_inflation': 'mascotrl.spectrum.policy_mode.risk_aversion_multiplier', 'archetype_crisis': 'mascotrl.spectrum.policy_mode.amihud_drop_pct_for_mode'}}

def _resolve_symbol(path: str):
    mod_name, _, attr = path.rpartition('.')
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)

def test_every_registry_id_has_implementation_or_refusal_flag() -> None:
    for axis in AXES:
        for opt in all_options(axis):
            mapping = _IMPLEMENTATIONS.get(axis, {})
            assert opt.id in mapping, f'missing implementation map for {axis}/{opt.id}'
            _resolve_symbol(mapping[opt.id])
            if axis == 'objective' and opt.requires_episode_returns:
                with pytest.raises(ValueError, match='requires_episode_returns'):
                    validate_cfg({'objective': opt.id, 'algo': 'sac'})
                assert validate_cfg({'objective': opt.id, 'algo': 'ppo'})['objective'] == opt.id

def test_no_silent_alias_across_axis_ids() -> None:
    """Distinct option ids must not collapse to each other via validate_cfg."""
    from mascotrl.spectrum.registry import _EPISODE_RETURN_ALGOS
    for axis in AXES:
        ids = [o.id for o in all_options(axis)]
        assert len(ids) == len(set(ids))
        for oid in ids:
            cfg: dict = {axis: oid}
            if axis == 'objective':
                cfg['algo'] = 'ppo'
            elif axis == 'algo' and oid not in _EPISODE_RETURN_ALGOS:
                # Step-only / non-episode algos: use mtm_pnl (rrl refuses
                # differential_sharpe via double-DSR; others are fine either way).
                cfg['objective'] = 'mtm_pnl'
            resolved = validate_cfg(cfg)
            assert resolved[axis] == oid
            assert get_option(axis, oid).id == oid
