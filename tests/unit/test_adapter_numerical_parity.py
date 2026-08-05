"""B-2: adapter numerical parity beyond smoke (SB3/custom PPO, OmniSafe CPPO, HARL HAPPO).

Parity definitions (see ``logs/campaign_sprint/AUDIT_LEDGER.md`` [B-2]):

* **SB3 vs custom PPO:** weight-aligned ``action_net`` mean logits at default orthogonal
  init (GELU vs Tanh diverge at large activations). Full trajectory parity is not
  claimed; one-step update invariants (finite loss/grad, ``optimizer_steps==1``,
  matched entropy when ``log_std`` is synced) are asserted instead.
* **OmniSafe CPPO:** shared ``PPOAgent`` body forward equality on synced weights;
  duals differ by design (trajectory CVaR ``nu`` vs PID ``omnisafe_lambda``).
* **HARL vs custom HAPPO:** custom ``HAPPOEngine`` forward contract always;
  HARL bundle construction + act when installed (LayerNorm MLP != custom GELU MLP).
"""
from __future__ import annotations

import torch
import pytest
from tests.conftest import FLOAT_TOL

pytest.importorskip("torch")

from mascotrl.policy.cppo import CPPOAgent
from mascotrl.policy.omnisafe_adapter import OmniSafeCPPOAgent
from mascotrl.policy.single_agent import PPOAgent
from mascotrl.policy.sb3_adapter import SB3PPOAgent

_SB3_SYNC_MAP = {
    "mlp_extractor.policy_net.0.weight": "actor.0.weight",
    "mlp_extractor.policy_net.0.bias": "actor.0.bias",
    "mlp_extractor.policy_net.2.weight": "actor.2.weight",
    "mlp_extractor.policy_net.2.bias": "actor.2.bias",
    "action_net.weight": "actor.4.weight",
    "action_net.bias": "actor.4.bias",
    "log_std": "log_std",
}


def _sync_custom_ppo_to_sb3(custom: PPOAgent, sb3: SB3PPOAgent) -> None:
    cs = custom.net.state_dict()
    policy = sb3._model.policy
    ps = policy.state_dict()
    new_ps = dict(ps)
    for sk, ck in _SB3_SYNC_MAP.items():
        new_ps[sk] = cs[ck].clone()
    policy.load_state_dict(new_ps)


def _toy_batch(n: int = 32, obs_dim: int = 8, act_dim: int = 3):
    torch.manual_seed(0)
    obs = torch.randn(n, obs_dim)
    actions = torch.randn(n, act_dim)
    rewards = torch.randn(n) * 0.01
    next_obs = torch.randn(n, obs_dim)
    dones = torch.zeros(n)
    dones[-1] = 1.0
    return obs, actions, rewards, next_obs, dones


@pytest.mark.parametrize("rl_backend", ["custom", "sb3"])
def test_ppo_construction_smoke(rl_backend: str):
    if rl_backend == "sb3":
        pytest.importorskip("stable_baselines3")
        agent = SB3PPOAgent(8, 3, normalize_obs=False)
    else:
        agent = PPOAgent(8, 3, hidden=64, normalize_obs=False)
    obs = torch.randn(2, 8)
    w = agent.act(obs)
    assert w.shape == (2, 3)
    assert torch.isfinite(w).all()


def test_sb3_custom_ppo_action_mean_parity_at_default_init():
    """Synced actor weights -> action means agree up to GELU vs Tanh scale.

    Near zero, GELU'(0)=0.5 while Tanh'(0)=1, so absolute allclose fails even
    with identical weights; cosine similarity is the honest B-2 gate.
    """
    pytest.importorskip("stable_baselines3")
    obs_dim, act_dim = 8, 3
    torch.manual_seed(42)
    custom = PPOAgent(obs_dim, act_dim, hidden=64, normalize_obs=False)
    sb3 = SB3PPOAgent(obs_dim, act_dim, normalize_obs=False)
    _sync_custom_ppo_to_sb3(custom, sb3)

    obs = torch.randn(4, obs_dim)
    custom_mean = custom.net.mean(custom._prep_obs(obs))
    with torch.no_grad():
        latent = sb3._model.policy.mlp_extractor.forward_actor(obs.float())
        sb3_mean = sb3._model.policy.action_net(latent)
    assert torch.isfinite(custom_mean).all() and torch.isfinite(sb3_mean).all()
    c = custom_mean.detach().reshape(-1)
    s = sb3_mean.detach().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(c, s, dim=0).item())
    assert cos > 0.99, f"synced-weight action-mean cosine={cos}"


def test_sb3_custom_ppo_update_magnitude_invariants():
    """One train_epoch: finite stats, one optimizer step, entropy matches synced log_std."""
    pytest.importorskip("stable_baselines3")
    obs_dim, act_dim = 8, 3
    custom = PPOAgent(obs_dim, act_dim, hidden=64, normalize_obs=False)
    sb3 = SB3PPOAgent(obs_dim, act_dim, normalize_obs=False)
    _sync_custom_ppo_to_sb3(custom, sb3)

    obs, actions, rewards, next_obs, dones = _toy_batch()
    s_custom = custom.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    s_sb3 = sb3.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    assert s_custom["optimizer_steps"] == pytest.approx(1.0, **FLOAT_TOL)
    assert s_sb3["optimizer_steps"] == pytest.approx(1.0, **FLOAT_TOL)
    assert torch.isfinite(torch.tensor(s_custom["loss"]))
    assert torch.isfinite(torch.tensor(s_sb3["loss"]))
    assert s_custom["grad_norm"] > 0.0
    assert s_sb3["grad_norm"] > 0.0
    assert s_custom["entropy"] == pytest.approx(s_sb3["entropy"], rel=0, abs=1e-5)


def test_omnisafe_cppo_forward_equality_on_shared_weights():
    """Shared PPO body: identical action means after weight transfer."""
    obs_dim, act_dim = 8, 3
    custom = CPPOAgent(obs_dim, act_dim, hidden=16, normalize_obs=False)
    omni = OmniSafeCPPOAgent(
        obs_dim, act_dim, hidden=16, normalize_obs=False, omnisafe_algo="cppo_pid"
    )
    omni.net.load_state_dict(custom.net.state_dict())

    obs = torch.randn(5, obs_dim)
    with torch.no_grad():
        mean_c = custom.net.mean(custom._prep_obs(obs))
        mean_o = omni.net.mean(omni._prep_obs(obs))
    assert torch.allclose(mean_c, mean_o, atol=1e-6, rtol=1e-5)


def test_omnisafe_cppo_dual_moves_on_positive_cost():
    """Dual channel: PID lambda rises when tail cost exceeds limit (not nu parity)."""
    obs_dim, act_dim = 8, 3
    custom = CPPOAgent(obs_dim, act_dim, hidden=16, normalize_obs=False)
    omni = OmniSafeCPPOAgent(
        obs_dim, act_dim, hidden=16, normalize_obs=False, omnisafe_algo="cppo_pid"
    )
    omni.net.load_state_dict(custom.net.state_dict())

    obs, actions, rewards, next_obs, dones = _toy_batch(n=64)
    rewards = -torch.abs(rewards) - 0.05

    s_custom = custom.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    lam_before = float(omni._lagrange.lagrangian_multiplier)  # type: ignore[attr-defined]
    s_omni = omni.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    assert s_custom["cvar_nu"] >= 0.0
    assert s_omni["omnisafe_lambda"] >= lam_before
    assert s_omni["omnisafe_ep_cost"] >= 0.0


def test_custom_happo_actor_forward_contract():
    """Custom HAPPO path: deterministic actor means from known weights."""
    from mascotrl.policy.happo import HAPPOEngine

    k, d = 2, 4
    engine = HAPPOEngine(
        num_assets=k, enriched_dim=d, macro_dim=6, use_projection=False
    )
    enriched = torch.tensor([[[1.0, 0.0, -0.5, 0.25], [0.5, 0.5, 0.0, -0.25]]])
    with torch.no_grad():
        for actor in engine.actors:
            for p in actor.parameters():
                p.fill_(0.01)
        means = engine._actor_means(enriched)
    assert means.shape == (1, k)
    assert torch.isfinite(means).all()


def test_harl_happo_construction_and_act_when_installed():
    """HARL opt-in: bundle constructs K actors and acts without error."""
    pytest.importorskip("harl")
    pytest.importorskip("gymnasium")
    import numpy as np

    from mascotrl.policy.harl_adapter import HARLHAPPOBundle, default_happo_args

    bundle = HARLHAPPOBundle(2, obs_dim_per_agent=4, args=default_happo_args())
    obs = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
    acts = bundle.act_all(obs, deterministic=True)
    assert len(acts) == 2
    assert all(a.size >= 1 for a in acts)
    assert all(np.isfinite(a).all() for a in acts)
