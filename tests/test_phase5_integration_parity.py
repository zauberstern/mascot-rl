"""Phase 5 item 41: expanded interface parity across algos and backends."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.plumbing
import torch

from src.policy.single_agent import make_single_agent

_ARTIFACT_KEYS = (
    "mean_reward",
    "n_steps",
    "rl_backend",
    "optimizer_steps",
    "train_stats",
)

# SB3 ships these; train_research_hist still forces custom for dqn/mcpg/rrl/cppo*.
_SB3_INTERFACE_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "dqn"})
_CUSTOM_ONLY_ALGOS = frozenset({"mcpg", "rrl", "cppo", "cppo_omnisafe"})
_ALL_ALGOS = ("ppo", "sac", "td3", "ddpg", "dqn", "mcpg", "rrl", "cppo", "cppo_omnisafe")


def _toy_panel(t: int = 24, k: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, factors


def _objective(algo: str) -> str:
    return "mean_std_cao" if algo in ("ppo", "cppo", "cppo_omnisafe") else "mtm_pnl"


def _make(algo: str, backend: str, *, obs_dim: int = 8, action_dim: int = 2):
    if algo == "cppo_omnisafe":
        pytest.importorskip("src.policy.omnisafe_adapter")
    kw: dict = {"hidden": 8, "lr": 1e-3}
    if algo == "dqn" and backend == "sb3":
        kw["n_bins"] = 3
    if algo in ("cppo", "cppo_omnisafe"):
        kw["normalize_obs"] = False
    if algo == "cppo_omnisafe":
        kw["omnisafe_algo"] = "cppo_pid"
    return make_single_agent(
        algo, obs_dim=obs_dim, action_dim=action_dim, rl_backend=backend, **kw
    )


def _backend_pairs():
    pairs = []
    for algo in _ALL_ALGOS:
        pairs.append((algo, "custom"))
        if algo in _SB3_INTERFACE_ALGOS:
            pairs.append((algo, "sb3"))
    return pairs


def _has_usable_opt(agent) -> bool:
    """PPO exposes policy.optimizer; SB3 off-policy uses actor/critic optimizers."""
    opt = getattr(agent, "opt", None)
    if opt is not None and hasattr(opt, "state_dict"):
        return True
    model = getattr(agent, "_model", None)
    if model is None:
        return False
    pol = getattr(model, "policy", None)
    for attr in ("optimizer", "actor", "critic"):
        obj = getattr(pol, attr, None)
        if obj is None:
            continue
        inner = getattr(obj, "optimizer", obj)
        if hasattr(inner, "state_dict"):
            return True
    return False


@pytest.mark.parametrize("algo,backend", _backend_pairs())
def test_phase5_interface_surface(algo: str, backend: str):
    """act / train_epoch / raw_to_weights / net|actor|q / opt / backend contract."""
    if backend == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
    agent = _make(algo, backend)
    assert hasattr(agent, "act")
    assert hasattr(agent, "train_epoch")
    assert _has_usable_opt(agent), f"{algo}/{backend} missing usable optimizer"
    assert getattr(agent, "backend", None) == backend

    # Policy module: at least one of net / actor / q.
    has_mod = any(
        isinstance(getattr(agent, attr, None), torch.nn.Module)
        for attr in ("net", "actor", "q")
    )
    assert has_mod, f"{algo}/{backend} missing net|actor|q"

    # raw_to_weights where the adapter exposes it (custom off-policy uses _head).
    if hasattr(agent, "raw_to_weights") and not (algo == "dqn" and backend == "sb3"):
        raw = torch.randn(2, 2)
        w_rt = agent.raw_to_weights(raw)
        assert torch.isfinite(w_rt).all()
    elif algo in ("ppo", "mcpg", "rrl", "cppo", "cppo_omnisafe") or (
        algo == "dqn" and backend == "custom"
    ):
        assert hasattr(agent, "raw_to_weights")

    obs = torch.randn(3, 8)
    w = agent.act(obs, deterministic=True)
    assert w.shape[0] == 3
    assert torch.isfinite(w).all()


@pytest.mark.parametrize(
    "algo,backend",
    [
        ("ppo", "custom"),
        ("ppo", "sb3"),
        ("sac", "custom"),
        ("sac", "sb3"),
        ("td3", "custom"),
        ("td3", "sb3"),
        ("ddpg", "custom"),
        ("ddpg", "sb3"),
        ("dqn", "custom"),  # train path forces custom even if sb3 requested
        ("mcpg", "custom"),
        ("rrl", "custom"),
        ("cppo", "custom"),
    ],
)
def test_phase5_train_research_hist_smoke(algo: str, backend: str):
    """Backends that actually train return finite mean_reward + shared keys."""
    if backend == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")

    from src.eval.research_alpha_train import train_research_hist

    rets, fac = _toy_panel(t=24, k=2)
    if algo in ("ppo", "cppo"):
        weight_head = "softmax"
    elif algo == "dqn":
        weight_head = "discrete"
    else:
        weight_head = "tanh_l1"
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": 2,
        "train_epochs": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "algo": algo,
        "objective": _objective(algo),
        "lr": 1e-3,
        "rl_backend": backend,
        "ppo_hidden": 8,
        "weight_head": weight_head,
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    for key in _ARTIFACT_KEYS:
        assert key in out, f"{algo}/{backend} missing {key}"
    mr = float(out["mean_reward"])
    assert math.isfinite(mr), f"{algo}/{backend} mean_reward={mr}"
    assert out["n_steps"] > 0
    # Forced-custom algos stamp actual backend=custom regardless of request.
    expected_backend = "custom" if algo in _CUSTOM_ONLY_ALGOS | {"dqn"} else backend
    assert out["rl_backend"] == expected_backend


def test_phase5_cppo_omnisafe_train_epoch_smoke():
    """cppo_omnisafe is not a spectrum registry algo; smoke train_epoch instead."""
    pytest.importorskip("src.policy.omnisafe_adapter")
    agent = _make("cppo_omnisafe", "custom")
    t, od, ad = 24, 8, 2
    stats = agent.train_epoch(
        obs=torch.randn(t, od),
        actions=torch.randn(t, ad),
        rewards=torch.randn(t) * 0.01,
        next_obs=torch.randn(t, od),
        dones=torch.zeros(t),
    )
    assert stats
    w = agent.act(torch.randn(1, od), deterministic=True)
    assert torch.isfinite(w).all()
    assert getattr(agent, "backend", None) == "custom"


def test_phase5_happo_construction_only():
    """HAPPO is multi-agent; assert construction, skip full train_research_hist."""
    from src.policy.happo import HAPPOEngine
    from src.policy.trainer import HAPPOTrainer

    eng = HAPPOEngine(num_assets=2, enriched_dim=4, macro_dim=2, turnover_limit=0.15)
    trainer = HAPPOTrainer(eng, use_compile=False)
    assert trainer.engine is eng
    assert hasattr(trainer, "update")
