"""Step 16: CMDPEnv transition_source must be a declared spectrum world."""
from __future__ import annotations

import pytest
import torch

from mascotrl.env.cmdp_env import ALLOWED_TRANSITION_SOURCES, CMDPEnv
from mascotrl.features.extractor import AlphaFeatureExtractor
from mascotrl.policy.happo import HAPPOEngine


def _tiny_env_kwargs():
    surfaces = torch.rand(2, 3, 8, 2, 2) * 0.2 + 0.1
    fe = AlphaFeatureExtractor(num_assets=3, d_model=4, d_state=8, use_dhgnn=False)
    policy = HAPPOEngine(
        num_assets=3,
        enriched_dim=4,
        macro_dim=2,
        use_projection=False,
    )
    return dict(
        surfaces=surfaces,
        feature_extractor=fe,
        policy=policy,
        d_model=4,
        macro_dim=2,
        use_gpu=False,
        seq_len=4,
    )


def test_unknown_transition_raises():
    kw = _tiny_env_kwargs()
    with pytest.raises(ValueError, match="unknown transition_source"):
        CMDPEnv(**kw, acceptance_mode=True, transition_source="not_real")


def test_acceptance_plus_declared_rbergomi_allowed():
    kw = _tiny_env_kwargs()
    env = CMDPEnv(**kw, acceptance_mode=True, transition_source="rbergomi")
    assert env.transition_source == "rbergomi"
    assert "rbergomi" in ALLOWED_TRANSITION_SOURCES


def test_ablation_rbergomi_allowed():
    kw = _tiny_env_kwargs()
    env = CMDPEnv(**kw, acceptance_mode=False, transition_source="rbergomi")
    assert env is not None
    out = env.reset(path=0, start_t=1, episode_seed=0)
    assert out.reward is not None


def test_synthetic_alias_maps_to_rbergomi():
    kw = _tiny_env_kwargs()
    env = CMDPEnv(**kw, acceptance_mode=False, transition_source="synthetic")
    assert env.transition_source == "rbergomi"
