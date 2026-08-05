"""TDD: train_world spectrum - allowlist not exclusive rBergomi ban."""
from __future__ import annotations

import pytest
import torch

from mascotrl.arms.spec import ALLOWED_TRAIN_DISTRIBUTIONS, ArmSpec
from mascotrl.env.cmdp_env import ALLOWED_TRANSITION_SOURCES, CMDPEnv
from mascotrl.features.extractor import AlphaFeatureExtractor
from mascotrl.policy.happo import HAPPOEngine
from mascotrl.reporting.research_alpha_router import (
    RESEARCH_PRIMARY_ALLOWED,
    resolve_research_primary_train,
)
from mascotrl.spectrum.registry import allowed_ids


def test_arm_spec_allows_spectrum_train_worlds() -> None:
    for world in ("historical", "rbergomi", "gbm", "heston", "garch", "hybrid_pretrain_finetune"):
        assert world in ALLOWED_TRAIN_DISTRIBUTIONS
        ArmSpec(id="opt", option_slots=2, equity_slots=0, train_distribution=world)


def test_arm_spec_train_worlds_subset_of_registry() -> None:
    for w in ALLOWED_TRAIN_DISTRIBUTIONS:
        assert w in allowed_ids("train_world") or w == "hybrid_pretrain_finetune"


def test_cmdp_acceptance_allows_declared_historical() -> None:
    surfaces = torch.rand(2, 3, 8, 2, 2) * 0.2 + 0.1
    fe = AlphaFeatureExtractor(num_assets=3, d_model=4, d_state=8, use_dhgnn=False)
    policy = HAPPOEngine(num_assets=3, enriched_dim=4, macro_dim=2, use_projection=False)
    env = CMDPEnv(
        surfaces=surfaces,
        feature_extractor=fe,
        policy=policy,
        d_model=4,
        macro_dim=2,
        use_gpu=False,
        seq_len=4,
        acceptance_mode=True,
        transition_source="historical",
    )
    assert env.transition_source == "historical"
    assert "historical" in ALLOWED_TRANSITION_SOURCES


def test_cmdp_acceptance_allows_declared_gbm() -> None:
    surfaces = torch.rand(2, 3, 8, 2, 2) * 0.2 + 0.1
    fe = AlphaFeatureExtractor(num_assets=3, d_model=4, d_state=8, use_dhgnn=False)
    policy = HAPPOEngine(num_assets=3, enriched_dim=4, macro_dim=2, use_projection=False)
    env = CMDPEnv(
        surfaces=surfaces,
        feature_extractor=fe,
        policy=policy,
        d_model=4,
        macro_dim=2,
        use_gpu=False,
        seq_len=4,
        acceptance_mode=True,
        transition_source="gbm",
    )
    assert env.transition_source == "gbm"


def test_cmdp_unknown_transition_raises() -> None:
    surfaces = torch.rand(2, 3, 8, 2, 2) * 0.2 + 0.1
    fe = AlphaFeatureExtractor(num_assets=3, d_model=4, d_state=8, use_dhgnn=False)
    policy = HAPPOEngine(num_assets=3, enriched_dim=4, macro_dim=2, use_projection=False)
    with pytest.raises(ValueError, match="unknown transition_source"):
        CMDPEnv(
            surfaces=surfaces,
            feature_extractor=fe,
            policy=policy,
            d_model=4,
            macro_dim=2,
            use_gpu=False,
            seq_len=4,
            transition_source="not_a_world",
        )


def test_research_primary_allows_hybrid() -> None:
    assert "hybrid_pretrain_finetune" in RESEARCH_PRIMARY_ALLOWED
    out = resolve_research_primary_train({"primary_train": "hybrid_pretrain_finetune"})
    assert out == "hybrid_pretrain_finetune"
