"""TDD: architecture spectrum - mlp/lstm/transformer/gru/mamba selectable."""
from __future__ import annotations

import torch

from mascotrl.features.extractor import AlphaFeatureExtractor
from mascotrl.plugins.resolve import _ALLOWED_ACTOR, resolve_plugins
from mascotrl.spectrum.registry import allowed_ids, validate_choice


def test_temporal_backends_from_registry() -> None:
    for backend in ("mlp", "gru", "lstm", "transformer", "mamba"):
        assert backend in allowed_ids("architecture")
        fe = AlphaFeatureExtractor(
            num_assets=2, d_model=8, d_state=4, temporal_backend=backend, use_dhgnn=False
        )
        x = torch.randn(1, 2, 6, 8)
        iv = torch.randn(1, 2, 8)
        out = fe(x, iv)
        assert out.shape == (1, 2, 8)


def test_shared_mappo_in_actor_allowlist() -> None:
    assert "shared_mappo" in _ALLOWED_ACTOR
    plugins = resolve_plugins({"plugins": {"actor_backend": "shared_mappo"}})
    assert plugins["actor_backend"] == "shared_mappo"


def test_validate_architecture_choice() -> None:
    assert validate_choice("architecture", "mlp") == "mlp"
