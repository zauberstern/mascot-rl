"""Surface signals wired into equity feature cube."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.features.blocks.assemble import assemble_equity_feature_cube
from src.features.blocks.obs_builder import PanelObservationBuilder
from src.features.surface_cnn import SurfaceImageEncoder


def test_iv_surface_dict_channels_enter_cube():
    t, k = 30, 4
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=(t, k))
    skew = rng.normal(0.05, 0.01, size=(t, k))
    cube, names = assemble_equity_feature_cube(
        rets,
        extras={"iv_surface": {"iv_skew_30d": skew}, "include_iv_surface": True},
        normalize=False,
    )
    assert "iv_skew_30d" in names
    assert cube.shape == (t, k, len(names))
    assert cube.shape[-1] > 0


def test_panel_obs_builder_with_surface():
    t, k = 20, 3
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, size=(t, k))
    surf = {"cw_vol_spread": rng.normal(0, 0.01, size=(t, k))}
    b = PanelObservationBuilder(
        rets, extras={"iv_surface": surf, "include_iv_surface": True}
    )
    obs = b(5, np.zeros(k))
    assert obs.ndim == 1
    assert obs.size == k * b.obs_channels_per_asset


def test_surface_cnn_forward_shape():
    enc = SurfaceImageEncoder(embed_dim=8)
    x = torch.randn(4, 1, 11, 34)
    y = enc(x)
    assert y.shape == (4, 8)


def test_include_surface_image_encoder_without_kelly_images_raises():
    t, k = 10, 3
    rets = np.zeros((t, k))

    with pytest.raises(ValueError, match="kelly_images"):
        assemble_equity_feature_cube(
            rets, extras={"include_surface_image_encoder": True}, normalize=False
        )


def test_kelly_images_flatten_into_374_raw_pixel_channels():
    t, k = 6, 2
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=(t, k))
    images = rng.normal(0.2, 0.05, size=(t, k, 11, 34))
    cube, names = assemble_equity_feature_cube(
        rets,
        extras={"include_surface_image_encoder": True, "kelly_images": images},
        normalize=False,
    )
    assert sum(n.startswith("kelly_px_") for n in names) == 11 * 34
    assert cube.shape[-1] >= 11 * 34


def test_asset_temporal_actor_critic_surface_image_encoder_gets_gradient():
    """B4: SurfaceImageEncoder is trainable end to end, not a shape-only stub.

    A backward pass through the actor/critic heads must produce a nonzero
    gradient on the CNN's own weights, proving the raw Kelly IV-surface
    pixels flow through the encoder inside the autograd graph rather than
    being precomputed and detached.
    """
    from src.policy.single_agent import _AssetTemporalActorCritic

    num_assets, seq_len, base_channels, embed_dim = 3, 2, 5, 4
    image_channels = 11 * 34
    d_model = base_channels + image_channels
    net = _AssetTemporalActorCritic(
        num_assets=num_assets,
        d_model=d_model,
        action_dim=num_assets,
        seq_len=seq_len,
        temporal_backend="gru",
        use_surface_image_encoder=True,
        image_channels=image_channels,
        surface_image_embed_dim=embed_dim,
    )
    obs = torch.randn(2, num_assets * seq_len * d_model, requires_grad=False)
    value = net.value(obs)
    mean = net.mean(obs)
    loss = value.sum() + mean.sum()
    loss.backward()
    conv_weight = net.image_encoder.net[0].weight
    assert conv_weight.grad is not None
    assert torch.any(conv_weight.grad != 0)


def test_asset_temporal_actor_critic_rejects_wrong_image_channels():
    from src.policy.single_agent import _AssetTemporalActorCritic

    with pytest.raises(ValueError, match="374"):
        _AssetTemporalActorCritic(
            num_assets=2,
            d_model=20,
            action_dim=2,
            use_surface_image_encoder=True,
            image_channels=10,
        )
