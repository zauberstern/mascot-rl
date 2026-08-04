"""WP-P3: w_base observation channel on PanelObservationBuilder."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.blocks.obs_builder import PanelObservationBuilder


def test_w_base_channel_equals_ew_on_mask() -> None:
    t, k = 12, 5
    rets = np.random.randn(t, k) * 0.01
    builder = PanelObservationBuilder(rets, seq_len=1, normalize=False)
    assert builder.obs_channels_per_asset == builder.n_channels + 4
    assert "w_base" in builder.portfolio_channel_names

    mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    builder.set_slot_mask(mask)
    w_prev = np.zeros(k)
    obs = builder(5, w_prev)
    feats = obs.reshape(k, builder.obs_channels_per_asset)
    w_base_col = feats[:, -1]
    expected = mask / mask.sum()
    assert np.allclose(w_base_col, expected, atol=1e-8)


def test_w_base_shifts_when_mask_changes() -> None:
    t, k = 8, 4
    rets = np.random.randn(t, k) * 0.01
    builder = PanelObservationBuilder(rets, seq_len=1, normalize=False)
    builder.set_slot_mask(np.ones(k))
    a = builder(3, np.zeros(k)).reshape(k, -1)[:, -1]
    builder.set_slot_mask(np.array([1.0, 0.0, 1.0, 0.0]))
    b = builder(3, np.zeros(k)).reshape(k, -1)[:, -1]
    assert not np.allclose(a, b)
    assert np.allclose(b, np.array([0.5, 0.0, 0.5, 0.0]))


def test_seq_obs_dim_includes_w_base() -> None:
    t, k = 10, 3
    rets = np.random.randn(t, k) * 0.01
    builder = PanelObservationBuilder(rets, seq_len=4, normalize=False)
    obs = builder(7, np.zeros(k))
    assert obs.size == k * 4 * builder.obs_channels_per_asset
