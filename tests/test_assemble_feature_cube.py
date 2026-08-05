"""B4: assemble_equity_feature_cube fundamentals fail-closed + surface channel sync."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.features.blocks.assemble import assemble_equity_feature_cube
from mascotrl.features.blocks.iv_surface import DEFAULT_SURFACE_CHANNELS
from mascotrl.data.surface_signals import SURFACE_SIGNAL_NAMES


def test_default_surface_channels_covers_every_gate_admissible_signal() -> None:
    """The gate can admit any name in SURFACE_SIGNAL_NAMES; the observation
    cube must be able to carry every one of them, or an admitted signal
    would silently never reach the policy."""
    missing = set(SURFACE_SIGNAL_NAMES) - set(DEFAULT_SURFACE_CHANNELS)
    assert not missing, f"signals admissible by the gate but unreachable as obs channels: {missing}"


def test_include_fundamentals_true_without_data_raises() -> None:
    r = np.random.default_rng(0).normal(size=(20, 5)) * 0.01
    with pytest.raises(ValueError, match="include_fundamentals"):
        assemble_equity_feature_cube(r, extras={"include_fundamentals": True})


def test_include_fundamentals_true_with_real_panel_is_used() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(size=(20, 5)) * 0.01
    fund = rng.normal(size=(20, 5))
    cube, names = assemble_equity_feature_cube(
        r, extras={"include_fundamentals": True, "fundamentals": fund}, normalize=False
    )
    assert "fund_0" in names
    assert cube.shape[0] == 20 and cube.shape[1] == 5


def test_include_fundamentals_false_is_a_pure_noop() -> None:
    r = np.random.default_rng(0).normal(size=(20, 5)) * 0.01
    cube, names = assemble_equity_feature_cube(r, extras={})
    assert all(not n.startswith("fund_") for n in names)
