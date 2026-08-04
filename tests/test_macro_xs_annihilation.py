"""Macro channels must survive default cross-section normalize."""
from __future__ import annotations

import numpy as np

from src.features.blocks.assemble import assemble_equity_feature_cube


def test_macro_survives_default_normalize() -> None:
    t, k, f = 30, 5, 3
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=(t, k))
    macro = np.linspace(-1.0, 1.0, t * f).reshape(t, f)
    cube, names = assemble_equity_feature_cube(
        r,
        {
            "macro": macro,
            "macro_names": [f"m{i}" for i in range(f)],
        },
        normalize=True,
    )
    idx = [names.index(f"m{i}") for i in range(f)]
    assert idx
    np.testing.assert_allclose(cube[:, 0, idx], cube[:, 1, idx], equal_nan=True)
    assert np.nanstd(cube[2:, 0, idx[0]]) > 1e-6


def test_inactive_slots_nan_before_xs_normalize() -> None:
    t, k = 20, 4
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=(t, k))
    r[:, 3] = 0.0
    mask = np.ones((t, k), dtype=bool)
    mask[:, 3] = False
    cube_masked, _ = assemble_equity_feature_cube(
        r, {"slot_valid_mask": mask}, normalize=True
    )
    cube_ref, _ = assemble_equity_feature_cube(r[:, :3], normalize=True)
    np.testing.assert_allclose(
        cube_masked[:, :3, : cube_ref.shape[-1]],
        cube_ref,
        rtol=1e-5,
        atol=1e-5,
    )


def test_kelly_missing_is_nan_not_zero() -> None:
    t, k = 8, 2
    r = np.zeros((t, k))
    img = np.full((t, k, 11, 34), np.nan, dtype=np.float64)
    cube, names = assemble_equity_feature_cube(
        r,
        {"kelly_images": img, "include_surface_image_encoder": True},
        normalize=False,
    )
    assert names
    assert np.isnan(cube).any()
    assert not np.all(cube == 0.0)
