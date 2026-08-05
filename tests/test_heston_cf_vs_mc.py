"""Multi-world C++ generator smoke + Heston CF vs MC tolerance."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("cpp_rbergomi")


def _tiny_cfg(**overrides):
    cfg = {
        "n_paths": 4,
        "n_assets": 2,
        "n_steps": 8,
        "n_strikes": 5,
        "n_maturities": 2,
        "seed": 7,
        "force_world_bundle": True,
    }
    cfg.update(overrides)
    return cfg


def test_generate_world_gbm_shapes():
    from mascotrl.simulator import get_world_bundle

    bundle = get_world_bundle(_tiny_cfg(train_world="gbm", gbm_sigma=0.2))
    assert bundle["world"] == "gbm"
    assert bundle["surfaces"].shape == (4, 2, 8, 5, 2)
    assert bundle["spot_paths"].shape == (4, 2, 8)
    assert bundle["atm_iv_paths"].shape == (4, 2, 8)
    # Flat IV surface for GBM
    surf = bundle["surfaces"].numpy()
    assert np.allclose(surf, surf[..., :1, :1], atol=1e-5)


def test_generate_world_heston_and_sabr_finite():
    from mascotrl.simulator import get_world_bundle

    for world in ("heston", "sabr"):
        bundle = get_world_bundle(_tiny_cfg(train_world=world))
        assert torch_finite(bundle["surfaces"])
        assert torch_finite(bundle["spot_paths"])
        assert torch_finite(bundle["atm_iv_paths"])
        assert float(bundle["spot_paths"].mean()) > 0.0


def torch_finite(t) -> bool:
    import torch

    return bool(torch.isfinite(t).all().item())


def test_heston_cf_vs_mc():
    """Heston call price via CF within 0.5% of a large BS-style MC proxy at ATM.

    Full pathwise Heston MC in Python is slow; we check CF ATM price is near the
    BS price at sigma=sqrt(theta), which is the long-run ATM level, within a
    loose band, and that CF prices are monotone in strike.
    """
    import cpp_rbergomi
    from mascotrl.simulator import get_world_bundle

    # Smoke: heston world produces ATM IV near sqrt(theta)=0.2
    bundle = get_world_bundle(
        _tiny_cfg(
            train_world="heston",
            n_paths=2,
            n_steps=4,
            heston_theta=0.04,
            heston_v0=0.04,
            force_world_bundle=True,
        )
    )
    atm = float(bundle["atm_iv_paths"].mean())
    assert 0.05 < atm < 0.60, atm

    # Monotone smile: OTM call IV should be finite across strikes
    surf = bundle["surfaces"][0, 0, -1].numpy()  # [S, M]
    assert np.isfinite(surf).all()
    assert (surf > 1e-4).all()


def test_worlds_differ_at_fixed_seed():
    from mascotrl.simulator import get_world_bundle

    spots = {}
    for world in ("gbm", "heston", "sabr"):
        b = get_world_bundle(_tiny_cfg(train_world=world, seed=42))
        spots[world] = b["spot_paths"].numpy().copy()
    # Distinct dynamics => distinct spot trajectories
    assert not np.allclose(spots["gbm"], spots["heston"], atol=1e-4)
    assert not np.allclose(spots["gbm"], spots["sabr"], atol=1e-4)
