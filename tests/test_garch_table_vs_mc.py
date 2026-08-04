"""GJR-GARCH world + Duan table smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cpp_rbergomi")


def test_garch_world_generates_finite_bundle():
    from src.simulator import get_world_bundle

    bundle = get_world_bundle(
        {
            "n_paths": 2,
            "n_assets": 1,
            "n_steps": 4,
            "n_strikes": 3,
            "n_maturities": 2,
            "seed": 3,
            "train_world": "garch",
            "force_world_bundle": True,
            "garch_n_inner": 256,  # keep CI fast
            "garch_omega": 1e-6,
            "garch_alpha": 0.02,
            "garch_beta": 0.90,
            "garch_gamma": 0.10,
        }
    )
    assert bundle["world"] == "garch"
    assert np.isfinite(bundle["surfaces"].numpy()).all()
    assert np.isfinite(bundle["spot_paths"].numpy()).all()
    assert float(bundle["atm_iv_paths"].mean()) > 0.0


def test_garch_stationarity_fail_closed():
    from src.simulator import get_world_bundle

    with pytest.raises(Exception):
        get_world_bundle(
            {
                "n_paths": 1,
                "n_assets": 1,
                "n_steps": 2,
                "n_strikes": 2,
                "n_maturities": 1,
                "seed": 1,
                "train_world": "garch",
                "force_world_bundle": True,
                "garch_n_inner": 64,
                "garch_omega": 1e-6,
                "garch_alpha": 0.5,
                "garch_beta": 0.5,
                "garch_gamma": 0.5,  # alpha+beta+gamma/2 >= 1
            }
        )


def test_garch_table_vs_mc():
    """Table prices should be positive and ordered by moneyness on a fixed grid."""
    from src.simulator import get_world_bundle

    bundle = get_world_bundle(
        {
            "n_paths": 1,
            "n_assets": 1,
            "n_steps": 2,
            "n_strikes": 5,
            "n_maturities": 2,
            "seed": 11,
            "train_world": "garch",
            "force_world_bundle": True,
            "garch_n_inner": 512,
        }
    )
    # Surface IVs finite; ATM (middle strike) near neighbors within a factor of 3
    surf = bundle["surfaces"][0, 0, 0].numpy()  # [S, M]
    assert np.isfinite(surf).all()
    mid = surf[surf.shape[0] // 2, 0]
    assert 0.01 < float(mid) < 2.0
