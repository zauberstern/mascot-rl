"""JKP / lottery block math checks."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.features.blocks.jkp_lottery import (
    beta_asym,
    build_jkp_lottery_block,
    idio_vol_ff4,
    max_ret,
    min_ret,
)


def test_max_min_ret_match_numpy() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, size=(40, 2))
    mx = max_ret(r, 21)
    mn = min_ret(r, 21)
    for t in range(20, 40):
        np.testing.assert_allclose(mx[t], np.max(r[t - 20 : t + 1], axis=0))
        np.testing.assert_allclose(mn[t], np.min(r[t - 20 : t + 1], axis=0))


def test_idio_vol_recovers_noise_scale() -> None:
    rng = np.random.default_rng(2)
    t, k = 120, 2
    factors = rng.normal(0, 0.01, size=(t, 4))
    true_beta = np.array([[0.5, -0.2, 0.1, 0.0], [1.0, 0.0, -0.3, 0.2]])
    noise = rng.normal(0, 0.02, size=(t, k))
    r = factors @ true_beta.T + noise
    iv = idio_vol_ff4(r, factors, window=63)
    # Annualized noise std ≈ 0.02 * sqrt(252) ≈ 0.317
    late = iv[80:]
    assert np.nanmean(late) == pytest.approx(0.02 * np.sqrt(252), rel=0.35)


def test_beta_asym_sign() -> None:
    rng = np.random.default_rng(3)
    t = 100
    mkt = rng.normal(0, 0.01, size=t)
    # Asset with higher downside beta.
    r = np.where(mkt[:, None] < 0, 1.5 * mkt[:, None], 0.5 * mkt[:, None])
    r = r + rng.normal(0, 1e-4, size=r.shape)
    ba = beta_asym(r, mkt, window=63)
    assert np.nanmean(ba[70:]) > 0.3


def test_build_jkp_lottery_block_names() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, size=(80, 3))
    f = np.random.default_rng(1).normal(0, 0.01, size=(80, 4))
    jkp = {
        "log_me": np.ones((80, 3)),
        "ivol_capm_21d": np.ones((80, 3)) * 0.2,
        "ret_1_0": np.zeros((80, 3)),
    }
    cube, names = build_jkp_lottery_block(r, jkp=jkp, factors=f)
    assert "max_ret_21" in names
    assert "idio_vol_ff4_21" in names
    assert "log_me" in names
    assert cube.shape[-1] == len(names)
