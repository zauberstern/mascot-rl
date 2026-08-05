"""Causal expert loss maps for mixable / Variable-Share mixers."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.expert_losses import expanding_unit_interval, log_wealth_loss


def test_log_wealth_loss_zero_return() -> None:
    R = np.zeros((5, 3), dtype=np.float64)
    ell = log_wealth_loss(R)
    np.testing.assert_allclose(ell, 0.0, atol=1e-15)


def test_expanding_unit_interval_prefix_stable() -> None:
    rng = np.random.default_rng(0)
    ell = rng.normal(0, 0.01, size=(80, 4))
    full = expanding_unit_interval(ell)
    t = 40
    pref = expanding_unit_interval(ell[:t])
    np.testing.assert_allclose(full[:t], pref, rtol=1e-12, atol=1e-12)


def test_expanding_unit_interval_no_lookahead() -> None:
    rng = np.random.default_rng(1)
    ell = rng.normal(0, 0.01, size=(50, 3))
    L = expanding_unit_interval(ell)
    t = 25
    ell2 = ell.copy()
    ell2[t + 1 :] = 999.0
    L2 = expanding_unit_interval(ell2)
    np.testing.assert_allclose(L[: t + 1], L2[: t + 1], rtol=1e-12, atol=1e-12)


def test_expanding_t0_is_half() -> None:
    ell = np.array([[0.1, -0.2], [0.3, 0.0]], dtype=np.float64)
    L = expanding_unit_interval(ell)
    assert L[0, 0] == pytest.approx(0.5)
    assert L[0, 1] == pytest.approx(0.5)
    assert np.all((L >= 0.0) & (L <= 1.0))
