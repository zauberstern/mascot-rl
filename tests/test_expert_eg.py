"""Tests for Helmbold EG-experts and Wintenberger BOA."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.expert_eg import boa_experts, eg_experts
from tests.test_variable_share import _four_regime_losses_01


def test_eg_experts_concentrates_on_dominant() -> None:
    t, n = 250, 4
    R = np.full((t, n), 0.0005, dtype=np.float64)
    R[:, 2] += 0.001  # +10 bps extra every day on expert 2
    w = eg_experts(R, eta=0.05)
    assert w.shape == R.shape
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    assert int(np.argmax(w[200])) == 2


def test_eg_experts_prefix_stable() -> None:
    rng = np.random.default_rng(1)
    R = rng.normal(0.0005, 0.01, size=(120, 3))
    w_full = eg_experts(R, eta=0.05)
    t = 60
    w_pref = eg_experts(R[:t], eta=0.05)
    np.testing.assert_allclose(w_full[:t], w_pref, rtol=1e-10, atol=1e-12)


def test_boa_on_unit_losses_tracks_blocks() -> None:
    L = _four_regime_losses_01(150, 3)
    w = boa_experts(L, eta=1.0, alpha=0.02)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    for i in range(3):
        mid = i * 150 + 75
        assert int(np.argmax(w[mid])) == i


def test_boa_rejects_out_of_range() -> None:
    import pytest

    L = np.array([[0.0, 1.5]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        boa_experts(L, eta=1.0)
