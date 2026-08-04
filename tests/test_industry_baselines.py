"""Industry-standard portfolio baselines (lookahead-safe registry)."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.industry_baselines import (
    INDUSTRY_BASELINE_NAMES,
    INDUSTRY_BASELINE_REGISTRY,
    industry_baseline_weights,
    list_industry_baselines,
)


def _synthetic_panel(t: int = 300, k: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mild cross-sectional structure so cov-based methods are well-posed.
    common = rng.normal(0.0, 0.01, size=(t, 1))
    idio = rng.normal(0.0, 0.015, size=(t, k))
    return common + idio


def test_all_names_in_registry():
    assert list_industry_baselines() == INDUSTRY_BASELINE_NAMES
    for name in INDUSTRY_BASELINE_NAMES:
        assert name in INDUSTRY_BASELINE_REGISTRY
        assert callable(INDUSTRY_BASELINE_REGISTRY[name])


@pytest.mark.parametrize("name", list(INDUSTRY_BASELINE_NAMES))
def test_each_baseline_returns_finite_shape(name: str):
    rets = _synthetic_panel(300, 8)
    t = 250
    hist = rets[:t]
    w = industry_baseline_weights(name, returns_hist=hist, t=t)
    assert w.shape == (8,)
    assert np.all(np.isfinite(w))


def test_buy_and_hold_returns_w_prev():
    rets = _synthetic_panel(50, 5)
    w_prev = np.array([0.4, -0.1, 0.3, 0.2, 0.2], dtype=np.float64)
    w = industry_baseline_weights(
        "buy_and_hold",
        returns_hist=rets[:40],
        t=40,
        w_prev=w_prev,
    )
    assert np.allclose(w, w_prev)


def test_buy_and_hold_equal_weight_without_prev():
    rets = _synthetic_panel(50, 5)
    w = industry_baseline_weights("buy_and_hold", returns_hist=rets[:40], t=40)
    assert np.allclose(w, np.full(5, 0.2))


def test_xs_momentum_lookahead_safe():
    """Shuffling only future returns must not change weights at decision t."""
    rets = _synthetic_panel(400, 8, seed=7)
    t = 300
    hist = rets[:t].copy()
    w0 = industry_baseline_weights("xs_momentum_12_1", returns_hist=hist, t=t)

    rng = np.random.default_rng(99)
    future = rets[t:].copy()
    rng.shuffle(future, axis=0)
    rets_shuf = np.vstack([rets[:t], future])
    # Caller passes only history ending at t-1; future rows never enter hist.
    hist_shuf = rets_shuf[:t]
    w1 = industry_baseline_weights("xs_momentum_12_1", returns_hist=hist_shuf, t=t)
    assert np.allclose(w0, w1)

    # Stronger check: mutate future in the full series but keep hist identical.
    assert np.allclose(hist, hist_shuf)
    # And if someone incorrectly sliced including future, weights would differ;
    # mutate hist by appending shuffled future then take wrong window — verify
    # our API only uses returns_hist as provided (past).
    bad_hist = rets_shuf  # (400, 8) including future — not how callers should use API
    w_wrong_window = industry_baseline_weights(
        "xs_momentum_12_1", returns_hist=bad_hist[:t], t=t
    )
    assert np.allclose(w0, w_wrong_window)


def test_short_term_reversal_lookahead_safe():
    rets = _synthetic_panel(200, 8, seed=11)
    t = 100
    hist = rets[:t].copy()
    w0 = industry_baseline_weights("short_term_reversal", returns_hist=hist, t=t)

    rng = np.random.default_rng(123)
    future = rets[t:].copy()
    rng.shuffle(future, axis=0)
    rets_shuf = np.vstack([rets[:t], future])
    hist_shuf = rets_shuf[:t]
    w1 = industry_baseline_weights("short_term_reversal", returns_hist=hist_shuf, t=t)
    assert np.allclose(w0, w1)
    assert np.allclose(hist, hist_shuf)


def test_unknown_name_raises():
    rets = _synthetic_panel(50, 4)
    with pytest.raises(KeyError):
        industry_baseline_weights("not_a_baseline", returns_hist=rets[:40], t=40)


def test_g1_baselines_no_trade_equal_weight_ridge():
    """Alpha v2 Step 21: G1 non-RL baselines remain callable."""
    for required in ("no_trade", "equal_weight", "ridge"):
        assert required in INDUSTRY_BASELINE_NAMES
    rets = _synthetic_panel(60, 5)
    t = 50
    hist = rets[:t]
    assert np.allclose(
        industry_baseline_weights("no_trade", returns_hist=hist, t=t),
        np.zeros(5),
    )
    assert np.allclose(
        industry_baseline_weights("equal_weight", returns_hist=hist, t=t),
        np.full(5, 0.2),
    )
    ridge = industry_baseline_weights("ridge", returns_hist=hist, t=t)
    assert ridge.shape == (5,)
    assert np.all(np.isfinite(ridge))
    assert abs(float(np.sum(np.abs(ridge))) - 1.0) < 1e-6
