"""Phase G: non-RL ceiling arms (zscore / ridge / Kelly CNN)."""
from __future__ import annotations

import numpy as np
import torch

import pytest

from src.eval.ceiling_arms import (
    CEILING_ARM_NAMES,
    KellyEnsemble,
    KellySurfaceCNN,
    ceiling_arm_weight_fn,
    ridge_composite_weights,
    zscore_composite_weights,
)


def _signals_and_returns(t: int = 80, k: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    s1 = rng.normal(size=(t, k))
    s2 = rng.normal(size=(t, k))
    noise = rng.normal(0.0, 0.01, size=(t, k))
    returns = np.zeros((t, k), dtype=np.float64)
    returns[1:] = 0.03 * s1[:-1] + noise[1:]
    returns[0] = noise[0]
    return {"s1": s1, "s2": s2}, returns


def test_ceiling_arm_names_registered():
    assert "zscore_composite" in CEILING_ARM_NAMES
    assert "ridge_composite" in CEILING_ARM_NAMES
    assert "kelly_cnn" in CEILING_ARM_NAMES


def test_zscore_composite_l1_normalized():
    signals, _ = _signals_and_returns()
    w = zscore_composite_weights(signals, t=40, long_only=True)
    assert w.shape == (6,)
    assert np.all(np.isfinite(w))
    assert abs(float(np.sum(np.abs(w))) - 1.0) < 1e-6
    assert np.all(w >= -1e-12)

    w_ls = zscore_composite_weights(signals, t=40, long_only=False)
    assert abs(float(np.sum(np.abs(w_ls))) - 1.0) < 1e-6


def test_ridge_composite_runs():
    signals, returns = _signals_and_returns()
    t = 50
    w = ridge_composite_weights(
        signals,
        returns[:t],
        t=t,
        l2=1.0,
        long_only=True,
    )
    assert w.shape == (6,)
    assert np.all(np.isfinite(w))
    assert abs(float(np.sum(np.abs(w))) - 1.0) < 1e-5


def test_kelly_cnn_forward_and_weights():
    model = KellySurfaceCNN()
    images = torch.randn(4, 1, 11, 34)
    scores = model(images)
    assert scores.shape == (4,) or scores.shape == (4, 1)

    pred = model.predict(images.numpy())
    assert pred.shape[0] == 4
    w = model.scores_to_weights(pred, long_only=True)
    assert w.shape == (4,)
    assert abs(float(np.sum(np.abs(w))) - 1.0) < 1e-6

    # Stub fit should not raise.
    model.fit_expanding(images.numpy(), pred)


def test_ceiling_arm_weight_fn_callable():
    signals, returns = _signals_and_returns(t=60, k=5)
    t = 40
    hist = returns[:t]
    w_prev = np.full(5, 0.2)
    kelly_images = np.random.default_rng(0).normal(size=(60, 5, 1, 11, 34))

    for name in CEILING_ARM_NAMES:
        fn = ceiling_arm_weight_fn(
            name,
            signals=signals,
            returns=returns,
            kelly_model=KellySurfaceCNN() if name == "kelly_cnn" else None,
            kelly_images=kelly_images,
        )
        w = fn(hist, t=t, w_prev=w_prev)
        assert w.shape == (5,)
        assert np.all(np.isfinite(w))
        assert abs(float(np.sum(np.abs(w))) - 1.0) < 1e-5


def test_kelly_cnn_raises_without_images_no_silent_equal_weight():
    """B3: kelly_cnn must fail closed, not fall back to equal weight."""
    signals, returns = _signals_and_returns(t=60, k=5)
    fn = ceiling_arm_weight_fn(
        "kelly_cnn",
        signals=signals,
        returns=returns,
        kelly_images=None,
    )
    with pytest.raises(RuntimeError, match="kelly_images"):
        fn(returns[:40], t=40, w_prev=np.full(5, 0.2))


def test_kelly_ensemble_refits_on_schedule_and_ensembles_seeds():
    """B3: expanding-window refit fires on the configured cadence and the
    ensemble prediction differs from any single member (real ensembling,
    not a pass-through to one seed)."""
    rng = np.random.default_rng(0)
    t_total, k = 60, 4
    images = rng.normal(size=(t_total, k, 1, 11, 34))
    returns = np.zeros((t_total, k), dtype=np.float64)
    returns[1:] = 0.01 * images[:-1, :, 0].mean(axis=(-1, -2)) + rng.normal(
        0.0, 0.001, size=(t_total - 1, k)
    )

    ens = KellyEnsemble(n_seeds=3, refit_every=10, epochs=2)
    assert ens._last_refit_t is None
    ens.refit_if_due(images, returns, t=15)
    assert ens._last_refit_t == 15
    assert len(ens._models) == 3

    batch = images[15]
    ensemble_pred = ens.predict_ensemble(batch)
    single_pred = ens._models[0].predict(batch)
    assert ensemble_pred.shape == (k,)
    assert not np.allclose(ensemble_pred, single_pred)

    # Not yet due: refitting at t=16 (< 15+10) must not move last_refit_t.
    ens.refit_if_due(images, returns, t=16)
    assert ens._last_refit_t == 15
    ens.refit_if_due(images, returns, t=26)
    assert ens._last_refit_t == 26


def test_kelly_cnn_equal_weights_during_warm_up_before_first_refit():
    """B3 regression: at small t (e.g. the very first decision of a walk-
    forward eval loop) there are zero training pairs available yet by
    construction, which is not the same failure as missing kelly_images.
    kelly_cnn must equal-weight through this natural warm-up window
    instead of crashing with 'predict_ensemble called before refit'."""
    rng = np.random.default_rng(2)
    t_total, k = 60, 5
    images = rng.normal(size=(t_total, k, 11, 34))
    returns = rng.normal(0.0, 0.01, size=(t_total, k))

    fn = ceiling_arm_weight_fn(
        "kelly_cnn",
        returns=returns,
        kelly_images=images,
    )
    # t=0 and t=1 have zero completed (image, next-return) pairs to train
    # on (max_tau <= 0 by construction); no model can exist yet.
    for t in (0, 1):
        w = fn(returns[:t], t=t, w_prev=np.full(k, 1.0 / k))
        assert w.shape == (k,)
        assert np.allclose(w, 1.0 / k)


def test_kelly_cnn_arm_uses_ensemble_refit_by_default():
    """The production ceiling_arm_weight_fn path (no kelly_model override)
    exercises the real KellyEnsemble end to end."""
    rng = np.random.default_rng(1)
    t_total, k = 50, 3
    images = rng.normal(size=(t_total, k, 1, 11, 34))
    returns = rng.normal(0.0, 0.01, size=(t_total, k))

    fn = ceiling_arm_weight_fn(
        "kelly_cnn",
        returns=returns,
        kelly_images=images,
        kelly_n_seeds=2,
        kelly_refit_every=5,
        kelly_epochs=2,
    )
    w = fn(returns[:30], t=30, w_prev=np.full(k, 1.0 / k))
    assert w.shape == (k,)
    assert np.all(np.isfinite(w))
    assert abs(float(np.sum(np.abs(w))) - 1.0) < 1e-5
