"""Classic online portfolio selection (OLPS) weight functions."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.olps import (
    OLPS_REGISTRY,
    bah_weights,
    best_stock_weights,
    eg_weights,
    olmar_weights,
    pamr_weights,
    up_weights,
    olps_weights,
)


def _hist(t: int = 40, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.02, size=(t, k))


@pytest.mark.parametrize(
    "fn",
    [bah_weights, eg_weights, up_weights, olmar_weights, pamr_weights],
)
def test_olps_shape_and_sum(fn):
    r = _hist()
    w = fn(r)
    assert w.shape == (r.shape[1],)
    assert np.all(np.isfinite(w))
    assert w.min() >= -1e-12
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_bah_equal_on_empty_then_drifts():
    r = _hist(30, 4)
    w0 = bah_weights(r[:0].reshape(0, 4))
    assert np.allclose(w0, 0.25)
    # With history, BAH is market-relative / buy-and-hold from equal start.
    w = bah_weights(r)
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_best_stock_is_lookahead_stamped():
    r = _hist(50, 4, seed=2)
    w, meta = best_stock_weights(r, return_meta=True)
    assert meta.get("look_ahead") is True
    assert abs(float(w.sum()) - 1.0) < 1e-6
    # Hindsight best name gets weight 1.
    cum = np.prod(1.0 + np.nan_to_num(r, nan=0.0), axis=0)
    assert int(np.argmax(w)) == int(np.argmax(cum))


def test_eg_responds_to_relative_returns():
    r = np.zeros((20, 3))
    r[:, 0] = 0.02
    r[:, 1] = -0.01
    r[:, 2] = 0.0
    w = eg_weights(r, eta=0.5)
    assert w[0] > w[1]


def test_registry_dispatch_and_stubs():
    r = _hist()
    w = olps_weights("eg", r)
    assert w.shape == (5,)
    assert "bah" in OLPS_REGISTRY
    assert "eg" in OLPS_REGISTRY
    assert "olmar" in OLPS_REGISTRY
    assert "pamr" in OLPS_REGISTRY
    assert "up" in OLPS_REGISTRY
    assert "best_stock" in OLPS_REGISTRY
    # Optional stubs exist and are stamped.
    for stub in ("corn", "bnn", "ons", "anticor", "cwmr", "rmr"):
        assert stub in OLPS_REGISTRY
        out = olps_weights(stub, r, return_meta=True)
        if isinstance(out, tuple):
            ww, meta = out
        else:
            ww, meta = out, {}
        assert ww.shape == (5,)
        assert abs(float(ww.sum()) - 1.0) < 1e-5
        # Stub either NotImplemented path or EG fallback with stamp.
        assert meta.get("stub") or meta.get("fallback") == "eg" or "NotImplemented" in str(
            meta.get("status", "")
        )
