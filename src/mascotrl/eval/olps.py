"""Classic online portfolio selection (OLPS) algorithms in pure numpy.

Long-only, sum-to-1 weight functions over ``returns_hist`` (T_hist, K).
BestStock is stamped as look-ahead (diagnostic upper bound only).

Stub algorithms (EG fallback, not distinct peers for gate3 counting)
--------------------------------------------------------------------
CORN, BNN, ONS, Anticor, CWMR, and RMR are **stubs**: their weight functions
call :func:`eg_weights` and stamp ``olps_stub_fallback: true``. They must not
be counted as distinct peers in gate3 / spectrum ``n_benchmarks`` / beat tallies
(use :func:`olps_claim_names` for the claimable panel).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

_EPS = 1e-12

# Names whose implementations are EG aliases (see ``_eg_fallback``).
OLPS_STUB_NAMES: frozenset[str] = frozenset(
    {"corn", "bnn", "ons", "anticor", "cwmr", "rmr"}
)


def _as_hist(returns_hist: np.ndarray) -> np.ndarray:
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _equal(k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(k, 1.0 / k, dtype=np.float64)


def _simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.nan_to_num(w, nan=0.0), 0.0)
    s = float(w.sum())
    if s < _EPS:
        return _equal(w.size)
    return w / s


def _price_relatives(returns_hist: np.ndarray) -> np.ndarray:
    """Convert simple returns to price relatives x_t = 1 + r_t."""
    r = _as_hist(returns_hist)
    return 1.0 + r


def bah_weights(returns_hist: np.ndarray, **_kw: Any) -> np.ndarray:
    """Buy-and-hold / market: equal start, drift with cumulative relatives."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] == 0 or k == 0:
        return _equal(k)
    x = _price_relatives(r)
    wealth = np.prod(x, axis=0)
    return _simplex(wealth)


def best_stock_weights(
    returns_hist: np.ndarray,
    *,
    return_meta: bool = False,
    **_kw: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Hindsight best single stock (LOOK-AHEAD diagnostic upper bound)."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    meta = {"look_ahead": True, "algorithm": "best_stock"}
    if r.shape[0] == 0 or k == 0:
        w = _equal(k)
        return (w, meta) if return_meta else w
    cum = np.prod(_price_relatives(r), axis=0)
    w = np.zeros(k, dtype=np.float64)
    w[int(np.argmax(cum))] = 1.0
    return (w, meta) if return_meta else w


def eg_weights(
    returns_hist: np.ndarray,
    *,
    eta: float = 0.05,
    **_kw: Any,
) -> np.ndarray:
    """Exponentiated Gradient (Helmbold et al.): multiplicative update on relatives."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if k == 0:
        return _equal(0)
    w = _equal(k)
    if r.shape[0] == 0:
        return w
    x = _price_relatives(r)
    for t in range(x.shape[0]):
        xt = x[t]
        port = float(np.dot(w, xt))
        if port < _EPS:
            port = _EPS
        w = w * np.exp(float(eta) * xt / port)
        w = _simplex(w)
    return w


def up_weights(
    returns_hist: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 0,
    **_kw: Any,
) -> np.ndarray:
    """Cover universal portfolio via Dirichlet(1,...,1) Monte-Carlo mix."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if k == 0:
        return _equal(0)
    if r.shape[0] == 0:
        return _equal(k)
    rng = np.random.default_rng(int(seed))
    # Dirichlet(ones) samples = normalized exponential of unit-rate Exp.
    alphas = np.ones(k, dtype=np.float64)
    samples = rng.dirichlet(alphas, size=int(n_samples))
    x = _price_relatives(r)
    # Wealth of each constant-rebalanced sample path.
    # log wealth = sum_t log(b · x_t)
    log_w = np.zeros(int(n_samples), dtype=np.float64)
    for t in range(x.shape[0]):
        port = samples @ x[t]
        port = np.maximum(port, _EPS)
        log_w += np.log(port)
    # Softmax over sample wealth → mixture weights on the simplex samples.
    log_w = log_w - float(np.max(log_w))
    weights = np.exp(log_w)
    weights = weights / max(float(weights.sum()), _EPS)
    mix = weights @ samples
    return _simplex(mix)


def olmar_weights(
    returns_hist: np.ndarray,
    *,
    window: int = 5,
    eps: float = 10.0,
    **_kw: Any,
) -> np.ndarray:
    """On-Line Moving Average Reversion (Li et al.): project toward predicted relatives."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if k == 0:
        return _equal(0)
    w = _equal(k)
    if r.shape[0] == 0:
        return w
    x = _price_relatives(r)
    w_win = max(1, int(window))
    for t in range(x.shape[0]):
        # Predicted relative via moving average of prices (cumulative product proxy).
        start = max(0, t - w_win + 1)
        # Simple MA of recent relatives as ã_t+1 predictor.
        pred = np.mean(x[start : t + 1], axis=0)
        pred = np.maximum(pred, _EPS)
        # OLMAR update: b_{t+1} = b_t + λ (pred - mean), projected to simplex.
        # λ = max(0, (eps - b·pred) / ||pred - mean||^2)
        mean_p = float(np.mean(pred))
        diff = pred - mean_p
        denom = float(np.dot(diff, diff))
        if denom < _EPS:
            continue
        lagrange = max(0.0, (float(eps) - float(np.dot(w, pred))) / denom)
        w = w + lagrange * diff
        w = _simplex(w)
    return w


def pamr_weights(
    returns_hist: np.ndarray,
    *,
    eps: float = 0.5,
    c: float = 500.0,
    **_kw: Any,
) -> np.ndarray:
    """Passive-Aggressive Mean Reversion (Li et al.)."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if k == 0:
        return _equal(0)
    w = _equal(k)
    if r.shape[0] == 0:
        return w
    x = _price_relatives(r)
    for t in range(x.shape[0]):
        xt = x[t]
        loss = max(0.0, float(np.dot(w, xt)) - float(eps))
        # τ = loss / (||x - mean||^2 + 1/(2C))
        mean_x = float(np.mean(xt))
        diff = xt - mean_x
        denom = float(np.dot(diff, diff)) + 1.0 / (2.0 * max(float(c), _EPS))
        tau = loss / max(denom, _EPS)
        w = w - tau * diff
        w = _simplex(w)
    return w


def _eg_fallback(
    returns_hist: np.ndarray,
    *,
    stub_name: str,
    return_meta: bool = False,
    **kw: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    w = eg_weights(returns_hist, **kw)
    meta = {
        "stub": True,
        "fallback": "eg",
        "olps_stub_fallback": True,
        "algorithm": stub_name,
        "status": f"NotImplemented:{stub_name};fallback=eg",
    }
    return (w, meta) if return_meta else w


def corn_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="corn", **kw)


def bnn_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="bnn", **kw)


def ons_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="ons", **kw)


def anticor_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="anticor", **kw)


def cwmr_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="cwmr", **kw)


def rmr_weights(returns_hist: np.ndarray, **kw: Any) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    return _eg_fallback(returns_hist, stub_name="rmr", **kw)


OLPS_REGISTRY: dict[str, Callable[..., Any]] = {
    "bah": bah_weights,
    "best_stock": best_stock_weights,
    "eg": eg_weights,
    "up": up_weights,
    "olmar": olmar_weights,
    "pamr": pamr_weights,
    "corn": corn_weights,
    "bnn": bnn_weights,
    "ons": ons_weights,
    "anticor": anticor_weights,
    "cwmr": cwmr_weights,
    "rmr": rmr_weights,
}


def olps_weights(
    name: str,
    returns_hist: np.ndarray,
    *,
    return_meta: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    try:
        fn = OLPS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown OLPS algorithm: {name!r}") from exc
    # Propagate return_meta when supported.
    try:
        return fn(returns_hist, return_meta=return_meta, **kwargs)
    except TypeError:
        w = fn(returns_hist, **kwargs)
        if return_meta:
            return w, {"algorithm": name}
        return w


def list_olps() -> tuple[str, ...]:
    return tuple(OLPS_REGISTRY.keys())


def olps_stub_names() -> frozenset[str]:
    """Algorithm keys that are EG-fallback stubs (not distinct gate3 peers)."""
    return OLPS_STUB_NAMES


def olps_claim_names() -> tuple[str, ...]:
    """Non-stub registry names suitable for gate3 / spectrum peer counting."""
    return tuple(n for n in OLPS_REGISTRY if n not in OLPS_STUB_NAMES)


def is_olps_stub_peer(name: str) -> bool:
    """True for bare stub keys or ``olps:<stub>`` peer labels (not ``olps_ons``)."""
    raw = str(name).strip()
    if not raw:
        return False
    if ":" in raw:
        key = raw.split(":", 1)[-1].strip().lower()
    else:
        key = raw.lower()
    return key in OLPS_STUB_NAMES


def filter_olps_stubs_from_peers(
    baseline_sharpes: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop stub OLPS peers so beat counts do not treat EG clones as distinct."""
    return {
        str(k): v for k, v in baseline_sharpes.items() if not is_olps_stub_peer(str(k))
    }
