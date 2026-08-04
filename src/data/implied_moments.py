"""Bakshi–Kapadia–Madan model-free implied moments (Phase C-2).

Recover risk-neutral variance (MFIV), skewness (MFIS), and excess kurtosis
(MFIK) from a cross-section of OTM calls and puts via trapezoidal integration
of the volatility / cubic / quartic contracts (Bakshi, Kapadia and Madan 2003,
RFS 16(1); trapezoidal discrete-strike implementation follows Bali–Hu–Murray).

Expected columns on ``otm_slice``: ``strike``, ``mid``, ``spot``, ``rate``,
``tau`` (years), ``cp_flag`` (``C``/``P`` or ``call``/``put``).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

_CALL = frozenset({"C", "CALL", "c", "call"})
_PUT = frozenset({"P", "PUT", "p", "put"})


def _as_flag(x: Any) -> str:
    s = str(x).strip()
    if s in _CALL or s.upper().startswith("C"):
        return "C"
    if s in _PUT or s.upper().startswith("P"):
        return "P"
    return s.upper()[:1]


def _trapz_contract(
    strikes: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Trapezoidal ∫ w(K) · price(K) dK on a sorted strike grid."""
    if strikes.size < 2:
        return float("nan")
    order = np.argsort(strikes)
    k = strikes[order]
    y = (weights * prices)[order]
    return float(np.trapezoid(y, k))


def compute_bkm_contracts(
    strike: np.ndarray,
    mid: np.ndarray,
    spot: float,
    rate: float,
    tau: float,
    cp_flag: np.ndarray,
) -> dict[str, float]:
    """
    Integrate OTM V / W / X contracts.

    OTM rule uses the forward ``F = S·e^{rτ}``: puts with K < F, calls with K > F.
    """
    K = np.asarray(strike, dtype=np.float64)
    P = np.asarray(mid, dtype=np.float64)
    S = float(spot)
    r = float(rate)
    t = float(tau)
    if not np.isfinite(S) or S <= 0 or not np.isfinite(t) or t <= 0:
        return {"V": float("nan"), "W": float("nan"), "X": float("nan"), "F": float("nan")}
    F = S * float(np.exp(r * t))
    flags = np.asarray([_as_flag(x) for x in cp_flag])
    ok = np.isfinite(K) & np.isfinite(P) & (K > 0) & (P >= 0)
    K, P, flags = K[ok], P[ok], flags[ok]
    if K.size < 4:
        return {"V": float("nan"), "W": float("nan"), "X": float("nan"), "F": F}

    put_m = (flags == "P") & (K < F)
    call_m = (flags == "C") & (K > F)
    if int(put_m.sum()) + int(call_m.sum()) < 4:
        # Fall back to spot cutoff if forward split is too thin.
        put_m = (flags == "P") & (K < S)
        call_m = (flags == "C") & (K > S)
    if int(put_m.sum()) + int(call_m.sum()) < 2:
        return {"V": float("nan"), "W": float("nan"), "X": float("nan"), "F": F}

    def _vw_put(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Relative to forward for numerical stability.
        x = np.log(F / k)
        # V put weight: 2(1 + ln(F/K)) / K²
        wV = 2.0 * (1.0 + x) / (k * k)
        # W put: −(6 ln(F/K) + 3 [ln(F/K)]²) / K²  (BKM put wing)
        wW = -(6.0 * x + 3.0 * x * x) / (k * k)
        # X put: (12 [ln(F/K)]² + 4 [ln(F/K)]³) / K²
        wX = (12.0 * x * x + 4.0 * x * x * x) / (k * k)
        return wV, wW, wX

    def _vw_call(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.log(k / F)
        wV = 2.0 * (1.0 - x) / (k * k)
        wW = (6.0 * x - 3.0 * x * x) / (k * k)
        wX = (12.0 * x * x - 4.0 * x * x * x) / (k * k)
        return wV, wW, wX

    V = W = X = 0.0
    n_legs = 0
    if put_m.any():
        k, p = K[put_m], P[put_m]
        wV, wW, wX = _vw_put(k)
        V += _trapz_contract(k, p, wV)
        W += _trapz_contract(k, p, wW)
        X += _trapz_contract(k, p, wX)
        n_legs += 1
    if call_m.any():
        k, p = K[call_m], P[call_m]
        wV, wW, wX = _vw_call(k)
        V += _trapz_contract(k, p, wV)
        W += _trapz_contract(k, p, wW)
        X += _trapz_contract(k, p, wX)
        n_legs += 1
    if n_legs == 0 or not np.isfinite(V) or V <= 0:
        return {"V": float("nan"), "W": float("nan"), "X": float("nan"), "F": F}
    return {"V": float(V), "W": float(W), "X": float(X), "F": float(F)}


def compute_mf_moments(otm_slice: pd.DataFrame | Mapping[str, Any]) -> dict[str, float]:
    """
    Bakshi–Kapadia–Madan model-free moments from an OTM option slice.

    Returns ``dict(mfiv, mfis, mfik)`` (variance, skewness, excess kurtosis).
    Missing / degenerate slices yield NaNs.
    """
    if isinstance(otm_slice, pd.DataFrame):
        df = otm_slice
    else:
        df = pd.DataFrame(otm_slice)
    required = ("strike", "mid", "spot", "rate", "tau", "cp_flag")
    for col in required:
        if col not in df.columns:
            raise ValueError(f"otm_slice missing column {col!r}")
    if df.empty:
        return {"mfiv": float("nan"), "mfis": float("nan"), "mfik": float("nan")}

    spot = float(np.nanmedian(pd.to_numeric(df["spot"], errors="coerce")))
    rate = float(np.nanmedian(pd.to_numeric(df["rate"], errors="coerce")))
    tau = float(np.nanmedian(pd.to_numeric(df["tau"], errors="coerce")))
    contracts = compute_bkm_contracts(
        strike=pd.to_numeric(df["strike"], errors="coerce").to_numpy(),
        mid=pd.to_numeric(df["mid"], errors="coerce").to_numpy(),
        spot=spot,
        rate=rate,
        tau=tau,
        cp_flag=df["cp_flag"].to_numpy(),
    )
    V, W, X = contracts["V"], contracts["W"], contracts["X"]
    if not np.isfinite(V) or V <= 0 or not np.isfinite(tau) or tau <= 0:
        return {"mfiv": float("nan"), "mfis": float("nan"), "mfik": float("nan")}

    er = float(np.exp(rate * tau))
    mu = er - 1.0 - (er / 2.0) * V - (er / 6.0) * W - (er / 24.0) * X
    base = er * V - mu * mu
    if not np.isfinite(base) or base <= 0:
        return {"mfiv": float("nan"), "mfis": float("nan"), "mfik": float("nan")}

    mfiv = base / tau
    denom_s = base ** 1.5
    mfis = (er * W - 3.0 * mu * er * V + 2.0 * mu**3) / denom_s if denom_s > 0 else float("nan")
    denom_k = base**2
    mfik = (
        (er * X - 4.0 * mu * er * W + 6.0 * (mu**2) * er * V - 3.0 * mu**4) / denom_k - 3.0
        if denom_k > 0
        else float("nan")
    )
    return {
        "mfiv": float(mfiv),
        "mfis": float(mfis),
        "mfik": float(mfik),
        "V": float(V),
        "W": float(W),
        "X": float(X),
        "mu": float(mu),
        "tau": float(tau),
        "forward": float(contracts["F"]),
    }
