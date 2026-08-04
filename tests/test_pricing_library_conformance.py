"""Optional QuantLib / py_vollib conformance (sim IV only; not OM production)."""
from __future__ import annotations

import math

import numpy as np
import pytest

# Production historical path uses OM vendor IV — these tests are sim/audit only.


def _bs_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    from math import erf, exp, log, sqrt

    if t <= 0 or sigma <= 0:
        return max(s - k, 0.0)
    d1 = (log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    n = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return s * n(d1) - k * exp(-r * t) * n(d2)


def test_py_vollib_iv_roundtrip():
    pytest.importorskip("py_vollib")
    from py_vollib.black_scholes.implied_volatility import implied_volatility

    s, k, t, r, sigma = 100.0, 100.0, 0.25, 0.01, 0.2
    price = _bs_call_price(s, k, t, r, sigma)
    iv = float(implied_volatility(price, s, k, t, r, "c"))
    assert iv == pytest.approx(sigma, rel=1e-3, abs=1e-3)


def test_quantlib_bs_iv_optional():
    ql = pytest.importorskip("QuantLib")
    s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.25
    price = _bs_call_price(s, k, t, r, sigma)
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, k)
    exercise = ql.EuropeanExercise(today + int(365 * t))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
        ),
    )
    opt = ql.VanillaOption(payoff, exercise)
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    assert float(opt.NPV()) == pytest.approx(price, rel=1e-3, abs=1e-2)

