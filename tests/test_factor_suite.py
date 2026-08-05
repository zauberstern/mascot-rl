"""Factor suite helpers: HLZ / BH / BCSZ / attach suite (Phase C)."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.factor_alpha import (
    attach_factor_alpha_suite,
    bh_fdr,
    build_bcsz_factors,
    hlz_hurdles,
)


def test_hlz_hurdles():
    h = hlz_hurdles(3.2)
    assert h["clears_t_3_0"] is True
    assert h["clears_t_3_9"] is False


def test_bh_fdr_rejects_small_p():
    out = bh_fdr([0.001, 0.2, 0.4], q=0.05)
    assert out["ok"] is True
    assert out["reject"][0] is True


def test_bcsz_and_suite_smoke():
    rng = np.random.default_rng(0)
    # Quintile long-short needs ≥10 names (2 × n_quantiles).
    T, K = 80, 12
    labels = rng.standard_normal((T, K)) * 0.01
    chars = {
        "option_spread": rng.random((T, K)),
        "option_price": 1.0 + rng.random((T, K)),
        "vol_deviation": rng.standard_normal((T, K)),
    }
    fac = build_bcsz_factors(labels, chars)
    assert "factors" in fac
    for name, series in fac["factors"].items():
        assert len(series) == T, f"{name} length {len(series)} != T={T}"
    pnl = list(np.nanmean(labels, axis=1))
    report: dict = {}
    suite = attach_factor_alpha_suite(
        report, strategy_pnl=pnl, labels=labels, characteristics=chars
    )
    assert "HVX" in suite["models"]
    assert "BCSZ" in suite["models"]
    assert suite["models"]["BCSZ"]["alpha"].get("factors_used"), (
        "BCSZ must retain aligned factors, not intercept-only"
    )
    assert "factor_alpha_suite" in report
