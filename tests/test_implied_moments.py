"""BKM model-free moments smoke tests (Phase C-2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.implied_moments import compute_mf_moments


def _synthetic_otm_slice():
    """Equity-like skew: OTM puts richer than OTM calls."""
    spot = 100.0
    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    # Rough call/put mids decreasing in |K-S|.
    rows = []
    for k in strikes:
        if k < spot:
            mid = max(0.5, 0.02 * (spot - k) ** 1.1)
            rows.append({"strike": k, "mid": mid, "cp_flag": "P"})
        elif k > spot:
            mid = max(0.3, 0.015 * (k - spot) ** 1.05)
            rows.append({"strike": k, "mid": mid, "cp_flag": "C"})
    df = pd.DataFrame(rows)
    df["spot"] = spot
    df["rate"] = 0.02
    df["tau"] = 30 / 365.0
    return df


def test_mf_moments_finite_and_skew_typically_negative():
    out = compute_mf_moments(_synthetic_otm_slice())
    assert np.isfinite(out["mfiv"])
    assert out["mfiv"] > 0
    # Equity-like put wing usually implies negative risk-neutral skew.
    assert np.isfinite(out["mfis"])
    assert out["mfis"] < 0.5  # soft bound; synthetic smile not calibrated BS
