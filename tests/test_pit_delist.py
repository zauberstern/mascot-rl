"""Step 7: PIT equity delist compounding — no silent zero DLRET."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.features.pit_universe import (
    build_equity_pit_returns,
    compound_equity_return,
    validate_delist_handling,
)


def test_compound_ret_dlret():
    assert compound_equity_return(0.01, -0.5) == pytest.approx((1.01) * (0.5) - 1.0)


def test_delist_missing_dlret_raises():
    with pytest.raises(ValueError, match="DLRET missing"):
        validate_delist_handling(delist_flag=True, dlret=None)


def test_delist_nan_dlret_raises():
    with pytest.raises(ValueError, match="non-finite"):
        validate_delist_handling(delist_flag=[False, True], dlret=[0.0, np.nan])


def test_build_equity_pit_returns_compounds():
    r = np.array([0.01, 0.0])
    d = np.array([0.0, -1.0])
    out = build_equity_pit_returns(r, d, delist_flag=[False, True])
    assert out[0] == pytest.approx(0.01)
    assert out[1] == pytest.approx(-1.0)
