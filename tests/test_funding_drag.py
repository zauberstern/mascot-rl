"""FundingDrag: GC borrow on short weights (default off)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import torch

from mascotrl.plugins.funding_drag import FundingDrag


def test_funding_drag_disabled_is_zero() -> None:
    fd = FundingDrag(enabled=False, gc_borrow_bps=25.0, dt_years=1.0 / 252.0)
    w = torch.tensor([[0.5, -0.5]])
    assert float(fd(w)) == pytest.approx(0.0, **FLOAT_TOL)


def test_funding_drag_enabled_on_short_weights() -> None:
    fd = FundingDrag(enabled=True, gc_borrow_bps=25.0, dt_years=1.0 / 252.0)
    w = torch.tensor([[0.5, -0.5]])
    drag = fd(w)
    # Only short notional 0.5: rate * dt * |w^-|
    expected = (25.0 / 1e4) * (1.0 / 252.0) * 0.5
    assert float(drag) == pytest.approx(expected)
    assert float(drag) > 0.0


def test_funding_drag_long_only_is_zero_when_enabled() -> None:
    fd = FundingDrag(enabled=True, gc_borrow_bps=25.0, dt_years=1.0 / 252.0)
    w = torch.tensor([[0.4, 0.6]])
    assert float(fd(w)) == pytest.approx(0.0, **FLOAT_TOL)
