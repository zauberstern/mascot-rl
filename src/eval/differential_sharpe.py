"""Moody differential Sharpe for research equity train reward."""
from __future__ import annotations

import math


class DifferentialSharpe:
    """Online differential Sharpe (Moody 2001) on a scalar after-cost return stream."""

    def __init__(self, *, eta: float = 0.01) -> None:
        self.eta = float(eta)
        self.A = 0.0
        self.B = 0.0
        self.n = 0

    def step(self, r: float) -> float:
        """Update moments; return differential Sharpe contribution for this step."""
        x = float(r)
        if not math.isfinite(x):
            return 0.0
        self.n += 1
        if self.n == 1:
            self.A = x
            self.B = x * x
            return 0.0
        dA = x - self.A
        dB = x * x - self.B
        denom = self.B - self.A * self.A
        if denom <= 1e-12:
            dt = 0.0
        else:
            # dDt/dt ∝ (B*dA - 0.5*A*dB) / (B-A^2)^{1.5}
            num = self.B * dA - 0.5 * self.A * dB
            dt = float(num / (denom ** 1.5))
        eta = self.eta
        self.A = self.A + eta * dA
        self.B = self.B + eta * dB
        if not math.isfinite(dt):
            return 0.0
        return dt
