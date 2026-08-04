# Vendored from OmniSafe (Apache-2.0).
# Upstream: https://github.com/PKU-Alignment/omnisafe
# Commit: 15603dd7a654a991d0a4648216b69d60b81a6366
# Source: omnisafe/common/pid_lagrange.py
# Stripped of Safety-Gymnasium / omnisafe package imports; concrete class (not ABC).
"""PID Lagrangian multiplier (CPPOPID / Stooke et al. 2020)."""
from __future__ import annotations

from collections import deque


class PIDLagrangian:
    """PID version of the Lagrangian multiplier (Stooke et al., 2020).

    More stable than naive Lagrange for constraint tracking. Used by OmniSafe
    ``CPPOPID`` to mix reward and cost advantages.
    """

    def __init__(
        self,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_d_delay: int = 10,
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float = 0.95,
        sum_norm: bool = True,
        diff_norm: bool = False,
        penalty_max: int = 100,
        lagrangian_multiplier_init: float = 0.0,
        cost_limit: float = 0.0,
    ) -> None:
        self._pid_kp = float(pid_kp)
        self._pid_ki = float(pid_ki)
        self._pid_kd = float(pid_kd)
        self._pid_d_delay = int(pid_d_delay)
        self._pid_delta_p_ema_alpha = float(pid_delta_p_ema_alpha)
        self._pid_delta_d_ema_alpha = float(pid_delta_d_ema_alpha)
        self._penalty_max = int(penalty_max)
        self._sum_norm = bool(sum_norm)
        self._diff_norm = bool(diff_norm)
        self._pid_i = float(lagrangian_multiplier_init)
        self._cost_ds: deque[float] = deque(maxlen=max(self._pid_d_delay, 1))
        self._cost_ds.append(0.0)
        self._delta_p = 0.0
        self._cost_d = 0.0
        self._cost_limit = float(cost_limit)
        self._cost_penalty = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        return float(self._cost_penalty)

    def pid_update(self, ep_cost_avg: float) -> None:
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, float(self._penalty_max))
        self._cost_ds.append(self._cost_d)
