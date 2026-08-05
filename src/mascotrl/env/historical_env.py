"""Historical arm environment: numpy hist panels + FrictionSpec + residualizer.

No rBergomi / synthetic surface tensors. Reward is residual net:
``gross - costs - borrow - rf - lagged_exp · f``.

Projection happens before fills; post-fill exposures are reported and never
silently re-projected after fill.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from src.data.slot_mask import apply_slot_mask
from src.eval.friction import FrictionSpec, apply_costs
from src.eval.residualization import (
    ResidualizerState,
    residualize_step,
    rolling_portfolio_ff4_beta,
    select_ff4_factor_matrix,
)

ProjectFn = Callable[..., np.ndarray]
FillFn = Callable[..., np.ndarray]


class HistoricalArmEnv:
    """Minimal Gymnasium-like env over a 2-D hist return panel ``(T, K)``."""

    def __init__(
        self,
        *,
        returns: np.ndarray,
        factors: np.ndarray,
        arm: Any,
        friction: FrictionSpec,
        residualizer: ResidualizerState | None,
        project_fn: ProjectFn | None = None,
        fill_fn: FillFn | None = None,
        feature_builder: Callable[[int, np.ndarray], np.ndarray] | None = None,
        rebalance_mask: np.ndarray | None = None,
        slot_valid_mask: np.ndarray | None = None,
        rf: np.ndarray | None = None,
        portfolio_beta_window: int = 252,
        risk_free_treatment: str = "daily_subtract_once",
        factor_frequency: str = "daily",
        post_fill_exposure_reporting: bool = True,
        off_rebalance_tol: float = 1e-12,
        reward_mode: str = "residual",
        marks: dict[str, np.ndarray] | None = None,
    ) -> None:
        rets = np.asarray(returns, dtype=np.float64)
        if rets.ndim != 2:
            raise ValueError(
                f"HistoricalArmEnv requires 2-D hist returns (T, K); "
                f"got ndim={rets.ndim} (rBergomi surfaces are not accepted)"
            )
        fac = np.asarray(factors, dtype=np.float64)
        if fac.ndim != 2:
            raise ValueError(f"factors must be 2-D (T, F); got ndim={fac.ndim}")
        if fac.shape[0] != rets.shape[0]:
            raise ValueError(
                f"returns/factors T mismatch {rets.shape[0]} vs {fac.shape[0]}"
            )
        n_slots = int(getattr(arm, "n_slots", rets.shape[1]))
        if rets.shape[1] != n_slots:
            raise ValueError(
                f"returns K={rets.shape[1]} != arm.n_slots={n_slots}"
            )
        self.returns = rets
        self.factors = fac
        self.arm = arm
        self.friction = friction
        self.residualizer = residualizer
        self.project_fn = project_fn
        self.fill_fn = fill_fn
        self.feature_builder = feature_builder
        self.rebalance_mask = (
            None
            if rebalance_mask is None
            else np.asarray(rebalance_mask, dtype=bool).reshape(-1)
        )
        if self.rebalance_mask is not None and self.rebalance_mask.size != rets.shape[0]:
            raise ValueError(
                f"rebalance_mask length {self.rebalance_mask.size} != T={rets.shape[0]}"
            )
        if rf is None:
            self.rf = np.zeros(rets.shape[0], dtype=np.float64)
        else:
            self.rf = np.asarray(rf, dtype=np.float64).reshape(-1)
            if self.rf.size != rets.shape[0]:
                raise ValueError(f"rf length {self.rf.size} != T={rets.shape[0]}")
        self.portfolio_beta_window = int(portfolio_beta_window)
        rf_mode = str(risk_free_treatment or "daily_subtract_once")
        if rf_mode not in ("daily_subtract_once", "monthly_average", "zero"):
            raise ValueError(f"unknown risk_free_treatment={rf_mode!r}")
        self.risk_free_treatment = rf_mode
        freq = str(factor_frequency or "daily")
        if freq != "daily":
            raise ValueError(
                f"factor_frequency must be 'daily' for alpha-v2; got {freq!r}"
            )
        self.factor_frequency = freq
        self.post_fill_exposure_reporting = bool(post_fill_exposure_reporting)
        self.off_rebalance_tol = float(off_rebalance_tol)
        rm = str(reward_mode or "residual").lower().strip()
        if rm not in ("residual", "mtm_pnl"):
            raise ValueError(
                f"unknown reward_mode={reward_mode!r}; expected 'residual' or 'mtm_pnl'"
            )
        self.reward_mode = rm
        self.T, self.K = int(rets.shape[0]), int(rets.shape[1])
        self.slot_valid_mask = (
            None if slot_valid_mask is None else np.asarray(slot_valid_mask, dtype=bool)
        )
        if self.slot_valid_mask is not None and self.slot_valid_mask.shape != (self.T, self.K):
            raise ValueError(
                f"slot_valid_mask shape {self.slot_valid_mask.shape} != "
                f"(T,K)=({self.T},{self.K})"
            )
        # OM marks for option friction (half_spread / delta / spot / capital_base).
        self.marks = self._validate_marks(marks)
        opt_slots = int(getattr(arm, "option_slots", 0) or 0)
        if (
            opt_slots > 0
            and bool(getattr(friction, "om_touch_enabled", False))
            and self.marks is None
        ):
            raise ValueError(
                "om_touch_enabled with option slots requires marks "
                "(half_spread/delta/spot/capital_base); refusing silent zero option costs"
            )
        self.t = 1
        # RC5: cold-start at EW (reset() also sets this; keep constructors consistent).
        self.w = np.full(self.K, 1.0 / max(self.K, 1), dtype=np.float64)
        # Realized portfolio returns for rolling beta (index-aligned with panel).
        self._port_rets = np.full(self.T, np.nan, dtype=np.float64)
        # Running RF sum for monthly_average treatment (calendar proxy: 21d blocks).
        self._rf_month_acc = 0.0
        self._rf_month_n = 0

    def _validate_marks(
        self, marks: dict[str, np.ndarray] | None
    ) -> dict[str, np.ndarray] | None:
        if marks is None:
            return None
        required = ("half_spread", "delta", "spot", "capital_base")
        out: dict[str, np.ndarray] = {}
        for key in required:
            if key not in marks:
                raise ValueError(f"marks missing required key {key!r}")
            arr = np.asarray(marks[key], dtype=np.float64)
            if arr.shape != (self.T, self.K):
                raise ValueError(
                    f"marks[{key!r}] shape {arr.shape} != (T,K)=({self.T},{self.K})"
                )
            out[key] = arr
        return out

    def _obs(self) -> np.ndarray:
        # Decision features: last observed return row (no look-ahead).
        idx = max(0, min(self.t - 1, self.T - 1))
        if self.feature_builder is not None:
            return np.asarray(self.feature_builder(idx, self.w), dtype=np.float64)
        # Sparse panels leave NaN in inactive name slots. Zero-fill so the
        # policy sees a defined state; slot_valid_mask still gates trading.
        obs = self.returns[idx].astype(np.float64, copy=True)
        if not np.isfinite(obs).all():
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _static_exposures(self) -> np.ndarray | None:
        """Legacy fold-frozen residualizer betas when rolling window unavailable."""
        if self.residualizer is None:
            return None
        betas = np.asarray(self.residualizer.betas, dtype=np.float64)
        n_f = int(self.factors.shape[1])
        if betas.ndim == 1 and betas.size == n_f:
            return betas
        if betas.ndim == 2 and betas.shape[1] == n_f:
            return self.w @ betas
        out = np.zeros(n_f, dtype=np.float64)
        n = min(n_f, betas.reshape(-1).size)
        out[:n] = betas.reshape(-1)[:n]
        return out

    def _portfolio_beta(self, t: int) -> np.ndarray:
        """Causal portfolio FF4 beta through t-1; fallback to frozen residualizer."""
        w = int(self.portfolio_beta_window)
        if w > 0 and t >= w and np.isfinite(self._port_rets[t - w : t]).all():
            return rolling_portfolio_ff4_beta(
                self._port_rets, self.factors, t=t, window=w
            )
        static = self._static_exposures()
        if static is not None:
            return np.asarray(static, dtype=np.float64).reshape(-1)[: self.factors.shape[1]]
        n_f = int(self.factors.shape[1])
        return np.zeros(n_f, dtype=np.float64)

    def _borrow_charge(self, w: np.ndarray) -> float:
        bps = float(getattr(self.friction, "borrow_floor_bps_annual", 0.0) or 0.0)
        if bps <= 0.0:
            return 0.0
        short_notional = float(np.maximum(-np.asarray(w, dtype=np.float64), 0.0).sum())
        return short_notional * (bps / 1e4) / 252.0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        # Episode start (t=1) is deterministic regardless of seed; seeding
        # numpy's global RNG only affects any RNG consumers downstream
        # (e.g. exploration noise), matching Gymnasium's reset(seed=...)
        # contract without inventing a per-instance RNG stream.
        if seed is not None:
            np.random.seed(seed)
        self.t = 1
        # RC5: start at equal weight so the turnover projector tilts away from
        # EW within tau, instead of building a book from a zero cold-start.
        self.w = np.full(self.K, 1.0 / max(self.K, 1), dtype=np.float64)
        self._port_rets = np.full(self.T, np.nan, dtype=np.float64)
        self._rf_month_acc = 0.0
        self._rf_month_n = 0
        fb = self.feature_builder
        if fb is not None and hasattr(fb, "reset_portfolio_state"):
            fb.reset_portfolio_state()
        return self._obs(), {}

    def step(
        self, weights: np.ndarray | list[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.t >= self.T:
            raise RuntimeError("HistoricalArmEnv episode already exhausted")
        w_raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w_raw.size != self.K:
            raise ValueError(f"weights size {w_raw.size} != K={self.K}")

        w_prev = self.w.copy()
        rebalance_today = True
        if self.rebalance_mask is not None:
            rebalance_today = bool(self.rebalance_mask[self.t])

        # Cadence: hold prior weights off rebalance dates (eq monthly / opt expiry).
        if not rebalance_today:
            target = w_prev.copy()
            w = w_prev.copy()
            # Fail closed: no position-changing fills off rebalance.
            if float(np.abs(w - w_prev).sum()) > self.off_rebalance_tol:
                raise RuntimeError(
                    f"off-rebalance weight drift at t={self.t}: "
                    f"turnover={float(np.abs(w - w_prev).sum())}"
                )
        else:
            if self.slot_valid_mask is not None:
                mask_t = self.slot_valid_mask[self.t]
                masked = apply_slot_mask(w_raw, mask_t)
                # Preserve the policy's chosen gross L1 budget by spreading it
                # over the surviving active slots; leave zeros when every
                # slot in the mask is inactive (nothing to renormalize onto).
                orig_l1 = float(np.abs(w_raw).sum())
                new_l1 = float(np.abs(masked).sum())
                if new_l1 > 1e-12 and orig_l1 > 1e-12:
                    masked = masked * (orig_l1 / new_l1)
                w_raw = masked
            target = w_raw
            if self.project_fn is not None:
                # Hard neutrality / liquidity clamps BEFORE every fill (AZB).
                target = np.asarray(
                    self.project_fn(w_raw, t=self.t, w_prev=w_prev), dtype=np.float64
                ).reshape(-1)
                if target.size != self.K:
                    raise ValueError(
                        f"project_fn returned size {target.size} != K={self.K}"
                    )
            if self.fill_fn is not None:
                w = np.asarray(
                    self.fill_fn(target, w_prev, t=self.t), dtype=np.float64
                ).reshape(-1)
                if w.size != self.K:
                    raise ValueError(f"fill_fn returned size {w.size} != K={self.K}")
            else:
                w = target.copy()
            # Never silently re-project after fill.

        ret_t = self.returns[self.t]
        fac_t = self.factors[self.t]
        fac_ff4 = select_ff4_factor_matrix(fac_t).reshape(-1)
        raw_rf = float(self.rf[self.t])
        rf_t = self._rf_for_step(raw_rf)

        cost_kw: dict[str, Any] = {}
        if self.marks is not None:
            t = int(self.t)
            t_prev = max(0, t - 1)
            cost_kw = {
                "half_spread": self.marks["half_spread"][t],
                "capital_base": self.marks["capital_base"][t],
                "deltas": self.marks["delta"][t],
                "deltas_prev": self.marks["delta"][t_prev],
                "spot": self.marks["spot"][t],
            }
        breakdown = apply_costs(
            torch.as_tensor(w, dtype=torch.float64),
            torch.as_tensor(w_prev, dtype=torch.float64),
            torch.as_tensor(ret_t, dtype=torch.float64),
            arm=self.arm,
            friction=self.friction,
            **cost_kw,
        )
        # Trading costs only (borrow / RF handled explicitly below).
        cost = float(
            breakdown.option_spread
            + breakdown.equity_spread
            + breakdown.hedge_leg
            + breakdown.funding
        )
        borrow = self._borrow_charge(w)
        beta = np.asarray(self._portfolio_beta(self.t), dtype=np.float64).reshape(-1)
        # Align beta to FF4 columns used for residualization (not full Gate2 panel).
        if beta.size < fac_ff4.size:
            pad = np.zeros(fac_ff4.size, dtype=np.float64)
            pad[: beta.size] = beta
            beta = pad
        else:
            beta = beta[: fac_ff4.size]
        factor = float(np.dot(beta, fac_ff4))
        residual = float(
            residualize_step(
                breakdown.gross, cost, beta, fac_ff4, borrow=borrow, rf=rf_t
            )
        )
        turnover = float(np.abs(w - w_prev).sum())
        if not rebalance_today and turnover > self.off_rebalance_tol:
            raise RuntimeError(
                f"non-zero turnover off rebalance at t={self.t}: {turnover}"
            )

        tw = np.asarray(target, dtype=np.float64).reshape(-1)
        fw = np.asarray(w, dtype=np.float64).reshape(-1)
        drift = {
            "l1": float(np.abs(fw - tw).sum()),
            "dollar_exposure": float(np.sum(fw)),
        }
        info = {
            "gross": float(breakdown.gross),
            "cost": cost,
            "borrow": borrow,
            "rf": rf_t,
            "factor": factor,
            "residual": residual,
            "turnover": turnover,
            "rebalance": bool(rebalance_today),
            "target_w": np.asarray(target, dtype=np.float64).copy(),
            "post_fill_w": np.asarray(w, dtype=np.float64).copy(),
            "post_fill_dollar_exposure": float(np.sum(w)),
            "post_fill_reprojected": False,
            "portfolio_beta": np.asarray(beta, dtype=np.float64).copy(),
            "post_fill_drift": drift,
            "risk_free_treatment": self.risk_free_treatment,
            "factor_frequency": self.factor_frequency,
            "n_nan_labels": int(getattr(breakdown, "n_nan_labels", 0) or 0),
        }
        if self.post_fill_exposure_reporting and drift.get("drifted"):
            info["post_fill_dollar_exposure"] = float(np.sum(w))

        # Record realized portfolio return at t for future rolling betas.
        self._port_rets[self.t] = float(breakdown.gross)
        self.w = w.copy()
        self.t += 1
        fb = self.feature_builder
        if fb is not None and hasattr(fb, "update_portfolio_state"):
            fb.update_portfolio_state(self.w, step_cost=float(cost) + float(borrow))
        truncated = self.t >= self.T - 1
        terminated = False
        # D.3: mtm_pnl is gross mark-to-market minus trading costs and borrow.
        if self.reward_mode == "mtm_pnl":
            reward = float(breakdown.gross) - float(cost) - float(borrow)
        else:
            reward = residual
        info["reward_mode"] = self.reward_mode
        info["mtm_pnl"] = float(breakdown.gross) - float(cost) - float(borrow)
        return self._obs(), reward, terminated, truncated, info

    def _rf_for_step(self, raw_rf: float) -> float:
        """Apply locked risk-free treatment from EstimandSpec."""
        if self.risk_free_treatment == "zero":
            return 0.0
        if self.risk_free_treatment == "daily_subtract_once":
            return float(raw_rf)
        # monthly_average: subtract running mean within 21-day blocks.
        self._rf_month_acc += float(raw_rf)
        self._rf_month_n += 1
        avg = self._rf_month_acc / max(1, self._rf_month_n)
        if self._rf_month_n >= 21:
            self._rf_month_acc = 0.0
            self._rf_month_n = 0
        return float(avg)
