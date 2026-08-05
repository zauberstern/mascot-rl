"""Unified transaction-cost / friction model for HAPPO and baselines.

Arm-aware: option slots pay OM-touch (or stylized bps); equity slots pay
``equity_bps`` on ``|Δw|`` plus optional square-root participation impact
(``impact_c_eq``; Bouchaud-style). Hedge-leg cost applies only to the option
block. Optional ``borrow_bps_annual`` charges short equity weights (L6b).

Cost-ladder note: equity spread + ``impact_c_eq`` participate in the same
after-cost measurement path as OM-touch / hedge-leg rungs (see
``src/eval/cost_ladder.py``); do not soft-fee overnight CMDP ``R_t``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class FrictionBreakdown:
    """Per-step gross PnL and friction components (all in return units)."""

    gross: float
    option_spread: float
    equity_spread: float
    hedge_leg: float
    funding: float
    net: float
    n_nan_labels: int = 0


@dataclass(frozen=True)
class FrictionSpec:
    """Frozen train/OOS friction contract (Alpha v2).

    ``cost_multiplier`` scales every cost component (1× / 2× / 3× stress).
    Identity is ``spec_id``; parity checks ignore multiplier so train@1× can
    match oos@2× on the structural fields.
    """

    spec_id: str = "v2_quote_touch"
    equity_bps: float = 5.0
    hedge_leg_bps: float = 5.0
    execution_spread_bps: float = 5.0
    execution_impact_coef: float = 0.5
    om_touch_enabled: bool = True
    om_touch_fee_bps: float = 0.0
    om_touch_spread_multiplier: float = 1.0
    equity_spread_multiplier: float = 1.0
    borrow_floor_bps_annual: float = 25.0
    impact_c_eq: float = 0.5
    impact_c_opt: float = 1.0
    cost_multiplier: float = 1.0
    hedge_impact_enabled: bool = False
    hedge_impact_coef: float = 1.0

    def scaled(self, multiplier: float) -> "FrictionSpec":
        """Return a copy with ``cost_multiplier`` set (does not bake into bps)."""
        m = float(multiplier)
        if m not in (1.0, 2.0, 3.0) and m <= 0:
            raise ValueError(f"cost_multiplier must be positive, got {m}")
        return FrictionSpec(
            spec_id=self.spec_id,
            equity_bps=self.equity_bps,
            hedge_leg_bps=self.hedge_leg_bps,
            execution_spread_bps=self.execution_spread_bps,
            execution_impact_coef=self.execution_impact_coef,
            om_touch_enabled=self.om_touch_enabled,
            om_touch_fee_bps=self.om_touch_fee_bps,
            om_touch_spread_multiplier=self.om_touch_spread_multiplier,
            equity_spread_multiplier=self.equity_spread_multiplier,
            borrow_floor_bps_annual=self.borrow_floor_bps_annual,
            impact_c_eq=self.impact_c_eq,
            impact_c_opt=self.impact_c_opt,
            cost_multiplier=m,
            hedge_impact_enabled=self.hedge_impact_enabled,
            hedge_impact_coef=self.hedge_impact_coef,
        )


def scale_friction(spec: FrictionSpec, multiplier: float) -> FrictionSpec:
    """Scale FrictionSpec for 1× / 2× / 3× cost stress."""
    return spec.scaled(float(multiplier))


def assert_friction_parity(train: FrictionSpec, oos: FrictionSpec) -> None:
    """Raise if train/OOS friction contracts differ (multiplier may differ)."""
    fields = (
        "spec_id",
        "equity_bps",
        "hedge_leg_bps",
        "execution_spread_bps",
        "execution_impact_coef",
        "om_touch_enabled",
        "om_touch_fee_bps",
        "om_touch_spread_multiplier",
        "equity_spread_multiplier",
        "borrow_floor_bps_annual",
        "impact_c_eq",
        "impact_c_opt",
        "hedge_impact_enabled",
        "hedge_impact_coef",
    )
    for name in fields:
        tv = getattr(train, name)
        ov = getattr(oos, name)
        if tv != ov:
            raise AssertionError(
                f"friction parity fail on {name}: train={tv!r} oos={ov!r}"
            )


def friction_spec_from_cfg(cfg: Mapping[str, Any] | None) -> FrictionSpec:
    """Build FrictionSpec from arm/overnight YAML keys."""
    from mascotrl.eval.yaml_honesty import track_copy

    cfg = track_copy(cfg or {})
    arm = dict(cfg.get("arm") or {})
    spec_id = str(
        arm.get("friction_spec_id")
        or cfg.get("friction_spec_id")
        or "v2_quote_touch"
    )
    plugins = dict(cfg.get("plugins") or {})
    om = dict(plugins.get("om_touch") or {})
    return FrictionSpec(
        spec_id=spec_id,
        equity_bps=float(cfg.get("equity_bps", cfg.get("capacity_spread_bps", 5.0))),
        hedge_leg_bps=float(
            cfg.get("hedge_leg_spread_bps", om.get("hedge_leg_spread_bps", 5.0))
        ),
        execution_spread_bps=float(cfg.get("execution_spread_bps", 5.0)),
        execution_impact_coef=float(cfg.get("execution_impact_coef", 0.5)),
        om_touch_enabled=bool(
            cfg.get("om_touch_enabled", om.get("enabled", True))
        ),
        om_touch_fee_bps=float(cfg.get("om_touch_fee_bps", om.get("fee_bps", 0.0))),
        om_touch_spread_multiplier=float(
            cfg.get("om_touch_spread_multiplier", om.get("spread_multiplier", 1.0))
        ),
        equity_spread_multiplier=float(cfg.get("equity_spread_multiplier", 1.0)),
        borrow_floor_bps_annual=float(cfg.get("borrow_floor_bps_annual", 25.0)),
        impact_c_eq=float(cfg.get("impact_c_eq", 0.5)),
        impact_c_opt=float(cfg.get("impact_c_opt", 1.0)),
        cost_multiplier=float(cfg.get("cost_multiplier", 1.0)),
        hedge_impact_enabled=bool(
            cfg.get(
                "hedge_impact_enabled",
                (plugins.get("hedge_impact") or {}).get("enabled", False),
            )
        ),
        hedge_impact_coef=float(
            cfg.get(
                "hedge_impact_coef",
                (plugins.get("hedge_impact") or {}).get("coef", 1.0),
            )
        ),
    )


def hedge_leg_cost(
    w: torch.Tensor,
    w_prev: torch.Tensor,
    *,
    spread_bps: float,
    deltas_now: np.ndarray | None,
    deltas_prev: np.ndarray | None,
    spot: np.ndarray | None,
    dh_denom: np.ndarray | None,
) -> float:
    """
    Stock-side cost of maintaining the delta hedge, in units of scaled return.

    Two components, both expressed per dollar of invested capital
    (the delta-hedged package costs ``dh_denom = Δ·S − C``):

      1. Package rebalancing: changing the position by ``|Δw|`` trades
         ``|Δw| · Δ·S / dh_denom`` of stock notional.
      2. Daily re-hedging: holding ``|w|`` while delta drifts by
         ``|Δ_t − Δ_{t-1}|`` trades ``|w| · |ΔΔ| · S / dh_denom`` of stock.

    O'Donovan and Yu (2025) find hedge frequency is first-order for after-cost
    delta-hedged option returns, so this term is charged explicitly rather than
    folded into the option spread.
    """
    if spread_bps <= 0.0 or spot is None or dh_denom is None or deltas_now is None:
        return 0.0
    w_np = w.detach().reshape(-1).cpu().numpy().astype(np.float64)
    wp_np = w_prev.detach().reshape(-1).cpu().numpy().astype(np.float64)
    n = min(w_np.size, spot.size, dh_denom.size, deltas_now.size)
    if n == 0:
        return 0.0
    w_np, wp_np = w_np[:n], wp_np[:n]
    s = np.nan_to_num(spot[:n], nan=0.0)
    d_now = np.nan_to_num(deltas_now[:n], nan=0.0)
    denom = np.nan_to_num(dh_denom[:n], nan=0.0)
    # Only names with a well-posed positive capital base contribute.
    safe = np.abs(denom) > 1e-8
    if not np.any(safe):
        return 0.0
    ratio = np.zeros(n, dtype=np.float64)
    ratio[safe] = np.abs(d_now[safe] * s[safe]) / np.abs(denom[safe])
    rebalance = np.abs(w_np - wp_np) * ratio
    drift = np.zeros(n, dtype=np.float64)
    if deltas_prev is not None:
        d_prev = np.nan_to_num(deltas_prev[:n], nan=0.0)
        dd = np.abs(d_now - d_prev)
        drift[safe] = np.abs(w_np[safe]) * dd[safe] * np.abs(s[safe]) / np.abs(denom[safe])
    total = float(np.sum(rebalance + drift))
    if not np.isfinite(total):
        return 0.0
    return (float(spread_bps) / 1e4) * total


def hedge_stock_notional(
    w: torch.Tensor,
    w_prev: torch.Tensor,
    *,
    deltas_now: np.ndarray | None,
    deltas_prev: np.ndarray | None,
    spot: np.ndarray | None,
) -> float:
    """Gross dollar stock notional traded for hedge rebalance + delta drift."""
    if spot is None or deltas_now is None:
        return 0.0
    w_np = w.detach().reshape(-1).cpu().numpy().astype(np.float64)
    wp_np = w_prev.detach().reshape(-1).cpu().numpy().astype(np.float64)
    n = min(w_np.size, spot.size, deltas_now.size)
    if n == 0:
        return 0.0
    s = np.nan_to_num(spot[:n], nan=0.0)
    d_now = np.nan_to_num(deltas_now[:n], nan=0.0)
    rebalance = np.abs(w_np[:n] - wp_np[:n]) * np.abs(d_now * s)
    drift = np.zeros(n, dtype=np.float64)
    if deltas_prev is not None:
        d_prev = np.nan_to_num(deltas_prev[:n], nan=0.0)
        drift = np.abs(w_np[:n]) * np.abs(d_now - d_prev) * np.abs(s)
    total = float(np.sum(rebalance + drift))
    return total if np.isfinite(total) else 0.0


def _as_1d_float_tensor(x: Any, *, n: int, device, dtype) -> torch.Tensor | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().to(device=device, dtype=dtype).reshape(-1)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype).reshape(-1)
    if t.numel() < n:
        return None
    return t[:n]


def _as_1d_numpy(x: Any, *, n: int) -> np.ndarray | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        arr = x.detach().reshape(-1).cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < n:
        return None
    return arr[:n]


def _block_indices(arm: Any | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (option_index, equity_index) for the weight vector of length ``n``."""
    if arm is None or getattr(arm, "id", None) == "opt":
        return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)
    arm_id = getattr(arm, "id", None)
    if arm_id == "eq":
        return np.array([], dtype=np.int64), np.arange(n, dtype=np.int64)
    if arm_id == "mix":
        opt = np.asarray(arm.option_index(), dtype=np.int64)
        eq = np.asarray(arm.equity_index(), dtype=np.int64)
        return opt, eq
    # Unknown arm: treat as options-only status quo.
    return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)


def _select_cols(t: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """Select columns from (B, K) or flat (K,) tensor."""
    if idx.size == 0:
        if t.dim() == 1:
            return t.new_zeros(0)
        return t.new_zeros(t.shape[0], 0)
    if t.dim() == 1:
        return t[idx]
    return t[:, idx]


def _funding_drag(
    w: torch.Tensor,
    funding: Any | None,
    *,
    prices: torch.Tensor | None = None,
    deltas: torch.Tensor | None = None,
    spot: torch.Tensor | None = None,
) -> float:
    if funding is None or not getattr(funding, "enabled", False):
        return 0.0
    fd = funding(w, prices=prices, deltas=deltas, spot=spot)
    if torch.is_tensor(fd):
        return float(fd.detach().reshape(-1)[0].item())
    return float(fd)


def option_execution_drag(
    w: torch.Tensor,
    w_prev: torch.Tensor | None,
    turn: float,
    *,
    atm_vol: float,
    execution_spread_bps: float = 0.0,
    execution_impact_coef: float = 0.0,
    execution_drag_mode: str = "fixed",
    execution_vol_ref: float = 0.20,
    execution_vol_floor: float = 0.05,
    execution_vol_cap: float = 1.0,
    om_touch_enabled: bool = False,
    om_touch_fee_bps: float = 0.0,
    spread_multiplier: float = 1.0,
    half_spreads: torch.Tensor | None = None,
    capital_base: torch.Tensor | None = None,
) -> float:
    """Option-block execution drag (OM-touch XOR stylized bps/impact)."""
    vol_mult = 1.0
    if execution_drag_mode == "vol_scaled":
        ref = max(float(execution_vol_ref), 1e-6)
        vol_mult = min(
            float(execution_vol_cap),
            max(float(execution_vol_floor), float(atm_vol) / ref),
        )
    drag = 0.0
    if om_touch_enabled:
        if w_prev is not None:
            from mascotrl.plugins.om_touch_execution import OMTouchCost

            touch = OMTouchCost(
                enabled=True,
                fee_bps=float(om_touch_fee_bps),
                spread_multiplier=float(spread_multiplier),
            )
            drag += float(
                touch.compute(w, w_prev, half_spreads, capital_base)
                .reshape(-1)[0]
                .item()
            )
    else:
        if execution_spread_bps > 0.0:
            drag += (execution_spread_bps / 1e4) * vol_mult * float(turn)
        if execution_impact_coef > 0.0:
            drag += execution_impact_coef * vol_mult * (max(float(turn), 0.0) ** 0.5)
    return float(drag)


def _equity_sqrt_impact(
    dw_abs: torch.Tensor,
    *,
    impact_c_eq: float,
    equity_adv: torch.Tensor | np.ndarray | float | None = None,
    aum: float = 1.0,
) -> float:
    """Square-root participation impact on equity ``|Δw|``.

    Plan / Phase-G convention (return units):
    ``impact = (impact_c_eq * sum_i sqrt(|dw_i| * AUM / ADV_i)) / 1e4``
    so ``impact_c_eq`` is a bps-scale coefficient (YAML default 0.5).

    When ADV is missing, use the unit-notional proxy ``AUM/ADV_i = 1`` which
    collapses to ``(impact_c_eq * sum_i sqrt(|dw_i|)) / 1e4``. Without the
    ``/1e4`` a coefficient of 0.5 would destroy NAV on ordinary rebalances.
    """
    c = float(impact_c_eq)
    if c <= 0.0 or dw_abs.numel() == 0:
        return 0.0
    dw = dw_abs.detach().reshape(-1).to(dtype=torch.float64).clamp(min=0.0)
    aum_f = float(aum) if float(aum) > 0.0 else 1.0
    if equity_adv is None:
        # Unit-notional proxy: AUM/ADV = 1 → sqrt(|dw_i|).
        part = dw
    else:
        if torch.is_tensor(equity_adv):
            adv = equity_adv.detach().reshape(-1).to(device=dw.device, dtype=dw.dtype)
        else:
            adv = torch.as_tensor(
                np.asarray(equity_adv, dtype=np.float64),
                device=dw.device,
                dtype=dw.dtype,
            ).reshape(-1)
        if adv.numel() == 1:
            adv = adv.expand_as(dw)
        elif adv.numel() < dw.numel():
            return 0.0
        else:
            adv = adv[: dw.numel()]
        # Safe ADV: non-positive → fall back to unit-notional |dw| proxy.
        part = torch.where(adv > 0, dw * aum_f / adv, dw)
    impact_bps = c * torch.sqrt(part.clamp(min=0.0)).sum()
    val = float(impact_bps.item()) / 1e4
    return val if np.isfinite(val) else 0.0


def apply_costs(
    w: torch.Tensor,
    w_prev: torch.Tensor,
    ret: torch.Tensor,
    *,
    arm: Any | None = None,
    friction: FrictionSpec | None = None,
    half_spread: torch.Tensor | np.ndarray | None = None,
    spot: torch.Tensor | np.ndarray | None = None,
    capital_base: torch.Tensor | np.ndarray | None = None,
    spread_multiplier: float = 1.0,
    om_touch_enabled: bool = False,
    om_touch_fee_bps: float = 0.0,
    equity_bps: float = 5.0,
    hedge_leg_bps: float = 5.0,
    hedge_frequency: str = "daily",
    deltas: torch.Tensor | np.ndarray | None = None,
    deltas_prev: torch.Tensor | np.ndarray | None = None,
    funding: Any | None = None,
    execution_spread_bps: float = 0.0,
    execution_impact_coef: float = 0.0,
    execution_drag_mode: str = "fixed",
    execution_vol_ref: float = 0.20,
    execution_vol_floor: float = 0.05,
    execution_vol_cap: float = 1.0,
    atm_iv: float | None = None,
    hedge_impact_enabled: bool = False,
    hedge_impact_coef: float = 1.0,
    hedge_adv: float | None = None,
    hedge_sigma: float | None = None,
    impact_c_eq: float = 0.0,
    equity_adv: torch.Tensor | np.ndarray | float | None = None,
    borrow_bps_annual: float | None = None,
    borrow_bps: float | None = None,
    funding_drag: float | None = None,
    borrow_dt_years: float = 1.0 / 252.0,
) -> FrictionBreakdown:
    """Apply shared friction to a step of weights / returns.

    Status quo (``arm is None`` or ``arm.id == "opt"``): all slots are options.
    Equity-only (``eq``): charge ``equity_bps`` on ``|Δw|`` plus optional
    ``impact_c_eq`` square-root participation; no option spread / hedge-leg.
    Mix: option block gets OM-touch / stylized option costs; equity block gets
    ``equity_bps`` (scaled by ``spread_multiplier``) + impact.

    ``borrow_bps_annual`` (alias ``borrow_bps``) charges short weights:
    ``(bps/1e4) * |w^-| * dt``. Extra ``funding_drag`` scalar is added when set.

    When ``friction`` is provided, its fields override the scalar kwargs and
    ``cost_multiplier`` scales all cost components.
    """
    mult = 1.0
    equity_spread_multiplier = 1.0
    if friction is not None:
        equity_bps = float(friction.equity_bps)
        hedge_leg_bps = float(friction.hedge_leg_bps)
        execution_spread_bps = float(friction.execution_spread_bps)
        execution_impact_coef = float(friction.execution_impact_coef)
        om_touch_enabled = bool(friction.om_touch_enabled)
        om_touch_fee_bps = float(friction.om_touch_fee_bps)
        spread_multiplier = float(friction.om_touch_spread_multiplier)
        equity_spread_multiplier = float(
            getattr(friction, "equity_spread_multiplier", 1.0) or 1.0
        )
        mult = float(friction.cost_multiplier)
        hedge_impact_enabled = bool(friction.hedge_impact_enabled)
        hedge_impact_coef = float(friction.hedge_impact_coef)
        impact_c_eq = float(friction.impact_c_eq)

    w_flat = w.detach()
    if w_flat.dim() == 1:
        w_b = w_flat.unsqueeze(0)
    else:
        w_b = w_flat
    wp = w_prev.detach()
    if wp.dim() == 1:
        wp_b = wp.unsqueeze(0)
    else:
        wp_b = wp

    ret_t = ret.detach()
    if ret_t.dim() > 1:
        ret_t = ret_t.reshape(-1)
    n = int(w_b.shape[-1])
    ret_t = ret_t.to(device=w_b.device, dtype=w_b.dtype)
    if ret_t.numel() < n:
        ret_t = torch.nn.functional.pad(ret_t, (0, n - ret_t.numel()))
    ret_t = ret_t[:n]

    # IEEE landmine: 0 * NaN == NaN; nan_to_num before the product so missing
    # marks contribute 0 rather than poisoning the whole portfolio gross.
    ret_np = ret_t.detach().cpu().numpy().astype(np.float64, copy=False)
    nan_mask = ~np.isfinite(ret_np)
    n_nan_labels = int(nan_mask.sum())
    if n_nan_labels:
        ret_np = np.nan_to_num(ret_np, nan=0.0, posinf=0.0, neginf=0.0)
        ret_t = torch.as_tensor(ret_np, device=w_b.device, dtype=w_b.dtype)
    gross = float((w_b.reshape(-1)[:n] * ret_t).sum().item())
    if not np.isfinite(gross):
        gross = 0.0

    opt_idx, eq_idx = _block_indices(arm, n)

    hs_t = _as_1d_float_tensor(half_spread, n=n, device=w_b.device, dtype=w_b.dtype)
    cb_t = _as_1d_float_tensor(capital_base, n=n, device=w_b.device, dtype=w_b.dtype)
    if hs_t is not None:
        hs_t = hs_t.unsqueeze(0)
    if cb_t is not None:
        cb_t = cb_t.unsqueeze(0)

    option_spread = 0.0
    if opt_idx.size > 0:
        w_opt = _select_cols(w_b, opt_idx)
        wp_opt = _select_cols(wp_b, opt_idx)
        hs_opt = _select_cols(hs_t, opt_idx) if hs_t is not None else None
        cb_opt = _select_cols(cb_t, opt_idx) if cb_t is not None else None
        turn_opt = float((w_opt - wp_opt).abs().sum().item())
        if not np.isfinite(turn_opt):
            turn_opt = 0.0
        option_spread = option_execution_drag(
            w_opt,
            wp_opt,
            turn_opt,
            atm_vol=float(atm_iv) if atm_iv is not None else 0.20,
            execution_spread_bps=float(execution_spread_bps),
            execution_impact_coef=float(execution_impact_coef),
            execution_drag_mode=str(execution_drag_mode),
            execution_vol_ref=float(execution_vol_ref),
            execution_vol_floor=float(execution_vol_floor),
            execution_vol_cap=float(execution_vol_cap),
            om_touch_enabled=bool(om_touch_enabled),
            om_touch_fee_bps=float(om_touch_fee_bps),
            spread_multiplier=float(spread_multiplier),
            half_spreads=hs_opt,
            capital_base=cb_opt,
        )

    equity_spread = 0.0
    if eq_idx.size > 0:
        w_eq = _select_cols(w_b, eq_idx)
        wp_eq = _select_cols(wp_b, eq_idx)
        dw_eq = (w_eq - wp_eq).abs()
        turn_eq = float(dw_eq.sum().item())
        if not np.isfinite(turn_eq):
            turn_eq = 0.0
        if float(equity_bps) > 0.0:
            equity_spread = (
                (float(equity_bps) / 1e4)
                * float(equity_spread_multiplier)
                * turn_eq
            )
        # Square-root participation impact (ADV missing → |dw| proxy).
        if float(impact_c_eq) > 0.0:
            eq_adv = None
            if equity_adv is not None:
                if torch.is_tensor(equity_adv) or isinstance(equity_adv, (int, float)):
                    eq_adv = equity_adv
                else:
                    arr = np.asarray(equity_adv, dtype=np.float64).reshape(-1)
                    if arr.size >= int(eq_idx.size):
                        eq_adv = arr[eq_idx] if arr.size >= n else arr
                    else:
                        eq_adv = arr
            equity_spread = float(equity_spread) + _equity_sqrt_impact(
                dw_eq, impact_c_eq=float(impact_c_eq), equity_adv=eq_adv
            )

    # Hedge-leg + optional Bouchaud impact on the option block.
    hedge = 0.0
    if opt_idx.size > 0 and (
        float(hedge_leg_bps) > 0.0 or bool(hedge_impact_enabled)
    ):
        w_opt = _select_cols(w_b, opt_idx)
        wp_opt = _select_cols(wp_b, opt_idx)
        d_now = _as_1d_numpy(deltas, n=n)
        d_prev = _as_1d_numpy(deltas_prev, n=n)
        if str(hedge_frequency) != "daily":
            d_prev = None
        spot_np = _as_1d_numpy(spot, n=n)
        denom_np = _as_1d_numpy(capital_base, n=n)
        if d_now is not None:
            d_now = d_now[opt_idx]
        if d_prev is not None:
            d_prev = d_prev[opt_idx]
        if spot_np is not None:
            spot_np = spot_np[opt_idx]
        if denom_np is not None:
            denom_np = denom_np[opt_idx]
        if float(hedge_leg_bps) > 0.0:
            hedge = hedge_leg_cost(
                w_opt,
                wp_opt,
                spread_bps=float(hedge_leg_bps),
                deltas_now=d_now,
                deltas_prev=d_prev,
                spot=spot_np,
                dh_denom=denom_np,
            )
        if bool(hedge_impact_enabled):
            from mascotrl.plugins.hedge_impact import hedge_impact_breakdown

            notion = hedge_stock_notional(
                w_opt,
                wp_opt,
                deltas_now=d_now,
                deltas_prev=d_prev,
                spot=spot_np,
            )
            sigma = float(
                hedge_sigma
                if hedge_sigma is not None
                else (atm_iv if atm_iv is not None else 0.20)
            )
            adv = float(hedge_adv) if hedge_adv is not None else 0.0
            impact = hedge_impact_breakdown(
                notion,
                adv,
                sigma,
                coef=float(hedge_impact_coef),
                enabled=True,
            )
            hedge = float(hedge) + float(impact.get("shortfall") or 0.0)

    # Funding on the full portfolio.
    fund_deltas = None
    if deltas is not None:
        if torch.is_tensor(deltas):
            fund_deltas = deltas.detach().to(device=w_b.device, dtype=w_b.dtype)
            if fund_deltas.dim() == 1:
                fund_deltas = fund_deltas.unsqueeze(0)
        else:
            fund_deltas = torch.as_tensor(
                np.asarray(deltas, dtype=np.float32), device=w_b.device, dtype=w_b.dtype
            ).unsqueeze(0)
    fund_spot = None
    if spot is not None:
        if torch.is_tensor(spot):
            fund_spot = spot.detach().to(device=w_b.device, dtype=w_b.dtype)
            if fund_spot.dim() == 1:
                fund_spot = fund_spot.unsqueeze(0)
        else:
            fund_spot = torch.as_tensor(
                np.asarray(spot, dtype=np.float32), device=w_b.device, dtype=w_b.dtype
            ).unsqueeze(0)
    fund = _funding_drag(w_b, funding, deltas=fund_deltas, spot=fund_spot)

    # L6b: optional annual borrow bps on short weights (om_borrate-style).
    bps_ann = borrow_bps_annual if borrow_bps_annual is not None else borrow_bps
    if bps_ann is not None and float(bps_ann) > 0.0:
        w_flat_n = w_b.reshape(-1)[:n]
        shorts = torch.clamp(-w_flat_n, min=0.0)
        borrow_drag = float(
            (shorts * (float(bps_ann) / 1e4) * float(borrow_dt_years)).sum().item()
        )
        if np.isfinite(borrow_drag):
            fund = float(fund) + float(borrow_drag)
    if funding_drag is not None and float(funding_drag) != 0.0:
        fund = float(fund) + float(funding_drag)

    if mult != 1.0:
        option_spread = float(option_spread) * mult
        equity_spread = float(equity_spread) * mult
        hedge = float(hedge) * mult
        fund = float(fund) * mult

    net = float(gross) - float(option_spread) - float(equity_spread) - float(hedge) - float(fund)
    return FrictionBreakdown(
        gross=float(gross),
        option_spread=float(option_spread),
        equity_spread=float(equity_spread),
        hedge_leg=float(hedge),
        funding=float(fund),
        net=float(net),
        n_nan_labels=int(n_nan_labels),
    )
