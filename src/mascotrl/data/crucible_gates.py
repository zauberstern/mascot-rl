"""CRUCIBLE discriminability gates, repair, and packing stages."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.crucible_types import CrucibleSpec
from mascotrl.data.crucible_screening import lottery_risk_budget_trim
from mascotrl.eval.friction import FrictionSpec, assert_friction_parity

def _maybe_friction_parity(
    friction_train: FrictionSpec | None,
    friction_oos: FrictionSpec | None,
    friction_spec: FrictionSpec | None = None,
) -> None:
    a = friction_train if friction_train is not None else friction_spec
    b = friction_oos if friction_oos is not None else friction_spec
    if a is not None and b is not None:
        assert_friction_parity(a, b)


def entropy_gap_upper_bound_l1(k: int, turnover_limit: float) -> float:
    """Max entropy gap for long-only weights with ``||w - ew||_1 <= tau``.

    Achieved by putting ``+tau/2`` on one name and spreading ``-tau/2`` equally
    across the rest (when that stays non-negative).
    """
    kk = int(k)
    tau = float(turnover_limit)
    if kk <= 1 or tau <= 0.0:
        return 0.0
    ew = 1.0 / float(kk)
    delta = 0.5 * tau
    # Shrink tau if equal-take would go negative.
    max_delta = ew * float(kk - 1)
    delta = min(delta, max_delta)
    w = np.full(kk, ew, dtype=np.float64)
    w[0] = ew + delta
    w[1:] = ew - delta / float(kk - 1)
    w = np.clip(w, 0.0, None)
    w = w / max(float(w.sum()), 1e-12)
    ent = float(-np.sum(w * np.log(w + 1e-12)))
    return float(np.log(kk) - ent)


def effective_g1_entropy_gap_floor(
    *,
    k: int,
    configured: float,
    turnover_limit: float | None,
) -> float:
    """Clamp YAML G1 gap floors that are unreachable under hard turnover.

    Unconstrained peaked softmax can clear ``0.60``; a hard CMDP projector
    that cold-starts from EW with ``tau=0.15`` cannot (bound ~0.07 at K=40).
    Without this clamp, CRUCIBLE fail-closes forever despite a live projector.
    """
    floor = float(configured)
    if turnover_limit is None:
        return floor
    ub = entropy_gap_upper_bound_l1(int(k), float(turnover_limit))
    # Probe draws N(0,1)→softmax→cap reach ~30% of the one-name bound; keep
    # a margin so mean gap clears the effective floor.
    return float(min(floor, max(0.0, 0.20 * ub)))


def feasible_action_diversity_probe(
    secids: Sequence[int],
    projector: Callable,
    *,
    n_draws: int = 512,
    rng: np.random.Generator | None = None,
    spec: CrucibleSpec | None = None,
    friction_train: FrictionSpec | None = None,
    friction_oos: FrictionSpec | None = None,
    friction_spec: FrictionSpec | None = None,
    turnover_limit: float | None = None,
) -> dict:
    _maybe_friction_parity(friction_train, friction_oos, friction_spec)
    spec = spec or CrucibleSpec(k=len(secids))
    k = len(secids)
    rng = rng or np.random.default_rng(0)
    ew = 1.0 / max(k, 1)
    l1s = []
    ents = []
    for _ in range(int(n_draws)):
        a = rng.normal(size=k)
        w = np.asarray(projector(a), dtype=np.float64).reshape(-1)
        if w.size != k:
            raise ValueError(f"projector returned size {w.size}, expected {k}")
        l1s.append(float(np.sum(np.abs(w - ew))))
        w_clip = np.clip(w, 0.0, None)
        # entropy on absolute weights normalised if needed
        if float(w_clip.sum()) <= 1e-12:
            p = np.full(k, ew)
        else:
            p = w_clip / w_clip.sum()
        ents.append(float(-np.sum(p * np.log(p + 1e-12))))
    l1 = float(np.mean(l1s))
    ent = float(np.mean(ents))
    gap = float(np.log(max(k, 1)) - ent)
    gap_floor = effective_g1_entropy_gap_floor(
        k=k,
        configured=float(spec.g1_entropy_gap_floor),
        turnover_limit=turnover_limit,
    )
    passed = l1 >= float(spec.g1_l1_floor) and gap >= gap_floor
    return {
        "g1_feasible_l1_vs_ew": l1,
        "g1_action_entropy": ent,
        "g1_entropy_gap": gap,
        "g1_entropy_gap_floor_effective": float(gap_floor),
        "g1_pass": bool(passed),
    }


def ridge_residual_signal(
    resid: pd.DataFrame,
    secids: Sequence[int],
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """G2 signal: ridge-shrunk cross-sectional mean of lookback residuals."""
    cols = [c for c in secids if c in resid.columns]
    if not cols:
        return np.zeros(len(secids), dtype=np.float64)
    r = resid[cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(r, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    grand = float(np.mean(mu)) if mu.size else 0.0
    signal = (mu + float(lam) * grand) / (1.0 + float(lam))
    # Align to requested secids order (zeros for missing)
    out = np.zeros(len(secids), dtype=np.float64)
    col_to_i = {c: i for i, c in enumerate(cols)}
    for j, s in enumerate(secids):
        if s in col_to_i:
            out[j] = float(signal[col_to_i[s]])
    return out


def transfer_coefficient_probe(
    secids: Sequence[int],
    signal: np.ndarray | pd.Series,
    projector: Callable,
    *,
    spec: CrucibleSpec | None = None,
    friction_train: FrictionSpec | None = None,
    friction_oos: FrictionSpec | None = None,
    friction_spec: FrictionSpec | None = None,
) -> dict:
    _maybe_friction_parity(friction_train, friction_oos, friction_spec)
    spec = spec or CrucibleSpec(k=len(secids))
    w_star = np.asarray(signal, dtype=np.float64).reshape(-1)
    if w_star.size != len(secids):
        raise ValueError("signal length must match secids")
    # Myopic MV: demean and normalise to unit L1 for correlation stability
    w_star = w_star - w_star.mean()
    if float(np.abs(w_star).sum()) > 1e-12:
        w_star = w_star / np.abs(w_star).sum()
    w_proj = np.asarray(projector(w_star), dtype=np.float64).reshape(-1)
    if w_proj.size != w_star.size:
        raise ValueError("projector size mismatch in G2")
    if float(np.std(w_star)) < 1e-15 or float(np.std(w_proj)) < 1e-15:
        tc = 0.0
    else:
        tc = float(np.corrcoef(w_star, w_proj)[0, 1])
    if not np.isfinite(tc):
        tc = 0.0
    return {
        "g2_tc_post_projection": tc,
        "g2_pass": bool(tc >= float(spec.g2_tc_floor)),
        "g2_signal_mode": "ridge_residual_or_provided",
    }


def _participation_cost(
    turnover: float, *, participation: float, impact_coef: float
) -> float:
    """Deprecated proxy; prefer FrictionSpec ``_equity_sqrt_impact`` path in G3."""
    return float(impact_coef) * float(np.sqrt(max(participation, 0.0))) * float(turnover)


def _g3_impact_cost(
    dw_abs: np.ndarray,
    *,
    friction_spec: FrictionSpec,
    participation: float,
    adv_row: np.ndarray | None,
    book_notional: float,
) -> float:
    """Match train FrictionSpec square-root impact (+ spread on L1 turnover)."""
    import torch

    from mascotrl.eval.friction import _equity_sqrt_impact

    dw = np.asarray(dw_abs, dtype=np.float64).reshape(-1)
    turn = float(np.sum(np.abs(dw)))
    spread_bps = float(getattr(friction_spec, "equity_bps", 5.0) or 5.0)
    spread = (spread_bps * 1e-4) * turn
    # Scale ADV so target participation ≈ book_notional / mean_ADV
    impact_c = float(getattr(friction_spec, "impact_c_eq", 0.5) or 0.5)
    if adv_row is None or not np.any(np.asarray(adv_row) > 0):
        # Unit ADV scaled so mean |dw| * AUM / ADV ≈ participation
        adv_eff = np.full_like(dw, max(float(book_notional) / max(float(participation), 1e-8), 1.0))
    else:
        adv_eff = np.asarray(adv_row, dtype=np.float64).reshape(-1)
        # Soft-scale so portfolio participation matches ladder level
        mean_adv = float(np.nanmean(adv_eff[adv_eff > 0])) if np.any(adv_eff > 0) else 1.0
        target_adv = max(float(book_notional) / max(float(participation), 1e-8), 1.0)
        if mean_adv > 0:
            adv_eff = adv_eff * (target_adv / mean_adv)
    impact = _equity_sqrt_impact(
        torch.as_tensor(np.abs(dw), dtype=torch.float64),
        impact_c_eq=impact_c,
        equity_adv=adv_eff,
        aum=float(book_notional),
    )
    return float(spread + impact)


def _ann_sharpe(rets: np.ndarray) -> float:
    x = np.asarray(rets, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    sd = float(np.std(x, ddof=0))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(x) / sd * np.sqrt(252.0))


def _g3_pass_from_ladder(ladder_sharpes: Mapping[str, float], *, floor: float) -> bool:
    s1 = float(ladder_sharpes.get("1pct", ladder_sharpes.get(0.01, 0.0)))
    s5 = float(ladder_sharpes.get("5pct", ladder_sharpes.get(0.05, 0.0)))
    s10 = float(ladder_sharpes.get("10pct", ladder_sharpes.get(0.10, 0.0)))
    return bool(s5 >= float(floor) and s1 >= 0.0 and s10 >= 0.0)


def structure_participation_gate(
    secids: Sequence[int],
    returns: pd.DataFrame,
    adv: pd.DataFrame,
    friction_spec: FrictionSpec,
    *,
    ladder: Sequence[float] = (0.01, 0.05, 0.10),
    spec: CrucibleSpec | None = None,
    friction_oos: FrictionSpec | None = None,
    book_notional: float = 1_000_000.0,
) -> dict:
    _maybe_friction_parity(friction_spec, friction_oos, friction_spec)
    spec = spec or CrucibleSpec(k=len(secids))
    cols = [c for c in secids if c in returns.columns]
    r = returns[cols].fillna(0.0).to_numpy(dtype=np.float64)
    t, k = r.shape
    if t < 5 or k < 1:
        empty = {
            "g3_ridge_minus_ew_sharpe_1pct": 0.0,
            "g3_ridge_minus_ew_sharpe_5pct": 0.0,
            "g3_ridge_minus_ew_sharpe_10pct": 0.0,
            "g3_pass": False,
            "g3_impact_mode": "equity_sqrt_impact",
        }
        return empty

    mu = np.nanmean(r, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    grand = float(np.mean(mu))
    lam = 1.0
    signal = (mu + lam * grand) / (1.0 + lam)
    w_ridge = signal - signal.mean()
    if float(np.abs(w_ridge).sum()) > 1e-12:
        w_ridge = w_ridge / np.abs(w_ridge).sum()
    else:
        w_ridge = np.full(k, 1.0 / k)
    w_ew = np.full(k, 1.0 / k)
    adv_cols = [c for c in cols if c in adv.columns]
    adv_mat = (
        adv[adv_cols].reindex(columns=cols).fillna(0.0).to_numpy(dtype=np.float64)
        if adv_cols
        else np.zeros_like(r)
    )

    out: dict[str, Any] = {"g3_impact_mode": "equity_sqrt_impact"}
    ladder_map: dict[str, float] = {}
    for p in ladder:
        def _path(w0: np.ndarray, part: float = float(p)) -> np.ndarray:
            w_prev = w_ew.copy()
            pnls = []
            for i in range(t):
                dw = w0 - w_prev
                cost = _g3_impact_cost(
                    dw,
                    friction_spec=friction_spec,
                    participation=part,
                    adv_row=adv_mat[i] if i < adv_mat.shape[0] else None,
                    book_notional=float(book_notional),
                )
                pnls.append(float(np.dot(w0, r[i]) - cost))
                w_prev = w0
            return np.asarray(pnls, dtype=np.float64)

        diff = _ann_sharpe(_path(w_ridge)) - _ann_sharpe(_path(w_ew))
        key = f"{int(round(float(p) * 100))}pct"
        ladder_map[key] = float(diff)
        out[f"g3_ridge_minus_ew_sharpe_{key}"] = float(diff)

    out["g3_pass"] = _g3_pass_from_ladder(ladder_map, floor=float(spec.g3_sharpe_floor))
    return out


def repair_swaps(
    sel: dict,
    ranked_candidates: Sequence[int],
    spec: CrucibleSpec,
    gates_fn: Callable[[dict], dict],
    *,
    community_of: Mapping[int, int] | None = None,
    stratum_of: Mapping[int, str] | None = None,
    resid: pd.DataFrame | None = None,
) -> tuple[dict, int]:
    """Swap lowest within-sleeve-rank names for candidates; re-check invariants."""
    cur = {
        "secids": list(sel["secids"]),
        "primary": dict(sel["primary"]),
        "membership": {k: list(v) for k, v in sel["membership"].items()},
    }
    within_rank = dict(sel.get("within_rank") or {})
    passes = 0
    for _ in range(int(spec.max_repair_passes)):
        g = gates_fn(cur)
        if g.get("g1_pass") and g.get("g2_pass") and g.get("g3_pass"):
            return cur, passes
        passes += 1
        victims = sorted(cur["secids"], key=lambda s: float(within_rank.get(s, 0.0)))
        if not victims:
            break
        drop = victims[0]
        sleeve = cur["primary"].get(drop, "core")
        repl = None
        for c in ranked_candidates:
            c = int(c)
            if c in cur["secids"]:
                continue
            # Community cap
            if community_of is not None:
                comm = int(community_of.get(c, -1))
                n_comm = sum(
                    1
                    for s in cur["secids"]
                    if s != drop and int(community_of.get(s, -1)) == comm
                )
                if n_comm >= int(spec.max_per_community):
                    continue
            repl = c
            break
        if repl is None:
            break
        cur["secids"] = [s for s in cur["secids"] if s != drop] + [repl]
        cur["primary"].pop(drop, None)
        cur["primary"][repl] = sleeve
        for m in cur["membership"].values():
            if drop in m:
                m.remove(drop)
        cur["membership"].setdefault(sleeve, []).append(repl)
        # Re-enforce lottery budget after swap
        if resid is not None:
            cur, _, _ = lottery_risk_budget_trim(
                cur,
                resid,
                cap=float(spec.lottery_resid_var_share_cap),
                refill_candidates=[
                    int(x)
                    for x in ranked_candidates
                    if int(x) not in set(cur["secids"])
                ],
            )
        # Stratum stamp only (targets are soft after initial fill)
        if stratum_of is not None:
            cur["stratum_of"] = {int(s): stratum_of.get(int(s), "unknown") for s in cur["secids"]}
    return cur, passes


def pack_slots_by_community(
    secids: Sequence[int], community_of: Mapping[int, int]
) -> tuple[list[int], list[int]]:
    """Sort names so communities are contiguous; emit HAPPO shard indices."""
    groups: dict[int, list[int]] = {}
    for s in secids:
        groups.setdefault(int(community_of.get(int(s), -1)), []).append(int(s))
    for g in groups.values():
        g.sort()
    ordered_comms = sorted(groups.keys())
    ordered: list[int] = []
    scores: list[int] = []
    for shard, c in enumerate(ordered_comms):
        for s in groups[c]:
            ordered.append(s)
            scores.append(int(shard))
    return ordered, scores


def apply_reselect_churn_cap(
    incumbent: Sequence[int],
    proposed: Sequence[int],
    cap: float,
    *,
    ranks: Mapping[int, float] | None = None,
    prefer_incumbent_on_tie: bool = True,
) -> list[int]:
    """Keep at most ``cap * K`` slot changes; incumbents win ties."""
    inc = [int(s) for s in incumbent]
    prop = [int(s) for s in proposed]
    k = len(inc) if inc else len(prop)
    if k == 0:
        return []
    if not inc:
        return prop[:k]
    max_change = int(np.floor(float(cap) * k + 1e-12))
    inc_set = set(inc)
    prop_set = set(prop)
    keep = [s for s in inc if s in prop_set]
    newcomers = [s for s in prop if s not in inc_set]
    dropped = [s for s in inc if s not in prop_set]

    def _rank(s: int) -> tuple:
        r = float(ranks.get(s, 0.0)) if ranks else 0.0
        # Higher rank first; on tie prefer incumbent (sort key: -rank, then not-incumbent)
        tie = 0 if (prefer_incumbent_on_tie and s in inc_set) else 1
        return (-r, tie, s)

    newcomers = sorted(newcomers, key=_rank)
    # Fill up to k: first kept incumbents, then limited newcomers, then remaining incumbents
    out = list(keep)
    n_change = 0
    for s in newcomers:
        if len(out) >= k:
            break
        if n_change >= max_change:
            break
        out.append(s)
        n_change += 1
    # Backfill with dropped incumbents (hysteresis) if still short
    for s in sorted(dropped, key=_rank):
        if len(out) >= k:
            break
        if s not in out:
            out.append(s)
    # If still short, take remaining proposed
    for s in prop:
        if len(out) >= k:
            break
        if s not in out:
            if s in inc_set or n_change < max_change:
                if s not in inc_set:
                    n_change += 1
                out.append(s)
    return out[:k]


def separate_turnover_keys(
    *, selection_turnover: float, policy_turnover: float
) -> dict[str, float]:
    """Record selection and policy turnover as distinct keys (never summed)."""
    return {
        "selection_turnover": float(selection_turnover),
        "policy_turnover": float(policy_turnover),
    }

