"""Diagnose oracle vs causal desk gap from sealed regime_desk_series.json."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from mascotrl.eval.adahedge import follow_the_leader
from mascotrl.eval.expert_losses import log_wealth_loss
from mascotrl.eval.regime_desk_metrics import sharpe_annualized


def oracle_segments(path: list[int], names: list[str]) -> list[dict[str, Any]]:
    """Contiguous runs of oracle_path. Each: start, end_exclusive, expert, n_days."""
    p = [int(x) for x in path]
    if not p:
        return []
    out: list[dict[str, Any]] = []
    start = 0
    for t in range(1, len(p) + 1):
        if t == len(p) or p[t] != p[start]:
            idx = p[start]
            name = names[idx] if 0 <= idx < len(names) else str(idx)
            out.append(
                {
                    "start": start,
                    "end_exclusive": t,
                    "expert": name,
                    "expert_idx": idx,
                    "n_days": t - start,
                }
            )
            start = t
    return out


def lag1_hit_rate(
    turb: np.ndarray,
    R: np.ndarray,
    names: list[str],
    specialist: str,
    owl: str = "owl",
) -> dict[str, Any]:
    """P(R_spec > R_owl | turb[t-1]==True) vs False. t=0 skipped. Causal probe only."""
    turb_arr = np.asarray(turb, dtype=bool).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    if specialist not in names or owl not in names:
        raise ValueError(f"need {specialist!r} and {owl!r} in names")
    i_s = names.index(specialist)
    i_o = names.index(owl)
    n_turb = n_calm = hit_turb = hit_calm = 0
    for t in range(1, R.shape[0]):
        wake = bool(turb_arr[t - 1])
        rs = float(R[t, i_s])
        ro = float(R[t, i_o])
        if not (np.isfinite(rs) and np.isfinite(ro)):
            continue
        win = rs > ro
        if wake:
            n_turb += 1
            hit_turb += int(win)
        else:
            n_calm += 1
            hit_calm += int(win)
    return {
        "specialist": specialist,
        "owl": owl,
        "n_turb_lag1": n_turb,
        "n_calm_lag1": n_calm,
        "hit_rate_turb": (hit_turb / n_turb) if n_turb else float("nan"),
        "hit_rate_calm": (hit_calm / n_calm) if n_calm else float("nan"),
    }


def diagnose_desk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build diagnosis dict from regime_desk_series.json keys."""
    names = list(payload.get("expert_names") or [])
    path = [int(x) for x in (payload.get("oracle_path") or [])]
    segs = oracle_segments(path, names)
    days_by: dict[str, int] = {n: 0 for n in names}
    for s in segs:
        days_by[s["expert"]] = days_by.get(s["expert"], 0) + int(s["n_days"])

    er = payload.get("expert_returns") or {}
    cols = [np.asarray(er[n], dtype=np.float64) for n in names]
    R = np.column_stack(cols) if cols else np.zeros((0, 0))
    turb = np.asarray(payload.get("turbulent") or [], dtype=bool)

    ell = log_wealth_loss(R) if R.size else np.zeros((0, 0))
    W_ftl = follow_the_leader(ell) if ell.size else np.zeros((0, 0))
    ftl_ret = (W_ftl * R).sum(axis=1) if R.size else np.zeros(0)
    ftl_dom = np.argmax(W_ftl, axis=1) if W_ftl.size else np.zeros(0, dtype=int)
    ftl_days: dict[str, int] = {n: 0 for n in names}
    for i in ftl_dom:
        if 0 <= int(i) < len(names):
            ftl_days[names[int(i)]] += 1

    hits = {}
    for spec in ("magpie", "tortoise"):
        if spec in names and "owl" in names and R.size:
            hits[spec] = lag1_hit_rate(turb, R, names, specialist=spec, owl="owl")

    return {
        "oracle_segments": segs,
        "n_switches": max(0, len(segs) - 1),
        "oracle_days_by_expert": days_by,
        "ftl_sharpe": float(sharpe_annualized(ftl_ret)) if ftl_ret.size else float("nan"),
        "ftl_days_by_expert": ftl_days,
        "lag1_hit_rates": hits,
        "mixer_sharpes": (payload.get("diagnostics") or {}).get("mixer_sharpes"),
        "primary_mixer": payload.get("primary_mixer"),
        "note": (
            "FTL concentrates but does not recover Owl; turb lag-1 is not a specialist router."
        ),
    }
