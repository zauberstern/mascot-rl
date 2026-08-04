#!/usr/bin/env python3
"""Assemble sealed Ch.10 regime-desk series from landed spectrum cell artifacts.

Picks one expert per archetype (highest archetype_confidence among sparse_tilt,
with mandate-preset tactical_rotator proxy and a softmax Owl control), runs
Kritzman turbulence + Herbster-Warmuth Fixed-Share + best-k-shift oracle, and
writes ``regime_desk_series.json`` in the schema consumed by
``scripts/render_regime_desk_figures.py``.

Usage:
    PYTHONPATH=. python scripts/assemble_regime_desk.py \\
        --cell-dir "/run/media/.../OUTPUT/rc6" \\
        --out logs/artifacts/regime_desk/regime_desk_series.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.adahedge import adahedge_fixed_share, flipflop, follow_the_leader
from src.eval.best_k_shift import best_k_shift, theoretical_regret_bound
from src.eval.expert_eg import boa_experts, eg_experts
from src.eval.expert_losses import expanding_unit_interval, log_wealth_loss
from src.eval.fixed_share import fixed_share, pre_register_alpha
from src.eval.hold_leader import hold_leader, rolling_leader
from src.eval.owl_hysteresis import owl_hysteresis
from src.eval.page_hinkley_switch import page_hinkley_switch
from src.eval.performance_sleeping import performance_sleeping
from src.eval.regime_desk_metrics import (
    best_solo_expert,
    book_table_row,
    per_regime_desk_stats,
    weight_turnover_l1,
)
from src.eval.regime_desk_peers import causal_rolling_panel_returns
from src.eval.regime_desk_seal import align_sealed_operational_mask
from src.eval.sleeping_experts import variable_share_sleeping
from src.eval.turbulence import classify_regime, turbulence_index
from src.eval.variable_share import variable_share
from src.eval.walk_forward_hmm import jaccard_turbulent

# Theory-ordered Gate A/B candidates (first passer is primary; not OOS Sharpe-max).
PRIMARY_MIXER_CANDIDATES = (
    "hold_leader_annual",
    "hold_leader_quarter",
    "follow_the_leader",
    "owl_hysteresis",
    "page_hinkley",
    "performance_sleeping",
    "rolling_leader_126",
    "flipflop",
    "variable_share_sleeping",
    "boa_share",
    "variable_share_log",
    "eg_experts",
    "adahedge_share",
)

DEFAULT_SEAL_PATH = (
    ROOT / "logs" / "artifacts" / "regime_scorecard" / "sealed" / "usb_kpt10_v3"
)

ARCHETYPE_TO_MASCOT = {
    "trend_follower": "cheetah",
    "contrarian": "fox",
    "risk_manager": "tortoise",
    "speculator": "magpie",
    "tactical_rotator": "hummingbird",
    "mixed": "owl",
    "balanced": "owl",
}

MASCOT_ORDER = ("cheetah", "fox", "tortoise", "magpie", "hummingbird", "owl")

# Sparse personalities that form the rotating desk; Owl is the softmax control.
DESK_ARCHETYPES = (
    "contrarian",
    "risk_manager",
    "trend_follower",
    "speculator",
    "tactical_rotator",
)

EVENT_MARKS_BY_DATE = (
    ("2020-02-20", "COVID-19 crash"),
    ("2020-03-23", "2020 recovery"),
    ("2022-03-16", "2022 rate hikes"),
)


def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _infer_head(stem: str, beh: dict[str, Any] | None = None) -> str:
    s = str(stem).lower()
    if "sparse_tilt" in s:
        return "sparse_tilt"
    if "tanh_l1" in s:
        return "tanh_l1"
    if "dirichlet" in s:
        return "dirichlet"
    if "softmax" in s:
        return "softmax"
    if beh:
        wh = str(beh.get("weight_head") or beh.get("head") or "").lower()
        if wh:
            return wh
    return "unknown"


def _load_behavior(cell_dir: Path, stem: str) -> dict[str, Any]:
    path = cell_dir / f"{stem}_policy_behavior.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _policy_returns(art: dict[str, Any]) -> np.ndarray | None:
    """Daily after-cost policy returns from path pnl (not shared panel_returns)."""
    ra = art.get("runner_artifact") or {}
    paths = ra.get("paths") or art.get("paths") or {}
    path0 = paths.get("0") or paths.get(0) or {}
    if not isinstance(path0, dict):
        return None
    pnl = path0.get("pnl")
    if pnl is None:
        return None
    r = np.asarray(pnl, dtype=np.float64).reshape(-1)
    if r.size < 2 or not np.isfinite(r).any():
        return None
    return r


def _panel_asset_returns(art: dict[str, Any]) -> np.ndarray | None:
    pr = art.get("panel_returns")
    if pr is None:
        ra = art.get("runner_artifact") or {}
        pr = ra.get("panel_returns")
    if pr is None:
        return None
    r = np.asarray(pr, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 2:
        return None
    return r


def _dates(art: dict[str, Any]) -> list[str]:
    raw = art.get("dates") or (art.get("runner_artifact") or {}).get("dates") or []
    return [str(d) for d in raw]


def load_candidate_cells(cell_dir: Path) -> list[dict[str, Any]]:
    """Load weight-bearing cells with behaviour metadata from ``cell_dir``."""
    out: list[dict[str, Any]] = []
    for art_path in sorted(cell_dir.glob("*.json")):
        name = art_path.name
        if name.endswith("_policy_behavior.json"):
            continue
        if name in (
            "index.json",
            "campaign_manifest.json",
            "behavior_refresh_summary.json",
            "behaviour_codenames.json",
        ):
            continue
        if "sha256" in name:
            continue
        try:
            art = json.loads(art_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(art, dict):
            continue
        stem = art_path.stem
        rets = _policy_returns(art)
        if rets is None:
            continue
        beh = _load_behavior(cell_dir, stem)
        head = _infer_head(stem, beh)
        arch = str(
            beh.get("archetype_primary")
            or beh.get("observed_personality")
            or beh.get("aa_primary")
            or "mixed"
        ).lower()
        conf = _num(beh.get("archetype_confidence")) or 0.0
        designed = str(beh.get("designed_personality") or "").lower()
        mandate = ""
        for token in ("archetype_carry", "archetype_crisis", "archetype_inflation"):
            if token in stem.lower():
                mandate = token
                break
        out.append(
            {
                "stem": stem,
                "head": head,
                "archetype": arch,
                "designed": designed,
                "confidence": conf,
                "mandate": mandate,
                "returns": rets,
                "dates": _dates(art),
                "panel_returns": _panel_asset_returns(art),
                "art": art,
                "beh": beh,
            }
        )
    return out


def select_experts(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pick best sparse_tilt cell per desk archetype + softmax Owl control.

    Selection: highest ``archetype_confidence`` among cells whose primary
    archetype matches. For ``tactical_rotator``, also accept mandate-preset
    cells whose designed personality is tactical_rotator when no sparse primary
    match exists.
    """
    selected: dict[str, dict[str, Any]] = {}

    for arch in DESK_ARCHETYPES:
        pool = [
            c
            for c in candidates
            if c["head"] == "sparse_tilt" and c["archetype"] == arch
        ]
        if not pool and arch == "tactical_rotator":
            # Mandate-preset proxy: accept any archetype_* stem even before
            # designed_personality is refreshed into behaviour exports.
            pool = [
                c
                for c in candidates
                if c["mandate"]
                and c["head"] in ("sparse_tilt", "unknown", "softmax")
            ]
            # Prefer sparse_tilt mandate cells.
            sparse_m = [c for c in pool if c["head"] == "sparse_tilt"]
            if sparse_m:
                pool = sparse_m
        if not pool:
            continue
        best = max(pool, key=lambda c: (c["confidence"], c["stem"]))
        if arch == "tactical_rotator" and best.get("mandate"):
            best = dict(best)
            best["hummingbird_proxy"] = True
        selected[arch] = best

    # Softmax Owl control: prefer high Sharpe / low L1 owl-trap style cell.
    soft = [c for c in candidates if c["head"] == "softmax"]
    if soft:

        def _owl_key(c: dict[str, Any]) -> tuple[float, float, str]:
            beh = c.get("beh") or {}
            sharpe = _num(beh.get("sharpe")) or _num(
                (c.get("art") or {}).get("gate2", {}).get("sharpe")
                if isinstance((c.get("art") or {}).get("gate2"), dict)
                else None
            )
            if sharpe is None:
                # Fall back to mean/std of returns (diagnostic only).
                r = c["returns"]
                sd = float(np.nanstd(r))
                sharpe = float(np.nanmean(r) / sd * math.sqrt(252)) if sd > 1e-12 else 0.0
            l1 = _num(beh.get("l1_vs_ew_mean")) or _num(beh.get("l1_vs_ew")) or 0.0
            # Prefer high Sharpe, then low L1 (collapse), then stem for stability.
            return (float(sharpe), -float(l1), c["stem"])

        selected["owl"] = max(soft, key=_owl_key)

    return selected


def _align_series(
    experts: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, list[str], np.ndarray | None]:
    """Intersect dates across experts; return (dates, R[T,n], names, panel[T,n])."""
    if len(experts) < 2:
        raise ValueError(f"need >=2 experts; got {len(experts)}")

    date_sets = []
    for meta in experts.values():
        dates = meta.get("dates") or []
        if not dates:
            raise ValueError(f"expert {meta.get('stem')} missing dates")
        date_sets.append(set(dates))
    common = set.intersection(*date_sets)
    if len(common) < 50:
        raise ValueError(f"date intersection too short: {len(common)}")
    dates_sorted = sorted(common)

    names: list[str] = []
    cols: list[np.ndarray] = []
    panel: np.ndarray | None = None
    for arch, meta in experts.items():
        mascot = ARCHETYPE_TO_MASCOT.get(arch, arch if arch in MASCOT_ORDER else "owl")
        if arch == "owl":
            mascot = "owl"
        names.append(mascot)
        dmap = {d: i for i, d in enumerate(meta["dates"])}
        idx = [dmap[d] for d in dates_sorted]
        cols.append(np.asarray(meta["returns"], dtype=np.float64)[idx])
        if panel is None and meta.get("panel_returns") is not None:
            pr = np.asarray(meta["panel_returns"], dtype=np.float64)
            if pr.ndim == 2 and pr.shape[0] == len(meta["dates"]):
                panel = pr[idx]

    R = np.column_stack(cols)
    return dates_sorted, R, names, panel


def _cumwealth(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    return np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))


def _expanding_threshold(turbulence: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Causal expanding quantile threshold series matching classify_regime."""
    turb = np.asarray(turbulence, dtype=np.float64).reshape(-1)
    thr = np.full(turb.shape[0], np.nan, dtype=np.float64)
    hist: list[float] = []
    for t, val in enumerate(turb):
        if not np.isfinite(val):
            continue
        hist.append(float(val))
        thr[t] = float(np.quantile(np.asarray(hist, dtype=np.float64), quantile))
    # Forward-fill leading NaNs for plotting.
    last = 0.0
    for t in range(thr.size):
        if np.isfinite(thr[t]):
            last = float(thr[t])
        else:
            thr[t] = last
    return thr


def _event_marks(dates: list[str]) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for day, label in EVENT_MARKS_BY_DATE:
        if day in dates:
            marks.append({"index": dates.index(day), "label": label, "date": day})
    return marks


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(252.0))


def _mixer_entry(
    name: str,
    W: np.ndarray,
    R: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    ret = (W * R).sum(axis=1)
    dom_idx = np.argmax(W, axis=1).astype(np.int32)
    row = book_table_row(ret)
    return {
        "name": name,
        "wealth": _cumwealth(ret).tolist(),
        "sharpe": _sharpe(ret),
        "mdd": row.get("max_drawdown"),
        "mean_max_weight": float(np.mean(np.max(W, axis=1))),
        "switch_count": int(np.sum(np.diff(dom_idx) != 0)),
        "weights": W.tolist(),
        "dominant_expert": [names[int(i)] for i in dom_idx],
        "_book_returns": ret,
    }


def select_primary_mixer(
    mixers: dict[str, dict[str, Any]],
    *,
    ew_sharpe: float,
    hrp_sharpe: float | None,
    olps_eg_sharpe: float | None,
    candidates: tuple[str, ...] = PRIMARY_MIXER_CANDIDATES,
) -> str:
    """First mixer in ``candidates`` that passes Gate A then Gate B; else fixed_share."""
    peers = [
        float(x)
        for x in (hrp_sharpe, olps_eg_sharpe)
        if x is not None and math.isfinite(float(x))
    ]
    peer_floor = max(peers) if peers else None
    ew = float(ew_sharpe) if math.isfinite(float(ew_sharpe)) else float("-inf")
    for name in candidates:
        m = mixers.get(name)
        if m is None:
            continue
        s = float(m["sharpe"])
        if not math.isfinite(s):
            continue
        if s <= ew:  # Gate A
            continue
        if peer_floor is not None and s <= peer_floor:  # Gate B
            continue
        return name
    return "fixed_share"


def assemble_regime_desk(
    *,
    cell_dir: Path,
    k_switches: int = 5,
    eta: float = 0.5,
    turb_window: int = 252,
    turb_quantile: float = 0.75,
    macro_cols: np.ndarray | None = None,
    seal_path: Path | None = None,
    prefer_seal_timeline: bool = True,
    require_roster_lock: Path | None = None,
    sleeping_experts: bool = True,
) -> dict[str, Any]:
    """Build sealed regime-desk payload from landed cells under ``cell_dir``."""
    candidates = load_candidate_cells(cell_dir)
    experts = select_experts(candidates)
    if len(experts) < 2:
        raise RuntimeError(
            f"insufficient experts in {cell_dir}: found {list(experts)}"
        )

    if require_roster_lock is not None:
        _assert_roster_lock(experts, Path(require_roster_lock))

    dates, R, names, panel = _align_series(experts)
    t_len, n_exp = R.shape
    losses = -R

    alpha = pre_register_alpha(k_switches=k_switches, sequence_length=t_len)
    W = fixed_share(losses, alpha=alpha, eta=eta)
    fs_returns = (W * R).sum(axis=1)
    ew_returns = R.mean(axis=1)
    dom_idx = np.argmax(W, axis=1).astype(np.int32)
    dominant_expert = [names[int(i)] for i in dom_idx]
    switch_count_dominant = int(np.sum(np.diff(dom_idx) != 0))
    mean_max_weight = float(np.mean(np.max(W, axis=1)))

    # Pre-registered alpha sensitivity (not OOS-tuned).
    alpha_grid = sorted(
        {float(alpha), 0.5 * float(alpha), 2.0 * float(alpha), max(float(alpha), 1e-6)}
    )
    alpha_sensitivity = []
    for a in alpha_grid:
        Wa = fixed_share(losses, alpha=float(a), eta=eta)
        ra = (Wa * R).sum(axis=1)
        alpha_sensitivity.append({"alpha": float(a), "sharpe": _sharpe(ra)})

    oracle_path, oracle_loss = best_k_shift(losses, k=k_switches)
    oracle_returns = R[np.arange(t_len), np.asarray(oracle_path, dtype=int)]
    fs_loss = float((-fs_returns).sum())
    regret_gap = float(fs_loss - oracle_loss)
    regret_bound = theoretical_regret_bound(
        n_experts=n_exp, k_switches=k_switches, sequence_length=t_len
    )

    if panel is None:
        panel = R.copy()
        panel_note = "expert_returns_proxy"
    else:
        panel_note = "cell_panel_returns"

    window = int(turb_window)
    panel_clean = np.asarray(panel, dtype=np.float64).copy()
    panel_clean[~np.isfinite(panel_clean)] = np.nan
    var = np.nanvar(panel_clean, axis=0)
    var = np.where(np.isfinite(var), var, -1.0)
    order = np.argsort(var)[::-1]
    max_assets = min(int(window) - 2, 40, panel_clean.shape[1])
    max_assets = max(max_assets, 2)
    panel_use = panel_clean[:, order[:max_assets]]
    panel_note = f"{panel_note};turb_assets={panel_use.shape[1]}"

    turb = None
    last_err: Exception | None = None
    for keep in (panel_use.shape[1], 20, 10, 5):
        if keep > panel_use.shape[1]:
            continue
        try:
            turb = turbulence_index(
                panel_use[:, :keep],
                window=window,
                min_names=min(5, keep),
                macro_cols=(
                    None
                    if macro_cols is None
                    else np.asarray(macro_cols, dtype=np.float64)[
                        : panel_use.shape[0]
                    ]
                ),
                scale_macro=True,
            )
            panel_note = f"{panel_note};turb_ok_n={keep}"
            break
        except Exception as exc:
            last_err = exc
            continue
    if turb is None:
        proxy = np.nanmean(np.abs(panel_use), axis=1)
        turb = proxy.astype(np.float64)
        panel_note = (
            f"{panel_note};turb_proxy_absmean;"
            f"err={type(last_err).__name__ if last_err else 'none'}"
        )

    turb_plot = np.asarray(turb, dtype=np.float64).copy()
    turb_plot[~np.isfinite(turb_plot)] = 0.0

    turbulent_q75_live = classify_regime(turb, quantile=turb_quantile)
    threshold = _expanding_threshold(turb, quantile=turb_quantile)

    # Live operational Markov (always computed for Jaccard vs seal).
    turbulent_live = np.asarray(turbulent_q75_live, dtype=bool)
    try:
        from src.eval.walk_forward_hmm import walk_forward_markov_filter

        hmm_w = min(252 * 3, max(80, t_len // 3))
        filt = walk_forward_markov_filter(
            np.asarray(turb, dtype=np.float64),
            window=hmm_w,
            step=21,
            k_regimes=2,
            growing=False,
        )
        hard = np.asarray(filt["hard"], dtype=np.int32)
        turbulent_live = hard == 1
        panel_note = f"{panel_note};operational_live=markov_filtered_p05"
    except Exception as exc:
        panel_note = (
            f"{panel_note};operational_live=q75_fallback;err={type(exc).__name__}"
        )

    turbulent = turbulent_live
    turbulent_q75 = np.asarray(turbulent_q75_live, dtype=bool)
    timeline_source = "live_refit"
    seal_name: str | None = None
    seal_matched_frac: float | None = None
    jaccard_live_vs_seal: float | None = None
    honesty_extra = ""

    seal_candidate = seal_path
    if seal_candidate is None and prefer_seal_timeline and DEFAULT_SEAL_PATH.is_dir():
        seal_candidate = DEFAULT_SEAL_PATH
    if prefer_seal_timeline and seal_candidate is not None and Path(seal_candidate).is_dir():
        aligned = align_sealed_operational_mask(seal_candidate, dates)
        if aligned["status"] in ("ok", "partial") and aligned["turbulent"] is not None:
            sealed_op = np.asarray(aligned["turbulent"], dtype=bool)
            sealed_q75 = np.asarray(aligned["turbulent_q75"], dtype=bool)
            sealed_d = np.asarray(aligned["turbulence"], dtype=np.float64)
            turbulent = sealed_op
            turbulent_q75 = sealed_q75
            seal_name = str(aligned.get("seal_name") or Path(seal_candidate).name)
            seal_matched_frac = float(aligned["matched_frac"])
            timeline_source = f"seal:{seal_name}"
            jaccard_live_vs_seal = float(
                jaccard_turbulent(turbulent_live, sealed_op)
            )
            finite_frac = float(np.isfinite(sealed_d).mean())
            if finite_frac >= 0.80:
                turb_plot = np.where(np.isfinite(sealed_d), sealed_d, turb_plot)
            else:
                panel_note = f"{panel_note};turb_plot=live_shade=seal"
            if aligned.get("limitation"):
                honesty_extra = f" {aligned['limitation']}"
            panel_note = f"{panel_note};timeline={timeline_source}"
        else:
            timeline_source = "live_refit"
            honesty_extra = f" seal_unavailable:{aligned.get('limitation')}"
            panel_note = f"{panel_note};timeline=live_refit"

    wealth: dict[str, list[float]] = {}
    expert_returns: dict[str, list[float]] = {}
    expert_meta: dict[str, Any] = {}
    for i, name in enumerate(names):
        wealth[name] = _cumwealth(R[:, i]).tolist()
        expert_returns[name] = R[:, i].tolist()
    for arch, meta in experts.items():
        mascot = "owl" if arch == "owl" else ARCHETYPE_TO_MASCOT.get(arch, arch)
        expert_meta[mascot] = {
            "archetype": arch,
            "stem": meta["stem"],
            "head": meta["head"],
            "confidence": meta["confidence"],
            "hummingbird_proxy": bool(meta.get("hummingbird_proxy")),
            "mandate": meta.get("mandate") or None,
        }

    missing = [m for m in MASCOT_ORDER if m not in wealth]
    for m in missing:
        wealth[m] = np.ones(t_len, dtype=np.float64).tolist()
        expert_returns[m] = np.zeros(t_len, dtype=np.float64).tolist()

    cap_ew = np.nanmean(panel_use, axis=1)
    cap_wealth = _cumwealth(cap_ew)

    # Peers on same panel calendar.
    hrp_peer = causal_rolling_panel_returns(
        panel_use, lookback=min(252, max(60, t_len // 3)), min_obs=40, mode="hrp"
    )
    eg_peer = causal_rolling_panel_returns(
        panel_use, lookback=min(252, max(60, t_len // 3)), min_obs=40, mode="eg"
    )

    solo = best_solo_expert(losses, R, names)
    solo_meta = expert_meta.get(solo["name"]) or {}
    solo["stem"] = solo_meta.get("stem")

    fs_turn = weight_turnover_l1(W)

    # Causal mixers + piecewise-constant switchers (baseline FS on -R kept).
    ell = log_wealth_loss(R)
    L01 = expanding_unit_interval(ell)
    owl_index = names.index("owl") if "owl" in names else None
    mixer_raw: dict[str, np.ndarray] = {
        "fixed_share": W,
        "eg_experts": eg_experts(R, eta=0.05),
        "variable_share_log": variable_share(L01, alpha=alpha, eta=0.5),
        "adahedge_share": adahedge_fixed_share(ell, alpha=alpha),
        "flipflop": flipflop(ell),
        "boa_share": boa_experts(L01, eta=1.0, alpha=alpha),
        "follow_the_leader": follow_the_leader(ell),
        "hold_leader_annual": hold_leader(R, lookback=252, hold=252),
        "hold_leader_quarter": hold_leader(R, lookback=63, hold=63),
        "rolling_leader_126": rolling_leader(R, lookback=126),
        "owl_hysteresis": owl_hysteresis(R, names, lookback=126, margin=0.25),
        "page_hinkley": page_hinkley_switch(R, names, delta=1e-4, lam=0.02),
        "performance_sleeping": performance_sleeping(
            R, alpha=alpha, lookback=126, owl_index=owl_index
        ),
    }
    if sleeping_experts:
        mixer_raw["variable_share_sleeping"] = variable_share_sleeping(
            R, turbulent, alpha=alpha, eta=0.5
        )

    mixers: dict[str, dict[str, Any]] = {}
    for mname, Wm in mixer_raw.items():
        mixers[mname] = _mixer_entry(mname, Wm, R, names)

    ew_sharpe = _sharpe(ew_returns)
    primary_name = select_primary_mixer(
        mixers,
        ew_sharpe=ew_sharpe,
        hrp_sharpe=hrp_peer.get("sharpe"),
        olps_eg_sharpe=eg_peer.get("sharpe"),
    )
    primary = mixers[primary_name]
    primary_returns = primary["_book_returns"]

    # Public mixer payload: drop internal returns arrays.
    mixers_public: dict[str, Any] = {}
    for mname, ment in mixers.items():
        mixers_public[mname] = {
            k: v for k, v in ment.items() if not k.startswith("_") and k != "weights"
        }
        # Keep weights only for primary and baseline FS (figures / continuity).
        if mname in (primary_name, "fixed_share"):
            mixers_public[mname]["weights"] = ment["weights"]

    returns_by_book: dict[str, np.ndarray] = {
        "fixed_share": fs_returns,
        "equal_weight": ew_returns,
        "oracle": oracle_returns,
        "cap_weight": cap_ew,
        "primary_mixer": primary_returns,
    }
    for i, name in enumerate(names):
        returns_by_book[name] = R[:, i]
    if hrp_peer["returns"] is not None:
        returns_by_book["hrp"] = np.asarray(hrp_peer["returns"], dtype=np.float64)
    if eg_peer["returns"] is not None:
        returns_by_book["olps_eg"] = np.asarray(eg_peer["returns"], dtype=np.float64)
    for mname, ment in mixers.items():
        if mname == "fixed_share":
            continue
        returns_by_book[mname] = ment["_book_returns"]

    by_regime = per_regime_desk_stats(returns_by_book, turbulent)

    table: dict[str, Any] = {
        "equal_weight": book_table_row(ew_returns),
        "cap_weight": book_table_row(cap_ew),
        "best_solo": {
            **book_table_row(R[:, solo["index"]]),
            "name": solo["name"],
            "stem": solo.get("stem"),
        },
        "owl": book_table_row(
            R[:, names.index("owl")] if "owl" in names else np.zeros(t_len)
        )
        if "owl" in names
        else None,
        "fixed_share": book_table_row(fs_returns, turnover=fs_turn),
        "oracle_best_k_shift": book_table_row(oracle_returns),
        "primary_mixer": {
            **book_table_row(primary_returns),
            "name": primary_name,
        },
        "hrp": (
            book_table_row(hrp_peer["returns"])
            if hrp_peer["returns"] is not None and hrp_peer["limitation"] is None
            else None
        ),
        "olps_eg": (
            book_table_row(eg_peer["returns"])
            if eg_peer["returns"] is not None and eg_peer["limitation"] is None
            else None
        ),
    }
    for mname, ment in mixers.items():
        if mname == "fixed_share":
            continue
        table[mname] = book_table_row(ment["_book_returns"])
    peer_limitations = []
    if hrp_peer.get("limitation"):
        peer_limitations.append(f"hrp:{hrp_peer['limitation']}")
    if eg_peer.get("limitation"):
        peer_limitations.append(f"olps_eg:{eg_peer['limitation']}")

    owl_sharpe = (
        _sharpe(R[:, names.index("owl")]) if "owl" in names else float("nan")
    )
    gate_c = bool(
        (
            float(primary["mean_max_weight"]) >= 0.45
            and math.isfinite(owl_sharpe)
            and float(primary["sharpe"]) >= 0.90 * float(owl_sharpe)
        )
        or (
            math.isfinite(owl_sharpe)
            and float(primary["sharpe"]) > float(owl_sharpe)
        )
    )

    payload: dict[str, Any] = {
        "synthetic": False,
        "dates": list(range(t_len)),
        "date_labels": dates,
        "turbulence": [float(v) for v in turb_plot],
        "threshold": [float(v) for v in threshold],
        "turbulent": [bool(v) for v in turbulent],
        "turbulent_q75": [bool(v) for v in turbulent_q75],
        "turbulent_live": [bool(v) for v in turbulent_live],
        "operational_label": "markov_filtered_p05",
        "timeline_source": timeline_source,
        "seal_matched_frac": seal_matched_frac,
        "jaccard_live_vs_seal_operational": jaccard_live_vs_seal,
        "wealth": wealth,
        "fixed_share": _cumwealth(fs_returns).tolist(),
        "equal_weight": _cumwealth(ew_returns).tolist(),
        "cap_weight": cap_wealth.tolist(),
        "fixed_share_weights": W.tolist(),
        "dominant_expert": dominant_expert,
        "dominant_expert_idx": [int(x) for x in dom_idx],
        "mixers": mixers_public,
        "primary_mixer": primary_name,
        "primary_wealth": primary["wealth"],
        "primary_dominant_expert": primary["dominant_expert"],
        "primary_weights": primary["weights"],
        "event_marks": _event_marks(dates),
        "expert_names": names,
        "expert_returns": expert_returns,
        "expert_meta": expert_meta,
        "missing_mascots": missing,
        "oracle_best_k_shift": _cumwealth(oracle_returns).tolist(),
        "oracle_path": [int(x) for x in oracle_path],
        "oracle_total_loss": float(oracle_loss),
        "fixed_share_total_loss": fs_loss,
        "regret_gap": regret_gap,
        "regret_bound": regret_bound,
        "alpha": float(alpha),
        "k_switches": int(k_switches),
        "eta": float(eta),
        "turb_window": window,
        "turb_quantile": float(turb_quantile),
        "panel_note": panel_note,
        "n_experts_active": int(n_exp),
        "n_candidates": int(len(candidates)),
        "diagnostics": {
            "fixed_share_sharpe": _sharpe(fs_returns),
            "equal_weight_sharpe": ew_sharpe,
            "oracle_sharpe": _sharpe(oracle_returns),
            "primary_mixer": primary_name,
            "primary_sharpe": float(primary["sharpe"]),
            "mean_max_weight_primary": float(primary["mean_max_weight"]),
            "gate_c_stretch": gate_c,
            "mixer_sharpes": {k: float(v["sharpe"]) for k, v in mixers.items()},
            "fs_beats_ew_sharpe": bool(
                _sharpe(fs_returns) >= ew_sharpe - 1e-12
                or not math.isfinite(ew_sharpe)
            ),
            "mean_max_weight": mean_max_weight,
            "switch_count_dominant": switch_count_dominant,
            "best_solo": solo,
            "by_regime": by_regime,
            "table": table,
            "peer_limitations": peer_limitations,
            "alpha_sensitivity": alpha_sensitivity,
            "alpha_grid_preregistered": True,
            "hrp_sharpe": hrp_peer.get("sharpe"),
            "olps_eg_sharpe": eg_peer.get("sharpe"),
            "olps_stub_fallback": bool(eg_peer.get("olps_stub_fallback")),
            "sleeping_experts": bool(sleeping_experts),
        },
        "honesty": {
            "capital_claim": False,
            "tradable_claim": False,
            "alpha_turbulence_gated": False,
            "loss_baseline": "-R",
            "loss_primary": primary_name,
            "wake_lag": 1 if sleeping_experts else None,
            "switcher_family": "piecewise_constant_plus_mixers",
            "hold_annual_H": 252,
            "hysteresis_margin": 0.25,
            "ph_delta": 1e-4,
            "ph_lambda": 0.02,
            "performance_sleep_lookback": 126,
            "timeline_source": timeline_source,
            "seal_name": seal_name,
            "switcher": "fixed_share_herbster_warmuth",
            "primary_switcher": primary_name,
            "note": (
                "Diagnostic Ch.10 desk assembly from landed spectrum cells; "
                "not a confirmatory HEAD-EQ result; CPCV folds are not nested WFO. "
                "Fixed-Share alpha is a Herbster-Warmuth prior, not turb-gated. "
                "Baseline FS uses losses=-R; primary selected by theory-ordered Gate A/B."
                + honesty_extra
            ),
        },
    }
    return payload


def build_roster_lock(
    payload: dict[str, Any],
    *,
    cell_dir: Path,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    seats: dict[str, Any] = {}
    for mascot, meta in (payload.get("expert_meta") or {}).items():
        seats[mascot] = {
            "stem": meta.get("stem"),
            "archetype": meta.get("archetype"),
            "confidence": meta.get("confidence"),
        }
    return {
        "schema_version": 1,
        "cell_dir": str(cell_dir),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seats": seats,
        "selection_rule": "max_archetype_confidence_sparse_tilt_plus_softmax_owl",
    }


def write_roster_lock(lock: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return path


def _assert_roster_lock(experts: dict[str, Any], lock_path: Path) -> None:
    if not lock_path.is_file():
        raise FileNotFoundError(f"roster lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    seats = lock.get("seats") or {}
    selected: dict[str, str] = {}
    for arch, meta in experts.items():
        mascot = "owl" if arch == "owl" else ARCHETYPE_TO_MASCOT.get(arch, arch)
        selected[mascot] = str(meta["stem"])
    for mascot, info in seats.items():
        want = str(info.get("stem"))
        got = selected.get(mascot)
        if got != want:
            raise RuntimeError(
                f"roster lock mismatch for {mascot}: want {want!r} got {got!r}"
            )


def write_regime_desk(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cell-dir",
        type=Path,
        required=True,
        help="Directory of landed cell finals + *_policy_behavior.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "regime_desk" / "regime_desk_series.json",
    )
    ap.add_argument("--k-switches", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--turb-window", type=int, default=252)
    ap.add_argument(
        "--macro-cols-npy",
        type=Path,
        default=None,
        help="Optional (T,m) .npy PIT macro cols for scaled y_t (eval only).",
    )
    ap.add_argument(
        "--seal-path",
        type=Path,
        default=None,
        help="SCHEMA>=3 seal dir (default: usb_kpt10_v3 if present).",
    )
    ap.add_argument(
        "--no-seal-timeline",
        action="store_true",
        help="Force live turb+Markov refit only (ignore seal).",
    )
    ap.add_argument(
        "--write-roster-lock",
        action="store_true",
        default=True,
        help="Write expert_roster_lock.json next to --out (default true).",
    )
    ap.add_argument(
        "--no-write-roster-lock",
        action="store_true",
        help="Skip writing expert_roster_lock.json.",
    )
    ap.add_argument(
        "--require-roster-lock",
        type=Path,
        default=None,
        help="Fail unless selected stems match this lock file exactly.",
    )
    ap.add_argument(
        "--sleeping-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include lag-1 sleeping Variable-Share mixer (default: on).",
    )
    args = ap.parse_args()

    macro_cols = None
    if args.macro_cols_npy is not None:
        macro_cols = np.load(args.macro_cols_npy)
    payload = assemble_regime_desk(
        cell_dir=args.cell_dir,
        k_switches=args.k_switches,
        eta=args.eta,
        turb_window=args.turb_window,
        macro_cols=macro_cols,
        seal_path=args.seal_path,
        prefer_seal_timeline=not args.no_seal_timeline,
        require_roster_lock=args.require_roster_lock,
        sleeping_experts=bool(args.sleeping_experts),
    )
    path = write_regime_desk(payload, args.out)
    if args.write_roster_lock and not args.no_write_roster_lock:
        lock = build_roster_lock(payload, cell_dir=args.cell_dir)
        lock_path = args.out.parent / "expert_roster_lock.json"
        write_roster_lock(lock, lock_path)
    diag = payload["diagnostics"]
    print(
        json.dumps(
            {
                "wrote": str(path),
                "n_experts_active": payload["n_experts_active"],
                "experts": list(payload["expert_meta"].keys()),
                "missing_mascots": payload["missing_mascots"],
                "regret_gap": payload["regret_gap"],
                "fixed_share_sharpe": diag["fixed_share_sharpe"],
                "equal_weight_sharpe": diag["equal_weight_sharpe"],
                "oracle_sharpe": diag["oracle_sharpe"],
                "primary_mixer": payload.get("primary_mixer"),
                "primary_sharpe": diag.get("primary_sharpe"),
                "timeline_source": payload.get("timeline_source"),
                "synthetic": payload["synthetic"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
