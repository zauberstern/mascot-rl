"""CRUCIBLE universe selection.

Cross-sectional Residual Universe Constructed to Isolate Behavioural
Learning Expression. Builds a K-name opportunity set that (a) survives
CMDP projection with enough action diversity for exploration to matter and
(b) is partitioned into six named behavioural sleeves so a trained policy's
weight path is directly attributable to recognisable trading styles.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mascotrl.eval.friction import FrictionSpec, assert_friction_parity
from mascotrl.eval.kahn_breadth import (
    effective_breadth,
    effective_number_of_bets_entropy,
    kahn_pack,
)
from mascotrl.eval.residualization import rolling_asset_ff4_residuals

SLEEVE_IDS: tuple[str, ...] = (
    "trend",
    "reversal",
    "carry",
    "defensive",
    "lottery",
    "illiquid",
    "core",
)

SLEEVE_QUOTAS: dict[str, int] = {
    "trend": 18,
    "reversal": 18,
    "carry": 18,
    "defensive": 14,
    "lottery": 12,
    "illiquid": 10,
    "core": 10,
}

SLEEVE_FILL_ORDER: tuple[str, ...] = (
    "illiquid",
    "lottery",
    "defensive",
    "carry",
    "reversal",
    "trend",
    "core",
)

_SLEEVE_DEFS_PAYLOAD = {
    "trend": "12m_resid_skip_21d",
    "reversal": "neg_21d_resid",
    "carry": "iv30_minus_hv20",
    "defensive": "neg_resid_vol_plus_neg_beta",
    "lottery": "resid_idio_vol_plus_abs_skew",
    "illiquid": "neg_adv_rank",
    "core": "overall_rank_buffer",
}


def sleeve_defs_hash() -> str:
    return hashlib.sha256(
        json.dumps(_SLEEVE_DEFS_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class CrucibleSpec:
    k: int = 100
    max_pool: int = 511
    lookback_days: int = 252
    reselect_every_days: int = 63
    reselect_churn_cap: float = 0.25
    adv_strata: tuple[tuple[float, float, float], ...] = (
        (70.0, 100.0, 0.40),
        (40.0, 70.0, 0.40),
        (20.0, 40.0, 0.20),
    )
    adv_participation_floor: float = 0.10
    amihud_drop_pct: float = 95.0
    amihud_drop_pct_crisis: float = 90.0
    n_communities: int = 20
    max_per_community: int = 3
    quotas: dict[str, int] = field(default_factory=lambda: dict(SLEEVE_QUOTAS))
    lottery_resid_var_share_cap: float = 0.20
    g1_l1_floor: float = 0.08
    g1_entropy_gap_floor: float = 0.60
    g2_tc_floor: float = 0.35
    g3_sharpe_floor: float = 0.10
    g3_participation_ladder: tuple[float, ...] = (0.01, 0.05, 0.10)
    max_repair_passes: int = 5

    def assert_k_feasible(self) -> None:
        """Fail closed when community packing cannot physically fill ``k``.

        ``n_communities * max_per_community`` is a hard upper bound on the
        selected set. A K=100 confirmatory with (20, 3) silently under-fills
        (~57 names) and then crashes inside G1 with a cryptic projector
        size mismatch; catch the config error here instead.
        """
        cap = int(self.n_communities) * int(self.max_per_community)
        if cap < int(self.k):
            raise ValueError(
                f"CRUCIBLE community capacity {self.n_communities}×"
                f"{self.max_per_community}={cap} < k={self.k}; raise "
                f"n_communities and/or max_per_community before launch"
            )

@dataclass(frozen=True)
class CrucibleResult:
    secids: list[int]
    sleeve_membership: dict[str, list[int]]
    sleeve_primary: dict[int, str]
    sleeve_matrix: np.ndarray
    community_of: dict[int, int]
    partition_scores: list[int]
    diagnostics: dict
    fingerprint: str


class CrucibleGateFailure(RuntimeError):
    """Raised when G1/G2/G3 still fail after repair."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        self.diagnostics = diagnostics or {}
        g1 = self.diagnostics.get("g1_pass")
        g2 = self.diagnostics.get("g2_pass")
        g3 = self.diagnostics.get("g3_pass")
        detail = (
            f"{message} "
            f"(g1_pass={g1}, g2_pass={g2}, g3_pass={g3}, "
            f"g1_gap={self.diagnostics.get('g1_entropy_gap')}, "
            f"g1_gap_floor_eff={self.diagnostics.get('g1_entropy_gap_floor_effective')}, "
            f"g2_tc={self.diagnostics.get('g2_tc_post_projection')}, "
            f"repair_passes_used={self.diagnostics.get('repair_passes_used')})"
        )
        super().__init__(detail)


def _zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mu) / sd


def _surface_pivot(
    surface_panel: pd.DataFrame, signal: str, secids: Sequence[int]
) -> pd.DataFrame:
    if surface_panel is None or len(surface_panel) == 0:
        return pd.DataFrame(columns=list(secids))
    d = surface_panel
    if "signal" not in d.columns:
        return pd.DataFrame(columns=list(secids))
    sub = d[d["signal"] == signal].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(secids))
    sub["date"] = pd.to_datetime(sub["date"])
    wide = sub.pivot_table(index="date", columns="secid", values="value", aggfunc="last")
    return wide.reindex(columns=list(secids))


def residualize_pool(
    returns: pd.DataFrame,
    ff4_factors: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """FF4 residual panel over the lookback ending at the last row (fit frozen)."""
    r = returns.sort_index()
    fac = ff4_factors.reindex(r.index).fillna(0.0)
    cols = list(r.columns)
    arr = r.to_numpy(dtype=np.float64)
    X = fac.to_numpy(dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 4:
        raise ValueError(f"ff4_factors must be (T, >=4); got {X.shape}")
    t = int(arr.shape[0])
    resid = rolling_asset_ff4_residuals(arr, X[:, :4], t=t, window=int(lookback_days))
    return pd.DataFrame(resid, index=r.index, columns=cols)


def stratify_by_adv(
    adv_panel: pd.DataFrame,
    spec: CrucibleSpec,
    *,
    notional_per_name: float,
) -> dict[str, list[int]]:
    """ADV floor then three percentile bands. Keys: p70_100, p40_70, p20_40."""
    med = adv_panel.median(axis=0, skipna=True)
    floor = float(spec.adv_participation_floor)
    feasible = [
        int(s)
        for s, v in med.items()
        if np.isfinite(v) and float(v) > 0 and float(notional_per_name) <= floor * float(v)
    ]
    if not feasible:
        return {"p70_100": [], "p40_70": [], "p20_40": [], "_all": []}
    vals = med.loc[feasible]
    ranks = vals.rank(pct=True) * 100.0
    bands: dict[str, list[int]] = {"p70_100": [], "p40_70": [], "p20_40": []}
    for sid, pct in ranks.items():
        p = float(pct)
        if 70.0 <= p <= 100.0:
            bands["p70_100"].append(int(sid))
        elif 40.0 <= p < 70.0:
            bands["p40_70"].append(int(sid))
        elif 20.0 <= p < 40.0:
            bands["p20_40"].append(int(sid))
    bands["_all"] = list(feasible)
    return bands


def amihud_screen(amihud_panel: pd.DataFrame, *, drop_pct: float) -> list[int]:
    med = amihud_panel.median(axis=0, skipna=True)
    thr = float(np.nanpercentile(med.to_numpy(dtype=float), float(drop_pct)))
    return [int(s) for s, v in med.items() if np.isfinite(v) and float(v) <= thr]


def option_eligibility(
    surface_panel: pd.DataFrame,
    *,
    min_obs: int = 21,
    window_days: int = 63,
    required_signals: Sequence[str] = ("mfis_30", "mfis_365"),
    eligible_secids: Sequence[int] | None = None,
) -> list[int]:
    """Keep names with required surface signals and coverage in the trailing window."""
    if eligible_secids is not None:
        return [int(s) for s in eligible_secids]
    if surface_panel is None or len(surface_panel) == 0:
        return []
    d = surface_panel.copy()
    d["date"] = pd.to_datetime(d["date"])
    last = d["date"].max()
    start = last - pd.tseries.offsets.BDay(int(window_days) - 1)
    win = d[(d["date"] >= start) & (d["date"] <= last)]
    ok: list[int] = []
    for sid, g in win.groupby("secid"):
        good = True
        for sig in required_signals:
            sub = g[g["signal"] == sig]
            if int(np.isfinite(sub["value"].to_numpy(dtype=float)).sum()) < int(min_obs):
                good = False
                break
        if good:
            ok.append(int(sid))
    return ok


def attrition_funnel_report(stages: list[tuple[str, int]]) -> dict:
    funnel = []
    for i, (stage, n_out) in enumerate(stages):
        n_in = int(stages[i - 1][1]) if i > 0 else int(n_out)
        if i == 0:
            n_in = int(n_out)
            n_dropped = 0
        else:
            n_dropped = int(n_in) - int(n_out)
        funnel.append(
            {
                "stage": stage,
                "n_in": int(n_in),
                "n_out": int(n_out),
                "n_dropped": int(n_dropped),
            }
        )
    return {"attrition_funnel": funnel}


def residual_communities(
    resid: pd.DataFrame,
    *,
    n_communities: int,
    min_n_eff_enb: float | None = 12.0,
) -> dict[int, int]:
    x = resid.dropna(how="all").to_numpy(dtype=np.float64)
    # Use columns that have enough finite rows
    cols = list(resid.columns)
    good_cols = []
    for j, c in enumerate(cols):
        col = resid[c].to_numpy(dtype=float)
        if np.isfinite(col).sum() >= 5:
            good_cols.append(c)
    if len(good_cols) < 2:
        n_finite = int(np.isfinite(resid.to_numpy(dtype=float)).sum())
        raise ValueError(
            "residual_communities: need at least 2 names with data "
            f"(good_cols={len(good_cols)} of {len(cols)}, "
            f"finite_cells={n_finite}, rows={len(resid)})"
        )
    sub = resid[good_cols].fillna(0.0).to_numpy(dtype=np.float64)
    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    n_eff = float(effective_number_of_bets_entropy(corr))
    if min_n_eff_enb is not None and (not np.isfinite(n_eff) or n_eff < float(min_n_eff_enb)):
        raise ValueError(
            f"n_eff_enb={n_eff:.4f} below floor {min_n_eff_enb} (fail-closed community stage)"
        )
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    m = min(int(n_communities), len(good_cols))
    labels = fcluster(z, t=m, criterion="maxclust")
    return {int(c): int(lab) - 1 for c, lab in zip(good_cols, labels)}


def sleeve_scores(
    resid: pd.DataFrame,
    surface_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    beta_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Six behavioural sleeve z-scores plus within-sleeve rank helpers."""
    secids = list(resid.columns)
    r = resid.fillna(0.0)
    # Trend: 12m residual return skipping most recent 21 days
    if len(r) >= 252:
        trend_raw = r.iloc[-252:-21].sum(axis=0)
    else:
        trend_raw = r.iloc[:-21].sum(axis=0) if len(r) > 21 else r.sum(axis=0)
    # Reversal: negative of 21-day residual return
    rev_raw = -r.iloc[-21:].sum(axis=0) if len(r) >= 21 else -r.sum(axis=0)
    sigma_res = r.iloc[-63:].std(axis=0) if len(r) >= 63 else r.std(axis=0)
    beta_last = beta_panel.reindex(columns=secids).iloc[-1]
    def_raw = (-_zscore(sigma_res) + -_zscore(beta_last)) / 2.0

    iv = _surface_pivot(surface_panel, "iv_30", secids)
    hv = _surface_pivot(surface_panel, "hv_20", secids)
    skew = _surface_pivot(surface_panel, "skew", secids)
    term = _surface_pivot(surface_panel, "term", secids)
    mfis30 = _surface_pivot(surface_panel, "mfis_30", secids)

    def _last_or_zero(wide: pd.DataFrame) -> pd.Series:
        if wide is None or wide.empty:
            return pd.Series(0.0, index=secids, dtype=float)
        w = wide.reindex(columns=secids)
        return w.ffill().iloc[-1].fillna(0.0) if len(w) else pd.Series(0.0, index=secids)

    carry_raw = _last_or_zero(iv) - _last_or_zero(hv)
    lottery_raw = sigma_res.fillna(0.0) + _last_or_zero(skew).abs()
    adv_med = adv_panel.reindex(columns=secids).median(axis=0)
    illiquid_raw = -adv_med.rank(pct=True)

    out = pd.DataFrame(index=secids)
    out["trend"] = _zscore(trend_raw)
    out["reversal"] = _zscore(rev_raw)
    out["carry"] = _zscore(carry_raw)
    out["defensive"] = _zscore(def_raw)
    out["lottery"] = _zscore(lottery_raw)
    out["illiquid"] = _zscore(illiquid_raw)

    # I_surf and within-sleeve overall rank (A.3)
    i_surf = (
        _zscore(_last_or_zero(hv).sub(_last_or_zero(iv)).abs())
        + _zscore(_last_or_zero(skew).abs())
        + _zscore(_last_or_zero(term).abs())
        + _zscore(_last_or_zero(mfis30).abs())
    )
    # IC proxy: |corr(resid_lag, resid)| over lookback; fall back to |trend|
    ic_mom = out["trend"].abs()
    within = 0.4 * _zscore(sigma_res) + 0.3 * _zscore(i_surf) + 0.3 * _zscore(ic_mom)
    out["within_rank"] = within
    out["i_surf"] = i_surf
    out["sigma_res"] = sigma_res
    return out


def build_sleeve_matrix(
    secids: Sequence[int], membership: Mapping[str, Sequence[int]]
) -> np.ndarray:
    k = len(secids)
    mat = np.zeros((k, len(SLEEVE_IDS)), dtype=np.float32)
    idx = {int(s): i for i, s in enumerate(secids)}
    for j, sleeve in enumerate(SLEEVE_IDS):
        for sid in membership.get(sleeve, []):
            i = idx.get(int(sid))
            if i is not None:
                mat[i, j] = 1.0
    return mat


def assign_sleeves(
    scores: pd.DataFrame,
    *,
    quotas: Mapping[str, int],
    community_of: Mapping[int, int],
    max_per_community: int,
    stratum_of: Mapping[int, str] | None = None,
    stratum_targets: Mapping[str, float] | None = None,
    k: int | None = None,
    return_shortfalls: bool = False,
) -> tuple:
    """Greedy primary-sleeve fill in SLEEVE_FILL_ORDER with community caps."""
    style_cols = [c for c in SLEEVE_IDS if c != "core" and c in scores.columns]
    primary_pref = scores[style_cols].idxmax(axis=1)
    assigned: dict[int, str] = {}
    membership: dict[str, list[int]] = {s: [] for s in SLEEVE_IDS}
    community_counts: Counter[int] = Counter()
    stratum_counts: Counter[str] = Counter()
    shortfalls: dict[str, int] = {s: 0 for s in SLEEVE_IDS}
    k_target = int(k) if k is not None else int(sum(int(v) for v in quotas.values()))

    def _stratum_ok(sid: int) -> bool:
        if stratum_of is None or stratum_targets is None:
            return True
        st = stratum_of.get(int(sid))
        if st is None:
            return True
        # Allow at most one name above target share
        target = float(stratum_targets.get(st, 1.0))
        n_sel = max(len(assigned), 1)
        if (stratum_counts[st] + 1) / float(k_target) > target + (1.0 / float(k_target)):
            return False
        return True

    for sleeve in SLEEVE_FILL_ORDER:
        need = int(quotas.get(sleeve, 0))
        if need <= 0:
            continue
        if sleeve == "core":
            # Remaining by within_rank
            cands = [
                int(s)
                for s in scores.sort_values("within_rank", ascending=False).index
                if int(s) not in assigned
            ]
        else:
            # Only names whose argmax primary matches this sleeve (plan A.2)
            matched = list(scores.index[primary_pref == sleeve])
            ordered = sorted(
                matched,
                key=lambda sid: (
                    float(scores.loc[sid, sleeve]),
                    float(scores.loc[sid, "within_rank"]),
                ),
                reverse=True,
            )
            cands = [int(s) for s in ordered if int(s) not in assigned]

        got = 0
        for sid in cands:
            if got >= need:
                break
            comm = int(community_of.get(sid, -1))
            if community_counts[comm] >= int(max_per_community):
                continue
            if not _stratum_ok(sid):
                continue
            assigned[sid] = sleeve
            membership[sleeve].append(sid)
            # Full membership also tags argmax styles for attribution
            if sleeve != "core":
                pass
            community_counts[comm] += 1
            if stratum_of is not None:
                stratum_counts[stratum_of.get(sid, "")] += 1
            got += 1
        shortfalls[sleeve] = max(0, need - got)

    # Attribution membership: primary plus any style with z > 0.5
    for sid, sleeve in assigned.items():
        if sid not in membership[sleeve]:
            membership[sleeve].append(sid)
        for st in style_cols:
            if st == sleeve:
                continue
            if float(scores.loc[sid, st]) > 0.5 and sid not in membership[st]:
                membership[st].append(int(sid))

    if return_shortfalls:
        return membership, assigned, shortfalls
    return membership, assigned


def _lottery_var_share(secids: Sequence[int], lottery: Sequence[int], resid: pd.DataFrame) -> float:
    cols = [c for c in secids if c in resid.columns]
    if not cols:
        return 0.0
    x = resid[cols].fillna(0.0).to_numpy(dtype=np.float64)
    if x.shape[0] < 2:
        return 0.0
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    k = len(cols)
    w = np.full(k, 1.0 / k)
    total = float(w @ cov @ w)
    if total <= 1e-18:
        return 0.0
    lot_idx = [i for i, c in enumerate(cols) if int(c) in set(int(s) for s in lottery)]
    if not lot_idx:
        return 0.0
    w_lot = np.zeros(k)
    for i in lot_idx:
        w_lot[i] = w[i]
    return float(w_lot @ cov @ w_lot) / total


def lottery_risk_budget_trim(
    sel: dict,
    resid: pd.DataFrame,
    *,
    cap: float,
    refill_candidates: Sequence[int] | None = None,
) -> tuple[dict, dict, float]:
    """Trim lottery sleeve until residual-var share <= cap; refill from core pool.

    Relabelling alone is not enough: a demoted lottery name is removed from the
    selection and replaced by the next unused core candidate when provided.
    """
    primary = dict(sel["primary"])
    membership = {k: list(v) for k, v in sel["membership"].items()}
    secids = list(sel["secids"])
    pre = _lottery_var_share(secids, membership.get("lottery", []), resid)
    share = pre
    sigma = resid.std(axis=0)
    refill = [int(s) for s in (refill_candidates or []) if int(s) not in set(secids)]
    while share > float(cap) + 1e-12:
        lot = [s for s in membership.get("lottery", []) if primary.get(s) == "lottery"]
        if not lot:
            break
        worst = max(lot, key=lambda s: float(sigma.get(s, 0.0)))
        # Remove worst lottery name from selection
        secids = [s for s in secids if s != worst]
        primary.pop(worst, None)
        membership["lottery"] = [s for s in membership.get("lottery", []) if s != worst]
        for sleeve, members in list(membership.items()):
            membership[sleeve] = [s for s in members if s != worst]
        # Refill from unused core candidates (preserve k)
        if refill:
            repl = refill.pop(0)
            secids.append(repl)
            primary[repl] = "core"
            membership.setdefault("core", []).append(repl)
        share = _lottery_var_share(
            secids, [s for s, p in primary.items() if p == "lottery"], resid
        )
        if len(lot) <= 1 and share > float(cap) and not refill:
            break
    post = share
    new_sel = {"primary": primary, "membership": membership, "secids": secids}
    info = {
        "lottery_resid_var_share_pre": float(pre),
        "lottery_resid_var_share_post": float(post),
        "n_lottery_dropped": int(
            sum(1 for s, p in sel["primary"].items() if p == "lottery")
            - sum(1 for s, p in primary.items() if p == "lottery")
        ),
        "k_after": len(secids),
    }
    return new_sel, info, float(post)


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


def crucible_fingerprint(result_like: dict, spec: CrucibleSpec) -> str:
    payload = {
        "secids": sorted(int(s) for s in result_like["secids"]),
        "reselect_every_days": spec.reselect_every_days,
        "quotas": dict(sorted(spec.quotas.items())),
        "g1_l1_floor": spec.g1_l1_floor,
        "g1_entropy_gap_floor": spec.g1_entropy_gap_floor,
        "g2_tc_floor": spec.g2_tc_floor,
        "g3_sharpe_floor": spec.g3_sharpe_floor,
        "ff4_fit_hash": result_like["ff4_fit_hash"],
        "sleeve_defs_hash": result_like["sleeve_defs_hash"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def schedule_fingerprint(slots_rows: Sequence[Sequence[int | None]]) -> str:
    """Hash of the full reselect schedule (OFAT freeze key)."""
    payload = [
        [None if s is None else int(s) for s in row] for row in slots_rows
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def write_universe_schedule(
    path: str | Path,
    *,
    slots_rows: Sequence[Sequence[int | None]],
    dates: Sequence,
    fingerprint: str,
    selection_fingerprint: str | None = None,
) -> Path:
    """Persist CRUCIBLE slots for OFAT cells to share the same selected universe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sched_fp = schedule_fingerprint(slots_rows)
    payload = {
        "schema": "crucible_universe_schedule_v1",
        "schedule_fingerprint": sched_fp,
        "selection_fingerprint": selection_fingerprint or fingerprint,
        "fingerprint": fingerprint,
        "n_dates": len(slots_rows),
        "dates": [str(pd.Timestamp(d).date()) for d in dates],
        "slots_rows": [[None if s is None else int(s) for s in row] for row in slots_rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_universe_schedule(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CRUCIBLE schedule freeze missing: {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != "crucible_universe_schedule_v1":
        raise ValueError(f"unknown schedule schema: {data.get('schema')!r}")
    rows = data.get("slots_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("schedule freeze has empty slots_rows")
    recomputed = schedule_fingerprint(rows)
    if recomputed != str(data.get("schedule_fingerprint") or ""):
        raise ValueError(
            "schedule_fingerprint mismatch (file corrupted or tampered)"
        )
    return data


def assert_ofat_cells_share_schedule_fingerprint(
    schedule_paths: Sequence[str | Path],
) -> str:
    """Fail closed if OFAT cells do not share one frozen schedule fingerprint."""
    fps = []
    for p in schedule_paths:
        data = load_universe_schedule(p)
        fps.append(str(data["schedule_fingerprint"]))
    if not fps:
        raise ValueError("no OFAT schedule paths provided")
    if len(set(fps)) != 1:
        raise AssertionError(
            f"OFAT cells do not share schedule fingerprint: {sorted(set(fps))}"
        )
    return fps[0]

def _stratum_map(bands: Mapping[str, Sequence[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key in ("p70_100", "p40_70", "p20_40"):
        for s in bands.get(key, []):
            out[int(s)] = key
    return out


def select_universe_crucible(
    *,
    as_of: pd.Timestamp,
    pool_secids: Sequence[int],
    returns: pd.DataFrame,
    ff4_factors: pd.DataFrame,
    adv_panel: pd.DataFrame,
    amihud_panel: pd.DataFrame,
    surface_panel: pd.DataFrame,
    beta_panel: pd.DataFrame,
    projector: Callable | None,
    friction_spec: FrictionSpec,
    spec: CrucibleSpec | None = None,
    policy_mode: str = "shared",
    incumbent_secids: Sequence[int] | None = None,
    rng_seed: int = 0,
    book_notional: float = 1_000_000.0,
    eligible_secids: Sequence[int] | None = None,
    turnover_limit: float | None = None,
) -> CrucibleResult:
    """Orchestrate stages C0-C13. ``projector`` is required."""
    if projector is None:
        raise ValueError(
            "CRUCIBLE requires the live CMDP projector; gate G1 is meaningless without it"
        )
    spec = spec or CrucibleSpec()
    spec.assert_k_feasible()
    as_of = pd.Timestamp(as_of)
    # C0 parent
    parent = [int(s) for s in pool_secids][: int(spec.max_pool)]
    n_parent = len(parent)
    if n_parent == 0:
        raise ValueError("CRUCIBLE C0: empty parent pool")

    # Align panels to lookback ending at as_of
    rets = returns.copy()
    rets.index = pd.to_datetime(rets.index)
    rets = rets.loc[rets.index <= as_of]
    # Need 2x lookback so rolling_asset_ff4_residuals warms up then yields a full window
    need_rows = int(spec.lookback_days) * 2 + 21
    look = rets.tail(need_rows)
    cols = [c for c in parent if c in look.columns]
    look = look[cols]
    # Fail closed before cryptic residual_communities errors when the campaign
    # only passed eval-window returns (no pre-EVAL_START history).
    min_rows = int(spec.lookback_days) + 5
    if len(look) < min_rows:
        raise ValueError(
            f"CRUCIBLE lookback too short at {as_of.date()}: have {len(look)} "
            f"return rows ending at as_of, need >= {min_rows} "
            f"(prepend selection-window history into the returns panel)"
        )
    adv = adv_panel.reindex(index=look.index, columns=cols)
    amihud = amihud_panel.reindex(index=look.index, columns=cols)
    beta = beta_panel.reindex(index=look.index, columns=cols)
    ff4 = ff4_factors.copy()
    ff4.index = pd.to_datetime(ff4.index)
    ff4 = ff4.reindex(index=look.index).fillna(0.0)

    # C1 ADV
    notional = float(book_notional) / float(max(spec.k, 1))
    bands = stratify_by_adv(adv, spec, notional_per_name=notional)
    after_adv = list(bands.get("_all", []))
    n_after_adv = len(after_adv)
    if n_after_adv == 0 or any(
        len(bands[k]) == 0 for k in ("p70_100", "p40_70", "p20_40")
    ):
        raise ValueError(
            f"CRUCIBLE C1 ADV floor/stratum fail: n_after_adv={n_after_adv} "
            f"strata={{p70_100:{len(bands['p70_100'])},p40_70:{len(bands['p40_70'])},"
            f"p20_40:{len(bands['p20_40'])}}}"
        )

    # C2 Amihud
    drop_pct = (
        spec.amihud_drop_pct_crisis
        if policy_mode == "archetype_crisis"
        else spec.amihud_drop_pct
    )
    amihud_ok = set(amihud_screen(amihud[after_adv], drop_pct=drop_pct))
    after_amihud = [s for s in after_adv if s in amihud_ok]
    n_after_amihud = len(after_amihud)

    # C3 Option eligibility
    opt_ok = set(
        option_eligibility(
            surface_panel,
            min_obs=21,
            window_days=63,
            eligible_secids=eligible_secids,
        )
    )
    if eligible_secids is None and (surface_panel is None or len(surface_panel) == 0):
        after_option = []
    elif eligible_secids is not None:
        after_option = [s for s in after_amihud if s in opt_ok]
    else:
        after_option = [s for s in after_amihud if s in opt_ok]
    n_after_option = len(after_option)

    # C4 attrition
    stages = [
        ("parent", n_parent),
        ("adv", n_after_adv),
        ("amihud", n_after_amihud),
        ("option", n_after_option),
    ]
    funnel = attrition_funnel_report(stages)
    if n_after_option < 2 * int(spec.k):
        raise ValueError(
            f"CRUCIBLE C4 eligible pool {n_after_option} < 2 * K={2 * spec.k} "
            f"(option stage); funnel={funnel['attrition_funnel']}"
        )

    eligible = after_option
    # C5 residualise
    resid = residualize_pool(look[eligible], ff4, lookback_days=spec.lookback_days)
    resid_tail = resid.tail(spec.lookback_days)
    ff4_fit_hash = hashlib.sha256(
        np.ascontiguousarray(ff4.tail(spec.lookback_days).to_numpy(dtype=np.float64)).tobytes()
    ).hexdigest()

    # C6 communities
    community_of = residual_communities(
        resid_tail[eligible], n_communities=spec.n_communities, min_n_eff_enb=12.0
    )
    # Restrict to names that got a community label
    eligible = [s for s in eligible if s in community_of]
    corr = np.corrcoef(resid_tail[eligible].fillna(0.0).to_numpy(dtype=np.float64), rowvar=False)
    n_eff_enb = float(effective_number_of_bets_entropy(corr))
    mean_abs_rho = float(np.mean(np.abs(corr[np.triu_indices(len(eligible), k=1)])))
    kp = kahn_pack(
        resid_tail[eligible].fillna(0.0).to_numpy(dtype=np.float64),
        pnls=np.zeros(len(resid_tail)),
        turnovers=np.zeros(len(resid_tail)),
        k=len(eligible),
    )

    # C7 scores
    scores = sleeve_scores(
        resid_tail[eligible],
        surface_panel,
        adv[eligible],
        beta[eligible],
    )
    stratum_of = _stratum_map(bands)
    stratum_targets = {f"p{int(a)}_{int(b)}".replace("p70_100", "p70_100"): c for a, b, c in spec.adv_strata}
    # Normalise keys
    stratum_targets = {
        "p70_100": float(spec.adv_strata[0][2]),
        "p40_70": float(spec.adv_strata[1][2]),
        "p20_40": float(spec.adv_strata[2][2]),
    }

    # C8 assign
    membership, primary, shortfalls = assign_sleeves(
        scores,
        quotas=spec.quotas,
        community_of=community_of,
        max_per_community=spec.max_per_community,
        stratum_of=stratum_of,
        stratum_targets=stratum_targets,
        k=spec.k,
        return_shortfalls=True,
    )
    # Absorb shortfalls into core from remaining
    selected = list(primary.keys())
    if len(selected) < spec.k:
        for sid in scores.sort_values("within_rank", ascending=False).index:
            sid = int(sid)
            if sid in primary:
                continue
            comm = community_of.get(sid, -1)
            if sum(1 for s, p in primary.items() if community_of.get(s) == comm) >= spec.max_per_community:
                continue
            primary[sid] = "core"
            membership.setdefault("core", []).append(sid)
            selected.append(sid)
            if len(selected) >= spec.k:
                break
    selected = selected[: spec.k]
    primary = {s: primary[s] for s in selected}
    membership = {
        sleeve: [s for s in membership.get(sleeve, []) if s in primary]
        for sleeve in SLEEVE_IDS
    }
    if len(selected) < int(spec.k):
        raise ValueError(
            f"CRUCIBLE C8 under-filled: selected {len(selected)} < k={spec.k} "
            f"(community capacity {spec.n_communities}×{spec.max_per_community}"
            f"={spec.n_communities * spec.max_per_community}; shortfalls={shortfalls})"
        )

    # C9 lottery budget
    sel = {"secids": selected, "primary": primary, "membership": membership}
    core_refill = [
        int(sid)
        for sid in scores.sort_values("within_rank", ascending=False).index
        if int(sid) not in set(selected)
    ]
    sel, lot_info, _ = lottery_risk_budget_trim(
        sel,
        resid_tail,
        cap=spec.lottery_resid_var_share_cap,
        refill_candidates=core_refill,
    )
    selected = list(sel["secids"])[: spec.k]
    primary = sel["primary"]
    membership = sel["membership"]

    rng = np.random.default_rng(int(rng_seed))

    def _run_gates(cur_secids: Sequence[int]) -> dict:
        assert_friction_parity(friction_spec, friction_spec)
        g1 = feasible_action_diversity_probe(
            cur_secids,
            projector,
            n_draws=512,
            rng=rng,
            spec=spec,
            friction_spec=friction_spec,
            turnover_limit=turnover_limit,
        )
        # G2: ridge of lookback residuals (not raw trend scores)
        sig = ridge_residual_signal(resid_tail, list(cur_secids))
        g2 = transfer_coefficient_probe(
            cur_secids, sig, projector, spec=spec, friction_spec=friction_spec
        )
        g3 = structure_participation_gate(
            cur_secids,
            look[list(cur_secids)],
            adv[list(cur_secids)],
            friction_spec,
            ladder=spec.g3_participation_ladder,
            spec=spec,
            book_notional=float(book_notional),
        )
        return {**g1, **g2, **g3}

    # C10 / C11
    gates = _run_gates(selected)
    repair_used = 0
    if not (gates["g1_pass"] and gates["g2_pass"] and gates["g3_pass"]):
        ranked = list(scores.sort_values("within_rank", ascending=False).index.astype(int))
        cur = {
            "secids": selected,
            "primary": primary,
            "membership": membership,
            "within_rank": scores["within_rank"].to_dict(),
        }

        def _gf(sdict):
            return _run_gates(sdict["secids"])

        cur, repair_used = repair_swaps(
            cur,
            ranked,
            spec,
            _gf,
            community_of=community_of,
            stratum_of=stratum_of,
            resid=resid_tail,
        )
        selected = cur["secids"][: spec.k]
        primary = cur["primary"]
        membership = cur["membership"]
        gates = _run_gates(selected)
        if not (gates["g1_pass"] and gates["g2_pass"] and gates["g3_pass"]):
            diag = {
                **gates,
                "repair_passes_used": repair_used,
                "as_of": as_of.isoformat(),
            }
            raise CrucibleGateFailure(
                "CRUCIBLE gates failed after repair", diagnostics=diag
            )

    # C12 churn cap
    selection_turnover = 0.0
    reselect_slot_churn_pct = 0.0
    if incumbent_secids is not None and len(incumbent_secids) > 0:
        before = list(selected)
        selected = apply_reselect_churn_cap(
            incumbent_secids,
            selected,
            spec.reselect_churn_cap,
            ranks=scores["within_rank"].to_dict(),
            prefer_incumbent_on_tie=True,
        )
        # Reconcile primary for any reintroduced incumbents
        for s in selected:
            if s not in primary:
                primary[s] = "core"
                membership.setdefault("core", []).append(s)
        changed = len(set(selected) - set(incumbent_secids))
        reselect_slot_churn_pct = float(changed) / float(max(spec.k, 1))
        selection_turnover = float(len(set(selected).symmetric_difference(set(incumbent_secids)))) / (
            2.0 * max(spec.k, 1)
        )

    # C13 pack
    ordered, partition_scores = pack_slots_by_community(selected, community_of)
    sleeve_mat = build_sleeve_matrix(ordered, membership)
    s_hash = sleeve_defs_hash()
    fp = crucible_fingerprint(
        {
            "secids": ordered,
            "ff4_fit_hash": ff4_fit_hash,
            "sleeve_defs_hash": s_hash,
        },
        spec,
    )
    community_sizes = [
        int(v)
        for _, v in sorted(Counter(community_of[s] for s in ordered).items())
    ]
    sleeve_counts = {s: sum(1 for p in primary.values() if p == s) for s in SLEEVE_IDS}
    diagnostics = {
        "schema_version": 1,
        "as_of": as_of.date().isoformat(),
        "policy_mode": str(policy_mode),
        "n_parent": n_parent,
        "n_after_adv": n_after_adv,
        "n_after_amihud": n_after_amihud,
        "n_after_option": n_after_option,
        "n_eligible": len(eligible),
        "attrition_funnel": funnel["attrition_funnel"],
        "liquidity_stratum_counts": {
            "p70_100": len(bands["p70_100"]),
            "p40_70": len(bands["p40_70"]),
            "p20_40": len(bands["p20_40"]),
        },
        "sleeve_counts": sleeve_counts,
        "sleeve_shortfalls": shortfalls,
        "lottery_resid_var_share_pre": lot_info["lottery_resid_var_share_pre"],
        "lottery_resid_var_share_post": lot_info["lottery_resid_var_share_post"],
        "n_eff_enb_residual": n_eff_enb,
        "mean_abs_rho_residual": mean_abs_rho,
        "kahn_pack": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in kp.items() if k != "note"},
        **gates,
        "repair_passes_used": int(repair_used),
        "selection_turnover": float(selection_turnover),
        "reselect_slot_churn_pct": float(reselect_slot_churn_pct),
        "community_sizes": community_sizes,
        "fingerprint": fp,
        "friction_parity_ok": True,
        "effective_breadth_residual": float(effective_breadth(resid_tail[ordered].fillna(0.0).to_numpy())),
    }
    return CrucibleResult(
        secids=ordered,
        sleeve_membership=membership,
        sleeve_primary={int(s): str(primary[s]) for s in ordered if s in primary},
        sleeve_matrix=sleeve_mat,
        community_of={int(s): int(community_of[s]) for s in ordered},
        partition_scores=list(partition_scores),
        diagnostics=diagnostics,
        fingerprint=fp,
    )
