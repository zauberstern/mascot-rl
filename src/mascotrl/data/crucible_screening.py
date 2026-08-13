"""CRUCIBLE screening, communities, and sleeve assignment stages."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mascotrl.data.crucible_types import SLEEVE_FILL_ORDER, SLEEVE_IDS, CrucibleSpec
from mascotrl.eval.kahn_breadth import effective_number_of_bets_entropy
from mascotrl.eval.residualization import rolling_asset_ff4_residuals

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


