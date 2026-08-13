"""CRUCIBLE universe selection.

Cross-sectional Residual Universe Constructed to Isolate Behavioural
Learning Expression. Builds a K-name opportunity set that (a) survives
CMDP projection with enough action diversity for exploration to matter and
(b) is partitioned into six named behavioural sleeves so a trained policy's
weight path is directly attributable to recognisable trading styles.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.crucible_schedule import (
    assert_ofat_cells_share_schedule_fingerprint,
    crucible_fingerprint,
    load_universe_schedule,
    schedule_fingerprint,
    write_universe_schedule,
)
from mascotrl.data.crucible_stages import (
    amihud_screen,
    apply_reselect_churn_cap,
    assign_sleeves,
    attrition_funnel_report,
    build_sleeve_matrix,
    effective_g1_entropy_gap_floor,
    entropy_gap_upper_bound_l1,
    feasible_action_diversity_probe,
    lottery_risk_budget_trim,
    option_eligibility,
    pack_slots_by_community,
    repair_swaps,
    residual_communities,
    residualize_pool,
    ridge_residual_signal,
    separate_turnover_keys,
    sleeve_scores,
    stratify_by_adv,
    structure_participation_gate,
    transfer_coefficient_probe,
)
from mascotrl.data.crucible_types import (
    SLEEVE_FILL_ORDER,
    SLEEVE_IDS,
    SLEEVE_QUOTAS,
    CrucibleGateFailure,
    CrucibleResult,
    CrucibleSpec,
    sleeve_defs_hash,
)
from mascotrl.eval.friction import FrictionSpec, assert_friction_parity
from mascotrl.eval.kahn_breadth import (
    effective_breadth,
    effective_number_of_bets_entropy,
    kahn_pack,
)

# Re-export private names for backward compatibility.
from mascotrl.data.crucible_gates import _g3_pass_from_ladder  # noqa: F401

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
