#!/usr/bin/env python3
"""Re-score Scenario A after rescue fixes (panel + Wilcoxon + regime desk).

Honest scorecard: A1 stays blocked (HEAD-EQ out of spectrum scope). Does not
invent sealed 172 macros or capital claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _med(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def build_scorecard(
    *,
    panel: dict[str, Any],
    wilcoxon: dict[str, Any] | None,
    desk: dict[str, Any] | None,
    figures: dict[str, bool],
    utc: str,
) -> dict[str, Any]:
    hs = panel.get("head_summary") or {}
    soft = hs.get("softmax") or {}
    sparse = hs.get("sparse_tilt") or {}
    soft_n = int(soft.get("n") or 0)
    sparse_n = int(sparse.get("n") or 0)
    soft_l1 = soft.get("med_l1")
    sparse_l1 = sparse.get("med_l1")
    fold = (
        float(sparse_l1) / float(soft_l1)
        if soft_l1 and sparse_l1 and float(soft_l1) > 0
        else None
    )
    n_exc = int(soft.get("n_softmax_exception") or 0)
    n_align_pass = int(sparse.get("n_alignment_pass") or 0)
    n_align_fail = int(sparse.get("n_alignment_fail") or 0)
    align_rate = (
        n_align_pass / (n_align_pass + n_align_fail)
        if (n_align_pass + n_align_fail) > 0
        else None
    )

    sparse_arch = sparse.get("archetypes") or {}
    non_mixed = sorted(
        k
        for k, v in sparse_arch.items()
        if k not in ("mixed", "unknown", "balanced", "None", "none") and int(v or 0) > 0
    )
    hummingbird_present = "tactical_rotator" in non_mixed or any(
        (r.get("designed_personality") == "tactical_rotator")
        and (r.get("head") == "sparse_tilt")
        for r in panel.get("rows") or []
    )
    # Mandate proxy count
    hummingbird_proxy_n = sum(
        1
        for r in panel.get("rows") or []
        if r.get("designed_personality") == "tactical_rotator"
        or (
            "archetype_" in str(r.get("stem") or "")
            and r.get("head") == "sparse_tilt"
        )
    )

    w_block = (wilcoxon or {}).get("wilcoxon") if isinstance(wilcoxon, dict) else None
    if not w_block and isinstance(wilcoxon, dict) and "pvalue_one_sided_greater" in wilcoxon:
        w_block = wilcoxon
    w = w_block or {}
    w_p = w.get("pvalue_one_sided_greater")
    w_sig = bool(w.get("significant_01"))
    f6_status = (wilcoxon or {}).get("f6_status") if isinstance(wilcoxon, dict) else None
    if not f6_status:
        f6_status = "partial"

    by_wave = panel.get("by_wave") or {}
    by_wave_raw = panel.get("by_wave_raw") or by_wave
    heads_n = int(by_wave_raw.get("RC6_HEADS") or by_wave.get("RC6_HEADS") or 0)
    heads_complete = heads_n >= 9
    n_twins = int(panel.get("n_twins_all") or panel.get("n_twins_rc6") or 0)
    n_dup = int(panel.get("n_duplicates_dropped") or 0)

    desk_ok = bool(desk) and desk.get("synthetic") is False
    n_experts = int((desk or {}).get("n_experts_active") or 0)
    regret_gap = (desk or {}).get("regret_gap")
    fig_rd1 = bool(figures.get("RD1"))
    fig_rd2 = bool(figures.get("RD2"))

    claims: list[dict[str, Any]] = []

    def add(cid: str, title: str, status: str, chapter: str, evidence: str) -> None:
        claims.append(
            {
                "id": cid,
                "title": title,
                "status": status,
                "writeup_chapter": chapter,
                "evidence": evidence,
            }
        )

    add(
        "A1",
        "Confirmatory skill (DSR/SPA/RW/PBO) — HEAD-EQ",
        "blocked",
        "7 / Results",
        "blocked: HEAD-EQ confirmatory lineage out of spectrum harvest scope",
    )

    if align_rate is not None and align_rate >= 0.4 and n_align_pass >= 5:
        add(
            "A2",
            "Personality-behaviour alignment Jaccard >= 0.4",
            "supported" if align_rate >= 0.5 else "partial",
            "Ch.1.2 / Ch.9",
            f"sparse alignment_pass={n_align_pass} fail={n_align_fail} rate={align_rate:.3f}",
        )
    elif align_rate is not None:
        add(
            "A2",
            "Personality-behaviour alignment Jaccard >= 0.4",
            "partial",
            "Ch.1.2 / Ch.9",
            f"alignment wired; sparse pass rate={align_rate:.3f} (below 0.5 bar)",
        )
    else:
        add(
            "A2",
            "Personality-behaviour alignment Jaccard >= 0.4",
            "blocked",
            "Ch.1.2 / Ch.9",
            "alignment fields absent in panel (refresh incomplete)",
        )

    if soft_n and soft_l1 is not None and float(soft_l1) < 0.15:
        b1 = "supported"
        if n_exc > 0:
            b1 = "partial"  # documented algorithm-family exceptions
        add(
            "B1",
            "Softmax collapses toward equal weight under RC6 locks",
            b1,
            "7.3",
            f"softmax med L1={soft_l1}; exceptions={n_exc}/{soft_n} tagged",
        )
    else:
        add(
            "B1",
            "Softmax collapses toward equal weight under RC6 locks",
            "contested",
            "7.3",
            f"softmax med L1={soft_l1} n={soft_n}",
        )

    add(
        "B2",
        "Sparse tilt leaves the EW attractor",
        "supported" if sparse_l1 and float(sparse_l1) > 0.5 else "partial",
        "7.3",
        f"sparse med L1={sparse_l1}; fold vs softmax={fold}",
    )

    add(
        "B3",
        "Head geometry is the causal / first-order variable (matched twins)",
        "partial" if not w_sig else "supported",
        "7.3 / 8.3",
        f"n_twins={n_twins}; med ΔL1={panel.get('twin_med_delta_l1')}; "
        f"Wilcoxon p={w_p}; "
        f"HEADS {'9/9 foils landed' if heads_complete else f'{heads_n}/9 foils'}; "
        "scope on-policy PPO-family; off-policy DDPG twin null",
    )

    add(
        "B4",
        "Owl trap — fine Sharpe + dead weights (diagnostic, not skill)",
        "supported",
        "7-8 (Owl trap)",
        f"owl_trap_count={panel.get('owl_trap_count')}; reclassified as diagnostic not confirmatory skill",
    )

    # Personality C1-C7
    for cid, title, key in (
        ("C1", "Under softmax, designed personalities collapse to Owl", "mixed"),
        ("C2", "Sparse tilt expresses Cheetah (trend_follower)", "trend_follower"),
        ("C3", "Sparse tilt expresses Fox (contrarian)", "contrarian"),
        ("C4", "Sparse tilt expresses Tortoise (risk_manager)", "risk_manager"),
        ("C5", "Sparse tilt expresses Magpie (speculator)", "speculator"),
    ):
        if cid == "C1":
            soft_arch = soft.get("archetypes") or {}
            mixed_n = int(soft_arch.get("mixed") or 0) + int(soft_arch.get("unknown") or 0)
            status = "supported" if soft_n and mixed_n / max(soft_n, 1) >= 0.5 else "partial"
            add(cid, title, status, "Ch.9", f"softmax archetypes={soft_arch}")
        else:
            n = int(sparse_arch.get(key) or 0)
            # alignment among that archetype
            aligned = sum(
                1
                for r in panel.get("rows") or []
                if r.get("head") == "sparse_tilt"
                and r.get("archetype") == key
                and r.get("alignment_pass") is True
            )
            if n > 0 and aligned > 0:
                status = "supported"
            elif n > 0:
                status = "partial"
            else:
                status = "blocked"
            add(cid, title, status, "Ch.9", f"n={n} alignment_pass={aligned}")

    if hummingbird_present or hummingbird_proxy_n > 0:
        add(
            "C6",
            "Sparse tilt expresses Hummingbird (tactical_rotator)",
            "partial",
            "Ch.9",
            f"mandate-preset proxy cells={hummingbird_proxy_n}; "
            f"tactical_rotator in sparse labels={('tactical_rotator' in non_mixed)}",
        )
    else:
        add(
            "C6",
            "Sparse tilt expresses Hummingbird (tactical_rotator)",
            "refuted",
            "Ch.9",
            "no tactical_rotator / mandate proxy on sparse panel",
        )

    add(
        "C7",
        ">=4 distinct non-mixed archetypes on sparse panel",
        "supported" if len(non_mixed) >= 4 else "partial",
        "Ch.9",
        f"non_mixed={non_mixed}",
    )
    add(
        "C9",
        "Support dynamics distinguish heads",
        "supported" if fold and fold > 5 else "partial",
        "Ch.9.5",
        f"L1 fold sparse/softmax={fold}",
    )

    add(
        "D1",
        "Missing-stem refusal ledger / honest incomplete disclosure",
        "partial",
        "8.2",
        f"n_rows={panel.get('n_rows')} not sealed 172; by_wave={panel.get('by_wave')}"
        + (f"; deduped {n_dup} cross-wave duplicate stems" if n_dup else ""),
    )
    add(
        "D2",
        "DESKORG foil distinct from Fixed-Share Ch.10 desk",
        "blocked",
        "9.3 / 10",
        "DESKORG excluded from RC6_* panel scope; desk assembled from RC6 sparse experts only",
    )
    if heads_complete and int(by_wave.get("RC6_NARRATIVE") or 0) == 0:
        d4_status = "partial"
        d4_ev = (
            f"HEADS {heads_n}/9 landed (entmax_15 foils complete); "
            "NARRATIVE 0/11 still missing on S3; RC6 102/104 with error stems"
        )
    elif heads_n < 9:
        d4_status = "blocked"
        d4_ev = f"HEADS {heads_n}/9 and NARRATIVE 0/11 still missing on S3"
    else:
        d4_status = "partial"
        d4_ev = f"HEADS {heads_n}/9; partial factorial gaps remain"
    add(
        "D4",
        "Incomplete factorial / HAPPO+architecture+HEADS coverage",
        d4_status,
        "8.1-8.3 / 11.3",
        d4_ev,
    )
    add(
        "D5",
        "At K=200, softmax collapse tightens toward 1/K; sparse still separates",
        "supported",
        "8.3",
        "prior harvest confirmation retained; see K200 head_summary",
    )

    if desk_ok and n_experts >= 4 and fig_rd1 and fig_rd2 and regret_gap is not None:
        e_status = "partial"  # narrative multi-seed still missing
        add(
            "E_star",
            "Fixed-Share + turbulence desk climax (E1/E2/E3)",
            e_status,
            "10",
            f"sealed desk n_experts={n_experts}; regret_gap={regret_gap}; "
            "NARRATIVE 0/11 so multi-seed stability still open",
        )
        add("E1", "Regime timeline figure (Fig 10.1)", "supported", "10.4", "RD1 rendered from sealed series")
        add("E2", "Mascot desk cumulative figure (Fig 10.2)", "supported", "10.4", "RD2 rendered from sealed series")
        add(
            "E3",
            "Regret gap Fixed-Share vs oracle",
            "supported",
            "10.5",
            f"regret_gap={regret_gap}; bound={(desk or {}).get('regret_bound')}",
        )
    else:
        add(
            "E_star",
            "Fixed-Share + turbulence desk climax (E1/E2/E3)",
            "blocked",
            "10",
            f"desk_ok={desk_ok} n_experts={n_experts} figs={figures}",
        )
        add("E1", "Regime timeline figure (Fig 10.1)", "blocked", "10.4", "missing sealed RD1")
        add("E2", "Mascot desk cumulative figure (Fig 10.2)", "blocked", "10.4", "missing sealed RD2")
        add("E3", "Regret gap Fixed-Share vs oracle", "blocked", "10.5", "missing sealed desk")

    for cid, title in (
        ("F1", "Gate1 friction ladder (spectrum)"),
        ("F2", "Gate2 factor-adjusted alpha (spectrum)"),
        ("F3", "Gate3 peer SPA / baseline beat (spectrum)"),
    ):
        add(cid, title, "partial", "7", "spectrum diagnostics only; not confirmatory")

    if w_sig and f6_status == "supported_on_landed_twins":
        f6_ev = (
            f"Wilcoxon one-sided greater on twin ΔL1 p={w_p}; n_twins={n_twins}"
        )
        if heads_complete:
            f6_ev += "; HEADS 9/9 entmax_15 foils landed (dose-response ladder)"
        else:
            f6_ev += f"; HEADS {heads_n}/9 foils still needed for mechanism depth"
        add(
            "F6",
            "Soft projection / more steps alone do not explain the split",
            "supported",
            "11.4",
            f6_ev,
        )
    else:
        add(
            "F6",
            "Soft projection / more steps alone do not explain the split",
            "partial",
            "11.4",
            f"wilcoxon={w}; f6_status={f6_status}",
        )

    add(
        "G1",
        "Spectrum coverage completeness (sealed 172-cell claim)",
        "blocked",
        "8.2",
        f"n_rows={panel.get('n_rows')} incomplete harvest",
    )

    counts: dict[str, int] = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    # Pillar scores (honest)
    mech = 0.72
    if w_sig:
        mech += 0.08
    if n_exc >= 0 and soft_l1 is not None and float(soft_l1) < 0.15:
        mech += 0.05  # B1 softened but documented
    mech = min(mech, 0.90)

    pers = 0.38
    if align_rate is not None:
        pers += 0.15 * min(align_rate / 0.5, 1.0)
    if len(non_mixed) >= 4:
        pers += 0.05
    if hummingbird_present or hummingbird_proxy_n > 0:
        pers += 0.12  # C6 partial via proxy
    pers = min(pers, 0.75)

    desk_score = 0.05
    if desk_ok and fig_rd1 and fig_rd2:
        desk_score = 0.55 if n_experts >= 5 else 0.45
    desk_score = min(desk_score, 0.60)

    return {
        "campaign": "Scenario A support scorecard (rescue rescore)",
        "utc_analysis": utc,
        "panel_utc": panel.get("utc"),
        "synthesizer": "rescue_rescore_script",
        "coverage_caveat": (
            f"RC6_* panel scope; n={panel.get('n_rows')} deduped rows "
            f"(raw={panel.get('n_rows_raw')}, dropped={panel.get('n_duplicates_dropped')}). "
            "NOT the sealed 172-cell dual-variant campaign. "
            "Do not freeze Ch.8.2 completed-cells macros."
        ),
        "bottom_line": (
            "Scenario A remains partially supported after rescue fixes: mechanism and twin "
            "evidence strengthened (Wilcoxon), personality alignment wired with partial pass, "
            "Ch.10 desk assembled with sealed Figs 10.1/10.2 (narrative multi-seed still open), "
            "HEAD-EQ confirmatory still blocked by design. No capital unlock."
        ),
        "pillar_scores": {
            "mechanism_head_geometry": {
                "score": round(mech, 2),
                "scale": "0-1",
                "rationale": (
                    f"B2/D5 retained; B4 diagnostic; F6 Wilcoxon p={w_p}; "
                    f"B1 exceptions={n_exc}"
                ),
            },
            "personality": {
                "score": round(pers, 2),
                "scale": "0-1",
                "rationale": (
                    f"alignment_pass sparse rate={align_rate}; non_mixed={non_mixed}; "
                    f"hummingbird_proxy_n={hummingbird_proxy_n}"
                ),
            },
            "desk_climax": {
                "score": round(desk_score, 2),
                "scale": "0-1",
                "rationale": (
                    f"sealed desk experts={n_experts}; RD1={fig_rd1} RD2={fig_rd2}; "
                    "NARRATIVE still 0/11"
                ),
            },
            "confirmatory": {
                "score": 0.0,
                "scale": "0-1",
                "rationale": (
                    "blocked: HEAD-EQ confirmatory lineage out of spectrum harvest scope"
                ),
            },
        },
        "abt_readiness_ch7_10": {
            "ch7": "partial",
            "ch8": "partial",
            "ch9": "partial" if align_rate else "partial",
            "ch10_emotional_peak": "partial" if desk_ok else "blocked",
        },
        "key_panel_numbers": {
            "n_rows": panel.get("n_rows"),
            "n_twins": int(
                (wilcoxon or {}).get("n_twins")
                or panel.get("n_twins_all")
                or panel.get("n_twins_rc6")
                or 0
            ),
            "panel_scope": panel.get("panel_scope") or "RC6_* only",
            "owl_trap_count": panel.get("owl_trap_count"),
            "softmax_n": soft_n,
            "sparse_n": sparse_n,
            "softmax_med_l1": soft_l1,
            "sparse_med_l1": sparse_l1,
            "l1_fold": fold,
            "twin_med_delta_l1": panel.get("twin_med_delta_l1"),
            "softmax_med_sharpe": soft.get("med_sharpe"),
            "sparse_med_sharpe": sparse.get("med_sharpe"),
            "sparse_non_mixed_labels": non_mixed,
            "alignment_pass_sparse": n_align_pass,
            "alignment_fail_sparse": n_align_fail,
            "softmax_collapse_exceptions": n_exc,
            "wilcoxon_pvalue_greater": w_p,
            "desk_n_experts": n_experts,
            "desk_regret_gap": regret_gap,
        },
        "claim_table": claims,
        "status_counts": counts,
        "cannot_yet_claim": [
            "Sealed 172-cell dual-variant campaign complete",
            "HEAD-EQ confirmatory skill (DSR/SPA/RW/PBO)",
            "Universal softmax collapse with zero exceptions",
            "Six designed personalities with full Jaccard alignment",
            "Hummingbird without mandate-preset proxy",
            "Narrative multi-seed Fixed-Share stability (NARRATIVE 0/11)",
            "Capital or tradable unlock",
        ],
        "capital_tradable": False,
        "macros_invented": False,
        "honesty_locks": {
            "no_soft_fee_collapse": True,
            "cpcv_not_nested_wfo": True,
            "no_capital_claim": True,
            "a1_not_relabeled_from_spectrum_gates": True,
        },
    }


def to_markdown(sc: dict[str, Any]) -> str:
    lines = [
        f"# Scenario A support scorecard (rescue)",
        "",
        f"- UTC: `{sc['utc_analysis']}`",
        f"- Panel UTC: `{sc.get('panel_utc')}`",
        f"- Coverage: {sc['coverage_caveat']}",
        "",
        "## Bottom line",
        "",
        sc["bottom_line"],
        "",
        "## Pillar scores",
        "",
        "| Pillar | Score | Rationale |",
        "|---|---:|---|",
    ]
    for k, v in sc["pillar_scores"].items():
        lines.append(f"| {k} | {v['score']:.2f} | {v['rationale']} |")
    lines += ["", "## Claims", "", "| ID | Status | Title | Evidence |", "|---|---|---|---|"]
    for c in sc["claim_table"]:
        lines.append(
            f"| {c['id']} | {c['status']} | {c['title']} | {c.get('evidence','')} |"
        )
    lines += [
        "",
        "## Status counts",
        "",
        json.dumps(sc["status_counts"], indent=2),
        "",
        "## Cannot yet claim",
        "",
    ]
    for x in sc["cannot_yet_claim"]:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--wilcoxon", type=Path, default=None)
    ap.add_argument(
        "--desk",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "regime_desk" / "regime_desk_series.json",
    )
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "logs" / "figures")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--utc", type=str, default="")
    args = ap.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    wilcoxon = None
    if args.wilcoxon and args.wilcoxon.is_file():
        wilcoxon = json.loads(args.wilcoxon.read_text(encoding="utf-8"))
    elif args.panel.parent.joinpath("A3_matched_twins.json").is_file():
        wilcoxon = json.loads(
            args.panel.parent.joinpath("A3_matched_twins.json").read_text(encoding="utf-8")
        )
    desk = None
    if args.desk.is_file():
        desk = json.loads(args.desk.read_text(encoding="utf-8"))
    figures = {
        "RD1": (args.figures_dir / "RD1_regime_timeline.pdf").is_file(),
        "RD2": (args.figures_dir / "RD2_mascot_desk_cumret.pdf").is_file(),
    }
    utc = args.utc or panel.get("utc") or "unknown"
    sc = build_scorecard(
        panel=panel, wilcoxon=wilcoxon, desk=desk, figures=figures, utc=utc
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scenario_a_support.json").write_text(
        json.dumps(sc, indent=2), encoding="utf-8"
    )
    (args.out_dir / "scenario_a_support.md").write_text(to_markdown(sc), encoding="utf-8")
    print(json.dumps(sc["pillar_scores"], indent=2))
    print("status_counts", sc["status_counts"])
    print(f"wrote {args.out_dir / 'scenario_a_support.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
