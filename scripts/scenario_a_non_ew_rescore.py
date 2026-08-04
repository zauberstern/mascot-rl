#!/usr/bin/env python3
"""Rescore Scenario A on RC6 panel with near-EW collapsed cells removed.

Excludes any cell with l1_vs_ew below --min-l1 (default 0.25, project
softmax-collapse boundary). Rebuilds twins, Wilcoxon, and scorecard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_landed_panel import filter_panel_exclude_near_ew
from scripts.scenario_a_rescue_scorecard import build_scorecard, to_markdown
from scripts.twin_delta_l1_wilcoxon import twin_wilcoxon


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--min-l1", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--utc", type=str, default="")
    ap.add_argument(
        "--desk",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "regime_desk" / "regime_desk_series.json",
    )
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "logs" / "figures")
    args = ap.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    filtered = filter_panel_exclude_near_ew(panel, min_l1=args.min_l1)
    utc = args.utc or panel.get("utc") or "unknown"
    tag = f"non_ew_l1_{str(args.min_l1).replace('.', 'p')}"

    twins_report = twin_wilcoxon(
        filtered.get("twins") or [],
        heads_n=int((panel.get("by_wave_raw") or panel.get("by_wave") or {}).get("RC6_HEADS") or 0),
    )
    twins_report["source_panel"] = str(args.panel)
    twins_report["panel_utc"] = utc
    twins_report["filter"] = filtered.get("filter")

    desk = None
    if args.desk.is_file():
        desk = json.loads(args.desk.read_text(encoding="utf-8"))
    figures = {
        "RD1": (args.figures_dir / "RD1_regime_timeline.pdf").is_file(),
        "RD2": (args.figures_dir / "RD2_mascot_desk_cumret.pdf").is_file(),
    }
    scorecard = build_scorecard(
        panel=filtered,
        wilcoxon=twins_report,
        desk=desk,
        figures=figures,
        utc=f"{utc}_{tag}",
    )
    scorecard["baseline_panel"] = str(args.panel)
    scorecard["filter"] = filtered.get("filter")
    scorecard["comparison_note"] = (
        f"Scenario A rescore excluding near-EW cells (l1_vs_ew < {args.min_l1})"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "landed_panel_non_ew.json").write_text(
        json.dumps(filtered, indent=2), encoding="utf-8"
    )
    (args.out_dir / "A3_matched_twins_non_ew.json").write_text(
        json.dumps(twins_report, indent=2), encoding="utf-8"
    )
    (args.out_dir / "scenario_a_support_non_ew.json").write_text(
        json.dumps(scorecard, indent=2), encoding="utf-8"
    )
    (args.out_dir / "scenario_a_support_non_ew.md").write_text(
        to_markdown(scorecard), encoding="utf-8"
    )

    summary = {
        "tag": tag,
        "min_l1": args.min_l1,
        "n_before": filtered["filter"]["n_before"],
        "n_after": filtered["filter"]["n_after"],
        "n_excluded": filtered["filter"]["n_excluded"],
        "excluded_by_head": filtered["filter"]["excluded_by_head"],
        "n_twins": twins_report.get("n_twins"),
        "wilcoxon_p": (twins_report.get("wilcoxon") or {}).get("pvalue_one_sided_greater"),
        "pillar_scores": scorecard.get("pillar_scores"),
        "status_counts": scorecard.get("status_counts"),
        "key_panel_numbers": scorecard.get("key_panel_numbers"),
    }
    (args.out_dir / "non_ew_rescore_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
