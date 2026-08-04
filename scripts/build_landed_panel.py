#!/usr/bin/env python3
"""Build landed_panel.json from RC6_* OUTPUT wave dirs after behavior refresh.

Panel scope is RC6 fleet only (RC6, RC6_K200, RC6_HAPPO, RC6_HEADS,
RC6_NARRATIVE). Legacy PICK/PICK2/DESKORG dirs are excluded.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


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


# RC6 fleet only. Higher rank wins when the same stem appears in multiple waves.
RC6_WAVE_DIR_NAMES: dict[str, str] = {
    "RC6": "rc6",
    "RC6_K200": "rc6_k200",
    "RC6_HAPPO": "rc6_happo_full",
    "RC6_HEADS": "rc6_heads",
    "RC6_NARRATIVE": "rc6_narrative",
}

WAVE_PRIORITY: dict[str, int] = {
    "RC6": 100,
    "RC6_K200": 90,
    "RC6_HAPPO": 80,
    "RC6_HEADS": 70,
    "RC6_NARRATIVE": 60,
}


def rc6_wave_dirs(out_root: Path) -> dict[str, Path]:
    """Map RC6_* wave labels to local OUTPUT subdirs (skip missing dirs)."""
    out: dict[str, Path] = {}
    for wave, dirname in RC6_WAVE_DIR_NAMES.items():
        d = out_root / dirname
        if d.is_dir():
            out[wave] = d
    return out


def _wave_rank(wave: str) -> int:
    return WAVE_PRIORITY.get(wave, 0)


def dedupe_rows_by_stem(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one row per stem; prefer higher-priority wave when duplicates exist."""
    best: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for r in rows:
        stem = str(r.get("stem") or "")
        if not stem:
            continue
        prev = best.get(stem)
        if prev is None:
            best[stem] = r
            continue
        if _wave_rank(str(r.get("wave") or "")) > _wave_rank(str(prev.get("wave") or "")):
            dropped.append(prev)
            best[stem] = r
        else:
            dropped.append(r)
    return list(best.values()), dropped


def _infer_head(stem: str, beh: dict[str, Any] | None = None) -> str:
    s = stem.lower()
    if "sparse_tilt" in s:
        return "sparse_tilt"
    if "entmax_15" in s or "entmax15" in s:
        return "entmax_15"
    if "tanh_l1" in s:
        return "tanh_l1"
    if "dirichlet" in s:
        return "dirichlet"
    if re.search(r"(^|_)discrete(_|$)", s) or "_dqn_" in s and "sparse" not in s and "softmax" not in s:
        if "discrete" in s:
            return "discrete"
    if "softmax" in s:
        return "softmax"
    if beh:
        for key in ("weight_head", "head", "action_head"):
            wh = str(beh.get(key) or "").lower()
            if wh in ("sparse_tilt", "softmax", "tanh_l1", "dirichlet", "discrete"):
                return wh
    return "unknown"


def _parse_stem(stem: str) -> dict[str, Any]:
    # eq_K100_single_ppo_mlp_sparse_tilt_cvar_ru
    parts = stem.split("_")
    out: dict[str, Any] = {
        "algo": None,
        "architecture": None,
        "objective": None,
        "policy_mode": None,
        "K": None,
        "train_world": None,
    }
    m = re.search(r"_K(\d+)_", stem)
    if m:
        out["K"] = int(m.group(1))
    for tok in ("single", "multi"):
        if f"_{tok}_" in f"_{stem}_":
            out["policy_mode"] = tok
    algos = (
        "happo",
        "cppo",
        "ppo",
        "ddpg",
        "td3",
        "sac",
        "dqn",
        "mcpg",
        "rrl",
    )
    for a in algos:
        if f"_{a}_" in f"_{stem}_":
            out["algo"] = a
            break
    for arch in ("mlp", "lstm", "gru", "mamba", "transformer"):
        if f"_{arch}_" in f"_{stem}_":
            out["architecture"] = arch
            break
    # objective: last known reward token after head
    objs = (
        "mean_std_cao",
        "cvar_ru",
        "mtm_pnl",
        "differential_sharpe",
        "mikkila_asym",
        "meanvar_kolm",
        "entropic_oce",
        "rsqp",
        "smse",
        "sdr_composite",
    )
    for o in objs:
        if o in stem:
            out["objective"] = o
            break
    for tw in (
        "rbergomi",
        "heston",
        "garch",
        "gbm",
        "sabr",
        "hybrid_pretrain_finetune",
        "uni-crucible",
        "hardtau",
    ):
        if tw in stem:
            out["train_world"] = tw
            break
    return out


def _twin_base(stem: str) -> str | None:
    if "sparse_tilt" in stem:
        return stem.replace("sparse_tilt", "HEAD", 1)
    if "softmax" in stem:
        return stem.replace("softmax", "HEAD", 1)
    return None


def build_panel(
    *,
    out_root: Path,
    utc: str,
    wave_dirs: dict[str, Path],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    enriched = 0

    for wave, d in wave_dirs.items():
        if not d.is_dir():
            continue
        for art_path in sorted(d.glob("*.json")):
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
            beh_path = d / f"{stem}_policy_behavior.json"
            beh: dict[str, Any] = {}
            if beh_path.is_file():
                try:
                    beh = json.loads(beh_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    beh = {}
                if not isinstance(beh, dict):
                    beh = {}
            measures = beh.get("behaviour") if isinstance(beh.get("behaviour"), dict) else {}
            head = _infer_head(stem, beh)
            meta = _parse_stem(stem)
            l1 = (
                _num(measures.get("l1_vs_ew_mean"))
                or _num(beh.get("l1_vs_ew_mean"))
                or _num(beh.get("l1_vs_ew"))
            )
            hhi = (
                _num(measures.get("hhi_mean"))
                or _num(beh.get("hhi_mean"))
                or _num(beh.get("hhi"))
            )
            turnover = (
                _num(measures.get("turnover_mean"))
                or _num(beh.get("turnover_mean"))
                or _num(beh.get("turnover"))
            )
            max_w = (
                _num(measures.get("max_weight_mean"))
                or _num(beh.get("max_weight_mean"))
                or _num(beh.get("max_weight"))
            )
            sharpe = _num(beh.get("sharpe")) or _num(measures.get("sharpe"))
            if sharpe is None and isinstance(art.get("gate2"), dict):
                # gate2 has alpha, not Sharpe; prefer path_summary.
                pass
            if sharpe is None:
                ra = art.get("runner_artifact") or {}
                ps = ra.get("path_summary") or {}
                sharpe = _num(ps.get("sharpe_mean")) or _num(ps.get("sharpe_median"))
            if sharpe is None:
                sharpe = _num((art.get("path_summary") or {}).get("sharpe_mean"))
            # Softmax exception may live under behaviour.
            soft_exc = beh.get("softmax_collapse_exception")
            if soft_exc is None:
                soft_exc = measures.get("softmax_collapse_exception")
            soft_note = beh.get("softmax_escape_note") or measures.get("softmax_escape_note")
            arch_label = (
                beh.get("archetype_primary")
                or beh.get("observed_personality")
                or beh.get("aa_primary")
            )
            row = {
                "stem": stem,
                "wave": wave,
                "head": head,
                "algo": meta["algo"] or beh.get("algo"),
                "architecture": meta["architecture"],
                "objective": meta["objective"] or beh.get("objective"),
                "policy_mode": meta["policy_mode"],
                "train_world": meta["train_world"],
                "K": meta["K"],
                "l1_vs_ew": l1,
                "hhi": hhi,
                "turnover": turnover,
                "max_weight": max_w,
                "sharpe": sharpe,
                "gate1": art.get("gate1"),
                "gate2": art.get("gate2"),
                "gate3": art.get("gate3"),
                "promotable": art.get("promotable"),
                "collapse": beh.get("collapse") or art.get("collapse_guard"),
                "archetype": arch_label,
                "alignment_pass": beh.get("alignment_pass"),
                "alignment_score": _num(beh.get("alignment_score")),
                "designed_personality": beh.get("designed_personality"),
                "observed_personality": beh.get("observed_personality"),
                "alignment_divergence": beh.get("alignment_divergence"),
                "softmax_collapse_exception": soft_exc,
                "softmax_escape_note": soft_note,
                "aa_primary": beh.get("aa_primary") or beh.get("archetype_primary"),
                "aa_confidence": _num(beh.get("archetype_confidence") or beh.get("aa_confidence")),
                "has_pb": bool(beh),
                "behaviour_export": bool(beh),
                "bytes": art_path.stat().st_size,
                "thin_or_dispatch": bool(art.get("dispatch_only"))
                or str(art.get("claim_tier") or "") == "dispatch_only",
                "sleeve_keys": list((beh.get("sleeve_tilts") or {}).keys())
                if isinstance(beh.get("sleeve_tilts"), dict)
                else None,
            }
            if beh:
                enriched += 1
            rows.append(row)

    raw_n = len(rows)
    by_wave_raw = Counter(str(r.get("wave") or "unknown") for r in rows)
    rows, dropped_dupes = dedupe_rows_by_stem(rows)
    return finalize_panel_rows(
        rows,
        utc=utc,
        panel_scope="RC6_* only",
        raw_n=raw_n,
        by_wave_raw=dict(by_wave_raw),
        dropped_dupes=dropped_dupes,
        enriched=enriched,
    )


def finalize_panel_rows(
    rows: list[dict[str, Any]],
    *,
    utc: str,
    panel_scope: str,
    raw_n: int | None = None,
    by_wave_raw: dict[str, int] | None = None,
    dropped_dupes: list[dict[str, Any]] | None = None,
    enriched: int | None = None,
    filter_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute head summaries, twins, and owl_trap from a row list."""
    by_wave = Counter(str(r.get("wave") or "unknown") for r in rows)
    by_head = Counter(str(r.get("head") or "unknown") for r in rows)
    if enriched is None:
        enriched = sum(1 for r in rows if r.get("has_pb"))

    # Head summaries
    head_summary: dict[str, Any] = {}
    for head, group in _groupby(rows, "head").items():
        l1s = [r["l1_vs_ew"] for r in group if r.get("l1_vs_ew") is not None]
        hhis = [r["hhi"] for r in group if r.get("hhi") is not None]
        turns = [r["turnover"] for r in group if r.get("turnover") is not None]
        maxws = [r["max_weight"] for r in group if r.get("max_weight") is not None]
        sharpes = [r["sharpe"] for r in group if r.get("sharpe") is not None]
        arches = Counter(str(r.get("archetype") or "unknown") for r in group)
        aa = Counter(str(r.get("aa_primary") or "unknown") for r in group)
        head_summary[head] = {
            "n": len(group),
            "med_l1": median(l1s) if l1s else None,
            "med_hhi": median(hhis) if hhis else None,
            "med_turnover": median(turns) if turns else None,
            "med_max_w": median(maxws) if maxws else None,
            "med_sharpe": median(sharpes) if sharpes else None,
            "archetypes": dict(arches),
            "aa_primaries": dict(aa),
            "n_l1_gt_0p25": sum(1 for v in l1s if v > 0.25),
            "n_to_gt_0p01": sum(1 for v in turns if v > 0.01),
            "n_non_mixed": sum(
                1
                for r in group
                if r.get("archetype") not in (None, "mixed", "unknown", "balanced")
            ),
            "n_alignment_pass": sum(1 for r in group if r.get("alignment_pass") is True),
            "n_alignment_fail": sum(1 for r in group if r.get("alignment_pass") is False),
            "n_softmax_exception": sum(
                1 for r in group if r.get("softmax_collapse_exception") is True
            ),
        }

    # Twins within RC6 (and K200) by HEAD base
    twins: list[dict[str, Any]] = []
    for wave in ("RC6", "RC6_K200"):
        wave_rows = [r for r in rows if r["wave"] == wave]
        by_base: dict[str, dict[str, dict[str, Any]]] = {}
        for r in wave_rows:
            base = _twin_base(r["stem"])
            if not base:
                continue
            by_base.setdefault(base, {})[r["head"]] = r
        for base, heads in by_base.items():
            if "sparse_tilt" in heads and "softmax" in heads:
                sl = heads["sparse_tilt"]["l1_vs_ew"]
                so = heads["softmax"]["l1_vs_ew"]
                if sl is None or so is None:
                    continue
                twins.append(
                    {
                        "base": base,
                        "wave": wave,
                        "sparse_l1": sl,
                        "softmax_l1": so,
                        "delta_l1": sl - so,
                        "sparse_stem": heads["sparse_tilt"]["stem"],
                        "softmax_stem": heads["softmax"]["stem"],
                    }
                )

    rc6_twins = [t for t in twins if t.get("wave") == "RC6"]
    twin_med = (
        median([t["delta_l1"] for t in rc6_twins]) if rc6_twins else None
    )

    owl_trap = 0
    for r in rows:
        if r["head"] != "softmax":
            continue
        l1 = r["l1_vs_ew"]
        sh = r["sharpe"]
        if l1 is not None and sh is not None and l1 < 0.10 and sh >= 0.5:
            owl_trap += 1

    out: dict[str, Any] = {
        "utc": utc,
        "panel_scope": panel_scope,
        "n_rows": len(rows),
        "by_wave": dict(by_wave),
        "by_head": dict(by_head),
        "head_summary": head_summary,
        "n_twins_rc6": len(rc6_twins),
        "n_twins_all": len(twins),
        "twin_med_delta_l1": twin_med,
        "twins": twins,
        "rows": rows,
        "enriched_behaviors": enriched,
        "owl_trap_count": owl_trap,
    }
    if raw_n is not None:
        out["n_rows_raw"] = raw_n
    if by_wave_raw is not None:
        out["by_wave_raw"] = by_wave_raw
    if dropped_dupes is not None:
        out["n_duplicates_dropped"] = len(dropped_dupes)
        out["duplicate_stems_dropped"] = [
            {
                "stem": d.get("stem"),
                "wave": d.get("wave"),
                "l1_vs_ew": d.get("l1_vs_ew"),
            }
            for d in dropped_dupes
        ]
    if filter_meta:
        out["filter"] = filter_meta
    return out


def filter_panel_exclude_near_ew(
    panel: dict[str, Any],
    *,
    min_l1: float = 0.25,
) -> dict[str, Any]:
    """Drop rows at or below min_l1 vs equal-weight (EW collapse / almost-EW)."""
    rows_in = list(panel.get("rows") or [])
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for r in rows_in:
        l1 = r.get("l1_vs_ew")
        if l1 is None or float(l1) < min_l1:
            excluded.append(r)
        else:
            kept.append(r)
    scope = str(panel.get("panel_scope") or "RC6_* only")
    filter_meta = {
        "kind": "exclude_near_ew",
        "min_l1_vs_ew": min_l1,
        "n_before": len(rows_in),
        "n_excluded": len(excluded),
        "n_after": len(kept),
        "excluded_by_head": dict(Counter(str(r.get("head") or "unknown") for r in excluded)),
        "excluded_stems": [r.get("stem") for r in excluded],
    }
    return finalize_panel_rows(
        kept,
        utc=str(panel.get("utc") or "unknown"),
        panel_scope=f"{scope}; L1>={min_l1}",
        raw_n=panel.get("n_rows_raw") or len(rows_in),
        by_wave_raw=panel.get("by_wave_raw"),
        dropped_dupes=panel.get("duplicate_stems_dropped"),
        filter_meta=filter_meta,
    )


def _groupby(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(key) or "unknown"), []).append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/volsurf/volsurf_raw/OUTPUT"),
    )
    ap.add_argument("--utc", type=str, default=None)
    args = ap.parse_args()
    out_root = args.out_root
    utc = args.utc or (out_root / "analysis" / "LATEST_UTC.txt").read_text().strip()
    wave_dirs = rc6_wave_dirs(out_root)
    if not wave_dirs:
        raise SystemExit(f"no RC6_* wave dirs under {out_root}")
    panel = build_panel(out_root=out_root, utc=utc, wave_dirs=wave_dirs)
    adir = out_root / "analysis" / utc
    adir.mkdir(parents=True, exist_ok=True)
    path = adir / "landed_panel.json"
    path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    latest = out_root / "analysis" / "landed_panel_latest.json"
    latest.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(path),
                "n_rows": panel["n_rows"],
                "by_wave": panel["by_wave"],
                "by_head": panel["by_head"],
                "n_twins_rc6": panel["n_twins_rc6"],
                "twin_med_delta_l1": panel["twin_med_delta_l1"],
                "owl_trap_count": panel["owl_trap_count"],
                "enriched_behaviors": panel["enriched_behaviors"],
                "alignment_pass_softmax": (panel["head_summary"].get("softmax") or {}).get(
                    "n_alignment_pass"
                ),
                "alignment_pass_sparse": (panel["head_summary"].get("sparse_tilt") or {}).get(
                    "n_alignment_pass"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
