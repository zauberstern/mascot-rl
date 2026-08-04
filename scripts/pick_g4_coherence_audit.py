#!/usr/bin/env python3
"""G4 coherence audit over pulled PICK finals (confirmatory gate)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_finals(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in root.rglob("*.json"):
        if p.name.endswith(".error.json") or p.name.endswith(".sha256"):
            continue
        if "_archive_" in str(p):
            continue
        try:
            art = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(art, dict):
            continue
        out[p.stem] = art
    return out


def sharpes(art: dict) -> list[float]:
    ps = art.get("path_summary") or {}
    if isinstance(ps.get("sharpe_by_path"), list):
        return [float(x) for x in ps["sharpe_by_path"]]
    paths = art.get("paths") or {}
    vals: list[float] = []
    if isinstance(paths, dict):
        for v in paths.values():
            if isinstance(v, dict) and "sharpe" in v:
                vals.append(float(v["sharpe"]))
    elif isinstance(paths, list):
        for v in paths:
            if isinstance(v, dict) and "sharpe" in v:
                vals.append(float(v["sharpe"]))
    return vals


def weight_hash(art: dict) -> str | None:
    paths = art.get("paths") or {}
    p0 = paths.get("0") if isinstance(paths, dict) else None
    if p0 is None and isinstance(paths, list) and paths:
        p0 = paths[0]
    w = (p0 or {}).get("weights") or art.get("weights") or []
    arr = np.asarray(w, dtype=np.float64)
    if arr.size == 0:
        return None
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _claim_stem(art: dict) -> str | None:
    if art.get("claim_label_stem"):
        return str(art["claim_label_stem"])
    prov = art.get("provenance") or {}
    if isinstance(prov, dict) and prov.get("claim_label_stem"):
        return str(prov["claim_label_stem"])
    return None


def audit(root: Path) -> dict:
    arts = load_finals(root)
    gates: dict = {
        "n_finals": len(arts),
        "all_147": len(arts) >= 147,
        "nan_path_cells": [],
        "all_zero_opt": [],
        "missing_behaviour": [],
        "bit_identical_clusters": [],
        "wrong_stem": [],
    }
    by_hash: dict[str, list[str]] = defaultdict(list)
    for stem, art in arts.items():
        sh = sharpes(art)
        if not sh or any(not math.isfinite(x) for x in sh):
            gates["nan_path_cells"].append(stem)
        if stem.startswith("opt_") and sh and all(abs(x) < 1e-15 for x in sh):
            gates["all_zero_opt"].append(stem)
        be = art.get("behaviour_export")
        if "happo" not in stem and be in (None, "unavailable"):
            gates["missing_behaviour"].append(stem)
        claim = _claim_stem(art)
        if stem.startswith("eq_") and claim not in (None, "stk_ret"):
            gates["wrong_stem"].append({"stem": stem, "claim_label_stem": claim})
        if stem.startswith("opt_") and claim not in (None, "dh_ret_lagdelta"):
            gates["wrong_stem"].append({"stem": stem, "claim_label_stem": claim})
        wh = weight_hash(art)
        if wh:
            by_hash[wh].append(stem)
    for h, stems in by_hash.items():
        uniq = sorted(set(stems))
        if len(uniq) < 2:
            continue
        # Flag clusters that mix different objectives or algos under one weight hash.
        objs = {s.split("_")[-1] for s in uniq}
        algos = set()
        for s in uniq:
            for tok in ("ppo", "cppo", "sac", "td3", "ddpg", "dqn", "mcpg", "rrl", "happo"):
                if f"_{tok}_" in f"_{s}_":
                    algos.add(tok)
        if len(objs) > 1 or len(algos) > 1:
            gates["bit_identical_clusters"].append(
                {"hash": h, "n": len(uniq), "stems": uniq[:12]}
            )
    gates["n_bit_clusters"] = len(gates["bit_identical_clusters"])
    gates["pass"] = bool(
        gates["all_147"]
        and not gates["nan_path_cells"]
        and not gates["all_zero_opt"]
        and not gates["missing_behaviour"]
        and not gates["wrong_stem"]
        and gates["n_bit_clusters"] == 0
    )
    return gates


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    root = Path(args[0] if args else "/tmp/pick_finals_v2")
    out = Path(args[1] if len(args) > 1 else "/tmp/pick_g4_gates.json")
    gates = audit(root)
    out.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))
    print(f"wrote {out} pass={gates['pass']}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
