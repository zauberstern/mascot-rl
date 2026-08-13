"""CRUCIBLE schedule fingerprints and persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from mascotrl.data.crucible_types import CrucibleSpec

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
