#!/usr/bin/env python3
"""Load the pseudo constituent panel and print CPCV fold geometry."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mascotrl.eval.cpcv import CPCVConfig, assign_paths, build_cpcv_folds

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_DIR = ROOT / "data" / "pseudo"
META_PATH = PSEUDO_DIR / "constituent_signals_pseudo.json"
PANEL_PATH = PSEUDO_DIR / "constituent_signals_pseudo.parquet"


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    n_days = int(meta["n_days"])
    panel = pd.read_parquet(PANEL_PATH)
    if panel.index.name != "date":
        panel = panel.set_index("date")
    dates = pd.to_datetime(panel.index)
    assert len(dates) == n_days, (len(dates), n_days)

    cfg = CPCVConfig()
    folds = build_cpcv_folds(dates, cfg)
    paths = assign_paths(cfg)
    print(
        f"Loaded panel {panel.shape[0]} days x {panel.shape[1]} columns; "
        f"CPCV folds={len(folds)}, paths={len(paths)} "
        f"(purge={cfg.purge_days}d, embargo={cfg.embargo_days}d)"
    )


if __name__ == "__main__":
    main()
