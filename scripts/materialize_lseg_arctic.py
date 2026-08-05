#!/usr/bin/env python3
"""Persist LSEG parallel panels to Arctic without date-dedup collapse."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mascotrl.data.arctic_store import ArcticStateStore  # noqa: E402
from mascotrl.data.paths import ARCTIC_ROOT, LAKE_ROOT  # noqa: E402
from mascotrl.logging_utils import setup_logging  # noqa: E402

PANEL_FILES = (
    ("lseg_eq_ohlc_unadj.parquet", "lseg_eq_ohlc_unadj"),
    ("lseg_eq_size.parquet", "lseg_eq_size"),
    ("lseg_spx_pit.parquet", "lseg_spx_pit"),
)


def _normalize_pit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pit_date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"pit_date": "date"})
    if "Constituent RIC" in out.columns and "secid" not in out.columns:
        out["secid"] = out["Constituent RIC"].astype("category").cat.codes
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake", default=str(LAKE_ROOT))
    p.add_argument("--arctic", default=str(ARCTIC_ROOT))
    args = p.parse_args()
    log = setup_logging(log_file=str(ROOT / "logs" / "materialize_lseg_arctic.log"))
    store = ArcticStateStore(db_path=args.arctic)
    macro = Path(args.lake) / "macro"
    for fname, symbol in PANEL_FILES:
        path = macro / fname
        if not path.is_file():
            log.warning("skip missing %s", path)
            continue
        df = pd.read_parquet(path)
        df = _normalize_pit(df)
        if "secid" not in df.columns or "date" not in df.columns:
            log.warning("skip %s: need date,secid", path)
            continue
        store.persist_panel(symbol, df)
        log.info("persisted %s rows=%d", symbol, len(df))


if __name__ == "__main__":
    main()
