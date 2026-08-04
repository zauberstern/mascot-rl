#!/usr/bin/env python3
"""Materialize OptionMetrics ATM marks → Arctic ``constituent_signals``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.oos_panel import materialize_oos_panel
from src.logging_utils import setup_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-assets", type=int, default=10)
    p.add_argument("--universe-start", default="2018-01-01")
    p.add_argument("--universe-end", default="2021-12-31")
    p.add_argument("--panel-start", default="2022-01-01")
    p.add_argument("--panel-end", default="2024-12-31")
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--arctic-db", default=None)
    p.add_argument("--arctic-library", default="hyper_volanet_features")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-memory", default="12GB")
    p.add_argument("--universe-mode", default="iv_hypergraph")
    p.add_argument("--liquidity-pool", type=int, default=200)
    p.add_argument("--iv-corr-threshold", type=float, default=0.35)
    p.add_argument("--tail-threshold", type=float, default=0.90)
    p.add_argument("--selection-metric", default="copula_tail")
    p.add_argument(
        "--attrition-out",
        default=str(ROOT / "logs" / "artifacts" / "filter_attrition.json"),
        help="Write filter attrition JSON for the data appendix",
    )
    p.add_argument("--log-file", default=str(ROOT / "logs" / "materialize_oos.log"))
    args = p.parse_args()

    log = setup_logging(log_file=args.log_file)
    log.info("materialize_oos_arctic start %s", vars(args))
    info = materialize_oos_panel(
        n_assets=args.n_assets,
        universe_start=args.universe_start,
        universe_end=args.universe_end,
        panel_start=args.panel_start,
        panel_end=args.panel_end,
        lake_base_dir=args.lake_dir,
        arctic_db_path=args.arctic_db,
        arctic_library=args.arctic_library,
        duckdb_threads=args.threads,
        max_memory=args.max_memory,
        universe_mode=args.universe_mode,
        liquidity_pool=args.liquidity_pool,
        iv_corr_threshold=args.iv_corr_threshold,
        universe_tail_threshold=args.tail_threshold,
        universe_selection_metric=args.selection_metric,
        attrition_out_path=args.attrition_out,
    )
    log.info("DONE %s", json.dumps({k: v for k, v in info.items() if k != "meta"}, default=str))
    print(json.dumps({k: v for k, v in info.items() if k != "meta"}, indent=2, default=str))


if __name__ == "__main__":
    main()
