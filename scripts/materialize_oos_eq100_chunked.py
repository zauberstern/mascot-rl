#!/usr/bin/env python3
"""Chunked liquid K=100 OOS rematerialize (fallback when full-panel OOM/dies).

Selects a liquid universe once, then materializes 10 x 10 secid batches into
temporary wide frames, merges by date, and writes ``constituent_signals`` +
``oos_universe`` into ``hyper_volanet_features_eq100``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.arctic_store import ArcticStateStore
from mascotrl.data.duckdb_engine import DuckDBFeatureEngine, OptionFilterConfig
from mascotrl.data.oos_panel import (
    FEATURE_STEMS,
    LABEL_STEM,
    SIGNALS_SYMBOL,
    UNIVERSE_SYMBOL,
    no_ffill_label_columns,
    pivot_long_marks_to_wide,
)
from mascotrl.data.pit_guards import membership_filter
from mascotrl.logging_utils import setup_logging


def _select_liquid_secids(
    eng: DuckDBFeatureEngine,
    *,
    n_assets: int,
    universe_start: str,
    universe_end: str,
) -> tuple[list[int], list[str], list[str]]:
    n_select = max(int(n_assets) * 2, int(n_assets) + 25)
    rows = eng.select_liquid_universe(n_select, universe_start, universe_end)
    members = eng.pit_membership_tickers(universe_end)
    rows, membership_meta = membership_filter(rows, members)
    rows = rows[: int(n_assets)]
    if len(rows) < n_assets:
        raise RuntimeError(
            f"Universe only returned {len(rows)} names; need {n_assets} "
            f"(membership={membership_meta})"
        )
    secids = [int(r["secid"]) for r in rows]
    tickers = [str(r["ticker"]) for r in rows]
    issuers = [str(r.get("issuer") or "") for r in rows]
    return secids, tickers, issuers


def _chunk_wide(
    eng: DuckDBFeatureEngine,
    secids: list[int],
    *,
    panel_start: str,
    panel_end: str,
    filters: OptionFilterConfig,
) -> pd.DataFrame:
    long_tbl = eng.compute_universe_daily_marks(
        secids, panel_start, panel_end, filters=filters
    )
    return pivot_long_marks_to_wide(long_tbl, secids)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-assets", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--universe-start", default="2018-01-01")
    p.add_argument("--universe-end", default="2021-12-31")
    p.add_argument("--panel-start", default="2019-01-01")
    p.add_argument("--panel-end", default="2024-12-31")
    p.add_argument("--lake-dir", required=True)
    p.add_argument("--arctic-db", required=True)
    p.add_argument("--arctic-library", default="hyper_volanet_features_eq100")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-memory", default="12GB")
    p.add_argument("--log-file", default=str(ROOT / "logs/materialize_oos_eq100_chunked.log"))
    args = p.parse_args(argv)

    log = setup_logging(log_file=args.log_file)
    log.info("chunked_eq100 start %s", vars(args))
    eng = DuckDBFeatureEngine(lake_base_dir=args.lake_dir)
    eng.con.execute(f"SET threads = {int(args.threads)};")
    eng.con.execute(f"SET max_memory = '{args.max_memory}';")

    secids, tickers, issuers = _select_liquid_secids(
        eng,
        n_assets=int(args.n_assets),
        universe_start=args.universe_start,
        universe_end=args.universe_end,
    )
    log.info("selected liquid n=%d", len(secids))
    print(f"secids_ready {len(secids)}", flush=True)

    filters = OptionFilterConfig()
    chunk = max(1, int(args.chunk_size))
    parts: list[pd.DataFrame] = []
    for i in range(0, len(secids), chunk):
        batch = secids[i : i + chunk]
        # Remap batch to local slots 0..len(batch)-1 then rename columns to
        # global slots so the merge keeps FEATURE_STEMS_{global_slot}.
        local = _chunk_wide(
            eng,
            batch,
            panel_start=args.panel_start,
            panel_end=args.panel_end,
            filters=filters,
        )
        rename = {}
        for local_i, global_i in enumerate(range(i, i + len(batch))):
            for stem in FEATURE_STEMS:
                src = f"{stem}_{local_i}"
                dst = f"{stem}_{global_i}"
                if src in local.columns:
                    rename[src] = dst
        local = local.rename(columns=rename)
        parts.append(local)
        log.info(
            "chunk %d-%d rows=%d cols=%d",
            i,
            i + len(batch) - 1,
            len(local),
            local.shape[1],
        )
        print(f"chunk_done {i}:{i + len(batch)}", flush=True)

    wide = parts[0]
    for part in parts[1:]:
        wide = wide.join(part, how="outer")
    wide = wide.sort_index()

    label_cols = [f"{LABEL_STEM}_{i}" for i in range(len(secids))]
    ret_cols = no_ffill_label_columns(wide, len(secids))
    coverage_cols = [c for c in label_cols if c in wide.columns] or ret_cols
    present = wide[coverage_cols].notna().sum(axis=1)
    need = max(1, (len(secids) + 1) // 2)
    wide = wide.loc[present >= need].copy()
    feat_cols = [c for c in wide.columns if c not in ret_cols]
    wide[feat_cols] = wide[feat_cols].ffill()
    if wide.empty:
        raise RuntimeError("Wide OOS panel empty after coverage filter")

    seen: dict[str, int] = {}
    display_names: list[str] = []
    for t, s in zip(tickers, secids):
        if t in seen:
            display_names.append(f"{t}_{s}")
        else:
            seen[t] = s
            display_names.append(t)

    meta = {
        "secids": secids,
        "tickers": list(tickers),
        "issuers": list(issuers),
        "display_names": display_names,
        "n_assets": len(secids),
        "universe_start": args.universe_start,
        "universe_end": args.universe_end,
        "panel_start": args.panel_start,
        "panel_end": args.panel_end,
        "feature_stems": list(FEATURE_STEMS),
        "label_stem": LABEL_STEM,
        "universe_mode": "liquid",
        "materialize_method": "chunked_secid_batches",
        "chunk_size": chunk,
        "knowledge_written_at": pd.Timestamp.utcnow().isoformat(),
    }
    store = ArcticStateStore(db_path=args.arctic_db, library_name=args.arctic_library)
    uni_df = pd.DataFrame(
        {"payload": [json.dumps(meta)]},
        index=pd.DatetimeIndex([pd.Timestamp(args.panel_end)], name="date"),
    )
    if UNIVERSE_SYMBOL in store.lib.list_symbols():
        store.lib.update(UNIVERSE_SYMBOL, uni_df, metadata=meta)
    else:
        store.lib.write(UNIVERSE_SYMBOL, uni_df, metadata=meta)

    out = wide.reset_index()
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    table = pa.Table.from_pandas(out, preserve_index=False)
    store.persist_features(SIGNALS_SYMBOL, table, metadata=meta)

    slim = {
        "secids": secids[:5],
        "n_secids": len(secids),
        "n_rows": int(len(wide)),
        "n_cols": int(wide.shape[1]),
        "start": str(wide.index.min().date()),
        "end": str(wide.index.max().date()),
        "arctic_symbols": store.list_available_features(),
        "method": "chunked_secid_batches",
    }
    print("DONE " + json.dumps(slim, default=str), flush=True)
    log.info("DONE %s", json.dumps(slim, default=str))
    out_path = ROOT / "logs/arctic_rematerialize_eq100.json"
    out_path.write_text(json.dumps({"status": "ok", **slim}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
