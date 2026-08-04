#!/usr/bin/env python3
"""Vendor fioracle macro CSVs into the mascotrl lake as PIT-ready parquet.

Does not import fioracle code. Copies eight series with provenance (sha256, lag).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SRC = Path("/home/kartik/Desktop/fioracle/fioracle")

# HY OAS file is named TOTAL_RETURN and the value column is "Total Return Index",
# but values are option-adjusted spread levels (e.g. ~3.29), not a total-return index.
SERIES_SPECS: dict[str, dict[str, Any]] = {
    "vix": {
        "source_file": "macro_universe/VIX.csv",
        "value_col": "Synthesized_VIX",
        "date_col": "Date",
        "lag_days": 1,
        "licence_note": "public (CBOE VIX; synthetic pre-1990 truncated)",
        "redistributable": True,
        "truncate_before": "1990-01-01",
    },
    "gpri": {
        "source_file": "macro_universe/GPRI.csv",
        "value_col": "GPRI",
        "date_col": "Date",
        "lag_days": 5,
        "licence_note": "published academic geopolitical risk index",
        "redistributable": True,
    },
    "hy_oas": {
        "source_file": "macro_universe/US_HY_CORP_OAS_TOTAL_RETURN.csv",
        "value_col": "Total Return Index",  # misnamed; values are OAS spread levels
        "date_col": "Date",
        "lag_days": 1,
        "licence_note": "vendor extract inside fioracle; proprietary",
        "redistributable": False,
    },
    "inflation": {
        "source_file": "macro_universe/US_INFLATION_RATE.csv",
        "value_col": "Inflation",
        "date_col": "Date",
        "lag_days": 15,
        "licence_note": "public macro series",
        "redistributable": True,
    },
    "unemployment": {
        "source_file": "macro_universe/US_UNEMPLOYMENT_RATE.csv",
        "value_col": "Unemployment",
        "date_col": "Date",
        "lag_days": 7,
        "licence_note": "public macro series",
        "redistributable": True,
    },
    "epu": {
        # Not macro_universe/EPU.csv (backfilled to 1945); real daily index from 1985.
        "source_file": "dataset/EPU/US_EPU_Daily.csv",
        "value_col": "EPU",
        "date_col": None,
        "lag_days": 7,
        "licence_note": "policyuncertainty.com Economic Policy Uncertainty",
        "redistributable": True,
        "epu_daily": True,
    },
    "yield_2y": {
        "source_file": "ancillary/US 2Y Yield (1976-present).csv",
        "value_col": "DGS2",
        "date_col": "observation_date",
        "lag_days": 0,
        "licence_note": "public FRED DGS2",
        "redistributable": True,
    },
    "term_spread": {
        "source_file": "ancillary/US 10Y2Y (1976-present).csv",
        "value_col": "T10Y2Y",
        "date_col": "observation_date",
        "lag_days": 0,
        "licence_note": "public FRED T10Y2Y",
        "redistributable": True,
    },
}

PARQUET_COMPRESSION = "zstd"


def default_dest() -> Path:
    env = os.environ.get("MASCOTRL_LAKE_DIR")
    if env:
        return Path(env) / "macro" / "fioracle"
    return Path("lake") / "macro" / "fioracle"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_source_csv(src_root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    path = src_root / spec["source_file"]
    if not path.is_file():
        raise FileNotFoundError(f"missing fioracle source: {path}")
    raw = pd.read_csv(path)
    if spec.get("epu_daily"):
        event = pd.to_datetime(
            dict(year=raw["year"], month=raw["month"], day=raw["day"])
        )
        value = pd.to_numeric(raw[spec["value_col"]], errors="coerce")
    else:
        event = pd.to_datetime(raw[spec["date_col"]])
        value = pd.to_numeric(raw[spec["value_col"]], errors="coerce")
    df = pd.DataFrame({"event_date": event, "value": value})
    df = df.dropna(subset=["event_date"]).sort_values("event_date")
    truncate_before = spec.get("truncate_before")
    if truncate_before:
        cut = pd.Timestamp(truncate_before)
        # Drop synthetic pre-cut rows (never emit finite synthetic VIX).
        df = df.loc[df["event_date"] >= cut].copy()
    df = df.drop_duplicates(subset=["event_date"], keep="last")
    return df.reset_index(drop=True)


def _series_frame(
    series_id: str,
    spec: dict[str, Any],
    src_root: Path,
) -> tuple[pd.DataFrame, str]:
    src_path = src_root / spec["source_file"]
    digest = _sha256_file(src_path)
    raw = _read_source_csv(src_root, spec)
    lag = int(spec["lag_days"])
    event = pd.to_datetime(raw["event_date"])
    available = event + pd.to_timedelta(lag, unit="D")
    out = pd.DataFrame(
        {
            "event_date": event.dt.date,
            "available_date": available.dt.date,
            "value": raw["value"].astype(np.float64),
            "series_id": series_id,
            "source_file": spec["source_file"],
            "source_sha256": digest,
            "lag_days": np.int32(lag),
        }
    )
    return out, digest


def _write_parquet_deterministic(df: pd.DataFrame, path: Path) -> None:
    """Fixed compression / no pandas index metadata for byte-identical reruns."""
    schema = pa.schema(
        [
            ("event_date", pa.date32()),
            ("available_date", pa.date32()),
            ("value", pa.float64()),
            ("series_id", pa.string()),
            ("source_file", pa.string()),
            ("source_sha256", pa.string()),
            ("lag_days", pa.int32()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    # Strip pandas metadata that can differ across versions
    table = table.replace_schema_metadata({})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def ingest_fioracle_macro(*, src: Path, dest: Path) -> dict[str, Any]:
    src = Path(src)
    dest = Path(dest)
    if not src.is_dir():
        raise FileNotFoundError(
            f"fioracle src missing: {src}; pass --src or synthesize fixtures for tests"
        )
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_root": str(src.resolve()),
        "dest": str(dest.resolve()),
        "parquet_compression": PARQUET_COMPRESSION,
        "series": {},
    }
    for series_id, spec in SERIES_SPECS.items():
        frame, digest = _series_frame(series_id, spec, src)
        out_path = dest / f"{series_id}.parquet"
        _write_parquet_deterministic(frame, out_path)
        event = pd.to_datetime(frame["event_date"])
        manifest["series"][series_id] = {
            "source_file": spec["source_file"],
            "source_sha256": digest,
            "lag_days": int(spec["lag_days"]),
            "row_count": int(len(frame)),
            "span_start": str(event.min().date()) if len(frame) else None,
            "span_end": str(event.max().date()) if len(frame) else None,
            "licence_note": spec["licence_note"],
            "redistributable": bool(spec["redistributable"]),
        }
    man_path = dest / "_manifest.json"
    # Stable JSON for idempotent manifest bytes
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    man_path.write_text(payload, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args(argv)
    dest = args.dest if args.dest is not None else default_dest()
    if not args.src.is_dir():
        print(f"ERROR: fioracle src does not exist: {args.src}", flush=True)
        print("Skipping live ingest; tests use tmp fixtures.", flush=True)
        return 2
    man = ingest_fioracle_macro(src=args.src, dest=dest)
    print(f"ingested {len(man['series'])} series -> {dest}", flush=True)
    print(f"manifest: {dest / '_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
