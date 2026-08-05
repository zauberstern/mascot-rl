"""Join LSEG P0/P2 onto existing lake tables without overwriting OM/CRSP SoT."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from mascotrl.data.file_fingerprints import file_fingerprints, fingerprints_match

P3_REFUSED = (
    "lseg_worldscope.parquet",
    "lseg_ibes.parquet",
    "lseg_short_interest.parquet",
)

RIC_MAP_P4_COLS = (
    "TR.BusinessSummary_p4",
    "TR.CommonName_p4",
    "TR.ExchangeName_p4",
    "TR.GICSIndustry_p4",
    "TR.GICSSector_p4",
    "TR.GICSSubIndustry_p4",
    "TR.InstrumentDescription_p4",
    "TR.OrganizationID_p4",
)

OM_SO_T = ("close", "return", "volume", "cfadj", "shrout", "ticker", "secid", "date")
OHLC_RENAME = {
    "BID": "lseg_bid",
    "ASK": "lseg_ask",
    "TRDPRC_1": "lseg_trdprc",
    "OPEN_PRC": "lseg_open",
    "HIGH_1": "lseg_high",
    "LOW_1": "lseg_low",
    "TRNOVR_UNS": "lseg_trnvr",
    "NUM_MOVES": "lseg_num_moves",
    "ACVOL_UNS": "lseg_acvol",
    "VWAP": "lseg_vwap",
    "VWAP_VOL": "lseg_vwap_vol",
    "BLKCOUNT": "lseg_blkcount",
    "BLKVOLUM": "lseg_blkvolum",
    "TRD_STATUS": "lseg_trd_status",
    "ric": "lseg_ric",
    "asof_ts": "lseg_asof_ts",
}


def refuse_valuation_path(path: Path) -> None:
    low = str(path).replace("\\", "/").lower()
    if "/valuation/" in low or "lseg_valuation" in low:
        raise ValueError(f"valuation refused: {path}")


def refuse_p3_path(path: Path) -> None:
    refuse_valuation_path(path)
    low = str(path).replace("\\", "/")
    if "/lseg_p3/" in low or any(name in path.name for name in P3_REFUSED):
        raise ValueError(f"P3 refused: {path}")
    for token in ("worldscope", "lseg_ibes", "short_interest"):
        if token in path.name.lower():
            raise ValueError(f"P3 refused: {path}")


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os_replace = dest
    tmp.replace(os_replace)


def overlay_sp500_sec(
    sec_path: Path,
    *,
    ohlc_path: Path,
    spread_path: Path,
) -> pd.DataFrame:
    refuse_p3_path(ohlc_path)
    refuse_p3_path(spread_path)
    sec = pd.read_parquet(sec_path)
    if "date" not in sec.columns or "secid" not in sec.columns:
        raise KeyError("sp500_sec requires date, secid")
    sec["date"] = pd.to_datetime(sec["date"])
    sec["secid"] = pd.to_numeric(sec["secid"], errors="coerce")
    ohlc = pd.read_parquet(ohlc_path)
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    ohlc["secid"] = pd.to_numeric(ohlc["secid"], errors="coerce")
    keep = ["secid", "date"] + [c for c in OHLC_RENAME if c in ohlc.columns]
    ohlc = ohlc[keep].rename(columns={k: v for k, v in OHLC_RENAME.items() if k in ohlc.columns})
    drop_existing = [c for c in ohlc.columns if c.startswith("lseg_") and c in sec.columns]
    sec = sec.drop(columns=drop_existing, errors="ignore")
    merged = sec.merge(ohlc, on=["secid", "date"], how="left")
    spread = pd.read_parquet(spread_path)
    spread["date"] = pd.to_datetime(spread["date"])
    spread["secid"] = pd.to_numeric(spread["secid"], errors="coerce")
    sp = spread[["secid", "date", "quoted_spread"]].rename(
        columns={"quoted_spread": "lseg_quoted_spread"}
    )
    merged = merged.drop(columns=["lseg_quoted_spread"], errors="ignore")
    merged = merged.merge(sp, on=["secid", "date"], how="left")
    _atomic_write_parquet(merged, Path(sec_path))
    return merged


def overlay_interest_rate(rates_path: Path, *, lseg_path: Path) -> pd.DataFrame:
    refuse_p3_path(lseg_path)
    rates = pd.read_parquet(rates_path)
    rates["date"] = pd.to_datetime(rates["date"])
    lseg = pd.read_parquet(lseg_path)
    lseg["date"] = pd.to_datetime(lseg["date"])
    yld = lseg["YLDTOMAT"] if "YLDTOMAT" in lseg.columns else pd.Series(index=lseg.index, dtype=float)
    fix = lseg["FIXING_1"] if "FIXING_1" in lseg.columns else pd.Series(index=lseg.index, dtype=float)
    lseg = lseg.assign(_yld=pd.to_numeric(yld, errors="coerce"), _fix=pd.to_numeric(fix, errors="coerce"))
    rows = []
    for ric, col, prefer_fix in (
        ("US2YT=RR", "lseg_us2y", False),
        ("US10YT=RR", "lseg_us10y", False),
        ("USDSOFR=", "lseg_sofr", True),
    ):
        sub = lseg.loc[lseg["ric"] == ric, ["date", "_yld", "_fix", "asof_ts"]].copy()
        sub[col] = sub["_fix"] if prefer_fix else sub["_yld"]
        if prefer_fix:
            sub[col] = sub[col].fillna(sub["_yld"])
        rows.append(sub[["date", col]].drop_duplicates("date"))
    wide = rows[0]
    for extra in rows[1:]:
        wide = wide.merge(extra, on="date", how="outer")
    asof = lseg.groupby("date", as_index=False)["asof_ts"].first().rename(
        columns={"asof_ts": "lseg_rates_asof_ts"}
    )
    wide = wide.merge(asof, on="date", how="left")
    drop_lseg = [c for c in wide.columns if c.startswith("lseg_") and c in rates.columns]
    rates = rates.drop(columns=drop_lseg, errors="ignore")
    out = rates.merge(wide, on="date", how="left")
    if "OAS_BID" in out.columns or "ZSPREAD" in out.columns or "INT_CDS" in out.columns:
        out = out.drop(columns=["OAS_BID", "ZSPREAD", "INT_CDS"], errors="ignore")
    _atomic_write_parquet(out, Path(rates_path))
    return out


def copy_parallel_lseg(*, src: Path, dest: Path) -> None:
    refuse_p3_path(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_lseg_dataset_dir(*, src_root: Path, dest_root: Path) -> dict[str, int]:
    """Mirror a hive dataset tree into the lake with incremental fingerprint skip."""
    refuse_p3_path(src_root)
    refuse_valuation_path(dest_root)
    if not src_root.is_dir():
        return {"copied": 0, "skipped": 0}
    copied = 0
    skipped = 0
    for src_file in sorted(src_root.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_root)
        dest_file = dest_root / rel
        if dest_file.is_file():
            if fingerprints_match(file_fingerprints(src_file), file_fingerprints(dest_file)):
                skipped += 1
                continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied += 1
    return {"copied": copied, "skipped": skipped}


def _parse_raw_cell(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {"_parse_error": True}
        return obj if isinstance(obj, dict) else {"_non_object": True}
    return {}


def validate_ipa_surface_raw(ipa_root: Path) -> dict[str, Any]:
    """Scan IPA shards; flag empty payloads and error JSON."""
    if not ipa_root.is_dir():
        return {"ipa_shards": 0, "ipa_payload_empty": True, "ipa_error_shards": 0}
    shards = sorted(ipa_root.rglob("data.parquet"))
    empty = 0
    errors = 0
    nonempty = 0
    for shard in shards:
        df = pd.read_parquet(shard)
        if df.empty:
            empty += 1
            continue
        obj = _parse_raw_cell(df.iloc[0].get("raw"))
        if not obj:
            empty += 1
        elif "_parse_error" in obj or "_non_object" in obj or "error" in obj or "Error" in obj:
            errors += 1
        else:
            nonempty += 1
    total = len(shards)
    rate = (nonempty / total) if total else 0.0
    return {
        "ipa_shards": total,
        "ipa_empty_shards": empty,
        "ipa_nonempty_shards": nonempty,
        "ipa_error_shards": errors,
        "ipa_payload_empty": rate < 0.01,
    }


def validate_ric_map_p4(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ric_map_rows": 0, "ric_map_p4_present": False}
    df = pd.read_parquet(path)
    present = [c for c in RIC_MAP_P4_COLS if c in df.columns]
    non_null = {c: int(df[c].notna().sum()) for c in present}
    return {
        "ric_map_rows": len(df),
        "ric_map_p4_present": len(present) == len(RIC_MAP_P4_COLS),
        "ric_map_p4_cols": present,
        "ric_map_p4_non_null": non_null,
    }


def ingest_lseg_overlays(*, lseg_raw: Path, lake_macro: Path) -> dict[str, Any]:
    refuse_p3_path(lseg_raw)
    refuse_valuation_path(lseg_raw)
    macro = Path(lseg_raw) / "macro"
    mapping = Path(lseg_raw) / "mapping"
    overlay_sp500_sec(
        lake_macro / "sp500_sec.parquet",
        ohlc_path=macro / "lseg_eq_ohlc_corax.parquet",
        spread_path=macro / "lseg_eq_spread.parquet",
    )
    overlay_interest_rate(
        lake_macro / "interest_rate.parquet",
        lseg_path=macro / "lseg_index_vol_rates.parquet",
    )
    copy_parallel_lseg(src=macro / "lseg_eq_ohlc_unadj.parquet", dest=lake_macro / "lseg_eq_ohlc_unadj.parquet")
    # Parallel SoT copies for Yang-Zhang overnight leg and index vol/rates
    # (also used as overlay sources above; keep stand-alone lake tables).
    if (macro / "lseg_eq_ohlc_corax.parquet").is_file():
        copy_parallel_lseg(
            src=macro / "lseg_eq_ohlc_corax.parquet",
            dest=lake_macro / "lseg_eq_ohlc_corax.parquet",
        )
    if (macro / "lseg_index_vol_rates.parquet").is_file():
        copy_parallel_lseg(
            src=macro / "lseg_index_vol_rates.parquet",
            dest=lake_macro / "lseg_index_vol_rates.parquet",
        )
    copy_parallel_lseg(src=macro / "lseg_eq_size.parquet", dest=lake_macro / "lseg_eq_size.parquet")
    copy_parallel_lseg(src=macro / "lseg_spx_pit.parquet", dest=lake_macro / "lseg_spx_pit.parquet")
    copy_parallel_lseg(src=macro / "lseg_gics.parquet", dest=lake_macro / "lseg_gics.parquet")
    ric_stats: dict[str, Any] = {"ric_map_rows": 0, "ric_map_p4_present": False}
    if (mapping / "ric_map.parquet").is_file():
        ric_dest = lake_macro / "lseg_ric_map.parquet"
        copy_parallel_lseg(src=mapping / "ric_map.parquet", dest=ric_dest)
        ric_stats = validate_ric_map_p4(ric_dest)
    ipa_src = macro / "lseg_ipa_surface"
    ipa_copy = {"copied": 0, "skipped": 0}
    ipa_stats: dict[str, Any] = {"ipa_shards": 0, "ipa_payload_empty": True, "ipa_error_shards": 0}
    if ipa_src.is_dir():
        ipa_copy = copy_lseg_dataset_dir(src_root=ipa_src, dest_root=lake_macro / "lseg_ipa_surface")
        ipa_stats = validate_ipa_surface_raw(lake_macro / "lseg_ipa_surface")
    lake_ipa = lake_macro / "lseg_ipa_surface"
    soft_seal = {
        "lseg_ric_map_exists": (lake_macro / "lseg_ric_map.parquet").is_file(),
        "lseg_ipa_surface_shards": len(list(lake_ipa.rglob("data.parquet"))) if lake_ipa.is_dir() else 0,
    }
    return {
        "ok": True,
        "ipa_copy": ipa_copy,
        **ipa_stats,
        **ric_stats,
        "soft_seal": soft_seal,
    }
