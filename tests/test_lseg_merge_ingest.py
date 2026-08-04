"""LSEG merge ingest: IPA dataset copy, ric_map p4, valuation guard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.lseg_overlay import (
    RIC_MAP_P4_COLS,
    copy_lseg_dataset_dir,
    ingest_lseg_overlays,
    refuse_valuation_path,
    validate_ipa_surface_raw,
)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_copy_lseg_dataset_dir_preserves_hive_and_skips_identical(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    shard = src / "ric=AAPL.O" / "date=2023-01-31" / "data.parquet"
    body = b"PAR1" + b"x" * 32
    shard.parent.mkdir(parents=True)
    shard.write_bytes(body)
    first = copy_lseg_dataset_dir(src_root=src, dest_root=dest)
    assert first["copied"] == 1
    assert (dest / "ric=AAPL.O" / "date=2023-01-31" / "data.parquet").is_file()
    second = copy_lseg_dataset_dir(src_root=src, dest_root=dest)
    assert second["skipped"] == 1
    assert second["copied"] == 0


def test_validate_ipa_surface_raw_flags_empty_payload(tmp_path: Path) -> None:
    root = tmp_path / "ipa"
    empty = root / "ric=A" / "date=2023-01-31" / "data.parquet"
    _write_parquet(
        empty,
        pd.DataFrame({"ric": ["A"], "calculation_date": ["2023-01-31"], "raw": ["{}"], "asof_ts": ["t"]}),
    )
    err = root / "ric=B" / "date=2023-12-31" / "data.parquet"
    _write_parquet(
        err,
        pd.DataFrame(
            {"ric": ["B"], "calculation_date": ["2023-12-31"], "raw": ['{"error":"x"}'], "asof_ts": ["t"]}
        ),
    )
    stats = validate_ipa_surface_raw(root)
    assert stats["ipa_shards"] == 2
    assert stats["ipa_payload_empty"] is True
    assert stats["ipa_error_shards"] == 1


def test_refuse_valuation_path() -> None:
    with pytest.raises(ValueError, match="valuation"):
        refuse_valuation_path(Path("/tmp/lseg/valuation/lake/x.parquet"))


def test_ingest_lseg_ric_map_p4_and_ipa(tmp_path: Path) -> None:
    lseg = tmp_path / "lseg"
    lake = tmp_path / "lake" / "macro"
    lake.mkdir(parents=True)
    sec = pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "close": [10.0],
            "return": [0.01],
            "volume": [100.0],
            "cfadj": [1.0],
        }
    )
    _write_parquet(lake / "sp500_sec.parquet", sec)
    ohlc = pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "BID": [9.9],
            "ASK": [10.1],
            "TRDPRC_1": [10.0],
            "ric": ["A.K"],
            "asof_ts": ["t"],
        }
    )
    spread = pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "quoted_spread": [0.01],
            "ric": ["A.K"],
            "asof_ts": ["t"],
        }
    )
    rates = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "dtb3": [0.01], "sofr": [0.02], "effr": [0.03]})
    lseg_rates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "ric": ["US2YT=RR"],
            "YLDTOMAT": [0.04],
            "FIXING_1": [0.04],
            "asof_ts": ["t"],
        }
    )
    macro = lseg / "macro"
    _write_parquet(macro / "lseg_eq_ohlc_corax.parquet", ohlc)
    _write_parquet(macro / "lseg_eq_spread.parquet", spread)
    _write_parquet(macro / "lseg_index_vol_rates.parquet", lseg_rates)
    _write_parquet(lake / "interest_rate.parquet", rates)
    for name in ("lseg_eq_ohlc_unadj", "lseg_eq_size", "lseg_spx_pit", "lseg_gics"):
        _write_parquet(macro / f"{name}.parquet", ohlc)
    ric = pd.DataFrame({"secid": [1], "ric": ["A.K"]})
    for col in RIC_MAP_P4_COLS:
        ric[col] = ["x"]
    _write_parquet(lseg / "mapping" / "ric_map.parquet", ric)
    ipa = macro / "lseg_ipa_surface" / "ric=A.K" / "date=2023-01-31" / "data.parquet"
    _write_parquet(
        ipa,
        pd.DataFrame({"ric": ["A.K"], "calculation_date": ["2023-01-31"], "raw": ["{}"], "asof_ts": ["t"]}),
    )
    info = ingest_lseg_overlays(lseg_raw=lseg, lake_macro=lake)
    assert info["ric_map_p4_present"] is True
    assert info["ipa_shards"] == 1
    assert info["ipa_payload_empty"] is True
    assert (lake / "lseg_ipa_surface" / "ric=A.K" / "date=2023-01-31" / "data.parquet").is_file()
    assert json.loads(json.dumps(info))["ok"] is True
