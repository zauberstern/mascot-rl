"""Unit tests for IBES + LSEG P3 disclosure ingest scripts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_ingest_ibes_noncontract_writes_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import ingest_ibes_noncontract as mod

    src = tmp_path / "wrds_ibes_financial_ratios_sp500.csv"
    src.write_text("ticker,date,ratio\nAAPL,2003-01-02,1.2\nMSFT,2003-01-03,0.9\n", encoding="utf-8")
    lake = tmp_path / "lake"
    lake.mkdir()
    monkeypatch.setattr(mod, "assert_raw_mounted", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(mod, "assert_lake_mounted", lambda p=None: Path(p or lake))
    info = mod.ingest_ibes(src=src, lake=lake)
    dest = lake / "macro" / "ibes_financial_ratios.parquet"
    assert dest.is_file()
    assert info["n_rows"] == 2
    assert info["feature_admitted"] is False
    assert info["non_contract"] is True
    prov = json.loads((lake / "macro" / "ibes_financial_ratios_provenance.json").read_text())
    assert prov["role"] == "disclosure-only"


def test_ingest_lseg_p3_disclosure_copies_with_provenance(tmp_path: Path) -> None:
    from scripts import ingest_lseg_p3_disclosure as mod

    src_dir = tmp_path / "lseg" / "macro" / "p3"
    src_dir.mkdir(parents=True)
    for name in mod.P3_FILES:
        pd.DataFrame({"a": [1, 2]}).to_parquet(src_dir / name, index=False)
    lake = tmp_path / "lake"
    lake.mkdir()
    info = mod.ingest_lseg_p3_disclosure(src_dir=src_dir, lake=lake)
    dest_dir = lake / "macro" / "p3"
    assert (dest_dir / "_provenance.json").is_file()
    assert info["feature_admitted"] is False
    assert info["overlay_refused"] is True
    for name in mod.P3_FILES:
        assert (dest_dir / name).is_file()
    prov = json.loads((dest_dir / "_provenance.json").read_text())
    assert "not feature-admitted" in prov["note"]


def test_build_data_lake_reconvert_flag_help() -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output(
        [sys.executable, str(root / "scripts" / "build_data_lake.py"), "--help"],
        text=True,
    )
    assert "--reconvert-mismatch-years" in out
