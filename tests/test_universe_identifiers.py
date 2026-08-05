"""Gold count for the ever-in OM S&P 500 optionable universe CSV."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mascotrl.data.paths import MOUNT_ROOT, TIER_B, UNIVERSE_IDENTIFIERS

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "universe_identifiers_gold.json"
GOLD = json.loads(FIXTURE.read_text(encoding="utf-8"))
REQUIRED_COLS = tuple(GOLD["required_cols"])
FORBIDDEN_COLS = tuple(GOLD["forbidden_cols"])
GOLD_N_SECID = int(GOLD["gold_n_secid"])


def test_universe_gold_fixture_always_on() -> None:
    assert GOLD_N_SECID == 511
    assert REQUIRED_COLS == ("secid", "cusip", "ticker", "first_date", "last_date", "n_days")
    assert "ric" in FORBIDDEN_COLS
    assert "permid" in FORBIDDEN_COLS
    assert "ric" not in REQUIRED_COLS


@pytest.mark.integration
def test_universe_identifiers_matches_om_sec_gold_count() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    src = TIER_B["sp500_sec"]
    out = UNIVERSE_IDENTIFIERS
    if not MOUNT_ROOT.exists() and (not src.is_file() or not out.is_file()):
        pytest.skip("USB unmounted and identifier gold not on this host")
    if not src.is_file():
        pytest.fail(f"OM sec CSV missing while USB is mounted: {src}")
    if not out.is_file():
        pytest.fail(f"universe identifiers CSV missing while USB is mounted: {out}")

    src_df = pd.read_csv(src, usecols=["secid"], dtype={"secid": "int64"})
    assert int(src_df["secid"].nunique()) == GOLD_N_SECID

    ids = pd.read_csv(out)
    for col in REQUIRED_COLS:
        assert col in ids.columns, f"missing column {col}"
    for col in FORBIDDEN_COLS:
        assert col not in ids.columns
    assert len(ids) == GOLD_N_SECID
    assert int(ids["secid"].nunique()) == GOLD_N_SECID
    assert ids["secid"].is_monotonic_increasing
    src_set = set(src_df["secid"].unique().tolist())
    id_set = set(ids["secid"].astype("int64").tolist())
    assert src_set == id_set
