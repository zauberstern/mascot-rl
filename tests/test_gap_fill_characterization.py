"""Gap-fill characterization tests for post-cull modules (Phase 4).

Expected values are taken from running the current implementation, not from
textbook ideals. See REFACTOR_PLAN characterize-then-prune protocol.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.file_fingerprints import (
    file_fingerprints,
    fingerprints_match,
    header_sha256,
)
from mascotrl.data.pit_guards import selection_pit_status
from mascotrl.data.slot_mask import (
    _lookup_eligible,
    build_members_by_date_from_intervals,
)
from mascotrl.eval.cpcv import (
    CPCVConfig,
    build_cpcv_folds,
    residual_equity_cpcv_config,
    stamp_reselect_purge_meta,
)
from mascotrl.eval.differential_sharpe import DifferentialSharpe
from mascotrl.spectrum import cell_schema


@pytest.fixture
def tiny_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"hello\n")
    return p


def test_file_fingerprints_tiny_file_characterization(tiny_file: Path) -> None:
    fp = file_fingerprints(tiny_file)
    assert fp == {
        "size": 6,
        "header_sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
        "head_1mib_sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
        "mid_1mib_sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
        "tail_1mib_sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
    }
    assert header_sha256(tiny_file) == fp["header_sha256"]


def test_file_fingerprints_empty_file_characterization(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    fp = file_fingerprints(p)
    empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert fp["size"] == 0
    assert fp["header_sha256"] == empty_hash
    assert fp["head_1mib_sha256"] == empty_hash


def test_fingerprints_match_legacy_rows_without_mid(tiny_file: Path) -> None:
    live = file_fingerprints(tiny_file)
    stored = {
        "size": live["size"],
        "header_sha256": live["header_sha256"],
        "head_1mib_sha256": live["head_1mib_sha256"],
        "tail_1mib_sha256": live["tail_1mib_sha256"],
    }
    assert fingerprints_match(live, stored) is True
    assert fingerprints_match(live, None) is False
    assert fingerprints_match(live, {**stored, "size": stored["size"] + 1}) is False


def test_fingerprints_match_rejects_stale_mid(tiny_file: Path) -> None:
    live = file_fingerprints(tiny_file)
    stored = dict(live, mid_1mib_sha256="deadbeef")
    assert fingerprints_match(live, stored) is False


def test_cpcv_extra_purge_expands_train_exclusion() -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=60))
    cfg = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=2, embargo_days=2)
    folds = build_cpcv_folds(
        dates, cfg, extra_purge_indices=[30], extra_purge_radius=1
    )
    fold0 = folds[0]
    assert fold0.n_purged_days == 3
    assert fold0.n_embargoed_days == 2
    assert fold0.n_train_days == 35


def test_stamp_reselect_purge_meta_characterization() -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=60))
    mask = np.zeros(60, dtype=bool)
    mask[30] = True
    meta = stamp_reselect_purge_meta(dates, mask, purge_radius=1)
    assert meta == {
        "n_reselect_days": 1,
        "n_purged_at_reselect": 3,
        "reselect_indices": [30],
        "purge_radius": 1,
    }


def test_residual_equity_cpcv_config_locked_geometry() -> None:
    cfg = residual_equity_cpcv_config()
    assert (cfg.n_splits, cfg.n_test_groups, cfg.purge_days, cfg.embargo_days) == (
        6,
        2,
        21,
        21,
    )


def test_selection_pit_same_day_boundary_is_dirty() -> None:
    st = selection_pit_status(
        universe_end="2021-06-15", eval_start="2021-06-15", phase="OOS"
    )
    assert st["pit_clean"] is False
    assert st["overlap_days"] == 1
    assert st["universe_protocol"] == "frozen"


def test_build_members_by_date_from_intervals_characterization() -> None:
    mdf = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA"],
            "start_date": ["2020-01-01", "2020-01-10", "2020-02-01"],
            "end_date": ["2020-01-31", "2020-01-20", pd.NaT],
        }
    )
    dlist = ["2020-01-05", "2020-01-15", "2020-02-05"]
    out = build_members_by_date_from_intervals(dlist, mdf)
    assert sorted(out["2020-01-05"]) == ["AAA"]
    assert sorted(out["2020-01-15"]) == ["AAA", "BBB"]
    assert sorted(out["2020-02-05"]) == ["AAA"]


def test_lookup_eligible_accepts_timestamp_keys() -> None:
    assert _lookup_eligible({"2020-01-05": [1, 2]}, "2020-01-05") == [1, 2]
    assert _lookup_eligible({pd.Timestamp("2020-01-05"): [9]}, "2020-01-05") == [9]
    assert _lookup_eligible({}, "2020-01-05") is None


@pytest.mark.unit
def test_differential_sharpe_nested_apply_not_idempotent() -> None:
    stream = [0.01, -0.005, 0.02, 0.0, -0.01, 0.015]
    single = DifferentialSharpe(eta=0.01)
    inner = DifferentialSharpe(eta=0.01)
    outer = DifferentialSharpe(eta=0.01)
    single_out = [single.step(r) for r in stream]
    nested_out = [outer.step(inner.step(r)) for r in stream]
    assert single_out != nested_out


@pytest.mark.unit
def test_reward_shaping_ablation_schema_stays_bool() -> None:
    spec = cell_schema.SCHEMA["reward_shaping_ablation"]
    assert spec.typ is bool
