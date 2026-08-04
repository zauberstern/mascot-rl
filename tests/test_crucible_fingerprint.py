"""CRUCIBLE fingerprint stability and OFAT schedule freeze invariance."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
import yaml
from src.data.crucible import CrucibleSpec, assert_ofat_cells_share_schedule_fingerprint, crucible_fingerprint, load_universe_schedule, schedule_fingerprint, write_universe_schedule

def _base_result_like(secids):
    return {'secids': list(secids), 'ff4_fit_hash': 'abc123', 'sleeve_defs_hash': 'sleevedefs01'}

def test_fingerprint_stable_across_secid_reorder():
    spec = CrucibleSpec()
    a = crucible_fingerprint(_base_result_like([3, 1, 2]), spec)
    b = crucible_fingerprint(_base_result_like([1, 2, 3]), spec)
    assert a == b
    assert len(a) == 64

def test_fingerprint_changes_when_locked_knob_changes():
    spec = CrucibleSpec()
    base = crucible_fingerprint(_base_result_like([1, 2, 3]), spec)
    other = crucible_fingerprint(_base_result_like([1, 2, 3]), CrucibleSpec(g1_l1_floor=0.09))
    assert base != other
    other2 = crucible_fingerprint(_base_result_like([1, 2, 3]), CrucibleSpec(reselect_every_days=64))
    assert base != other2

def test_ofat_cells_share_frozen_schedule_fingerprint(tmp_path: Path):
    """Real OFAT freeze: cells must load the same schedule_fingerprint."""
    dates = list(pd.bdate_range('2020-01-02', periods=5))
    slots = [[10, 20, 30], [10, 20, 30], [10, 21, 30], [10, 21, 30], [10, 21, 31]]
    paths = []
    for i in range(3):
        p = tmp_path / f'cell_{i}' / 'crucible_universe_schedule.json'
        write_universe_schedule(p, slots_rows=slots, dates=dates, fingerprint='sel_abc')
        paths.append(p)
    shared = assert_ofat_cells_share_schedule_fingerprint(paths)
    assert shared == schedule_fingerprint(slots)
    loaded = load_universe_schedule(paths[0])
    assert loaded['schedule_fingerprint'] == shared

def test_ofat_schedule_mismatch_fails_closed(tmp_path: Path):
    dates = list(pd.bdate_range('2020-01-02', periods=3))
    p1 = tmp_path / 'a.json'
    p2 = tmp_path / 'b.json'
    write_universe_schedule(p1, slots_rows=[[1, 2], [1, 2], [1, 3]], dates=dates, fingerprint='x')
    write_universe_schedule(p2, slots_rows=[[9, 8], [9, 8], [9, 7]], dates=dates, fingerprint='y')
    with pytest.raises(AssertionError, match='do not share'):
        assert_ofat_cells_share_schedule_fingerprint([p1, p2])
