"""Tests for desk gap diagnosis (oracle segments, lag-1 hit rates)."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.desk_gap_diagnosis import lag1_hit_rate, oracle_segments


def test_oracle_segments_two_runs() -> None:
    path = [0, 0, 1, 1, 1]
    names = ["a", "b"]
    segs = oracle_segments(path, names)
    assert len(segs) == 2
    assert segs[0] == {
        "start": 0,
        "end_exclusive": 2,
        "expert": "a",
        "expert_idx": 0,
        "n_days": 2,
    }
    assert segs[1] == {
        "start": 2,
        "end_exclusive": 5,
        "expert": "b",
        "expert_idx": 1,
        "n_days": 3,
    }


def test_oracle_segments_prefix_clips_last() -> None:
    path = [0, 0, 1, 1, 1]
    names = ["a", "b"]
    full = oracle_segments(path, names)
    pref = oracle_segments(path[:3], names)
    assert pref[0] == full[0]
    assert pref[1]["start"] == 2
    assert pref[1]["end_exclusive"] == 3
    assert pref[1]["n_days"] == 1


def test_lag1_hit_rate_skips_t0() -> None:
    # t=0 turb unused; t=1 uses turb[0], t=2 uses turb[1]
    turb = np.array([True, False, True], dtype=bool)
    R = np.array(
        [
            [0.0, 0.0],  # t=0 skipped
            [0.02, 0.01],  # turb[0]=True: spec>owl
            [0.00, 0.01],  # turb[1]=False: spec<owl
        ],
        dtype=np.float64,
    )
    out = lag1_hit_rate(turb, R, ["spec", "owl"], specialist="spec", owl="owl")
    assert out["n_turb_lag1"] == 1
    assert out["n_calm_lag1"] == 1
    assert out["hit_rate_turb"] == 1.0
    assert out["hit_rate_calm"] == 0.0
