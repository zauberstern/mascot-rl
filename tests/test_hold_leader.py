"""Tests for hold_leader and rolling_leader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mascotrl.eval.hold_leader import hold_leader, rolling_leader


def test_hold_leader_locks_dominant_after_review() -> None:
    t, n = 200, 3
    rng = np.random.default_rng(7)
    R = rng.normal(0.0002, 0.002, size=(t, n))
    R[:, 2] += 0.005
    hold = 50
    W = hold_leader(R, lookback=50, hold=hold, min_obs=20)
    assert W.shape == R.shape
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(W[:hold], np.ones((hold, n)) / n)
    assert int(np.argmax(W[hold])) == 2
    np.testing.assert_allclose(W[hold : 2 * hold], np.broadcast_to(W[hold], (hold, n)))


def test_hold_leader_prefix_stable() -> None:
    rng = np.random.default_rng(2)
    R = rng.normal(0.0005, 0.01, size=(180, 3))
    hold = 40
    w_full = hold_leader(R, lookback=40, hold=hold)
    t = 100
    w_pref = hold_leader(R[:t], lookback=40, hold=hold)
    np.testing.assert_allclose(w_full[:t], w_pref, rtol=1e-10, atol=1e-12)


def test_hold_leader_R_t_does_not_affect_W_t() -> None:
    rng = np.random.default_rng(3)
    R = rng.normal(0.0, 0.01, size=(120, 3))
    hold = 30
    w0 = hold_leader(R, lookback=30, hold=hold)
    day = 45  # mid-hold, not a review
    R2 = R.copy()
    R2[day] += 0.05
    w1 = hold_leader(R2, lookback=30, hold=hold)
    np.testing.assert_allclose(w0[day], w1[day], rtol=1e-12)


def test_hold_leader_switch_count_bounded() -> None:
    rng = np.random.default_rng(4)
    R = rng.normal(0.0, 0.02, size=(300, 4))
    hold = 60
    W = hold_leader(R, lookback=60, hold=hold)
    dom = np.argmax(W, axis=1)
    switches = int(np.sum(np.diff(dom) != 0))
    assert switches <= (300 - 1) // hold + 1


def test_rolling_leader_concentrates_on_dominant() -> None:
    t, n = 200, 3
    rng = np.random.default_rng(8)
    R = rng.normal(0.0002, 0.002, size=(t, n))
    R[:, 1] += 0.005
    W = rolling_leader(R, lookback=50, min_obs=20)
    assert int(np.argmax(W[150])) == 1


def test_hold_leader_no_turbulence_import() -> None:
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "mascotrl" / "eval" / "hold_leader.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "turbulence" not in alias.name
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "turbulence" not in mod
