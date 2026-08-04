"""Tests for honest RD2 dominant-expert transition markers and RD3 runs."""
from __future__ import annotations

import numpy as np

from scripts.render_regime_desk_figures import archetype_runs, dominant_at_transitions


def test_dominant_at_transitions_uses_fs_names() -> None:
    turb = np.array([False, False, True, True, False, False], dtype=bool)
    dom = ["cheetah", "cheetah", "fox", "tortoise", "owl", "owl"]
    marks = dominant_at_transitions(turb, dom)
    assert marks
    # First transition at idx=1 (False->True), dominant at idx+1=2 -> fox
    assert marks[0] == (1, "fox")


def test_dominant_missing_returns_empty() -> None:
    turb = np.array([False, True, False], dtype=bool)
    assert dominant_at_transitions(turb, None) == []
    assert dominant_at_transitions(turb, ["a", "b"]) == []  # length mismatch


def test_never_invents_tortoise_cheetah_from_turb() -> None:
    turb = np.array([False] * 10 + [True] * 10, dtype=bool)
    dom = ["magpie"] * 20
    marks = dominant_at_transitions(turb, dom)
    assert all(name == "magpie" for _, name in marks)


def test_archetype_runs_basic() -> None:
    assert archetype_runs(["owl", "owl", "fox", "fox", "fox"]) == [
        (0, 2, "owl"),
        (2, 5, "fox"),
    ]
    assert archetype_runs(None) == []


def test_archetype_runs_ignore_turb_fiction() -> None:
    dom = ["magpie"] * 20
    runs = archetype_runs(dom)
    assert runs == [(0, 20, "magpie")]
