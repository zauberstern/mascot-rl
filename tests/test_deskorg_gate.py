"""DESKORG prior-gate accepts complete watch even if index is stale."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aws_submit_wave import assert_deskorg_priors_complete


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deskorg_gate_prefers_complete_watch_over_stale_index(tmp_path: Path) -> None:
    root = tmp_path
    # Incomplete index would previously short-circuit the gate.
    _write(
        root / "logs/artifacts/spectrum/cherrypick_smoke/index.json",
        {"complete": False, "n_accepted": 0, "n_expected": 1},
    )
    _write(
        root / "logs/aws_burst_watch_PICK_SMOKE.json",
        {
            "polled_at": "2026-08-28T00:00:00+00:00",
            "complete": True,
            "n_errors": 0,
            "n_found": 1,
            "n_expected": 1,
        },
    )
    # Remaining priors: complete watches.
    for wave, subdir in [
        ("PICK", "cherrypick"),
        ("PICK2", "cherrypick/narrative"),
        ("K200", "cherrypick/k200"),
        ("FEATNET", "cherrypick_featnet"),
        ("HYBRID", "cherrypick_hybrid"),
    ]:
        _write(
            root / f"logs/aws_burst_watch_{wave}.json",
            {
                "polled_at": "2026-08-28T00:00:00+00:00",
                "complete": True,
                "n_errors": 0,
                "n_found": 1,
                "n_expected": 1,
            },
        )
        # Stale incomplete indexes must not block.
        _write(
            root / "logs/artifacts/spectrum" / subdir / "index.json",
            {"complete": False, "n_accepted": 0, "n_expected": 1},
        )

    assert_deskorg_priors_complete(root)


def test_deskorg_gate_fails_when_all_evidence_incomplete(tmp_path: Path) -> None:
    root = tmp_path
    for wave in ("PICK_SMOKE", "PICK", "PICK2", "K200", "FEATNET", "HYBRID"):
        _write(
            root / f"logs/aws_burst_watch_{wave}.json",
            {
                "polled_at": "2026-08-28T00:00:00+00:00",
                "complete": False,
                "n_errors": 1,
                "n_found": 0,
                "n_expected": 1,
            },
        )
    with pytest.raises(ValueError, match="deskorg_gate_prior_incomplete"):
        assert_deskorg_priors_complete(root)
