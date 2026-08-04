"""Tests for scripts/run_local_excluded_cells.py (excluded mamba cells)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_local_excluded_cells as runner


def _g0_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "AUDIT_LEDGER.md"
    path.write_text(f"# ledger\n\nSentinel: `{runner.G0_SENTINEL}`\n", encoding="utf-8")
    return path


def _mock_lake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    lake.mkdir()
    monkeypatch.setattr(runner, "check_lake_mounted", lambda: lake)
    return lake


def test_dry_run_lists_three_mamba_stems_with_out_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _g0_ledger(tmp_path)
    monkeypatch.setattr(runner, "AUDIT_LEDGER", ledger)
    _mock_lake(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "collect_env_snapshot",
        lambda **_: {
            "lake_root": "/mock/lake",
            "mem_available_gib": 32.0,
            "arctic_symbol_counts": {"opt100": 0, "mix100": 0},
        },
    )

    report = runner.dry_run_report(root=ROOT)
    stems = [c["stem"] for c in report["cells"]]
    assert stems == [
        "eq_K100_single_ppo_mamba_softmax_mean_std_cao",
        "eq_K200_single_ppo_mamba_softmax_mean_std_cao",
        "eq_K100_single_ppo_mamba_dirichlet_tilt_cvar_ru",
    ]
    out_dirs = [c["out_dir"] for c in report["cells"]]
    assert out_dirs == [
        str(ROOT / "logs/artifacts/spectrum/cherrypick"),
        str(ROOT / "logs/artifacts/spectrum/cherrypick/k200"),
        str(ROOT / "logs/artifacts/spectrum/cherrypick/narrative"),
    ]


def test_deferred_refused_while_arctic_empty() -> None:
    with pytest.raises(SystemExit, match="deferred_opt_mix_refused"):
        runner.check_deferred_opt_mix_allowed(
            allow_deferred_opt_mix=True,
            arctic_counts={"opt100": 0, "mix100": 0},
        )


def test_deferred_allowed_when_arctic_has_symbols() -> None:
    runner.check_deferred_opt_mix_allowed(
        allow_deferred_opt_mix=True,
        arctic_counts={"opt100": 3, "mix100": 0},
    )


def test_skip_if_complete_on_fixture_artifact(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    out.mkdir()
    stem = "eq_K100_single_ppo_mamba_softmax_mean_std_cao"
    art_path = out / f"{stem}.json"
    art_path.write_text(
        json.dumps(
            {
                "spectrum_cell_id": stem,
                "dry_run": False,
                "strict_degraded": False,
                "promotable": True,
            }
        ),
        encoding="utf-8",
    )
    assert runner.artifact_looks_complete(art_path) is True

    dry_art = out / "dry.json"
    dry_art.write_text(json.dumps({"spectrum_cell_id": "x", "dry_run": True}), encoding="utf-8")
    assert runner.artifact_looks_complete(dry_art) is False


def test_run_campaign_skips_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _g0_ledger(tmp_path)
    monkeypatch.setattr(runner, "AUDIT_LEDGER", ledger)
    _mock_lake(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "check_mem_available", lambda **_: 32.0)

    out = tmp_path / "cherrypick"
    out.mkdir()
    stem = runner.EXCLUDED_CELLS[0].stem
    (out / f"{stem}.json").write_text(
        json.dumps(
            {
                "spectrum_cell_id": stem,
                "dry_run": False,
                "strict_degraded": False,
                "promotable": True,
            }
        ),
        encoding="utf-8",
    )

    cells = [
        {
            "stem": stem,
            "config_dir": str(ROOT / runner.EXCLUDED_CELLS[0].config_dir_relpath),
            "config_glob": runner.EXCLUDED_CELLS[0].config_glob,
            "out_dir": str(out),
            "artifact_path": str(out / f"{stem}.json"),
        }
    ]
    monkeypatch.setattr(runner, "resolve_cells", lambda **_: cells)
    ledger_path = tmp_path / "campaign.jsonl"

    with patch.object(runner, "run_cell_subprocess") as mock_run:
        code = runner.run_campaign(root=ROOT, ledger_path=ledger_path)
        mock_run.assert_not_called()

    assert code == 0
    rows = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["skipped"] == "complete"
    assert row["stem"] == stem


def test_memory_guard_refusal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:       8388608 kB\n", encoding="utf-8")  # 8 GiB
    with pytest.raises(SystemExit, match="mem_guard"):
        runner.check_mem_available(min_gib=12.0, meminfo_path=meminfo)


def test_g0_sentinel_refusal_when_missing(tmp_path: Path) -> None:
    bad = tmp_path / "AUDIT_LEDGER.md"
    bad.write_text("no sentinel here\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        runner.check_g0_sentinel(bad)
    assert exc.value.code == 2


def test_main_dry_run_exit_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _g0_ledger(tmp_path)
    monkeypatch.setattr(runner, "AUDIT_LEDGER", ledger)
    _mock_lake(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "collect_env_snapshot",
        lambda **_: {"lake_root": "/mock", "mem_available_gib": 16.0},
    )
    assert runner.main(["--dry-run"]) == 0


def test_max_parallel_refused() -> None:
    with pytest.raises(SystemExit, match="max-parallel"):
        runner.main(["--no-dry-run", "--max-parallel", "2"])
