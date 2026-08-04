from __future__ import annotations

from pathlib import Path

from scripts.build_landed_panel import RC6_WAVE_DIR_NAMES, rc6_wave_dirs


def test_rc6_wave_dirs_only_rc6_prefix(tmp_path: Path) -> None:
    (tmp_path / "rc6").mkdir()
    (tmp_path / "rc6_k200").mkdir()
    (tmp_path / "pick2").mkdir()
    (tmp_path / "cherrypick_deskorg").mkdir()
    waves = rc6_wave_dirs(tmp_path)
    assert set(waves) == {"RC6", "RC6_K200"}
    assert all(k.startswith("RC6") for k in waves)
    assert "pick2" not in {p.name for p in waves.values()}
    assert len(RC6_WAVE_DIR_NAMES) == 5
