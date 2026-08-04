"""Strict spectrum YAML loader preserves scr_mix: off as string."""
from __future__ import annotations

from pathlib import Path

from src.spectrum.yaml_loader import load_cell_yaml, load_cell_yaml_text


def test_scr_mix_off_stays_string_from_file(tmp_path: Path) -> None:
    p = tmp_path / "cell.yaml"
    p.write_text("scr_mix: off\nspectrum_cell_id: test\n", encoding="utf-8")
    cfg = load_cell_yaml(p)
    assert cfg["scr_mix"] == "off"
    assert cfg["scr_mix"] is not False


def test_scr_mix_off_stays_string_from_text() -> None:
    cfg = load_cell_yaml_text("scr_mix: off\n")
    assert cfg["scr_mix"] == "off"


def test_true_false_still_bool() -> None:
    cfg = load_cell_yaml_text("flag: true\nother: false\n")
    assert cfg["flag"] is True
    assert cfg["other"] is False
