"""HAPPO USB macro inject helper (default off)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.spectrum.happo_macro_inject import (
    HAPPO_USB_COLS,
    happo_usb_macro_enabled,
    maybe_load_happo_macro,
)


def test_happo_usb_macro_default_false() -> None:
    assert happo_usb_macro_enabled({}) is False
    assert happo_usb_macro_enabled({"happo_usb_macro": False}) is False
    assert happo_usb_macro_enabled(None) is False


def test_maybe_load_returns_none_when_disabled(tmp_path: Path) -> None:
    assert maybe_load_happo_macro({"happo_usb_macro": False}, tmp_path) is None


def test_maybe_load_tensor_when_enabled(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    macro = tmp_path / "macro"
    macro.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=40)
    pd.DataFrame(
        {
            "date": dates,
            "vix": np.linspace(12, 20, 40),
            "vxn": np.linspace(13, 21, 40),
            "vxd": np.linspace(11, 19, 40),
        }
    ).to_parquet(macro / "cboe_vix.parquet")
    pd.DataFrame(
        {
            "date": dates,
            "sofr": np.linspace(0.1, 1.0, 40),
            "effr": np.linspace(0.1, 1.0, 40),
            "dtb3": np.linspace(0.05, 0.9, 40),
        }
    ).to_parquet(macro / "interest_rate.parquet")
    ten = maybe_load_happo_macro(
        {"happo_usb_macro": True}, tmp_path, n_rows=20
    )
    assert ten is not None
    assert tuple(ten.shape) == (20, len(HAPPO_USB_COLS))


def test_no_spectrum_yaml_enables_happo_usb_macro() -> None:
    root = Path(__file__).resolve().parents[1]
    spectrum = root / "config" / "spectrum"
    if not spectrum.is_dir():
        pytest.skip("no spectrum configs")
    hits = 0
    for path in spectrum.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "happo_usb_macro: true" in text or "happo_usb_macro: True" in text:
            hits += 1
    assert hits == 0
