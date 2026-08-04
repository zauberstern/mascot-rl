"""Optional HAPPO USB macro series inject (designed 6-col path).

Default off. Does not admit fioracle onto the confirmatory train cube.
Campaign remains dispatch-only unless ``happo_usb_macro: true`` in cell YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


HAPPO_USB_VIX_COLS: tuple[str, ...] = ("vix", "vxn", "vxd")
HAPPO_USB_RATE_COLS: tuple[str, ...] = ("sofr", "effr", "dtb3")
HAPPO_USB_COLS: tuple[str, ...] = HAPPO_USB_VIX_COLS + HAPPO_USB_RATE_COLS


def happo_usb_macro_enabled(cfg: Mapping[str, Any] | None) -> bool:
    """True only when YAML explicitly sets ``happo_usb_macro: true``."""
    if not cfg:
        return False
    return bool(cfg.get("happo_usb_macro", False))


def maybe_load_happo_macro(
    cfg: Mapping[str, Any] | None,
    usb_root: Path | str | None,
    *,
    n_rows: int | None = None,
) -> Any | None:
    """Load designed USB VIX+rates tensor when enabled; else None.

    Returns a float32 torch.Tensor of shape ``(T, 6)`` or None. Never invents
    white-noise macro. Missing files → None (caller keeps zeros macro).
    """
    if not happo_usb_macro_enabled(cfg):
        return None
    if usb_root is None:
        return None
    root = Path(usb_root)
    vix_path = root / "macro" / "cboe_vix.parquet"
    ir_path = root / "macro" / "interest_rate.parquet"
    if not vix_path.is_file() or not ir_path.is_file():
        return None

    import pandas as pd
    import torch

    try:
        vix = pd.read_parquet(vix_path)
        ir = pd.read_parquet(ir_path)
    except Exception:
        return None

    def _dated(df: pd.DataFrame) -> pd.DataFrame:
        lower = {str(c).lower(): c for c in df.columns}
        date_col = lower.get("date")
        if date_col is None:
            raise ValueError("missing date")
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col)
        return out.set_index(date_col)

    try:
        vix_i = _dated(vix)
        ir_i = _dated(ir)
    except Exception:
        return None

    lower_v = {str(c).lower(): c for c in vix_i.columns}
    lower_r = {str(c).lower(): c for c in ir_i.columns}
    pieces: list[pd.Series] = []
    for name in HAPPO_USB_VIX_COLS:
        col = lower_v.get(name)
        if col is None:
            return None
        pieces.append(pd.to_numeric(vix_i[col], errors="coerce").rename(name))
    for name in HAPPO_USB_RATE_COLS:
        col = lower_r.get(name)
        if col is None:
            return None
        pieces.append(pd.to_numeric(ir_i[col], errors="coerce").rename(name))

    frame = pd.concat(pieces, axis=1).sort_index().ffill()
    # Align calendars: inner join on dates present in VIX (core).
    frame = frame.dropna(how="all")
    if frame.empty:
        return None
    if n_rows is not None and n_rows > 0:
        frame = frame.iloc[-int(n_rows) :]
    arr = frame.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(arr, dtype=torch.float32)
