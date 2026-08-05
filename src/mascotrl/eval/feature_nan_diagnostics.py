"""Feature-channel all-NaN diagnostics (fail-closed for admitted surface signals)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def feature_nan_diagnostics(
    cube: np.ndarray,
    *,
    channel_names: Sequence[str],
    admitted_channels: Sequence[str] | None = None,
    max_all_nan_frac: float = 0.20,
) -> dict[str, Any]:
    """Count all-NaN cross-sections per channel on a ``(T, K, C)`` feature cube.

    An admitted surface channel fails closed when more than ``max_all_nan_frac``
    of eval dates have an all-NaN cross-section (every name missing).
    """
    x = np.asarray(cube, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"feature cube must be (T, K, C); got shape {x.shape}")
    t, _k, c = x.shape
    names = [str(n) for n in channel_names]
    if len(names) != c:
        raise ValueError(f"channel_names length {len(names)} != C={c}")
    admitted = {str(n) for n in (admitted_channels or names)}
    per_channel: dict[str, Any] = {}
    fail_channels: list[str] = []
    for i, name in enumerate(names):
        # all-NaN across names on each date
        all_nan = np.all(~np.isfinite(x[:, :, i]), axis=1)
        n_bad = int(np.sum(all_nan))
        frac = float(n_bad / t) if t else float("nan")
        entry = {
            "n_dates": int(t),
            "n_all_nan_dates": n_bad,
            "all_nan_frac": frac,
            "admitted": name in admitted,
        }
        per_channel[name] = entry
        if name in admitted and np.isfinite(frac) and frac > float(max_all_nan_frac):
            fail_channels.append(name)
    return {
        "max_all_nan_frac": float(max_all_nan_frac),
        "per_channel": per_channel,
        "fail_channels": fail_channels,
        "pass": len(fail_channels) == 0,
    }


def assert_feature_nan_ok(diag: Mapping[str, Any]) -> None:
    """Raise ``SystemExit`` when admitted channels exceed the all-NaN budget."""
    if bool(diag.get("pass", False)):
        return
    fails = diag.get("fail_channels") or []
    raise SystemExit(
        "feature_nan_diagnostics fail-closed: admitted surface channel(s) "
        f"all-NaN on >{diag.get('max_all_nan_frac')} of eval dates: {fails}"
    )
